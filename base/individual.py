# base/individual.py
import copy, logging
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx

import tensorflow as tf
keras = tf.keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

from .shared_layers import REGISTRY, default_key
from .config import basepath
from .structures import Blueprint

basepath = "./"


class Individual:
    """
    A single candidate network built from a Blueprint.
    Handles assembly, training, and scoring with live shared weights.

    Sharing mode is controlled by self.population.module_share_mode (if present):
      - "module"  : share per module species (extra = (species_id, node_tag))
      - "layer"   : share per node tag only (extra = (node_tag,))
      - "global"  : share across all (extra = ())
      - "off"     : disable sharing (unique extra so keys never collide)
    """

    def __init__(self, blueprint: Blueprint, compiler=None, birth=None,
                 model=None, name=None, population=None):
        self.blueprint = blueprint
        self.compiler = compiler
        self.birth = birth
        self.model = model
        self.name = name
        self.scores = [0, 0]
        self.population = population  # provides module_share_mode and datasets

    # -----------------------------
    # helpers: label mode / num_classes / head
    # -----------------------------
    def _infer_num_classes(self) -> int:
        """Infer number of classes from population datasets."""
        try:
            y = np.asarray(self.population.datasets.training[1])
            if y.ndim >= 2 and y.shape[-1] > 1:
                return int(y.shape[-1])  # one-hot
            return int(np.max(y)) + 1    # integer ids
        except Exception:
            logging.warning("[Individual] Could not infer num_classes; defaulting to 10.")
            return 10

    def _infer_label_mode_and_loss(self):
        """Return ('onehot'|'sparse', loss_obj)."""
        try:
            y = np.asarray(self.population.datasets.training[1])
            if y.ndim >= 2 and y.shape[-1] > 1:
                return "onehot", keras.losses.CategoricalCrossentropy()
            else:
                return "sparse", keras.losses.SparseCategoricalCrossentropy()
        except Exception:
            return "onehot", keras.losses.CategoricalCrossentropy()

    def _ensure_classifier_head(self, x, num_classes: int):
        """Append GAP/Flatten + Dense(num_classes, softmax)."""
        if len(x.shape) == 4:
            x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
        elif len(x.shape) > 2:
            x = keras.layers.Flatten(name="flatten")(x)
        x = keras.layers.Dense(num_classes, name="logits")(x)
        x = keras.layers.Activation("softmax", name="predictions")(x)
        return x

    # -----------------------------
    # live-share layer factory
    # -----------------------------
    def _share_extra_scope(self, species_id, module_node_tag):
        """
        Compute the 'extra' tuple for the registry key based on the selected sharing mode.
        """
        mode = getattr(self.population, "module_share_mode", "module")

        if mode == "module":
            return (species_id, str(module_node_tag))
        if mode == "layer":
            return (str(module_node_tag),)
        if mode == "global":
            return ()
        if mode == "off":
            return ("nosync", id(self), str(module_node_tag))
        return (species_id, str(module_node_tag))

    def _make_shared_layer(self, comp, inbound_tensor, species_id, module_node_tag):
        """
        Return a keras Layer object. If the (kind, in_ch, params, extra) key repeats,
        everyone gets the SAME object -> live shared variables (unless sharing is "off").
        """
        comp_type = comp.component_type  # "conv2d", "dense", "bn", "maxpool", etc.
        comp_def  = comp.representation[0]
        params    = dict(comp.representation[1])  # copy to avoid mutation

        # Decide 'extra' sharing scope
        extra = self._share_extra_scope(species_id, module_node_tag)

        logging.log(21, f"[LIVE-SHARE][mode={getattr(self.population, 'module_share_mode', 'module')}] extra={extra}")

        # Map shorthand → keras args and pick input channels from inbound
        shape = K.int_shape(inbound_tensor)
        in_ch = shape[-1] if shape and shape[-1] is not None else -1

        # --- Recognize common shorthand types and share them (BN and Pooling included) ---
        if comp_type == "conv2d":
            if "kernel" in params:
                params["kernel_size"] = params.pop("kernel")
            if "stride" in params:
                params["strides"] = params.pop("stride")
            params.setdefault("padding", "same")
            params.setdefault("activation", None)

            key = default_key("conv2d", in_ch, params, extra=extra)
            if extra and extra[0] == "nosync":
                layer = keras.layers.Conv2D(**params)
            else:
                layer = REGISTRY.get_or_create(key, lambda: keras.layers.Conv2D(**params))

        elif comp_type == "dense":
            params.setdefault("activation", "relu")
            key = default_key("dense", in_ch, params, extra=extra)
            if extra and extra[0] == "nosync":
                layer = keras.layers.Dense(**params)
            else:
                layer = REGISTRY.get_or_create(key, lambda: keras.layers.Dense(**params))

        elif comp_type in ("bn", "batch_norm", "BatchNormalization", "batchnormalization"):
            # BatchNorm has trainables -> allow sharing
            p = {"momentum": 0.99, "epsilon": 1e-3}
            p.update(params)
            key = default_key("batch_norm", in_ch, p, extra=extra)
            if extra and extra[0] == "nosync":
                layer = keras.layers.BatchNormalization(**p)
            else:
                layer = REGISTRY.get_or_create(key, lambda: keras.layers.BatchNormalization(**p))

        elif comp_type in ("maxpool", "max_pooling2d", "MaxPooling2D", "maxpool2d"):
            # MaxPool is stateless, but we keep registry symmetry
            p = {"pool_size": (2, 2), "strides": 2, "padding": "valid"}
            p.update(params)
            key = default_key("max_pooling2d", in_ch, p, extra=extra)
            if extra and extra[0] == "nosync":
                layer = keras.layers.MaxPooling2D(**p)
            else:
                layer = REGISTRY.get_or_create(key, lambda: keras.layers.MaxPooling2D(**p))

        else:
            # Fallback: use the provided keras class (no sharing)
            # e.g., Flatten, Dropout, etc.
            layer = comp_def(**params)

        # Tag for diagnostics
        setattr(layer, "_species_id", species_id)
        setattr(layer, "_share_mode", getattr(self.population, "module_share_mode", "module"))
        setattr(layer, "_share_extra", extra)
        return layer

    # -----------------------------
    # assembly / training / scoring
    # -----------------------------
    def generate(self, save_fig=False, generation=""):
        """
        Assemble Keras model from blueprint graphs.
        Uses _make_shared_layer to reuse layer objects (live sharing).
        """
        logging.log(21, f"Starting assembling of blueprint {self.blueprint.mark}.")
        module_graph = self.blueprint.module_graph
        layer_map = {}

        # union component subgraphs
        assembled = nx.DiGraph()
        out_nodes = {}
        for node in module_graph.nodes():
            sub = module_graph.nodes[node]["node_def"].component_graph
            assembled = nx.union(assembled, sub, rename=(None, f'{node}-'))
            order = list(nx.algorithms.dag.topological_sort(sub))
            out_nodes[node] = order[-1] if order else None

        # connect module outputs to successors’ inputs
        for node in module_graph.nodes():
            for suc in module_graph.successors(node):
                if out_nodes[node] is not None:
                    assembled.add_edge(f'{node}-{out_nodes[node]}', f'{suc}-0')

        logging.log(21, f"Generated assembled graph for blueprint {self.blueprint.mark}: {list(assembled.nodes())}")

        # keras input
        model_input = keras.layers.Input(shape=self.blueprint.input_shape)
        logging.log(21, f"Added Input layer: {model_input}")

        # build layers in topo order
        node_order = list(nx.algorithms.dag.topological_sort(assembled))
        for cid in node_order:
            module_node = str(cid).rsplit('-', 1)[0]
            try:
                mod_obj = module_graph.nodes[module_node]["node_def"]
                sid = mod_obj.species.name if getattr(mod_obj, "species", None) is not None else None
            except Exception:
                sid = None

            comp = copy.deepcopy(assembled.nodes[cid]["node_def"])
            indeg = assembled.in_degree(cid)
            layer_stack = []

            # optional complementary component
            if comp.complementary_component is not None:
                c_def, c_param = comp.complementary_component
                comp.keras_complementary_component = c_def(**c_param)
                comp.keras_complementary_component._species_id = sid

            if indeg == 0:
                inbound = model_input
                if comp.component_type == "dense" and len(inbound.shape) >= 3:
                    inbound = keras.layers.Flatten()(inbound)
                L = self._make_shared_layer(comp, inbound, sid, module_node)
                out = L(inbound)
                layer_stack = [out]

            elif indeg == 1:
                preds = [layer_map[p] for p in assembled.predecessors(cid)][0]
                inbound = preds[-1]
                if comp.component_type == "dense" and len(inbound.shape) == 4:
                    inbound = keras.layers.Flatten()(inbound)
                L = self._make_shared_layer(comp, inbound, sid, module_node)
                out = L(inbound)
                layer_stack = [out]

            elif indeg == 2:
                preds = [layer_map[p] for p in assembled.predecessors(cid)]
                p0 = preds[0][-1]; p1 = preds[1][-1]
                if comp.component_type == "dense":
                    if len(p0.shape) == 4: p0 = keras.layers.Flatten()(p0)
                    if len(p1.shape) == 4: p1 = keras.layers.Flatten()(p1)
                merged = keras.layers.concatenate([p0, p1])
                L = self._make_shared_layer(comp, merged, sid, module_node)
                out = L(merged)
                layer_stack = [out]

            else:
                preds = [layer_map[p][-1] for p in assembled.predecessors(cid)]
                if comp.component_type == "dense":
                    preds = [keras.layers.Flatten()(t) if len(t.shape) == 4 else t for t in preds]
                merged = keras.layers.concatenate(preds)
                L = self._make_shared_layer(comp, merged, sid, module_node)
                out = L(merged)
                layer_stack = [out]

            # add complementary on top if exists
            if comp.complementary_component is not None:
                layer_stack.append(comp.keras_complementary_component(layer_stack[-1]))

            layer_map[cid] = layer_stack

        # finalize model: append classifier head to produce (batch, num_classes)
        last_key = node_order[-1] if node_order else None
        outputs = layer_map[last_key][-1] if last_key is not None else model_input
        num_classes = self._infer_num_classes()
        outputs = self._ensure_classifier_head(outputs, num_classes=num_classes)

        self.model = keras.models.Model(inputs=model_input, outputs=outputs)

        # compile: prefer provided compiler; if loss == 'auto' or None, pick from labels
        if self.compiler:
            compiler_cfg = dict(self.compiler)
            loss_cfg = compiler_cfg.get("loss", "auto")
            if loss_cfg in (None, "auto"):
                _, loss_obj = self._infer_label_mode_and_loss()
                compiler_cfg["loss"] = loss_obj
            # default optimizer/metrics if missing
            compiler_cfg.setdefault("optimizer", keras.optimizers.Adam(learning_rate=0.001))
            compiler_cfg.setdefault("metrics", ["accuracy"])
            self.model.compile(**compiler_cfg)
        else:
            # sensible defaults
            _, loss_obj = self._infer_label_mode_and_loss()
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=loss_obj,
                metrics=["accuracy"],
            )

    def step_on_batch(self, xb, yb):
        """One micro-batch SGD step (uses compiled optimizer)."""
        return self.model.train_on_batch(xb, yb)

    def fit(self, input_x, input_y, training_epochs=1, validation_split=0.15,
            current_generation="", custom_fit_args=None):
        logging.info(f"Fitting one individual for {training_epochs} epochs")
        self.generate(generation=current_generation)
        if custom_fit_args is not None:
            # TF2 uses fit() for arrays and generators
            fitness = self.model.fit(**custom_fit_args)
        else:
            fitness = self.model.fit(
                input_x, input_y, epochs=training_epochs,
                validation_split=validation_split, batch_size=128, verbose=1
            )
        logging.info(f"Fitness for individual {self.name} using blueprint {self.blueprint.mark} "
                     f"after {training_epochs} epochs: {fitness.history}")
        return fitness

    def score(self, test_x, test_y):
        logging.info("Scoring one individual")
        scores = self.model.evaluate(test_x, test_y, verbose=1)
        self.blueprint.update_scores(scores)
        self.scores = scores
        logging.info(f"Test scores for individual {self.name} using blueprint {self.blueprint.mark}: {scores}")
        return scores

    def extract_blueprint(self):
        """(stub) back-extract a Blueprint from self.model if needed."""
        raise NotImplementedError

