# structures.py
import copy
import logging
from enum import Enum
import random
from typing import List, Any, Tuple, Optional

import numpy as np
import networkx as nx


class HistoricalMarker:
    def __init__(self):
        self.module_counter = 0
        self.blueprint_counter = 0
        self.individual_counter = 0

    def mark_module(self):
        self.module_counter += 1
        return self.module_counter

    def mark_blueprint(self):
        self.blueprint_counter += 1
        return self.blueprint_counter

    def mark_individual(self):
        self.individual_counter += 1
        return self.individual_counter


class NameGenerator:
    def __init__(self):
        self.counter = 0

    def generate(self):
        self.counter += 1
        return self.counter


class ModuleComposition(Enum):
    INPUT = "input"
    INTERMED = "intermed"
    CONV = "conv2d"
    DENSE = "dense"
    OUTPUT = "output"
    COMPILER = "compiler"


class Datasets:
    def __init__(self, complete=None, test=None, training=None, validation=None):
        self.complete = complete
        self.training = training
        self.validation = validation
        self.test = test
        self.custom_fit_args = None
        # Defensive defaults
        self.SAMPLE_SIZE = len(training[0]) if training is not None else 0
        self.TEST_SAMPLE_SIZE = len(test[0]) if test is not None else 0

    @property
    def shape(self):
        return getattr(self.complete, "shape", None)

    def split_complete(self):
        pass  # unchanged


class Component:
    """
    representation = [keras_layer_cls, params_dict]
    component_type = a tag like "conv2d", "dense", "maxpool", etc. (from your runner dict keys)
    """
    def __init__(
        self,
        representation: Optional[List[Any]],
        keras_component=None,
        complementary_component=None,
        keras_complementary_component=None,
        component_type: Optional[str] = None,
    ):
        self.representation = representation
        self.keras_component = keras_component
        self.complementary_component = complementary_component
        self.keras_complementary_component = keras_complementary_component
        self.component_type = component_type

    def _params(self) -> dict:
        if isinstance(self.representation, (list, tuple)) and len(self.representation) >= 2:
            return self.representation[1] or {}
        return {}

    def get_layer_size(self) -> int:
        """
        Return a non-negative integer proxy for the layer's "size".
        Must NEVER return None.
        """
        params = self._params()
        ctype = (self.component_type or "").lower()

        # Convolution: use number of filters
        if ctype in {"conv2d", "conv"}:
            return int(params.get("filters", 0) or 0)

        # Dense: units
        if ctype in {"dense", "fc"}:
            return int(params.get("units", 0) or 0)

        # Pooling (Max/Avg): use kernel area as a proxy
        if ctype in {"maxpooling2d", "avgpooling2d", "averagepooling2d", "maxpool", "avgpool"}:
            k = params.get("pool_size") or params.get("kernel_size") or params.get("kernel")
            if isinstance(k, (list, tuple)) and len(k) >= 2:
                return int((k[0] or 0) * (k[1] or 0))
            if isinstance(k, int):
                return int(k * k)
            return 0

        # Global Average Pooling: 1 channel-wise aggregate
        if ctype in {"globalaveragepooling2d", "gap"}:
            return 1

        # Flatten: proxy 0 (no params)
        if ctype in {"flatten"}:
            return 0

        # Dropout: scale by rate (0..1) * 100 for a small contribution
        if ctype in {"dropout"}:
            rate = params.get("rate", 0.0) or 0.0
            try:
                return int(max(0.0, float(rate)) * 100)
            except Exception:
                return 0

        # BatchNorm: small constant proxy
        if ctype in {"batchnormalization", "batch_norm", "batchnorm"}:
            return 1

        # Unknown layer types: be safe, contribute 0
        return 0


class Module:
    def __init__(
        self,
        components: Optional[dict],
        layer_type: ModuleComposition = ModuleComposition.INPUT,
        mark=None,
        component_graph: Optional[nx.DiGraph] = None,
        parents=None,
    ):
        self.components = components
        self.component_graph = component_graph
        self.layer_type = layer_type
        self.mark = mark
        self.weighted_scores = [99, 0]
        self.score_log: List[Tuple[float, float]] = []
        self.species = None
        self.parents = parents
        self.use_count = 0

    def __getitem__(self, item):
        return self.components[item]

    def get_module_size(self) -> int:
        s = 0
        if self.component_graph is None:
            return 0
        for node in self.component_graph.nodes():
            nd = self.component_graph.nodes[node].get("node_def", None)
            if isinstance(nd, Component):
                try:
                    s += int(nd.get_layer_size() or 0)
                except Exception:
                    # If anything goes wrong, treat as 0 to avoid crashing speciation
                    s += 0
            else:
                # missing/invalid node_def; treat as 0
                s += 0
        return int(s)

    def get_kmeans_representation(self) -> Tuple[int, int, int]:
        n = len(self.component_graph.nodes()) if self.component_graph is not None else 0
        e = len(self.component_graph.edges()) if self.component_graph is not None else 0
        return n, e, self.get_module_size()

    def update_scores(self, scores):
        self.score_log.append(scores)

    def update_weighted_scores(self):
        if len(self.score_log) > 0:
            arr = np.array(self.score_log)
            self.weighted_scores = [arr[:, 0].mean(), arr[:, 1].mean()]

    def simple_crossover(self, parent_2, mark):
        g1 = self.component_graph
        g2 = parent_2.component_graph
        child = self.component_graph.copy()
        nodes = list(child.nodes())
        g1_nodes = list(g1.nodes())
        g2_nodes = list(g2.nodes())
        for i, nid in enumerate(nodes):
            p1 = g1.nodes[g1_nodes[i]].get("node_def", None)
            # if parent2 shorter, sample a random node from it
            p2 = (
                g2.nodes[g2_nodes[i]].get("node_def", None)
                if i < len(g2_nodes)
                else g2.nodes[random.choice(g2_nodes)].get("node_def", None)
            )
            if isinstance(p1, Component) and isinstance(p2, Component) and (p1.component_type == p2.component_type):
                child.nodes[nid]["node_def"] = copy.deepcopy(random.choice([p1, p2]))
        return Module(None, layer_type=self.layer_type, mark=mark, component_graph=child)


class Blueprint:
    def __init__(self, modules: List[Module], input_shape=None, module_graph: Optional[nx.DiGraph] = None, mark=None, parents=None):
        self.modules = modules
        self.input_shape = input_shape
        self.module_graph = module_graph
        self.mark = mark
        self.weighted_scores = [99, 0]
        self.score_log: List[Tuple[float, float]] = []
        self.species = None
        self.parents = parents
        self.use_count = 0

    def __getitem__(self, item):
        return self.modules[item]

    def get_blueprint_size(self) -> int:
        s = 0
        if self.module_graph is None:
            return 0
        for node in self.module_graph.nodes():
            md = self.module_graph.nodes[node].get("node_def", None)
            if isinstance(md, Module):
                try:
                    s += int(md.get_module_size() or 0)
                except Exception:
                    s += 0
            else:
                s += 0
        return int(s)

    def get_kmeans_representation(self) -> Tuple[int, int, int]:
        n = len(self.module_graph.nodes()) if self.module_graph is not None else 0
        e = len(self.module_graph.edges()) if self.module_graph is not None else 0
        return n, e, self.get_blueprint_size()

    def update_scores(self, scores):
        self.score_log.append(scores)
        if self.module_graph is None:
            return
        for node in self.module_graph.nodes():
            md = self.module_graph.nodes[node].get("node_def", None)
            if isinstance(md, Module):
                md.update_scores(scores)

    def update_weighted_scores(self):
        if len(self.score_log) > 0:
            arr = np.array(self.score_log)
            self.weighted_scores = [arr[:, 0].mean(), arr[:, 1].mean()]

    def simple_crossover(self, parent_2, mark):
        g1 = self.module_graph
        g2 = parent_2.module_graph
        child = self.module_graph.copy()
        nlist = list(child.nodes())
        g1n = list(g1.nodes())
        g2n = list(g2.nodes())
        for i, nid in enumerate(nlist):
            p1 = g1.nodes[g1n[i]].get("node_def", None)
            p2 = (
                g2.nodes[g2n[i]].get("node_def", None)
                if i < len(g2n)
                else g2.nodes[random.choice(g2n)].get("node_def", None)
            )
            if isinstance(p1, Module) and isinstance(p2, Module) and (p1.layer_type == p2.layer_type):
                child.nodes[nid]["node_def"] = copy.deepcopy(random.choice([p1, p2]))
        return Blueprint(None, input_shape=self.input_shape, module_graph=child, mark=mark)


class Species:
    def __init__(
        self,
        name=None,
        species_type=None,
        group=None,
        properties=None,
        starting_generation=None,
    ):
        self.name = name
        self.species_type = species_type
        self.group = group
        self.properties = properties
        self.starting_generation = starting_generation
        self.weighted_scores = [99, 0]

    def update_weighted_scores(self):
        if self.group and len(self.group) > 0:
            ws = [it.weighted_scores for it in self.group if getattr(it, "weighted_scores", [99, 0]) != [99, 0]]
            if ws:
                arr = np.array(ws)
                self.weighted_scores = [arr[:, 0].mean(), arr[:, 1].mean()]
                logging.log(21, f"Updated weighted scores for species {self.name}: {self.weighted_scores}")
