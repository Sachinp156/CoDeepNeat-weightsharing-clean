# base/population.py
import os
import copy
import math
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import tensorflow as tf
keras = tf.keras
from tensorflow.keras import layers as KL
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from base.shared_layers import REGISTRY

# local
from base.structures import (
    HistoricalMarker,
    NameGenerator,
    ModuleComposition,
    Datasets,
    Component,
    Module,
    Blueprint,
    Species,
)
from base.shared_layers import REGISTRY
from base.config import LOG_DETAIL  # custom logging level 21

# ---------------------------
# Helpers
# ---------------------------

RNG = random.Random(1337)


def _ensure_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


def _sorted_items(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Stable param tuple for hashing keys."""
    return tuple(sorted(d.items(), key=lambda x: x[0]))


def _layer_key(
    component_type: str,
    in_channels: Optional[int],
    params: Dict[str, Any],
    extra_scope: Optional[Tuple[Any, Any]],
    share_mode: str = "module",
) -> Tuple:
    """
    Build a deterministic sharing key like your logs:
    ('conv2d', 3, (('activation','relu'),('filters',56)...), (species_id,'intermed-0')).
    """
    params_tuple = _sorted_items(params)
    if share_mode == "off":
        return (component_type, in_channels, params_tuple, ("nonce", id(object())))
    if share_mode == "global":
        return (component_type, in_channels, params_tuple, None)
    if share_mode == "layer":
        # only layer tagging
        label = extra_scope[1] if extra_scope is not None else None
        return (component_type, in_channels, params_tuple, ("L", label))
    # default: module scope -> (module_species_id, layer_tag)
    return (component_type, in_channels, params_tuple, extra_scope)


def _key_to_safe_name(key: Tuple) -> str:
    """Deterministic, unique, short layer name from the share key."""
    h = abs(hash(key)) % 10**10
    base = str(key[0]).replace(" ", "_")
    return f"{base}__{h}"


def _safe_eval_optimizer(opt_str: Any):
    """
    Evaluate legacy optimizer strings safely.
    Supports patterns like: 'keras.optimizers.Adam(lr=0.005)'
    or returns the object unchanged if already an optimizer.
    """
    if not isinstance(opt_str, str):
        return opt_str
    safe_globals = {
        "keras": keras,
        "tf": tf,
    }
    opt_str = opt_str.replace("lr=", "learning_rate=")
    try:
        return eval(opt_str, safe_globals, {})
    except Exception as e:
        logging.warning(f"[OPTIMIZER] Could not eval '{opt_str}': {e}. Falling back to Adam(0.001).")
        return keras.optimizers.Adam(learning_rate=0.001)


def _make_layer(component_type: str, params: Dict[str, Any], *, name: Optional[str] = None) -> KL.Layer:
    """
    Create a fresh Keras layer from a component definition.
    Accept an explicit `name` to guarantee uniqueness inside a model graph.
    """
    def make(cls, p):
        return cls(name=name, **p) if name is not None else cls(**p)

    if component_type == "conv2d":
        return make(KL.Conv2D, params)
    if component_type == "dense":
        return make(KL.Dense, params)
    if component_type in ("dropout", "Dropout"):
        return make(KL.Dropout, params)
    if component_type in ("batch_norm", "BatchNormalization", "batchnormalization", "bn"):
        p = {"momentum": 0.99, "epsilon": 1e-3}
        p.update(params)
        return make(KL.BatchNormalization, p)
    if component_type in ("max_pooling2d", "MaxPooling2D", "maxpool2d", "maxpool"):
        p = {"pool_size": (2, 2), "strides": 2, "padding": "valid"}
        p.update(params)
        return make(KL.MaxPooling2D, p)

    logging.warning(f"[LAYER] Unknown component_type '{component_type}', defaulting to Identity via Lambda.")
    return make(KL.Lambda, {"function": lambda x: x})


def _get_or_create_shared_layer(
    component_type: str,
    in_channels: Optional[int],
    params: Dict[str, Any],
    extra_scope: Optional[Tuple[Any, Any]],
    share_mode: str,
) -> KL.Layer:
    """
    Return a layer with live sharing. Creates + stores in REGISTRY on first use.
    Uses a deterministic unique name to avoid Keras name collisions.
    """
    key = _layer_key(component_type, in_channels, params, extra_scope, share_mode)
    if key in REGISTRY:
        logging.log(LOG_DETAIL, f"[LIVE-SHARE] REUSE  key={key} layer_id={id(REGISTRY[key])}")
        return REGISTRY[key]
    lname = _key_to_safe_name(key)
    layer = _make_layer(component_type, params, name=lname)
    REGISTRY[key] = layer
    logging.log(LOG_DETAIL, f"[LIVE-SHARE] CREATE key={key} name={lname} layer_id={id(layer)}")
    return layer


def _infer_in_channels_from_tensor(x):
    # channels_last assumption
    try:
        return int(x.shape[-1])
    except Exception:
        return None


# ---------------------------
# Gradient conflict utilities (PCGrad-style for shared vars)
# ---------------------------

def _flatten(t: tf.Tensor) -> tf.Tensor:
    return tf.reshape(t, [-1]) if t is not None else None


def _cosine(a: tf.Tensor, b: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
    a_f = _flatten(a)
    b_f = _flatten(b)
    if a_f is None or b_f is None:
        return tf.constant(float("nan"), dtype=tf.float32)
    na = tf.norm(a_f)
    nb = tf.norm(b_f)
    denom = tf.maximum(na * nb, eps)
    return tf.reduce_sum(a_f * b_f) / denom


def _pcgrad_pairwise(grads_i: List[Optional[tf.Tensor]], grads_j: List[Optional[tf.Tensor]]):
    """
    Project grads_i to avoid conflict with grads_j:
      g_i <- g_i - proj_{g_j}(g_i)  if <g_i, g_j> < 0
    Only projects on entries where both have a tensor (shared variables).
    """
    out = []
    for gi, gj in zip(grads_i, grads_j):
        if gi is None or gj is None:
            out.append(gi)
            continue
        gi_f = _flatten(gi)
        gj_f = _flatten(gj)
        dot = tf.reduce_sum(gi_f * gj_f)
        if dot < 0:
            denom = tf.maximum(tf.reduce_sum(gj_f * gj_f), 1e-12)
            proj = (dot / denom) * gj_f
            gi_new = tf.reshape(gi_f - proj, tf.shape(gi))
            out.append(gi_new)
        else:
            out.append(gi)
    return out


# NEW: map each variable id to the list of model indices that own it
def _shared_var_owners(models: List[keras.Model]) -> Dict[int, List[int]]:
    """
    Map id(var) -> list of model indices that own that exact Variable object.
    If a var appears in 2+ models, it's a candidate for conflict resolution.
    """
    var2owners: Dict[int, List[int]] = {}
    for mi, m in enumerate(models):
        for v in m.trainable_variables:
            vid = id(v)
            var2owners.setdefault(vid, []).append(mi)
    return var2owners


# NEW: align any collection of grads by a provided var-id list
def _align_grads_on_ids(
    models: List[keras.Model],
    per_model_grads: List[List[Optional[tf.Tensor]]],
    var_ids: List[int],
) -> List[List[Optional[tf.Tensor]]]:
    """
    For each model, build id(var)->grad, then align to the provided var_ids list.
    Missing entries become None.
    """
    dicts = []
    for m, grads in zip(models, per_model_grads):
        d = {}
        for v, g in zip(m.trainable_variables, grads):
            d[id(v)] = g
        dicts.append(d)
    aligned = [[d.get(vid, None) for vid in var_ids] for d in dicts]
    return aligned


def _pairwise_conflict_report_from_aligned(aligned_grads: List[List[Optional[tf.Tensor]]]) -> Dict[str, float]:
    """
    aligned_grads: per-model list of gradients aligned on the same var-id list.
    We count pairs only among models that actually share the variable at a slot.
    """
    num_models = len(aligned_grads)
    if num_models < 2 or not aligned_grads or not aligned_grads[0]:
        return {"pairs_evaluated": 0.0, "pct_negative": 0.0, "neg_pairs": 0}

    total = 0
    neg = 0
    L = len(aligned_grads[0])
    for k in range(L):
        owners = [i for i in range(num_models) if aligned_grads[i][k] is not None]
        for a in range(len(owners)):
            for b in range(a + 1, len(owners)):
                gi = aligned_grads[owners[a]][k]
                gj = aligned_grads[owners[b]][k]
                c = _cosine(gi, gj).numpy()
                if not np.isfinite(c):
                    continue
                total += 1
                if c < 0:
                    neg += 1

    pct = (100.0 * neg / total) if total > 0 else 0.0
    return {"pairs_evaluated": float(total), "pct_negative": float(pct), "neg_pairs": int(neg)}


def _varid_to_layername_map() -> Dict[int, str]:
    """
    Map id(variable) -> layer.name using the shared-layer REGISTRY.
    This lets us aggregate conflict stats per shared layer.
    """
    mapping = {}
    for layer in REGISTRY.values():
        _ = layer.weights  # ensure built
        for w in layer.weights:
            mapping[id(w)] = layer.name
    return mapping


def _per_layer_conflict_table(shared_keys: List[int], aligned: List[List[Optional[tf.Tensor]]]) -> List[Tuple[str, int, int, float]]:
    """
    Build per-layer conflict counts using the REGISTRY var->layer map.
    Returns a list of rows: (layer_name, pairs, neg, pct_neg)
    """
    id2lname = _varid_to_layername_map()
    num_models = len(aligned)
    agg: Dict[str, List[int]] = {}

    if num_models < 2 or not shared_keys:
        return []

    for pos, var_id in enumerate(shared_keys):
        lname = id2lname.get(var_id, f"var_{var_id}")
        total = 0
        neg = 0
        gks = [aligned[i][pos] for i in range(num_models)]
        owners = [i for i in range(num_models) if gks[i] is not None]
        if len(owners) < 2:
            continue
        for i_idx in range(len(owners)):
            for j_idx in range(i_idx + 1, len(owners)):
                i = owners[i_idx]
                j = owners[j_idx]
                c = _cosine(gks[i], gks[j]).numpy()
                if not np.isfinite(c):
                    continue
                total += 1
                if c < 0:
                    neg += 1
        if total > 0:
            if lname not in agg:
                agg[lname] = [0, 0]
            agg[lname][0] += total
            agg[lname][1] += neg

    rows = []
    for lname, (pairs, neg) in sorted(agg.items(), key=lambda x: (-x[1][0], x[0])):
        pct = (100.0 * neg / pairs) if pairs > 0 else 0.0
        rows.append((lname, pairs, neg, pct))
    return rows


def _print_per_layer_table(rows: List[Tuple[str, int, int, float]]):
    w = 40
    print("\n{:<{w}} | {:>7} | {:>7} | {:>6}".format("Layer", "Pairs", "Neg", "%Neg", w=w))
    print("-" * (w + 29))
    if not rows:
        print("(no shared-variable pairs evaluated)")
        return
    for lname, pairs, neg, pct in rows:
        print("{:<{w}} | {:>7d} | {:>7d} | {:>5.2f}%".format(lname, pairs, neg, pct, w=w))
    print("")


# ---------------------------
# PCGRAD AUDIT (per-layer + global)
# ---------------------------

class _PcgradAuditAggregator:
    __slots__ = ("pairs_before", "pairs_after", "neg_before", "neg_after")
    def __init__(self):
        self.reset()
    def reset(self):
        self.pairs_before = 0
        self.pairs_after  = 0
        self.neg_before   = 0
        self.neg_after    = 0
    def add(self, pairs_before, neg_before, pairs_after, neg_after):
        self.pairs_before += int(pairs_before)
        self.neg_before   += int(neg_before)
        self.pairs_after  += int(pairs_after)
        self.neg_after    += int(neg_after)

_pcgrad_audit_global = _PcgradAuditAggregator()

def _flatten_np(g):
    if g is None:
        return None
    arr = g
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return arr.reshape(-1)

def _pairwise_dots(vecs):
    dots = []
    for i in range(len(vecs)):
        vi = vecs[i]
        if vi is None:
            continue
        for j in range(i + 1, len(vecs)):
            vj = vecs[j]
            if vj is None:
                continue
            dots.append(float(np.dot(vi.astype(np.float64, copy=False),
                                     vj.astype(np.float64, copy=False))))
    return np.asarray(dots, dtype=np.float64)

def pcgrad_audit_layer(layer_name: str,
                       grads_before_per_model: List[List[Optional[tf.Tensor]]],
                       grads_after_per_model: List[List[Optional[tf.Tensor]]]):
    # flatten per model into one vector per model for this layer
    def to_flat_per_model(L):
        flats = []
        for tensors in L:
            parts = []
            for t in tensors:
                if t is None:
                    continue
                parts.append(_flatten_np(t))
            flats.append(np.concatenate(parts, axis=0) if parts else None)
        return flats

    fb = to_flat_per_model(grads_before_per_model)
    fa = to_flat_per_model(grads_after_per_model)
    db = _pairwise_dots([v for v in fb if v is not None])
    da = _pairwise_dots([v for v in fa if v is not None])
    if db.size == 0 or da.size == 0:
        return
    pairs_b = int(db.size)
    pairs_a = int(da.size)
    neg_b = int((db < 0.0).sum())
    neg_a = int((da < 0.0).sum())
    fixed = max(0, neg_b - neg_a)
    mean_dot_b = float(db.mean()) if pairs_b else 0.0
    mean_dot_a = float(da.mean()) if pairs_a else 0.0
    logging.log(
        LOG_DETAIL,
        "[PCGRAD][AUDIT] layer=%s pairs=%d neg_before=%d (%.2f%%) neg_after=%d (%.2f%%) fixed=%d "
        "mean_dot_before=%.4g mean_dot_after=%.4g mean_proj_norm=nan",
        str(layer_name),
        pairs_b,
        neg_b, 100.0 * (neg_b / max(1, pairs_b)),
        neg_a, 100.0 * (neg_a / max(1, pairs_a)),
        fixed,
        mean_dot_b, mean_dot_a,
    )
    _pcgrad_audit_global.add(pairs_b, neg_b, pairs_a, neg_a)

def pcgrad_audit_global_flush():
    try:
        if _pcgrad_audit_global.pairs_before > 0:
            logging.log(
                LOG_DETAIL,
                "[PCGRAD][AUDIT][GLOBAL] pairs=%d neg_before=%d (%.2f%%) → neg_after=%d (%.2f%%); fixed=%d",
                _pcgrad_audit_global.pairs_before,
                _pcgrad_audit_global.neg_before,
                100.0 * (_pcgrad_audit_global.neg_before / max(1, _pcgrad_audit_global.pairs_before)),
                _pcgrad_audit_global.neg_after,
                100.0 * (_pcgrad_audit_global.neg_after / max(1, _pcgrad_audit_global.pairs_after)),
                max(0, _pcgrad_audit_global.neg_before - _pcgrad_audit_global.neg_after),
            )
    finally:
        _pcgrad_audit_global.reset()


# ---------------------------
# Synchronous multi-individual trainer (instant sharing + Magic-T)
# ---------------------------

def _apply_ema_to_registry(ema_beta: float):
    """Simple self-EMA on live weights (no shadow). Use >0 to smooth."""
    if ema_beta <= 0.0:
        return
    for layer in REGISTRY.values():
        _ = layer.weights  # force build
        for w in layer.weights:
            w.assign(ema_beta * w + (1.0 - ema_beta) * w)


def _warmup_optimizer_variables(models: List[keras.Model], optimizers: List[keras.optimizers.Optimizer]):
    """
    Ensure optimizer slot variables are created OUTSIDE any tf.function.
    Works with TF 2.x OptimizerV2 and Keras3-style optimizers.
    """
    for m, opt in zip(models, optimizers):
        try:
            if hasattr(opt, "_create_all_weights"):  # TF2 OptimizerV2 path
                opt._create_all_weights(m.trainable_variables)
            else:  # Keras 3 path
                opt.build(m.trainable_variables)
        except Exception as e:
            logging.debug(f"[OPT] warmup skipped: {e}")


def _layer_grads_map_from_aligned(shared_var_ids: List[int],
                                  aligned: List[List[Optional[tf.Tensor]]]) -> Dict[str, List[List[Optional[tf.Tensor]]]]:
    """
    Build: layer_name -> per-model list (list of grads for that layer's vars)
    using REGISTRY to map var-id -> layer.
    """
    id2lname = _varid_to_layername_map()
    num_models = len(aligned)
    out: Dict[str, List[List[Optional[tf.Tensor]]]] = {}
    for pos, vid in enumerate(shared_var_ids):
        lname = id2lname.get(vid, f"var_{vid}")
        if lname not in out:
            out[lname] = [[]
                          for _ in range(num_models)]
        for m in range(num_models):
            out[lname][m].append(aligned[m][pos])
    return out


def train_individuals_synchronously(
    models: List[keras.Model],
    optimizers: List[keras.optimizers.Optimizer],
    loss_fn,
    train_ds,  # tf.data.Dataset or (x,y)
    steps_per_epoch: int,
    epochs: int,
    validation_data=None,
    ema_beta: float = 0.0,
    magic_t: bool = False,
    verbose: int = 1,
    report_every: int = 3,
):
    """
    Round-robin: on each batch, every model gets a step; shared weights update instantly.
    If magic_t=True, apply PCGrad-style projection on shared conflicting grads before applying.
    Also prints per-layer conflict table every `report_every` steps.
    """
    if isinstance(train_ds, tuple):
        x, y = train_ds
        train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(128).repeat()
    train_it = iter(train_ds)

    # Make sure optimizers are ready (avoids tf.function variable-creation errors)
    _warmup_optimizer_variables(models, optimizers)

    for epoch in range(epochs):
        for m in models:
            m.reset_metrics()

        for step in range(steps_per_epoch):
            xb, yb = next(train_it)

            # 1) compute grads for all models (eager)
            per_model_grads: List[List[Optional[tf.Tensor]]] = []
            per_model_preds = []

            for i, m in enumerate(models):
                with tf.GradientTape() as tape:
                    y_pred = m(xb, training=True)
                    loss = loss_fn(yb, y_pred)
                    if m.losses:
                        loss += tf.add_n(m.losses)
                grads = tape.gradient(loss, m.trainable_variables)
                per_model_grads.append(grads)
                per_model_preds.append(y_pred)

            # 2) Build var ownership map and align grads on vars shared by ≥2 models
            var2owners = _shared_var_owners(models)
            shared_var_ids = [vid for vid, owners in var2owners.items() if len(owners) >= 2]
            aligned = _align_grads_on_ids(models, per_model_grads, shared_var_ids)

            # 3) Report conflicts (pairwise over only models that share a var)
            stats = _pairwise_conflict_report_from_aligned(aligned)
            logging.log(
                LOG_DETAIL,
                f"[GRAD-CONFLICT] Overall conflict: {stats['pct_negative']:.2f}% "
                f"({int(stats.get('neg_pairs', 0))}/{int(stats['pairs_evaluated'])})."
            )
            if ((step + 1) % max(1, report_every)) == 0:
                rows = _per_layer_conflict_table(shared_var_ids, aligned)
                _print_per_layer_table(rows)

            # Snapshot per-layer BEFORE for AUDIT (only if we might project)
            before_layer_map = _layer_grads_map_from_aligned(shared_var_ids, aligned) if (magic_t and shared_var_ids) else {}

            # 4) Magic-T mitigation (PCGrad-like) only across models that share a var
            adjusted_grads = [list(g) for g in per_model_grads]  # shallow copies
            new_aligned = [list(a) for a in aligned]
            changed = 0
            total_slots = 0
            if magic_t and shared_var_ids:
                # For each shared var id position k, project grads among owners
                for k, vid in enumerate(shared_var_ids):
                    owners = [i for i in range(len(models)) if aligned[i][k] is not None]
                    if len(owners) < 2:
                        continue
                    for i in owners:
                        gi = new_aligned[i][k]
                        if gi is None:
                            continue
                        gi_proj = gi
                        for j in owners:
                            if i == j:
                                continue
                            gj = aligned[j][k]
                            if gj is None:
                                continue
                            gi_old = gi_proj
                            gi_proj = _pcgrad_pairwise([gi_proj], [gj])[0]
                            if gi_old is not None and gi_proj is not None:
                                diff = _flatten(gi_old) - _flatten(gi_proj)
                                mag = float(tf.norm(diff).numpy())
                                if mag > 0:
                                    changed += 1
                            total_slots += 1
                        new_aligned[i][k] = gi_proj

                # map projected shared grads back into original per-variable order
                id2idx_list = [{id(v): idx for idx, v in enumerate(m.trainable_variables)} for m in models]
                for model_i in range(len(models)):
                    id2idx = id2idx_list[model_i]
                    for pos, key in enumerate(shared_var_ids):
                        idx = id2idx.get(key)
                        if idx is not None:
                            adjusted_grads[model_i][idx] = new_aligned[model_i][pos]

                proj_frac = (changed / max(total_slots, 1))
                logging.log(
                    LOG_DETAIL,
                    f"[PCGRAD][ACTIVE] projected={changed}/{total_slots} (frac={proj_frac:.2f}), mean|Δg|=0.000000"
                )

                # ---- AUDIT per-layer + GLOBAL
                after_layer_map = _layer_grads_map_from_aligned(shared_var_ids, new_aligned)
                for lname, before_lists in before_layer_map.items():
                    after_lists = after_layer_map.get(lname, before_lists)
                    pcgrad_audit_layer(lname, before_lists, after_lists)
                pcgrad_audit_global_flush()

            else:
                if magic_t:
                    logging.log(LOG_DETAIL, "[PCGRAD][SKIP] No shared variables between any model pairs this step.")

            # 5) apply grads & update metrics
            for i, m in enumerate(models):
                grads = adjusted_grads[i]
                optimizers[i].apply_gradients(zip(grads, m.trainable_variables))

                # update accuracy metric if present
                for met in m.metrics:
                    if met.name in ("accuracy", "acc"):
                        met.update_state(yb, per_model_preds[i])

            if ema_beta and ema_beta > 0.0:
                _apply_ema_to_registry(ema_beta)

            if verbose and (step + 1) % max(1, steps_per_epoch // 5) == 0:
                acc0 = None
                for mm in models[0].metrics:
                    if mm.name in ("accuracy", "acc"):
                        acc0 = float(mm.result().numpy())
                        break
                logging.info(f"[SYNC] epoch {epoch+1}/{epochs} step {step+1}/{steps_per_epoch} acc(model0)={acc0}")

        if validation_data is not None and verbose:
            res = models[0].evaluate(validation_data[0], validation_data[1], verbose=0)
            names = models[0].metrics_names
            if isinstance(res, (list, tuple)) and "accuracy" in names:
                va = float(res[names.index("accuracy")])
                logging.info(f"[SYNC] epoch {epoch+1} val_accuracy={va:.4f}")

    return models


# ---------------------------
# Individual structure
# ---------------------------

@dataclass
class Individual:
    name: int
    blueprint_mark: int
    blueprint_ref: Blueprint
    model: Optional[keras.Model] = None
    scores: Optional[List[float]] = None  # [loss, acc]
    species: Optional[int] = None
    gen: Optional[int] = None


# ---------------------------
# Population
# ---------------------------

class Population:
    def __init__(
        self,
        datasets: Datasets,
        input_shape: Tuple[int, ...],
        population_size: int = 6,
        compiler: Optional[Dict[str, Any]] = None,
    ):
        self.datasets = datasets
        self.input_shape = input_shape
        self.population_size = population_size
        self.compiler = compiler or {
            "loss": "categorical_crossentropy",
            "optimizer": keras.optimizers.Adam(learning_rate=0.001),
            "metrics": ["accuracy"],
        }

        # evolution state
        self.historical_marker = HistoricalMarker()
        self.name_generator = NameGenerator()

        self.modules: Dict[int, Module] = {}         # mark -> Module
        self.blueprints: Dict[int, Blueprint] = {}   # mark -> Blueprint
        self.module_species: List[Species] = []
        self.blueprint_species: List[Species] = []

        # sharing + debug toggles (will be set by runner)
        self.module_share_mode: str = "module"  # "module" | "layer" | "global" | "off"
        self.parallel_sync: bool = True
        self.ema_beta: float = 0.0
        self.fast_debug: bool = True
        self.max_steps_per_epoch: int = 15
        self.disable_graph_plots: bool = True

        # MAGIC-T controls
        self.magic_t: bool = False   # if True -> PCGrad mitigation is applied
        self.magic_t_sync_every = 1
        self.magic_t_elastic_tau = 0.0
        self.magic_t_migrate_p = 0.1
        self.magic_t_noise_std = 0.0
        self.magic_t_log_keys = True

        # conflict table cadence
        self.conflict_report_every: int = 3

        # simple mutation policy
        self.magic_t_limit_mutations_per_child: int = 1  # when Magic-T on
        self.magic_t_burst_every: int = 5                 # every N generations
        self.magic_t_burst_minmax: Tuple[int, int] = (3, 5)  # 3–5 mutations in a burst

        # caching last iteration summary
        self._last_iteration: Optional[List[List[Any]]] = None

    # ---------------------------
    # Creation of search spaces
    # ---------------------------

    def _sample_param(self, space):
        """
        space: (range, type) as in your runner.
        """
        rng, typ = space
        if typ == "int":
            if isinstance(rng, (list, tuple)) and len(rng) == 2 and isinstance(rng[0], int) and isinstance(rng[1], int):
                return RNG.randint(rng[0], rng[1])
            return int(RNG.choice(rng))
        if typ == "float":
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                return RNG.uniform(rng[0], rng[1])
            return float(RNG.choice(rng))
        if typ == "list":
            return RNG.choice(rng)
        return RNG.choice(_ensure_list(rng))

    def _generate_component(self, comp_name: str, meta) -> Component:
        """
        meta: (keras_class, param_space_dict)
        """
        kclass, param_spaces = meta
        params = {}
        for k, spec in (param_spaces or {}).items():
            params[k] = self._sample_param(spec)
        if comp_name == "conv2d":
            ks = params.get("kernel_size", 3)
            params["kernel_size"] = ks if isinstance(ks, (tuple, list)) else int(ks)
            params.setdefault("padding", "same")
        return Component(
            representation=(comp_name, params),
            keras_component=kclass,
            complementary_component=None,
            keras_complementary_component=None,
            component_type=comp_name,
        )

    def _generate_module_graph(
        self,
        global_configs,
        possible_components,
        possible_complementary_components=None,
    ):
        """
        Create a simple linear module graph with N components.
        """
        n_comps = self._sample_param(global_configs.get("component_range", ([1, 3], "int")))
        g = nx.DiGraph()
        for i in range(n_comps):
            comp_key = RNG.choice(list(possible_components.keys()))
            comp = self._generate_component(comp_key, possible_components[comp_key])
            node_id = f"c{i}"
            g.add_node(node_id, node_def=comp)
            if i > 0:
                g.add_edge(f"c{i-1}", node_id)
        return g

    def _wrap_module(self, g, layer_type: ModuleComposition) -> Module:
        mark = self.historical_marker.mark_module()
        return Module(components=None, layer_type=layer_type, mark=mark, component_graph=g)

    def create_module_population(
        self,
        module_population_size: int,
        global_configs,
        possible_components,
        possible_complementary_components=None,
    ):
        logging.log(LOG_DETAIL, f"Generating {module_population_size} components")
        for _ in range(module_population_size):
            lt = RNG.choices(
                [ModuleComposition.INPUT, ModuleComposition.INTERMED, ModuleComposition.OUTPUT],
                weights=[1, 4, 1],
            )[0]
            g = self._generate_module_graph(global_configs, possible_components, possible_complementary_components)
            m = self._wrap_module(g, lt)
            self.modules[m.mark] = m

        if not self.disable_graph_plots:
            try:
                from base.graph_ops import plot_graph
                for m in self.modules.values():
                    plot_graph(m.component_graph, title=f"Module {m.mark}")
            except Exception:
                pass

    # ---------------------------
    # Speciation
    # ---------------------------

    def apply_kmeans_speciation(self, items: List[Any], n_species: int):
        """
        Cluster Modules or Blueprints based on their kmeans representation.
        """
        from sklearn.cluster import KMeans  # optional dep

        representations = [item.get_kmeans_representation() for item in items]
        X = np.array(representations, dtype=np.float32)
        k = max(1, min(n_species, len(items)))
        try:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=1337)
            labels = kmeans.fit_predict(X)
        except Exception as e:
            logging.warning(f"[KMEANS] Falling back to random species due to: {e}")
            labels = np.array([RNG.randint(0, k - 1) for _ in items], dtype=np.int32)

        species: Dict[int, List[Any]] = {i: [] for i in range(k)}
        for it, lab in zip(items, labels):
            species[int(lab)].append(it)
        return list(species.values()), labels.tolist()

    def create_module_species(self, n_module_species: int):
        items = list(self.modules.values())
        module_species, module_classifications = self.apply_kmeans_speciation(items, n_module_species)
        logging.log(
            LOG_DETAIL,
            f"KMeans generated {len(module_species)} species using: { [it.get_kmeans_representation() for it in items] }."
        )
        self.module_species = []
        for i, group in enumerate(module_species):
            # set module.species = i so we can build share keys with species id
            for m in group:
                try:
                    m.species = i
                except Exception:
                    pass
            sp = Species(name=i, species_type="module", group=group, properties=None, starting_generation=0)
            self.module_species.append(sp)
            logging.log(LOG_DETAIL, f"Created {len(module_species)} module species.")
            logging.log(LOG_DETAIL, f"Module species {i}: {[m.mark for m in group]}")
        return module_species, module_classifications

    # ---------------------------
    # Blueprints
    # ---------------------------

    def _choose_module_of_type(self, comp: ModuleComposition) -> Module:
        pool = [m for m in self.modules.values() if m.layer_type == comp]
        if not pool:
            pool = list(self.modules.values())
        return copy.deepcopy(RNG.choice(pool))

    def _generate_blueprint_graph(
        self,
        global_configs,
        possible_components,
        possible_complementary_components,
        input_configs,
        possible_inputs,
        possible_complementary_inputs,
        output_configs,
        possible_outputs,
        possible_complementary_outputs,
    ):
        """
        Simple stacked blueprint: [INPUT] -> [INTERMED x N] -> [OUTPUT]
        """
        g = nx.DiGraph()
        nodes = []

        in_mod = self._choose_module_of_type(ModuleComposition.INPUT)
        nid = "input-0-0"
        g.add_node(nid, node_def=in_mod)
        nodes.append(nid)

        n_mods = self._sample_param(global_configs.get("module_range", ([1, 3], "int")))
        for i in range(n_mods):
            m = self._choose_module_of_type(ModuleComposition.INTERMED)
            nid = f"intermed-{i}-0"
            g.add_node(nid, node_def=m)
            g.add_edge(nodes[-1], nid)
            nodes.append(nid)

        out_mod = self._choose_module_of_type(ModuleComposition.OUTPUT)
        nid = "output-0-0"
        g.add_node(nid, node_def=out_mod)
        g.add_edge(nodes[-1], nid)
        nodes.append(nid)

        mark = self.historical_marker.mark_blueprint()
        bp = Blueprint(modules=None, input_shape=self.input_shape, module_graph=g, mark=mark)
        return bp

    def create_blueprint_population(
        self,
        blueprint_population_size: int,
        global_configs,
        possible_components,
        possible_complementary_components,
        input_configs, possible_inputs, possible_complementary_inputs,
        output_configs, possible_outputs, possible_complementary_outputs,
    ):
        logging.log(LOG_DETAIL, f"Generating {blueprint_population_size} modules")
        for _ in range(blueprint_population_size):
            bp = self._generate_blueprint_graph(
                global_configs, possible_components, possible_complementary_components,
                input_configs, possible_inputs, possible_complementary_inputs,
                output_configs, possible_outputs, possible_complementary_outputs,
            )
            self.blueprints[bp.mark] = bp

        if not self.disable_graph_plots:
            try:
                from base.graph_ops import plot_graph
                for b in self.blueprints.values():
                    plot_graph(b.module_graph, title=f"Blueprint {b.mark}")
            except Exception:
                pass

    def create_blueprint_species(self, n_blueprint_species: int):
        items = list(self.blueprints.values())
        blueprint_species, blueprint_classifications = self.apply_kmeans_speciation(items, n_blueprint_species)
        logging.log(
            LOG_DETAIL,
            f"KMeans generated {len(blueprint_species)} species using: { [it.get_kmeans_representation() for it in items] }."
        )
        self.blueprint_species = []
        for i, group in enumerate(blueprint_species):
            sp = Species(name=i, species_type="blueprint", group=group, properties=None, starting_generation=0)
            self.blueprint_species.append(sp)
        logging.log(LOG_DETAIL, f"Created {len(blueprint_species)} blueprint species.")
        for i, sp in enumerate(self.blueprint_species):
            logging.log(LOG_DETAIL, f"Blueprint species {i}: {[b.mark for b in sp.group]}")
        return blueprint_species, blueprint_classifications

    # ---------------------------
    # Assembly (Keras)
    # ---------------------------

    def _module_species_id(self, module_obj: Module) -> int:
        sid = getattr(module_obj, "species", None)
        if sid is not None:
            return int(sid)
        # fallback: try to find it in recorded species
        for sp in self.module_species:
            if module_obj in sp.group:
                try:
                    module_obj.species = sp.name
                except Exception:
                    pass
                return int(sp.name)
        return -1  # unknown / unspeciated

    def _apply_module(self, x, module_obj: Module, module_scope: Tuple[int, str]):
        """
        Apply a Module's internal graph (assumed linear) on tensor x, using shared layers.

        Sharing policy (module_share_mode == "module"):
          key = (component_type, in_channels, params, (module_species_id, f"{outer_layer_tag}#{pos}"))

        So modules belonging to the same module species share weights *position-wise*
        under the same outer tag (input/intermed-k/output). This avoids reusing the
        exact same layer multiple times inside a single module while still enabling
        cross-individual sharing by species + position.
        """
        g = module_obj.component_graph
        order = list(nx.topological_sort(g)) if g.number_of_nodes() > 0 else []
        if not order:
            return x

        species_id = self._module_species_id(module_obj)
        outer_tag = module_scope[1]  # e.g., 'input-0', 'intermed-1', 'output-0'

        for pos, node in enumerate(order):
            comp: Component = g.nodes[node]["node_def"]
            ctype, cparams = comp.representation
            params = dict(cparams)
            in_ch = _infer_in_channels_from_tensor(x)

            # Position-aware tag for unambiguous sharing within the outer tag
            position_tag = f"{outer_tag}#{pos}"

            # Scope for the share key
            if self.module_share_mode == "module":
                scope = (species_id, position_tag)
            elif self.module_share_mode == "layer":
                scope = (None, position_tag)  # ('L', label) inside _layer_key
            elif self.module_share_mode == "global":
                scope = None
            else:  # 'off'
                scope = ("nonce", id(object()))

            lyr = _get_or_create_shared_layer(
                component_type=ctype,
                in_channels=in_ch,
                params=params,
                extra_scope=scope,
                share_mode=self.module_share_mode,
            )
            # call layer appropriately
            try:
                x = lyr(x)
            except TypeError:
                # Dense on spatial requires flatten
                if len(x.shape) > 2:
                    x = KL.Flatten()(x)
                x = lyr(x)
        return x

    def assemble_model_for_individual(self, indiv: Individual) -> keras.Model:
        bp = indiv.blueprint_ref
        logging.log(LOG_DETAIL, f"Starting assembling of blueprint {bp.mark}.")
        g = bp.module_graph
        nodes = list(g.nodes())
        logging.log(LOG_DETAIL, f"Generated assembled graph for blueprint {bp.mark}: {nodes}")

        # model IO
        inp = KL.Input(shape=self.input_shape, name=f"input_{bp.mark}")
        logging.log(LOG_DETAIL, f"Added Input layer: {inp}")

        # walk graph (linear chain assumed)
        topo = list(nx.topological_sort(g))
        x = inp
        for nid in topo:
            m: Module = g.nodes[nid]["node_def"]
            # derive layer tag from node id
            label = nid.rsplit("-", 1)[0] if "-" in nid else nid
            module_scope = (0, label)  # index kept for compatibility, not used in key
            x = self._apply_module(x, m, module_scope)

        # ensure a proper classification head with softmax
        y_train = self.datasets.training[1]
        num_classes = int(y_train.shape[1]) if len(y_train.shape) > 1 else int(np.max(y_train) + 1)

        if len(x.shape) > 2:
            x = KL.GlobalAveragePooling2D()(x)
        if int(x.shape[-1]) != num_classes:
            x = KL.Dense(num_classes)(x)
        out = KL.Activation("softmax", name="predictions")(x)

        model = Model(inputs=inp, outputs=out, name=f"indiv_{indiv.name}_bp{bp.mark}")

        # compile per self.compiler
        loss = self.compiler.get("loss", "categorical_crossentropy")
        metrics = self.compiler.get("metrics", ["accuracy"])
        optimizer = self.compiler.get("optimizer", keras.optimizers.Adam(learning_rate=0.001))
        optimizer = _safe_eval_optimizer(optimizer)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    # ---------------------------
    # Simple mutation utilities
    # ---------------------------

    def _mutate_component_params_in_place(self, comp: Component, possible_components: Dict[str, Any]):
        """Randomly tweak a single hyperparam of the component if we know its space."""
        ctype, params = comp.representation
        meta = possible_components.get(ctype)
        if not meta:
            return
        _, param_spaces = meta
        if not param_spaces:
            return
        key = RNG.choice(list(param_spaces.keys()))
        new_val = self._sample_param(param_spaces[key])
        params = dict(params)
        params[key] = new_val
        # normalize
        if ctype == "conv2d":
            ks = params.get("kernel_size", 3)
            params["kernel_size"] = ks if isinstance(ks, (tuple, list)) else int(ks)
            params.setdefault("padding", "same")
        comp.representation = (ctype, params)

    def _mutate_blueprint_once(self, bp: Blueprint, possible_components: Dict[str, Any]):
        """Pick a random module node and one component inside; mutate its params a bit."""
        g = bp.module_graph
        if g.number_of_nodes() == 0:
            return
        nid = RNG.choice(list(g.nodes()))
        m: Module = g.nodes[nid]["node_def"]
        mg = m.component_graph
        if mg.number_of_nodes() == 0:
            return
        cnid = RNG.choice(list(mg.nodes()))
        comp: Component = mg.nodes[cnid]["node_def"]
        self._mutate_component_params_in_place(comp, possible_components)

    # ---------------------------
    # Evolutionary loop
    # ---------------------------

    def _create_individuals_from_blueprints(self, gen_idx: int) -> List[Individual]:
        bp_marks = list(self.blueprints.keys())
        indivs: List[Individual] = []
        for i in range(self.population_size):
            bp_mark = bp_marks[i % len(bp_marks)]
            indivs.append(
                Individual(
                    name=self.historical_marker.mark_individual(),
                    blueprint_mark=bp_mark,
                    blueprint_ref=self.blueprints[bp_mark],
                    model=None,
                    scores=None,
                    species=None,
                    gen=gen_idx,
                )
            )
        logging.info(f"[POP] Created {len(indivs)} individuals: {[(indiv.name, indiv.blueprint_mark) for indiv in indivs]}")
        logging.log(LOG_DETAIL, f"Created individuals for blueprints: {[(indiv.name, indiv.blueprint_mark) for indiv in indivs]}")
        return indivs

    def _evaluate_generation_sync(self, individuals: List[Individual], training_epochs: int, validation_split: float):
        """
        Build all models, then train synchronously round-robin to enable instant weight sharing.
        """
        x_train, y_train = self.datasets.training
        x_test, y_test = self.datasets.test

        # assemble & compile models
        models: List[keras.Model] = []
        opts: List[keras.optimizers.Optimizer] = []
        for indiv in individuals:
            m = self.assemble_model_for_individual(indiv)
            models.append(m)
            indiv.model = m
            opts.append(m.optimizer)

        # training dataset (tf.data)
        batch_size = 128
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(8192).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        loss_fn = keras.losses.CategoricalCrossentropy()

        logging.info(f"[SYNC] Iterating fitness (parallel-style) over {len(models)} individuals")

        # synchronous round-robin with conflict detection + Magic-T mitigation
        train_individuals_synchronously(
            models=models,
            optimizers=opts,
            loss_fn=loss_fn,
            train_ds=train_ds,
            steps_per_epoch=self.max_steps_per_epoch,
            epochs=training_epochs,
            validation_data=(x_test, y_test),
            ema_beta=self.ema_beta,
            magic_t=self.magic_t,
            verbose=1,
            report_every=self.conflict_report_every,
        )

        # Final evaluation per model
        iteration_summary = []
        for indiv in individuals:
            scores = indiv.model.evaluate(x_test, y_test, verbose=0)  # [loss, acc]
            indiv.scores = [float(scores[0]), float(scores[1] if len(scores) > 1 else float("nan"))]
            feat = indiv.blueprint_ref.get_kmeans_representation()
            indiv.species = 0
            iteration_summary.append([indiv.name, indiv.blueprint_mark, indiv.scores, feat, indiv.species, indiv.gen])

        return iteration_summary

    def iterate_generations(
        self,
        generations: int,
        training_epochs: int,
        validation_split: float,
        mutation_rate: float,
        crossover_rate: float,
        elitism_rate: float,
        possible_components,
        possible_complementary_components,
    ):
        logging.info(f"Iterating over {generations} generations")

        # Banner to prove runtime config
        if self.magic_t:
            logging.info(
                "[MAGIC-T][ON] mutations/child=%d, burst_every=%d, burst_range=(%d, %d), "
                "pcgrad=ENABLED, ema_beta=%s",
                self.magic_t_limit_mutations_per_child,
                self.magic_t_burst_every,
                self.magic_t_burst_minmax[0],
                self.magic_t_burst_minmax[1],
                str(self.ema_beta),
            )

        all_iterations = []
        for gen_idx in range(generations):
            logging.info(f" -- Iterating generation {gen_idx} --")
            logging.log(LOG_DETAIL, f"Currently {len(self.modules)} modules, {len(self.blueprints)} blueprints, latest iteration: {self._last_iteration}")
            logging.log(LOG_DETAIL, f"Current modules: {list(self.modules.keys())}")
            logging.log(LOG_DETAIL, f"Current blueprints: {list(self.blueprints.keys())}")

            try:
                tf.compat.v1.reset_default_graph()
            except Exception:
                pass

            # === Mutation policy (very lightweight) ===
            total_mut = 0
            burst = False
            if gen_idx > 0:
                per_child = 1 if self.magic_t else RNG.randint(1, 3)
                bp_marks = list(self.blueprints.keys())
                RNG.shuffle(bp_marks)
                n_children = min(self.population_size, len(bp_marks))
                for k in range(n_children):
                    bp = self.blueprints[bp_marks[k]]
                    for _ in range(per_child):
                        if RNG.random() < mutation_rate:
                            self._mutate_blueprint_once(bp, possible_components)
                            total_mut += 1

                if self.magic_t and (gen_idx % max(1, self.magic_t_burst_every) == 0):
                    burst_lo, burst_hi = self.magic_t_burst_minmax
                    n_burst = RNG.randint(burst_lo, burst_hi)
                    for _ in range(n_burst):
                        bp = self.blueprints[RNG.choice(list(self.blueprints.keys()))]
                        self._mutate_blueprint_once(bp, possible_components)
                        total_mut += 1
                    burst = True

            # log the mutation summary per generation
            if self.magic_t:
                logging.info("[MAGIC-T][SEARCH] generation %d total_mutations=%d burst=%s", gen_idx, total_mut, str(burst))

            individuals = self._create_individuals_from_blueprints(gen_idx)
            iteration = self._evaluate_generation_sync(individuals, training_epochs, validation_split)
            all_iterations.append(["generation %d" % gen_idx, max(iteration, key=lambda r: r[2][1])])

            for ind in individuals:
                self.blueprints[ind.blueprint_mark].update_scores(ind.scores)
            for m in self.modules.values():
                m.update_weighted_scores()
            for b in self.blueprints.values():
                b.update_weighted_scores()

            self._log_generation_summary(individuals, gen_idx)
            self._last_iteration = iteration

        return all_iterations

    def _log_generation_summary(self, individuals: List[Individual], gen_idx: int):
        best = max(individuals, key=lambda it: it.scores[1] if it.scores else -1.0)
        logging.log(LOG_DETAIL, f"This iteration: {[[ind.name, ind.blueprint_mark, ind.scores, ind.blueprint_ref.get_kmeans_representation(), ind.species, ind.gen] for ind in individuals]}")
        logging.log(LOG_DETAIL, f"Best model chosen: {[best.name, best.blueprint_mark, best.scores, best.blueprint_ref.get_kmeans_representation(), best.species, best.gen]}")

        try:
            os.makedirs("models", exist_ok=True)
            best.model.save(os.path.join("models", f"best_gen{gen_idx}_indiv{best.name}.h5"))
        except Exception:
            pass

        print("\n --------------- Generation %d Summary --------------\n" % gen_idx)
        print(f"Current {len(individuals)} individuals: [", end="")
        print(", ".join(str(ind.name) for ind in individuals), end="")
        print("]")
        print("Current %d blueprints:" % len(self.blueprints))
        print("[Mark, Test loss, Test acc, Species, Node Count, Edge Count, Neuron Count]")
        bp_rows = []
        for b in self.blueprints.values():
            n = len(b.module_graph.nodes())
            e = len(b.module_graph.edges())
            s = b.get_blueprint_size()
            bp_rows.append([b.mark] + [int(b.weighted_scores[0]), int(b.weighted_scores[1])] + [b.species if b.species is not None else 0, n, e, s])
        try:
            print(np.array(bp_rows))
        except Exception:
            print(bp_rows)

        print(f"Current {len(self.blueprint_species)} blueprint species: {[i for i, _ in enumerate(self.blueprint_species)]}")
        print("NOTE: Scores reflect past iteration; current blueprints may not all be evaluated yet.")
        for i, sp in enumerate(self.blueprint_species):
            print(f"Blueprint species {i} scores: {sp.weighted_scores}. members: {[b.mark for b in sp.group]}")

        print(f"Current {len(self.modules)} modules:")
        print("[Mark, Test loss, Test acc, Species, Node Count, Edge Count, Neuron Count]")
        mod_rows = []
        for m in self.modules.values():
            n = len(m.component_graph.nodes())
            e = len(m.component_graph.edges())
            s = m.get_module_size()
            mod_rows.append([m.mark] + [int(m.weighted_scores[0]), int(m.weighted_scores[1])] + [m.species if getattr(m, 'species', None) is not None else 0, n, e, s])
        try:
            print(np.array(mod_rows))
        except Exception:
            print(mod_rows)

        print(f"Current {len(self.module_species)} module species: {[i for i, _ in enumerate(self.module_species)]}")
        print("NOTE: Scores reflect past iteration; current modules may not all be evaluated yet.")
        for i, sp in enumerate(self.module_species):
            print(f"Module species {i} scores: {sp.weighted_scores}. members: {[m.mark for m in sp.group]}")
        print("\n --------------------------------------------\n")

    # ---------------------------
    # Utilities used by the runner
    # ---------------------------

    def log_shared_state(self, tag: str):
        logging.info(f"[REGISTRY][{tag}] entries={len(REGISTRY)}")

    def return_best_individual(self) -> Individual:
        if not self._last_iteration:
            bp_mark = next(iter(self.blueprints.keys()))
            return Individual(
                name=1,
                blueprint_mark=bp_mark,
                blueprint_ref=self.blueprints[bp_mark],
                model=None,
                scores=None,
                species=None,
                gen=0,
            )
        rows = self._last_iteration
        best_row = max(rows, key=lambda r: r[2][1])
        best_indiv_name = best_row[0]
        best_bp_mark = best_row[1]
        return Individual(
            name=best_indiv_name,
            blueprint_mark=best_bp_mark,
            blueprint_ref=self.blueprints[best_bp_mark],
            model=None,
            scores=best_row[2],
            species=best_row[4],
            gen=best_row[5],
        )

    def train_full_model(
        self,
        indiv: Individual,
        epochs: int,
        validation_split: float,
        custom_fit_args: Optional[Dict[str, Any]] = None,
        warm_start: bool = False,
    ):
        """
        Final retraining (e.g. with augmentation) on the best architecture.
        """
        logging.info(f"[POP] Final retrain start for individual {indiv.name} (bp {indiv.blueprint_mark})")
        K.clear_session()
        if not warm_start:
            REGISTRY.clear()    # rebuild fresh layers to avoid shape carryover
    
        model = self.assemble_model_for_individual(indiv)
        indiv.model = model

        loss = self.compiler.get("loss", "categorical_crossentropy")
        metrics = self.compiler.get("metrics", ["accuracy"])
        optimizer = self.compiler.get("optimizer", keras.optimizers.Adam(learning_rate=0.001))
        optimizer = _safe_eval_optimizer(optimizer)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if custom_fit_args:
            fit_args = dict(custom_fit_args)
            if "generator" in fit_args and "x" not in fit_args:
                fit_args["x"] = fit_args.pop("generator")
            if "steps_per_epoch" not in fit_args and "x" in fit_args and hasattr(fit_args["x"], "__len__"):
                try:
                    fit_args["steps_per_epoch"] = len(fit_args["x"])
                except Exception:
                    pass
            history = model.fit(**fit_args)
        else:
            x_train, y_train = self.datasets.training
            x_test, y_test = self.datasets.test
            history = model.fit(
                x_train, y_train,
                validation_data=(x_test, y_test),
                epochs=epochs,
                verbose=1,
                batch_size=128,
            )

        x_test, y_test = self.datasets.test
        scores = model.evaluate(x_test, y_test, verbose=0)
        logging.info(f"[POP] Final retrain scores: {scores}")
        return scores
