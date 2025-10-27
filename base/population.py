# base/population.py
import os
import copy
import math
import random
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import csv as _csv
import numpy as np
import networkx as nx
import tensorflow as tf
keras = tf.keras
from tensorflow.keras import layers as KL
from tensorflow.keras import Model
from tensorflow.keras import backend as K

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
from base.config import LOG_DETAIL

RNG = random.Random(1337)

IGNORE_SHARE_TYPES = {
    "BatchNormalization", "BatchNorm", "FusedBatchNorm",
    "bn", "batch_norm", "batchnormalization"
}
# ---- weight-sharing counters (for per-generation stats) ----
WS_CREATED = 0
WS_REUSED = 0

def ws_reset():
    """Reset per-generation weight-sharing counters."""
    global WS_CREATED, WS_REUSED
    WS_CREATED = 0
    WS_REUSED = 0

def ws_counts():
    """Return (created, reused) counts since last reset."""
    return WS_CREATED, WS_REUSED

def _ctype_norm(ctype: str) -> str:
    return (ctype or "").strip()

def _ensure_list(x):
    return x if isinstance(x, (list, tuple)) else [x]

def _sorted_items(d: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(d.items(), key=lambda x: x[0]))

def _layer_key(component_type: str, in_channels: Optional[int], params: Dict[str, Any],
               extra_scope: Optional[Tuple[Any, Any]], share_mode: str = "module") -> Tuple:
    params_tuple = _sorted_items(params)
    if share_mode == "off":
        return (component_type, in_channels, params_tuple, ("nonce", id(object())))
    if share_mode == "global":
        return (component_type, in_channels, params_tuple, None)
    if share_mode == "layer":
        label = extra_scope[1] if extra_scope is not None else None
        return (component_type, in_channels, params_tuple, ("L", label))
    return (component_type, in_channels, params_tuple, extra_scope)

def _key_to_safe_name(key: Tuple) -> str:
    h = abs(hash(key)) % 10**10
    base = str(key[0]).replace(" ", "_")
    return f"{base}__{h}"

def _safe_eval_optimizer(opt_str: Any):
    if not isinstance(opt_str, str):
        return opt_str
    safe_globals = {"keras": keras, "tf": tf}
    opt_str = opt_str.replace("lr=", "learning_rate=")
    try:
        return eval(opt_str, safe_globals, {})
    except Exception as e:
        logging.warning(f"[OPTIMIZER] Could not eval '{opt_str}': {e}. Falling back to Adam(0.001).")
        return keras.optimizers.Adam(learning_rate=0.001)

def _make_layer(component_type: str, params: Dict[str, Any], *, name: Optional[str] = None) -> KL.Layer:
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
        p = {"pool_size": (2, 2), "strides": 2, "padding": "same"}
        p.update(params)
        return make(KL.MaxPooling2D, p)
    return make(KL.Lambda, {"function": lambda x: x})

def _infer_in_channels_from_tensor(x):
    try:
        return int(x.shape[-1])
    except Exception:
        return None

def _labels_are_one_hot(y):
    return (len(getattr(y, "shape", ())) == 2) and (int(y.shape[1]) > 1)

def _loss_and_metrics_for(y):
    if _labels_are_one_hot(y):
        return (keras.losses.CategoricalCrossentropy(label_smoothing=0.0), ["accuracy"])
    else:
        return (keras.losses.SparseCategoricalCrossentropy(), ["sparse_categorical_accuracy"])

def _get_or_create_shared_layer(component_type: str, in_channels: Optional[int], params: Dict[str, Any],
                                extra_scope: Optional[Tuple[Any, Any]], share_mode: str) -> KL.Layer:
    ctype = _ctype_norm(component_type)
    if ctype in IGNORE_SHARE_TYPES or share_mode == "off":
        lyr = _make_layer(component_type, params, name=None)
        logging.log(LOG_DETAIL, f"[LIVE-SHARE] NOSHARE ctype={ctype} layer_id={id(lyr)}")
        return lyr
    key = _layer_key(component_type, in_channels, params, extra_scope, share_mode)
    if key in REGISTRY:
        lyr = REGISTRY[key]
        try:
            global WS_REUSED
            WS_REUSED += 1
        except Exception:
            pass
        logging.log(LOG_DETAIL, f"[LIVE-SHARE] REUSE  key={key} layer_id={id(REGISTRY[key])}")
        return lyr
    lname = _key_to_safe_name(key)
    layer = _make_layer(component_type, params, name=lname)
    REGISTRY[key] = layer

    try:
        global WS_CREATED
        WS_CREATED += 1
    except Exception:
        pass
    logging.log(LOG_DETAIL, f"[LIVE-SHARE] CREATE key={key} name={lname} layer_id={id(layer)}")
    return layer


def _flatten(t: tf.Tensor) -> tf.Tensor:
    return tf.reshape(t, [-1]) if t is not None else None

def _cosine(a: tf.Tensor, b: tf.Tensor, eps: float = 1e-12) -> tf.Tensor:
    a_f = _flatten(a); b_f = _flatten(b)
    if a_f is None or b_f is None:
        return tf.constant(float("nan"), dtype=tf.float32)
    na = tf.norm(a_f); nb = tf.norm(b_f)
    denom = tf.maximum(na * nb, eps)
    return tf.reduce_sum(a_f * b_f) / denom

def _pcgrad_pairwise(grads_i: List[Optional[tf.Tensor]], grads_j: List[Optional[tf.Tensor]]):
    out = []
    for gi, gj in zip(grads_i, grads_j):
        if gi is None or gj is None:
            out.append(gi); continue
        gi_f = _flatten(gi); gj_f = _flatten(gj)
        dot = tf.reduce_sum(gi_f * gj_f)
        if dot < 0:
            denom = tf.maximum(tf.reduce_sum(gj_f * gj_f), 1e-12)
            proj = (dot / denom) * gj_f
            gi_new = tf.reshape(gi_f - proj, tf.shape(gi))
            out.append(gi_new)
        else:
            out.append(gi)
    return out

def _shared_var_owners(models: List[keras.Model]) -> Dict[int, List[int]]:
    var2owners: Dict[int, List[int]] = {}
    for mi, m in enumerate(models):
        for v in m.trainable_variables:
            var2owners.setdefault(id(v), []).append(mi)
    return var2owners

def _align_grads_on_ids(models: List[keras.Model], per_model_grads: List[List[Optional[tf.Tensor]]], var_ids: List[int]):
    dicts = []
    for m, grads in zip(models, per_model_grads):
        d = {}
        for v, g in zip(m.trainable_variables, grads):
            d[id(v)] = g
        dicts.append(d)
    aligned = [[d.get(vid, None) for vid in var_ids] for d in dicts]
    return aligned

def _pairwise_conflict_report_from_aligned(aligned_grads: List[List[Optional[tf.Tensor]]]) -> Dict[str, float]:
    num_models = len(aligned_grads)
    if num_models < 2 or not aligned_grads or not aligned_grads[0]:
        return {"pairs_evaluated": 0.0, "pct_negative": 0.0, "neg_pairs": 0}
    total = 0; neg = 0; L = len(aligned_grads[0])
    for k in range(L):
        owners = [i for i in range(num_models) if aligned_grads[i][k] is not None]
        for a in range(len(owners)):
            for b in range(a + 1, len(owners)):
                gi = aligned_grads[owners[a]][k]; gj = aligned_grads[owners[b]][k]
                c = _cosine(gi, gj).numpy()
                if not np.isfinite(c): continue
                total += 1
                if c < 0: neg += 1
    pct = (100.0 * neg / total) if total > 0 else 0.0
    return {"pairs_evaluated": float(total), "pct_negative": float(pct), "neg_pairs": int(neg)}

def _varid_to_layername_map() -> Dict[int, str]:
    mapping = {}
    for layer in REGISTRY.values():
        _ = layer.weights
        for w in layer.weights:
            mapping[id(w)] = layer.name
    return mapping

def _per_layer_conflict_table(shared_keys: List[int], aligned: List[List[Optional[tf.Tensor]]]) -> List[Tuple[str, int, int, float]]:
    id2lname = _varid_to_layername_map()
    num_models = len(aligned); agg: Dict[str, List[int]] = {}
    if num_models < 2 or not shared_keys:
        return []
    for pos, var_id in enumerate(shared_keys):
        lname = id2lname.get(var_id, f"var_{var_id}")
        total = 0; neg = 0
        gks = [aligned[i][pos] for i in range(num_models)]
        owners = [i for i in range(num_models) if gks[i] is not None]
        if len(owners) < 2:
            continue
        for i_idx in range(len(owners)):
            for j_idx in range(i_idx + 1, len(owners)):
                i = owners[i_idx]; j = owners[j_idx]
                c = _cosine(gks[i], gks[j]).numpy()
                if not np.isfinite(c): continue
                total += 1
                if c < 0: neg += 1
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

class _PcgradAuditAggregator:
    __slots__ = ("pairs_before", "pairs_after", "neg_before", "neg_after")
    def __init__(self): self.reset()
    def reset(self):
        self.pairs_before = 0; self.pairs_after = 0
        self.neg_before = 0; self.neg_after = 0
    def add(self, pb, nb, pa, na):
        self.pairs_before += int(pb); self.neg_before += int(nb)
        self.pairs_after  += int(pa); self.neg_after  += int(na)

_pcgrad_audit_global = _Pcgrad_audit = _PcgradAuditAggregator()

def pcgrad_audit_layer(layer_name: str,
                       grads_before_per_model: List[List[Optional[tf.Tensor]]],
                       grads_after_per_model: List[List[Optional[tf.Tensor]]]):
    def to_flat_per_model(L):
        flats = []
        for tensors in L:
            parts = []
            for t in tensors:
                if t is None: continue
                arr = t.numpy() if hasattr(t, "numpy") else t
                parts.append(arr.reshape(-1))
            flats.append(np.concatenate(parts, axis=0) if parts else None)
        return flats
    fb = to_flat_per_model(grads_before_per_model)
    fa = to_flat_per_model(grads_after_per_model)
    def _pairwise_dots(vecs):
        dots = []
        for i in range(len(vecs)):
            vi = vecs[i]
            if vi is None: continue
            for j in range(i + 1, len(vecs)):
                vj = vecs[j]
                if vj is None: continue
                dots.append(float(np.dot(vi.astype(np.float64, copy=False),
                                         vj.astype(np.float64, copy=False))))
        return np.asarray(dots, dtype=np.float64)
    db = _pairwise_dots([v for v in fb if v is not None])
    da = _pairwise_dots([v for v in fa if v is not None])
    if db.size == 0 or da.size == 0: return
    pairs_b = int(db.size); pairs_a = int(da.size)
    neg_b = int((db < 0.0).sum()); neg_a = int((da < 0.0).sum())
    fixed = max(0, neg_b - neg_a)
    mean_dot_b = float(db.mean()) if pairs_b else 0.0
    mean_dot_a = float(da.mean()) if pairs_a else 0.0
    logging.log(
        LOG_DETAIL,
        "[PCGRAD][AUDIT] layer=%s pairs=%d neg_before=%d (%.2f%%) neg_after=%d (%.2f%%) fixed=%d "
        "mean_dot_before=%.4g mean_dot_after=%.4g",
        str(layer_name), pairs_b, neg_b, 100.0 * (neg_b / max(1, pairs_b)),
        neg_a, 100.0 * (neg_a / max(1, pairs_a)), fixed, mean_dot_b, mean_dot_a,
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

def _apply_ema_to_registry(ema_beta: float):
    if ema_beta <= 0.0:
        return
    for layer in REGISTRY.values():
        _ = layer.weights
        for w in layer.weights:
            w.assign(ema_beta * w + (1.0 - ema_beta) * w)

def _layer_grads_map_from_aligned(shared_var_ids: List[int],
                                  aligned: List[List[Optional[tf.Tensor]]]) -> Dict[str, List[List[Optional[tf.Tensor]]]]:
    id2lname = _varid_to_layername_map()
    num_models = len(aligned)
    out: Dict[str, List[List[Optional[tf.Tensor]]]] = {}
    for pos, vid in enumerate(shared_var_ids):
        lname = id2lname.get(vid, f"var_{vid}")
        if lname not in out:
            out[lname] = [[] for _ in range(num_models)]
        for m in range(num_models):
            out[lname][m].append(aligned[m][pos])
    return out

# imports at top if missing

def train_individuals_synchronously(
    models: List[keras.Model],
    optimizers: List[keras.optimizers.Optimizer],
    loss_fn,
    train_ds,
    steps_per_epoch: int,
    epochs: int,
    validation_data=None,
    ema_beta: float = 0.0,
    magic_t: bool = False,
    verbose: int = 1,
    report_every: int = 3,
    *,
    pcgrad_every: int = 1,
    pcgrad_sample_frac: float = 1.0,
    epoch_recorders: Optional[List[list]] = None,   # one recorder per model
    step_recorders: Optional[List[list]] = None,    # one recorder per model
):
    """
    Synchronous EA trainer with optional PCGrad.
    Records:
      • per-step training loss for every model (step_recorders[i])
      • per-epoch validation metrics for every model (epoch_recorders[i])
    """
    if isinstance(train_ds, tuple):
        x, y = train_ds
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x, y))
            .shuffle(8192, reshuffle_each_iteration=True)
            .repeat()
            .batch(128, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
    train_it = iter(train_ds)

    # Warm-up optimizer slots
    for m, opt in zip(models, optimizers):
        try:
            if hasattr(opt, "_create_all_weights"):
                opt._create_all_weights(m.trainable_variables)
            else:
                opt.build(m.trainable_variables)
        except Exception:
            pass

    n_models = len(models)
    if step_recorders is None:
        step_recorders = [[] for _ in range(n_models)]
    if epoch_recorders is None:
        epoch_recorders = [[] for _ in range(n_models)]

    for epoch in range(epochs):
        for m in models:
            m.reset_metrics()

        for step in range(steps_per_epoch):
            try:
                xb, yb = next(train_it)
            except StopIteration:
                train_it = iter(train_ds)
                xb, yb = next(train_it)

            per_model_grads: List[List[Optional[tf.Tensor]]] = []
            per_model_preds = []
            batch_losses: List[float] = []

            # Forward + grads
            for m in models:
                with tf.GradientTape() as tape:
                    y_pred = m(xb, training=True)
                    loss = loss_fn(yb, y_pred)
                    # Ensure scalar loss
                    if isinstance(loss, (list, tuple)):
                        loss = tf.add_n([tf.convert_to_tensor(l) for l in loss])
                    if len(getattr(loss, "shape", ())) > 0:
                        loss = tf.reduce_mean(loss)
                    if m.losses:
                        loss += tf.add_n(m.losses)
                grads = tape.gradient(loss, m.trainable_variables)
                per_model_grads.append(grads)
                per_model_preds.append(y_pred)
                batch_losses.append(float(loss.numpy()))

            # Per-step training loss (for every model)
            for i in range(n_models):
                step_recorders[i].append(batch_losses[i])

            # === PCGrad (optional) ===
            do_pcgrad = magic_t and (
                pcgrad_every is None or pcgrad_every <= 1 or ((step + 1) % pcgrad_every == 0)
            )
            adjusted_grads = [list(g) for g in per_model_grads]

            if do_pcgrad:
                var2owners = _shared_var_owners(models)
                shared_var_ids_all = [vid for vid, owners in var2owners.items() if len(owners) >= 2]

                if 0.0 < pcgrad_sample_frac < 1.0 and len(shared_var_ids_all) > 0:
                    k = max(1, int(pcgrad_sample_frac * len(shared_var_ids_all)))
                    RNG.shuffle(shared_var_ids_all)
                    shared_var_ids = shared_var_ids_all[:k]
                else:
                    shared_var_ids = shared_var_ids_all

                aligned = _align_grads_on_ids(models, per_model_grads, shared_var_ids)

                if report_every and report_every > 0 and ((step + 1) % report_every == 0):
                    stats = _pairwise_conflict_report_from_aligned(aligned)
                    logging.log(
                        LOG_DETAIL,
                        f"[GRAD-CONFLICT] Overall conflict: {stats['pct_negative']:.2f}% "
                        f"({int(stats.get('neg_pairs', 0))}/{int(stats['pairs_evaluated'])})."
                    )

                before_layer_map = _layer_grads_map_from_aligned(shared_var_ids, aligned)
                new_aligned = [list(a) for a in aligned]
                changed = 0
                total_slots = 0

                for kpos, _ in enumerate(shared_var_ids):
                    owners = [i for i in range(n_models) if aligned[i][kpos] is not None]
                    if len(owners) < 2:
                        continue
                    for i in owners:
                        gi_proj = new_aligned[i][kpos]
                        if gi_proj is None:
                            continue
                        for j in owners:
                            if i == j:
                                continue
                            gj = aligned[j][kpos]
                            if gj is None:
                                continue
                            gi_old = gi_proj
                            gi_proj = _pcgrad_pairwise([gi_proj], [gj])[0]
                            if gi_old is not None and gi_proj is not None:
                                diff = _flatten(gi_old) - _flatten(gi_proj)
                                if float(tf.norm(diff).numpy()) > 0:
                                    changed += 1
                            total_slots += 1
                        new_aligned[i][kpos] = gi_proj

                id2idx_list = [{id(v): idx for idx, v in enumerate(m.trainable_variables)} for m in models]
                for mi in range(n_models):
                    id2idx = id2idx_list[mi]
                    for pos, key in enumerate(shared_var_ids):
                        idx = id2idx.get(key)
                        if idx is not None:
                            adjusted_grads[mi][idx] = new_aligned[mi][pos]

                if report_every and report_every > 0 and total_slots > 0:
                    logging.log(
                        LOG_DETAIL,
                        f"[PCGRAD][ACTIVE] projected={changed}/{total_slots} (frac={changed/total_slots:.3f})"
                    )

                if report_every and report_every > 0:
                    after_layer_map = _layer_grads_map_from_aligned(shared_var_ids, new_aligned)
                    for lname, before_lists in before_layer_map.items():
                        after_lists = after_layer_map.get(lname, before_lists)
                        pcgrad_audit_layer(lname, before_lists, after_lists)
                    pcgrad_audit_global_flush()

            # Apply grads + metrics
            for i, m in enumerate(models):
                grads = adjusted_grads[i]
                safe_pairs = []
                for g, v in zip(grads, m.trainable_variables):
                    if g is None:
                        g = tf.zeros_like(v)  # guard
                    safe_pairs.append((g, v))
                optimizers[i].apply_gradients(safe_pairs)

                for met in m.metrics:
                    if met.name in ("accuracy", "acc", "sparse_categorical_accuracy"):
                        met.update_state(yb, per_model_preds[i])

            if ema_beta and ema_beta > 0.0:
                _apply_ema_to_registry(ema_beta)

            if verbose and (step + 1) % max(1, steps_per_epoch // 5) == 0:
                acc0 = None
                for mm in models[0].metrics:
                    if mm.name in ("accuracy", "acc", "sparse_categorical_accuracy"):
                        acc0 = float(mm.result().numpy()); break
                logging.info(f"[SYNC] epoch {epoch+1}/{epochs} step {step+1}/{steps_per_epoch} acc(model0)={acc0}")

        # === end epoch: evaluate EVERY model and record ===
        if validation_data is not None:
            x_val, y_val = validation_data
            for i, m in enumerate(models):
                res = m.evaluate(x_val, y_val, verbose=0)
                names = m.metrics_names
                vloss = float(res[0]) if isinstance(res, (list, tuple)) else float(res)
                if isinstance(res, (list, tuple)):
                    if "sparse_categorical_accuracy" in names:
                        vacc = float(res[names.index("sparse_categorical_accuracy")])
                    elif "accuracy" in names:
                        vacc = float(res[names.index("accuracy")])
                    else:
                        vacc = float("nan")
                else:
                    vacc = float("nan")
                epoch_recorders[i].append({"epoch": int(epoch + 1), "val_acc": vacc, "val_loss": vloss})

    return models
@dataclass
class Individual:
    name: int
    blueprint_mark: int
    blueprint_ref: Blueprint
    model: Optional[keras.Model] = None
    scores: Optional[List[float]] = None
    species: Optional[int] = None
    gen: Optional[int] = None

class Population:
    def __init__(self, datasets: Datasets, input_shape: Tuple[int, ...],
                 population_size: int = 6, compiler: Optional[Dict[str, Any]] = None):
        self.datasets = datasets
        self.input_shape = input_shape
        self.population_size = population_size
        self.compiler = compiler or {
            "loss": "categorical_crossentropy",
            "optimizer": keras.optimizers.Adam(learning_rate=1e-3),
            "metrics": ["accuracy"],
        }
        self.historical_marker = HistoricalMarker()
        self.name_generator = NameGenerator()
        self.modules: Dict[int, Module] = {}
        self.blueprints: Dict[int, Blueprint] = {}
        self.module_species: List[Species] = []
        self.blueprint_species: List[Species] = []
        self.module_share_mode: str = "module"
        self.parallel_sync: bool = True
        self.ema_beta: float = 0.0
        self.fast_debug: bool = True
        self.max_steps_per_epoch: int = 15
        self.disable_graph_plots: bool = True
        self.ea_batch_size: int = 128
        self.magic_t: bool = False
        self.magic_t_sync_every = 1
        self.magic_t_elastic_tau = 0.0
        self.magic_t_migrate_p = 0.1
        self.magic_t_noise_std = 0.0
        self.magic_t_log_keys = True
        self.conflict_report_every: int = 3
        self._global_best_row = None      # tracks best row across all gens
        self.convergence_log = []         # list of dicts for convergence CSV
        self.convergence_epoch_log = []
        # NEW: performance knobs for PCGrad
        self.pcgrad_every: int = 2           # run PCGrad every N steps (default throttle)
        self.pcgrad_sample_frac: float = 0.5 # sample 50% shared vars per PCGrad pass

        self.magic_t_limit_mutations_per_child: int = 1
        self.magic_t_burst_every: int = 5
        self.magic_t_burst_minmax: Tuple[int, int] = (3, 5)
        self._last_iteration: Optional[List[List[Any]]] = None

    # ----- create spaces -----
    def _sample_param(self, space):
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

    def _generate_module_graph(self, global_configs, possible_components, possible_complementary_components=None):
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

    def create_module_population(self, module_population_size, global_configs, possible_components, possible_complementary_components=None):
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

    def apply_kmeans_speciation(self, items: List[Any], n_species: int):
        from sklearn.cluster import KMeans
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
        logging.log(LOG_DETAIL, f"KMeans generated {len(module_species)} species using: { [it.get_kmeans_representation() for it in items] }.")
        self.module_species = []
        for i, group in enumerate(module_species):
            for m in group:
                try: m.species = i
                except Exception: pass
            sp = Species(name=i, species_type="module", group=group, properties=None, starting_generation=0)
            self.module_species.append(sp)
            logging.log(LOG_DETAIL, f"Created {len(module_species)} module species.")
            logging.log(LOG_DETAIL, f"Module species {i}: {[m.mark for m in group]}")
        return module_species, module_classifications

    def _choose_module_of_type(self, comp: ModuleComposition) -> Module:
        pool = [m for m in self.modules.values() if m.layer_type == comp]
        if not pool:
            pool = list(self.modules.values())
        return copy.deepcopy(RNG.choice(pool))

    def _generate_blueprint_graph(
        self,
        global_configs, possible_components, possible_complementary_components,
        input_configs, possible_inputs, possible_complementary_inputs,
        output_configs, possible_outputs, possible_complementary_outputs,
    ):
        g = nx.DiGraph()
        nodes = []
        in_mod = self._choose_module_of_type(ModuleComposition.INPUT)
        nid = "input-0-0"; g.add_node(nid, node_def=in_mod); nodes.append(nid)
        n_mods = self._sample_param(global_configs.get("module_range", ([1, 3], "int")))
        for i in range(n_mods):
            m = self._choose_module_of_type(ModuleComposition.INTERMED)
            nid = f"intermed-{i}-0"; g.add_node(nid, node_def=m); g.add_edge(nodes[-1], nid); nodes.append(nid)
        out_mod = self._choose_module_of_type(ModuleComposition.OUTPUT)
        nid = "output-0-0"; g.add_node(nid, node_def=out_mod); g.add_edge(nodes[-1], nid); nodes.append(nid)
        mark = self.historical_marker.mark_blueprint()
        bp = Blueprint(modules=None, input_shape=self.input_shape, module_graph=g, mark=mark)
        return bp

    def create_blueprint_population(self, blueprint_population_size, global_configs, possible_components, possible_complementary_components, input_configs, possible_inputs, possible_complementary_inputs, output_configs, possible_outputs, possible_complementary_outputs):
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
        logging.log(LOG_DETAIL, f"KMeans generated {len(blueprint_species)} species using: { [it.get_kmeans_representation() for it in items] }.")
        self.blueprint_species = []
        for i, group in enumerate(blueprint_species):
            sp = Species(name=i, species_type="blueprint", group=group, properties=None, starting_generation=0)
            self.blueprint_species.append(sp)
        logging.log(LOG_DETAIL, f"Created {len(blueprint_species)} blueprint species.")
        for i, sp in enumerate(self.blueprint_species):
            logging.log(LOG_DETAIL, f"Blueprint species {i}: {[b.mark for b in sp.group]}")
        return blueprint_species, blueprint_classifications

    def _module_species_id(self, module_obj: Module) -> int:
        sid = getattr(module_obj, "species", None)
        if sid is not None:
            return int(sid)
        for sp in self.module_species:
            if module_obj in sp.group:
                try: module_obj.species = sp.name
                except Exception: pass
                return int(sp.name)
        return -1

    def _apply_module(self, x, module_obj: Module, module_scope: Tuple[int, str]):
        g = module_obj.component_graph
        order = list(nx.topological_sort(g)) if g.number_of_nodes() > 0 else []
        if not order:
            return x
        species_id = getattr(module_obj, "species", -1)
        outer_tag = module_scope[1]
        for pos, node in enumerate(order):
            comp: Component = g.nodes[node]["node_def"]
            ctype, cparams = comp.representation
            params = dict(cparams)
            in_ch = _infer_in_channels_from_tensor(x)
            position_tag = f"{outer_tag}#{pos}"

            # NEW: prevent maxpool crash on tiny spatial dims
            if ctype in ("max_pooling2d", "MaxPooling2D", "maxpool2d", "maxpool"):
                h = x.shape[1]
                w = x.shape[2]
                if (h is not None and int(h) < 2) or (w is not None and int(w) < 2):
                    # degrade pooling to identity if the map is already 1x1
                    x = KL.Lambda(lambda t: t, name=f"noop_pool_{outer_tag}_{pos}")(x)
                    continue
                params.setdefault("padding", "same")
            if self.module_share_mode == "module":
                scope = (species_id, position_tag)
            elif self.module_share_mode == "layer":
                scope = (None, position_tag)
            elif self.module_share_mode == "global":
                scope = None
            else:
                scope = ("nonce", id(object()))
            lyr = _get_or_create_shared_layer(ctype, in_ch, params, scope, self.module_share_mode)
            try:
                x = lyr(x)
            except TypeError:
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
        inp = KL.Input(shape=self.input_shape, name=f"input_{bp.mark}")
        logging.log(LOG_DETAIL, f"Added Input layer: {inp}")
        topo = list(nx.topological_sort(g))
        x = inp
        for nid in topo:
            m: Module = g.nodes[nid]["node_def"]
            label = nid.rsplit("-", 1)[0] if "-" in nid else nid
            module_scope = (0, label)
            x = self._apply_module(x, m, module_scope)
        y_train = self.datasets.training[1]
        num_classes = int(y_train.shape[1]) if len(y_train.shape) > 1 else int(np.max(y_train) + 1)
        if len(x.shape) > 2:
            x = KL.GlobalAveragePooling2D()(x)
        if int(x.shape[-1]) != num_classes:
            x = KL.Dense(num_classes)(x)
        out = KL.Activation("softmax", name="predictions")(x)
        model = Model(inputs=inp, outputs=out, name=f"indiv_{indiv.name}_bp{bp.mark}")
        loss, metrics = _loss_and_metrics_for(y_train)
        optimizer = _safe_eval_optimizer(self.compiler.get("optimizer", keras.optimizers.Adam(learning_rate=0.001)))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model


    # ----- mutations (minimal) -----
    def _mutate_component_params_in_place(self, comp: Component, possible_components: Dict[str, Any]):
        ctype, params = comp.representation
        meta = possible_components.get(ctype)
        if not meta:
            return
        _, param_spaces = meta
        if not param_spaces:
            return
        key = RNG.choice(list(param_spaces.keys()))
        new_val = self._sample_param(param_spaces[key])
        params = dict(params); params[key] = new_val
        if ctype == "conv2d":
            ks = params.get("kernel_size", 3)
            params["kernel_size"] = ks if isinstance(ks, (tuple, list)) else int(ks)
            params.setdefault("padding", "same")
        comp.representation = (ctype, params)

    def _mutate_blueprint_once(self, bp: Blueprint, possible_components: Dict[str, Any]):
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

    # ----- EA loop -----
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
                    model=None, scores=None, species=None, gen=gen_idx,
                )
            )
        logging.info(f"[POP] Created {len(indivs)} individuals: {[(indiv.name, indiv.blueprint_mark) for indiv in indivs]}")
        logging.log(LOG_DETAIL, f"Created individuals for blueprints: {[(indiv.name, indiv.blueprint_mark) for indiv in indivs]}")
        return indivs
    
    def _evaluate_generation_sync(
    self,
    individuals: List[Individual],
    training_epochs: int,
    validation_split: float,
    gen_idx: int | None = None,   # used for filenames
    ):
        x_train, y_train = self.datasets.training
        x_val, y_val = self.datasets.test

        models: List[keras.Model] = []
        opts: List[keras.optimizers.Optimizer] = []
        for indiv in individuals:
            m = self.assemble_model_for_individual(indiv)
            models.append(m)
            indiv.model = m
            opts.append(m.optimizer)

        batch_size = int(getattr(self, "ea_batch_size", 128))
        train_ds = (
            tf.data.Dataset
            .from_tensor_slices((x_train, y_train))
            .shuffle(8192, reshuffle_each_iteration=True)
            .repeat()
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        loss_fn, _ = _loss_and_metrics_for(y_train)

        logging.info(f"[SYNC] Iterating fitness (parallel-style) over {len(models)} individuals")

        # Per-model recorders
        epoch_recorders: List[list] = [[] for _ in range(len(models))]  # one list per model
        step_recorders:  List[list] = [[] for _ in range(len(models))]  # one list per model

        # Train
        train_individuals_synchronously(
            models=models,
            optimizers=opts,
            loss_fn=loss_fn,
            train_ds=train_ds,
            steps_per_epoch=self.max_steps_per_epoch,
            epochs=training_epochs,
            validation_data=(x_val, y_val),
            ema_beta=self.ema_beta,
            magic_t=self.magic_t,
            verbose=1,
            report_every=int(getattr(self, "conflict_report_every", 3)),
            pcgrad_every=int(getattr(self, "pcgrad_every", 1)),
            pcgrad_sample_frac=float(getattr(self, "pcgrad_sample_frac", 1.0)),
            epoch_recorders=epoch_recorders,   # per-model epochs
            step_recorders=step_recorders,     # per-model steps
        )

        # Score all individuals
        iteration_summary = []
        for indiv in individuals:
            scores = indiv.model.evaluate(x_val, y_val, verbose=0)
            indiv.scores = [
                float(scores[0]),
                float(scores[1] if len(scores) > 1 else float("nan")),
            ]
            feat = indiv.blueprint_ref.get_kmeans_representation()
            indiv.species = 0
            iteration_summary.append([
                indiv.name, indiv.blueprint_mark, indiv.scores, feat, indiv.species, indiv.gen
            ])

        # Best-of-generation (max acc, tie-break lower loss)
        gen_best_row = max(iteration_summary, key=lambda r: (r[2][1], -r[2][0]))
        best_name = gen_best_row[0]
        name2idx = {indiv.name: i for i, indiv in enumerate(individuals)}
        best_idx = name2idx.get(best_name, 0)

        # Filenames (+ A/B tag)
        try:
            os.makedirs("models", exist_ok=True)
        except Exception:
            pass
        eff_gen = int(gen_idx) if gen_idx is not None else int(individuals[0].gen or 0)
        label = getattr(self, "ab_label", None)     # "A", "B", or None/"single"
        tag = f"{label}_" if label else ""

        # Write best model's per-step training loss
        try:
            steps = step_recorders[best_idx]
            out_csv = os.path.join("models", f"best_steps_{tag}gen{eff_gen}.csv")
            import csv as _csv
            with open(out_csv, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=["iteration", "train_loss"])
                w.writeheader()
                for i, loss_val in enumerate(steps, start=1):
                    w.writerow({"iteration": i, "train_loss": float(loss_val)})
            logging.info("[EA][GEN %d] wrote best-model per-step CSV: %s (n=%d)", eff_gen, out_csv, len(steps))
        except Exception as e:
            logging.warning("Could not write best-model per-step CSV: %s", e)

        # Write best model's per-epoch validation curve (optional)
        try:
            epochs_best = epoch_recorders[best_idx]
            if epochs_best:
                ep_csv = os.path.join("models", f"best_epochs_{tag}gen{eff_gen}.csv")
                import csv as _csv
                with open(ep_csv, "w", newline="") as f:
                    w = _csv.DictWriter(f, fieldnames=["epoch", "val_loss", "val_acc"])
                    w.writeheader()
                    for row in epochs_best:
                        w.writerow({
                            "epoch": int(row.get("epoch", 0)),
                            "val_loss": float(row.get("val_loss", float("nan"))),
                            "val_acc": float(row.get("val_acc", float("nan"))),
                        })
                logging.info("[EA][GEN %d] wrote best-model per-epoch CSV: %s (n=%d)", eff_gen, ep_csv, len(epochs_best))
        except Exception as e:
            logging.warning("Could not write best-model per-epoch CSV: %s", e)

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
        all_iterations = []

        for gen_idx in range(generations):
            logging.info(f" -- Iterating generation {gen_idx} --")
            logging.log(
                LOG_DETAIL,
                f"Currently {len(self.modules)} modules, {len(self.blueprints)} blueprints, latest iteration: {self._last_iteration}"
            )

            # --- lightweight mutations
            total_mut = 0
            if gen_idx > 0:
                bp_marks = list(self.blueprints.keys())
                RNG.shuffle(bp_marks)
                n_children = min(self.population_size, len(bp_marks))
                for k in range(n_children):
                    bp = self.blueprints[bp_marks[k]]
                    if RNG.random() < mutation_rate:
                        self._mutate_blueprint_once(bp, possible_components)
                        total_mut += 1
            logging.log(LOG_DETAIL, f"[SEARCH] generation {gen_idx} total_mutations={total_mut}")

            # --- assemble & evaluate this generation
            ws_reset()  # reset share counters for this gen (keeps layers themselves unless cleared elsewhere)

            gen_t0 = time.time()
            individuals = self._create_individuals_from_blueprints(gen_idx)
            iteration = self._evaluate_generation_sync(individuals, training_epochs, validation_split, gen_idx=gen_idx)
            gen_t1 = time.time()
            gen_wall = gen_t1 - gen_t0

            # per-gen best (prefer higher acc, tie-break lower loss)
            gen_best = max(iteration, key=lambda r: (r[2][1], -r[2][0]))

            # update global best
            if (self._global_best_row is None) or (
                (gen_best[2][1], -gen_best[2][0]) > (self._global_best_row[2][1], -self._global_best_row[2][0])
            ):
                self._global_best_row = gen_best

            # weight-sharing reuse stats (optional)
            created = reused = 0
            try:
                created, reused = ws_counts()
            except Exception:
                pass
            reuse_ratio = reused / max(1, (created + reused))

            logging.info(
                "[WS][GEN] gen=%d created=%d reused=%d reuse_ratio=%.3f wall=%.2fs",
                gen_idx, int(created), int(reused), reuse_ratio, gen_wall
            )

            # convergence log row (for CSV/plotting)
            self.convergence_log.append({
                "gen": int(gen_idx),
                "gen_wall_sec": float(gen_wall),
                "ea_best_acc_gen": float(gen_best[2][1]),
                "ea_best_loss_gen": float(gen_best[2][0]),
                "ea_best_acc_cum": float(self._global_best_row[2][1]),
                "ea_best_loss_cum": float(self._global_best_row[2][0]),
                "ws_created": int(created),
                "ws_reused": int(reused),
                "ws_reuse_ratio": float(reuse_ratio),
            })

            logging.info(
                "[EA][GEN %d] wall=%.2fs best(gen): acc=%.4f loss=%.4f | best(cum): acc=%.4f loss=%.4f | reuse c=%d r=%d (%.3f)",
                gen_idx, gen_wall,
                self.convergence_log[-1]["ea_best_acc_gen"],  self.convergence_log[-1]["ea_best_loss_gen"],
                self.convergence_log[-1]["ea_best_acc_cum"],  self.convergence_log[-1]["ea_best_loss_cum"],
                created, reused, reuse_ratio
            )

            # keep the original list for compatibility, but store gen_best (not just last gen)
            all_iterations.append([f"generation {gen_idx}", gen_best])

            # --- bookkeeping
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
        print(f"Current {len(individuals)} individuals: [", end=""); print(", ".join(str(ind.name) for ind in individuals), end=""); print("]")

        print("Current %d blueprints:" % len(self.blueprints))
        print("[Mark, Test loss, Test acc, Species, Node Count, Edge Count, Neuron Count]")
        bp_rows = []
        for b in self.blueprints.values():
            n = len(b.module_graph.nodes()); e = len(b.module_graph.edges()); s = b.get_blueprint_size()
            bp_rows.append([b.mark] + [int(b.weighted_scores[0]), int(b.weighted_scores[1])] + [b.species if b.species is not None else 0, n, e, s])
        try: print(np.array(bp_rows))
        except Exception: print(bp_rows)

        print(f"Current {len(self.module_species)} modules:")
        print("[Mark, Test loss, Test acc, Species, Node Count, Edge Count, Neuron Count]")
        mod_rows = []
        for m in self.modules.values():
            n = len(m.component_graph.nodes()); e = len(m.component_graph.edges()); s = m.get_module_size()
            mod_rows.append([m.mark] + [int(m.weighted_scores[0]), int(m.weighted_scores[1])] + [m.species if getattr(m, 'species', None) is not None else 0, n, e, s])
        try: print(np.array(mod_rows))
        except Exception: print(mod_rows)
        print("\n --------------------------------------------\n")

    # utilities for runner
    def log_shared_state(self, tag: str):
        logging.info(f"[REGISTRY][{tag}] entries={len(REGISTRY)}")

    def return_best_individual(self) -> Individual:
        """
        Always return the globally best individual discovered so far.
        Prefers self._global_best_row, which tracks the highest validation accuracy
        (and lowest loss in case of tie) across all generations.
        Falls back to best from last iteration, or any blueprint if no data yet.
        """
        # 1️⃣ If no generations have been evaluated yet — fallback
        if getattr(self, "_global_best_row", None) is None and not self._last_iteration:
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

        # 2️⃣ Prefer global best if available
        best_row = getattr(self, "_global_best_row", None)

        # 3️⃣ Fallback to best of last generation if global not yet set
        if best_row is None and self._last_iteration:
            candidates = [r for r in self._last_iteration
                        if isinstance(r, (list, tuple)) and len(r) >= 3 and isinstance(r[2], (list, tuple))]
            if candidates:
                best_row = max(candidates, key=lambda r: (float(r[2][1]), -float(r[2][0])))

        # 4️⃣ Still nothing — pick first available blueprint
        if best_row is None:
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

        # 5️⃣ Build the Individual from best row
        try:
            name, bp_mark, scores, feat, species, gen = best_row[:6]
        except Exception:
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

        return Individual(
            name=name,
            blueprint_mark=bp_mark,
            blueprint_ref=self.blueprints[bp_mark],
            model=None,
            scores=scores,
            species=species,
            gen=gen,
        )

    def train_full_model(
        self,
        indiv: Individual,
        epochs: int,
        validation_split: float,
        custom_fit_args: Optional[Dict[str, Any]] = None,
        warm_start: bool = False,
    ):
        logging.info(f"[POP] Final retrain start for individual {indiv.name} (bp {indiv.blueprint_mark})")
        K.clear_session()
        if not warm_start:
            REGISTRY.clear()    # fresh weights
        model = self.assemble_model_for_individual(indiv)
        indiv.model = model
        y_train = self.datasets.training[1]
        loss, metrics = _loss_and_metrics_for(y_train)
        optimizer = _safe_eval_optimizer(self.compiler.get("optimizer", keras.optimizers.Adam(learning_rate=0.001)))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if custom_fit_args:
            fit_args = dict(custom_fit_args)
            if "generator" in fit_args and "x" not in fit_args:
                fit_args["x"] = fit_args.pop("generator")
            if "steps_per_epoch" not in fit_args and "x" in fit_args and hasattr(fit_args["x"], "__len__"):
                try: fit_args["steps_per_epoch"] = len(fit_args["x"])
                except Exception: pass
            history = model.fit(**fit_args)
        else:
            x_train, y_train = self.datasets.training
            x_val, y_val = self.datasets.test
            history = model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                verbose=1,
                batch_size=128,
            )
        x_val, y_val = self.datasets.test
        scores = model.evaluate(x_val, y_val, verbose=0)
        logging.info(f"[POP] Final retrain scores: {scores}")
        return scores

