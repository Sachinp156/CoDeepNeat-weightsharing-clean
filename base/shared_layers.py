# base/shared_layers.py
from typing import Dict, Hashable, Callable, Tuple, Iterable, Optional
import threading
from contextlib import contextmanager
import tensorflow as tf

keras = tf.keras


# ---- key helpers -------------------------------------------------------------

def _canon_params(d: dict) -> Tuple:
    return tuple(sorted(d.items()))


def default_key(kind: str, in_ch: int, params: dict, extra: Tuple = ()) -> Tuple:
    """
    Canonical key for a layer to be shared live across models.
    kind: e.g., "conv2d", "dense", "maxpool"
    in_ch: input channels (shape-dependent discriminator)
    params: layer hyperparameters (sorted)
    extra: optional tuple to further scope sharing (species id, node tag, etc.)
    """
    return (kind, in_ch, _canon_params(params), extra)


# ---- registry ---------------------------------------------------------------

class SharedLayerRegistry:
    """
    Thread-safe map: key -> keras.layers.Layer
    Implements mapping protocol so you can use: `key in REGISTRY`, `REGISTRY[key]`,
    `REGISTRY[key] = layer`, and `len(REGISTRY)`.
    """

    def __init__(self):
        self._layers: Dict[Hashable, keras.layers.Layer] = {}
        self._masks: Dict[tf.Variable, tf.Variable] = {}
        self._lock = threading.Lock()

        # thread-local scope stack to optionally qualify keys
        self._tls = threading.local()
        self._tls.scope_stack = []

    # ---------- scope helpers ----------
    def _ensure_stack(self):
        if not hasattr(self._tls, "scope_stack"):
            self._tls.scope_stack = []

    @contextmanager
    def scope(self, *scope_parts: Hashable):
        """Context: all layers created inside will inherit this tuple in their keys."""
        self._ensure_stack()
        self._tls.scope_stack.append(tuple(scope_parts))
        try:
            yield
        finally:
            # pop defensively
            if getattr(self._tls, "scope_stack", []):
                self._tls.scope_stack.pop()

    def current_scope(self) -> Tuple:
        """Return the flattened current scope tuple (may be empty)."""
        self._ensure_stack()
        if not self._tls.scope_stack:
            return ()
        flat: Tuple[Hashable, ...] = ()
        for t in self._tls.scope_stack:
            flat += t
        return flat
    # -----------------------------------

    # ---------- core ops ----------
    def clear(self):
        with self._lock:
            self._layers.clear()
            self._masks.clear()

    def get_or_create(
        self,
        key: Hashable,
        factory: Callable[[], keras.layers.Layer]
    ) -> keras.layers.Layer:
        with self._lock:
            if key not in self._layers:
                layer = factory()
                self._layers[key] = layer
                print(f"[LIVE-SHARE] CREATE key={key} layer_id={id(layer)}")
            else:
                layer = self._layers[key]
                print(f"[LIVE-SHARE] REUSE  key={key} layer_id={id(layer)}")
            return layer

    # optional: progressive pruning helpers
    def mask_for(self, var: tf.Variable) -> tf.Variable:
        with self._lock:
            if var not in self._masks:
                self._masks[var] = tf.Variable(
                    tf.ones_like(var), trainable=False,
                    name=var.name.split(':')[0] + "/mask"
                )
            return self._masks[var]

    def apply_masks_after_step(self):
        """Apply masks to all trainable variables of shared layers."""
        with self._lock:
            layers = list(self._layers.values())
        for layer in layers:
            for v in layer.trainable_variables:
                m = self._masks.get(v)
                if m is not None:
                    v.assign(v * m)

    # ---------- python mapping protocol ----------
    def __len__(self) -> int:
        # tolerant even if called very early
        lock = getattr(self, "_lock", None)
        if lock is None:
            return len(getattr(self, "_layers", {}))
        with lock:
            return len(self._layers)

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._layers

    def __getitem__(self, key: Hashable) -> keras.layers.Layer:
        with self._lock:
            return self._layers[key]

    def __setitem__(self, key: Hashable, value: keras.layers.Layer) -> None:
        if not isinstance(value, keras.layers.Layer):
            raise TypeError("SharedLayerRegistry only stores keras.layers.Layer instances.")
        with self._lock:
            self._layers[key] = value

    def __repr__(self) -> str:
        return f"SharedLayerRegistry(size={len(self)})"

    # ---------- convenience ----------
    def size(self) -> int:
        return len(self)

    def keys(self) -> Iterable[Hashable]:
        with self._lock:
            return list(self._layers.keys())

    def values(self) -> Iterable[keras.layers.Layer]:
        with self._lock:
            return list(self._layers.values())

    def items(self):
        with self._lock:
            return list(self._layers.items())

    def stats(self) -> dict:
        with self._lock:
            return {
                "n_layers": len(self._layers),
                "n_masks": len(self._masks),
                "scopes_active": list(getattr(self._tls, "scope_stack", [])),
            }


# Global instance
REGISTRY = SharedLayerRegistry()
