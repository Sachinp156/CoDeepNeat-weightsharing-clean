# base/shared_layers.py
from typing import Dict, Hashable, Callable, Tuple, Iterable, Optional
import threading
import itertools
from contextlib import contextmanager
import tensorflow as tf

keras = tf.keras

# ---------- key helpers ----------
def _canon_params(d: dict) -> Tuple:
    return tuple(sorted(d.items()))

def default_key(kind: str, in_ch: int, params: dict, extra: Tuple = ()) -> Tuple:
    return (kind, in_ch, _canon_params(params), extra)

# ---------- ignore list for sharing ----------
_IGNORE_SHARE_TYPES = {"batchnormalization", "batch_norm", "batchnorm", "bn", "fusedbatchnorm"}

def _should_ignore_sharing(component_type: Optional[str]) -> bool:
    if not component_type:
        return False
    return component_type.replace(" ", "").lower() in _IGNORE_SHARE_TYPES

# Unique suffix counter for non-shared layers (e.g., BN) to avoid name collisions
_unique_nonce = itertools.count()

def _unique_name(base: str) -> str:
    return f"{base}_noshare__{next(_unique_nonce)}"

# ---------- simple global counters for CREATE/REUSE ----------
WS_CREATE = 0
WS_REUSE  = 0

def ws_reset():
    global WS_CREATE, WS_REUSE
    WS_CREATE = 0
    WS_REUSE  = 0

def ws_counts():
    return WS_CREATE, WS_REUSE

# ---------- registry ----------
class SharedLayerRegistry:
    def __init__(self):
        self._layers: Dict[Hashable, keras.layers.Layer] = {}
        self._masks: Dict[tf.Variable, tf.Variable] = {}
        self._lock = threading.Lock()

        self._tls = threading.local()
        self._tls.scope_stack = []

    # ----- scope helpers -----
    def _ensure_stack(self):
        if not hasattr(self._tls, "scope_stack"):
            self._tls.scope_stack = []

    @contextmanager
    def scope(self, *scope_parts: Hashable):
        self._ensure_stack()
        self._tls.scope_stack.append(tuple(scope_parts))
        try:
            yield
        finally:
            if getattr(self._tls, "scope_stack", []):
                self._tls.scope_stack.pop()

    def current_scope(self) -> Tuple:
        self._ensure_stack()
        if not self._tls.scope_stack:
            return ()
        flat: Tuple[Hashable, ...] = ()
        for t in self._tls.scope_stack:
            flat += t
        return flat

    # ----- core ops -----
    def clear(self):
        with self._lock:
            self._layers.clear()
            self._masks.clear()

    def get_or_create(
        self,
        key: Hashable,
        factory: Callable[[], keras.layers.Layer]
    ) -> keras.layers.Layer:
        global WS_CREATE, WS_REUSE
        with self._lock:
            if key not in self._layers:
                layer = factory()
                self._layers[key] = layer
                WS_CREATE += 1
                print(f"[LIVE-SHARE] CREATE key={key} layer_id={id(layer)}")
            else:
                layer = self._layers[key]
                WS_REUSE += 1
                print(f"[LIVE-SHARE] REUSE  key={key} layer_id={id(layer)}")
            return layer

    def mask_for(self, var: tf.Variable) -> tf.Variable:
        with self._lock:
            if var not in self._masks:
                self._masks[var] = tf.Variable(
                    tf.ones_like(var), trainable=False,
                    name=var.name.split(':')[0] + "/mask"
                )
            return self._masks[var]

    def apply_masks_after_step(self):
        with self._lock:
            layers = list(self._layers.values())
        for layer in layers:
            for v in layer.trainable_variables:
                m = self._masks.get(v)
                if m is not None:
                    v.assign(v * m)

    # ----- python mapping protocol -----
    def __len__(self) -> int:
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

    # ----- convenience -----
    def size(self) -> int:
        return len(self)

    def keys(self):
        with self._lock:
            return list(self._layers.keys())

    def values(self):
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

# ---------- convenience used by Population._get_or_create_shared_layer ----------
def make_or_share_layer(
    *,
    component_type: str,
    in_channels: Optional[int],
    params: dict,
    scope_key: Tuple,
    share_mode: str,
    name_factory: Callable[[Tuple], str],
    ctor: Callable[[dict, Optional[str]], keras.layers.Layer],
):
    """
    Centralized policy:
      - If BN (or listed type): do NOT share; create a fresh layer with a guaranteed-unique name.
      - Else: share via REGISTRY using the provided key.
    """
    if _should_ignore_sharing(component_type):
        lname = _unique_name(component_type.lower())
        layer = ctor(params, lname)
        print(f"[LIVE-SHARE][SKIP:{component_type}] name={lname} layer_id={id(layer)}")
        return layer

    key = (component_type, in_channels, _canon_params(params),
           scope_key if share_mode != "off" else ("nonce", id(object())))
    def factory():
        lname = name_factory(key)
        return ctor(params, lname)
    return REGISTRY.get_or_create(key, factory)

