"""NumPy <-> JAX backend dispatch.

All numerical modules import ``xp`` (the array module) and ``jit`` from here so
the same code runs on NumPy and JAX without conditional imports scattered
throughout the codebase.

Backend selection order:
1. ``hopfion.backend.use("jax")`` / ``use("numpy")`` at runtime
2. ``HOPFION_BACKEND`` environment variable
3. Default: ``numpy``
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as _np

_BACKEND: str = "numpy"
_xp: Any = _np
_jit: Callable = lambda f, *a, **kw: f  # noqa: E731
_vmap: Callable | None = None


def _load_jax():
    import jax
    import jax.numpy as jnp

    return jax, jnp


def use(name: str) -> None:
    """Switch the active backend. ``name`` is ``"numpy"`` or ``"jax"``."""
    global _BACKEND, _xp, _jit, _vmap
    name = name.lower()
    if name == "numpy":
        _xp = _np
        _jit = lambda f, *a, **kw: f  # noqa: E731
        _vmap = None
        _BACKEND = "numpy"
    elif name == "jax":
        jax, jnp = _load_jax()
        _xp = jnp
        _jit = jax.jit
        _vmap = jax.vmap
        _BACKEND = "jax"
    else:
        raise ValueError(f"Unknown backend: {name!r}. Choose 'numpy' or 'jax'.")


def name() -> str:
    return _BACKEND


def xp():
    """Return the active array module (numpy or jax.numpy)."""
    return _xp


def jit(fn: Callable, *args: Any, **kwargs: Any) -> Callable:
    """JIT-compile if backend supports it, else return the function unchanged."""
    return _jit(fn, *args, **kwargs)


def vmap(fn: Callable, *args: Any, **kwargs: Any) -> Callable:
    """Vectorizing transform (JAX only). Falls back to np.vectorize semantics."""
    if _vmap is not None:
        return _vmap(fn, *args, **kwargs)
    return _np.vectorize(fn, signature=kwargs.get("signature"))


def asarray(x: Any, dtype: Any = None) -> Any:
    return _xp.asarray(x, dtype=dtype) if dtype is not None else _xp.asarray(x)


def to_numpy(x: Any) -> _np.ndarray:
    """Convert backend array to plain NumPy (no-op in numpy mode)."""
    if _BACKEND == "jax":
        return _np.asarray(x)
    return _np.asarray(x)


_env = os.environ.get("HOPFION_BACKEND")
if _env:
    use(_env)
