"""JAX lax.scan relaxation path agrees with the NumPy loop.

Guarded by importorskip — runs wherever JAX is installed. Uses x64 so the
comparison is at double precision (the jax backend defaults to float32, where
agreement is only ~1e-6)."""
import numpy as np
import pytest


def test_jax_scan_relax_matches_numpy():
    pytest.importorskip("jax")
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from hopfion import backend
    from hopfion.energy import EnergyParams
    from hopfion.field import add_perturbation, hopfion
    from hopfion.grid import Grid
    from hopfion.llg import relax

    g = Grid(20, 20, 20, 0.4, 0.4, 0.4, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7, easy_axis=(0, 0, 1))
    m0 = np.asarray(add_perturbation(hopfion(g, R=1.5), amplitude=0.05, seed=2))

    try:
        backend.use("numpy")
        m_np = np.asarray(relax(m0, g, ep, n_steps=40, dt=0.004))
        backend.use("jax")
        assert backend.name() == "jax"
        # step_callback is None -> compiled lax.scan path
        m_j = np.asarray(relax(jnp.asarray(m0), g, ep, n_steps=40, dt=0.004))
    finally:
        backend.use("numpy")   # never leak the jax backend into other tests

    assert np.abs(m_j - m_np).max() < 1e-6
    norm = np.sqrt((m_j * m_j).sum(axis=0))
    assert np.abs(norm - 1.0).max() < 1e-6
