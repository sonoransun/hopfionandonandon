"""Advanced integrators: CFL-free implicit midpoint + JAX full-dynamics scan."""
import numpy as np
import pytest

from hopfion.energy import EnergyParams, total_energy
from hopfion.field import add_perturbation, hopfion
from hopfion.grid import Grid
from hopfion.physics.integrators import heun_step, implicit_midpoint_step


def _setup():
    g = Grid(20, 20, 20, 0.4, 0.4, 0.4, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7, easy_axis=(0, 0, 1))
    m0 = np.asarray(add_perturbation(hopfion(g, R=1.5), amplitude=0.05, seed=3))
    return g, ep, m0


def test_implicit_midpoint_conserves_energy_in_conservative_limit():
    """With damping off (α=0), explicit Heun's energy drifts/diverges while the
    near-symplectic implicit midpoint keeps it bounded — its raison d'être for
    long-time precessional dynamics."""
    g = Grid(16, 16, 16, 0.5, 0.5, 0.5, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.0, Ku=0.3, easy_axis=(0, 0, 1))
    m0 = np.asarray(add_perturbation(hopfion(g, R=1.8), amplitude=0.02, seed=3))
    dt, N = 0.01, 400
    E0 = total_energy(m0, g, ep)
    m_h, m_i = m0, m0
    for _ in range(N):
        m_h = heun_step(m_h, g, ep, 1.0, 0.0, dt)
        m_i = implicit_midpoint_step(m_i, g, ep, 1.0, 0.0, dt)
    drift_h = abs(total_energy(m_h, g, ep) - E0)
    drift_i = abs(total_energy(m_i, g, ep) - E0)
    assert drift_i < 0.2 * abs(E0)                    # midpoint energy stays bounded
    assert drift_i < 0.05 * drift_h                   # ≫ better than Heun
    n = np.sqrt((np.asarray(m_i) ** 2).sum(axis=0))
    assert np.abs(n - 1.0).max() < 1e-10


def test_implicit_midpoint_agrees_with_heun_at_small_dt():
    g, ep, m0 = _setup()
    dt = 0.002
    m_h, m_i = m0, m0
    for _ in range(15):
        m_h = heun_step(m_h, g, ep, 1.0, 0.1, dt)
        m_i = implicit_midpoint_step(m_i, g, ep, 1.0, 0.1, dt, iters=6)
    assert np.abs(np.asarray(m_h) - np.asarray(m_i)).max() < 1e-2   # 2nd-order agreement


def test_jax_integrate_scan_matches_numpy():
    pytest.importorskip("jax")
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from hopfion import backend
    from hopfion.llg import LLGParams, integrate

    g, ep, m0 = _setup()
    lp = LLGParams(gamma=1.0, alpha=0.1, dt=0.004)
    try:
        backend.use("numpy")
        m_np = np.asarray(integrate(m0, g, ep, lp, n_steps=30))
        backend.use("jax")
        m_j = np.asarray(integrate(jnp.asarray(m0), g, ep, lp, n_steps=30))  # → integrate_scan
    finally:
        backend.use("numpy")
    assert np.abs(m_j - m_np).max() < 1e-6
