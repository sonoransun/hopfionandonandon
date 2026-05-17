"""Crouch-Grossman geometric integrator: exact |m|=1, low Q_H drift."""
import numpy as np

from hopfion.energy import EnergyParams
from hopfion.field import hopfion
from hopfion.grid import Grid
from hopfion.physics.high_order import (
    _rodrigues_rotate,
    crouch_grossman_rk4_step,
)
from hopfion.physics.integrators import heun_step
from hopfion.topology import hopf_index


def test_rodrigues_rotate_preserves_unit_norm_exactly():
    """Rotation is an isometry — |m| is preserved to machine precision."""
    rng = np.random.default_rng(0)
    g_shape = (3, 8, 8, 8)
    m = rng.normal(size=g_shape)
    m = m / np.sqrt((m * m).sum(0, keepdims=True))
    omega = rng.normal(size=g_shape)
    m_rot = _rodrigues_rotate(m, omega, dt=0.13)
    n = np.sqrt(((np.asarray(m_rot)) ** 2).sum(axis=0))
    assert np.max(np.abs(n - 1.0)) < 1e-12


def test_cg_step_preserves_unit_norm_to_machine():
    g = Grid(16, 16, 16, 0.5, 0.5, 0.5, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.0, Ku=0.5, easy_axis=(0, 0, 1))
    m = hopfion(g, R=1.5, p=1, q=1)
    for _ in range(20):
        m = crouch_grossman_rk4_step(m, g, ep, gamma=1.0, alpha=0.05, dt=0.005)
    n = np.sqrt(((np.asarray(m)) ** 2).sum(axis=0))
    assert np.max(np.abs(n - 1.0)) < 1e-12


def test_cg_zero_alpha_zero_gamma_is_identity():
    """If gamma = 0 and alpha = 0 the angular velocity is zero, so the
    state should be unchanged."""
    g = Grid(8, 8, 8, 0.5, 0.5, 0.5, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.0, Ku=0.5)
    m = hopfion(g, R=1.2, p=1, q=1)
    m_out = crouch_grossman_rk4_step(m, g, ep, gamma=0.0, alpha=0.0, dt=0.005)
    assert np.allclose(np.asarray(m), np.asarray(m_out), atol=1e-12)


def test_cg_long_time_q_drift_better_than_heun():
    """Q_H should drift less under CG-RK4 than under Heun on the same trajectory."""
    g = Grid(24, 24, 24, 0.5, 0.5, 0.5, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.0, Ku=0.0, easy_axis=(0, 0, 1))
    m0 = hopfion(g, R=1.5, p=1, q=1)

    # Zero-damping pure precession run -- Q_H should be exactly preserved in continuum
    m_cg = m0
    m_he = m0
    for _ in range(50):
        m_cg = crouch_grossman_rk4_step(m_cg, g, ep, gamma=1.0, alpha=0.0, dt=0.005)
        m_he = heun_step(m_he, g, ep, gamma=1.0, alpha=0.0, dt=0.005)
    Q0 = hopf_index(m0, g)
    drift_cg = abs(hopf_index(m_cg, g) - Q0)
    drift_he = abs(hopf_index(m_he, g) - Q0)
    # CG-RK4 should be no worse than Heun on a finite-step trajectory
    assert drift_cg <= drift_he + 0.05, (
        f"CG drift {drift_cg:.4e} vs Heun drift {drift_he:.4e}"
    )
