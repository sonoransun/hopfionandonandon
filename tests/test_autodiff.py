"""Differentiable simulation + inverse design (JAX backend, x64 for parity)."""
from dataclasses import replace

import numpy as np
import pytest


def _jax_setup():
    pytest.importorskip("jax")
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)


def test_energy_gradient_equals_minus_effective_field():
    """jax.grad of the differentiable energy reproduces the hand-coded
    -effective_field — proof the whole energy stack is differentiable."""
    _jax_setup()
    import jax.numpy as jnp

    from hopfion import backend
    from hopfion.energy import EnergyParams, effective_field
    from hopfion.field import add_perturbation, hopfion
    from hopfion.grid import Grid
    from hopfion.physics.autodiff import energy_gradient

    g = Grid(16, 16, 16, 0.4, 0.4, 0.4, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.2, D_interface=0.3, Ku=0.5, Kc=0.2,
                      easy_axis=(0, 0, 1), H_ext=(0, 0, 0.1))
    m = np.asarray(add_perturbation(hopfion(g, R=1.5), amplitude=0.05, seed=4))
    try:
        backend.use("jax")
        grad = np.asarray(energy_gradient(jnp.asarray(m), g, ep))
        H = np.asarray(effective_field(jnp.asarray(m), g, ep))
    finally:
        backend.use("numpy")
    assert np.abs(grad - (-H)).max() < 1e-8


def test_dE_dD_through_relaxation_matches_finite_difference():
    """End-to-end differentiable simulation: autodiff of the relaxed-state energy
    w.r.t. the DMI strength D, differentiating through the compiled relax_scan."""
    _jax_setup()
    import jax.numpy as jnp

    from hopfion import backend
    from hopfion.energy import EnergyParams
    from hopfion.field import add_perturbation, hopfion
    from hopfion.grid import Grid
    from hopfion.physics.autodiff import denergy_dparam_through_relax, energy_value_ep
    from hopfion.physics.integrators import relax_scan

    g = Grid(16, 16, 16, 0.4, 0.4, 0.4, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.2, Ku=0.5, easy_axis=(0, 0, 1))
    m = jnp.asarray(np.asarray(add_perturbation(hopfion(g, R=1.5), amplitude=0.05, seed=4)))
    try:
        backend.use("jax")
        d_ad = denergy_dparam_through_relax(m, g, ep, "D", n_steps=30, dt=0.003)

        def Efin(D):
            mf = relax_scan(m, g, replace(ep, D=D), 30, 0.003)
            return float(energy_value_ep(mf, g, replace(ep, D=D)))
        h = 1e-4
        d_fd = (Efin(ep.D + h) - Efin(ep.D - h)) / (2 * h)
    finally:
        backend.use("numpy")
    assert abs(d_ad - d_fd) < 1e-2 * max(abs(d_fd), 1.0)


def test_fit_scalar_inverse_design_recovers_target():
    """Gradient-descent inverse design: recover the anisotropy Ku that makes a
    fixed field hit a target energy (E is linear in Ku → exact recovery)."""
    _jax_setup()
    import jax.numpy as jnp

    from hopfion import backend
    from hopfion.field import add_perturbation, hopfion
    from hopfion.grid import Grid
    from hopfion.physics.autodiff import energy_value, fit_scalar

    g = Grid(10, 10, 10, 0.5, 0.5, 0.5, "periodic")
    m = jnp.asarray(np.asarray(add_perturbation(hopfion(g, R=1.5), amplitude=0.1, seed=5)))
    easy, H = (0, 0, 1), (0, 0, 0)
    Ku_true = 0.7
    try:
        backend.use("jax")
        # sensitivity S = -dE/dKu (constant since E is linear in Ku); set lr ~ 0.5/S^2
        S = float(np.sum(np.asarray(m)[2] ** 2) * g.dV)   # ≈ -dE/dKu
        target = float(energy_value(m, g, 1.0, 0.0, Ku_true, easy, H))

        def loss(Ku):
            return (energy_value(m, g, 1.0, 0.0, Ku, easy, H) - target) ** 2
        Ku_fit = fit_scalar(loss, x0=0.0, lr=0.5 / (S * S), n_iter=300)
    finally:
        backend.use("numpy")
    assert abs(Ku_fit - Ku_true) < 0.02
