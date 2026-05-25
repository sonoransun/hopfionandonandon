"""Differentiable energy + inverse design (JAX backend).

The pipeline's ``energy.total_energy`` / ``topology.hopf_index`` cast their result
to a Python ``float``, which severs ``jax.grad``. This module provides a
*differentiable re-expression* of the (local) micromagnetic energy — identical
discretizations, but returning the raw traced scalar — so the whole simulator
becomes differentiable: gradients w.r.t. the field (== ``-effective_field``),
gradients w.r.t. material parameters *through a relaxation*, and gradient-descent
inverse design.

Requires the JAX backend (``hopfion.backend.use("jax")``); set ``jax_enable_x64``
for double-precision parity with NumPy. Covers exchange / bulk + interfacial DMI /
uniaxial + cubic anisotropy / Zeeman (the local terms; not the FFT dipolar term).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Tuple

from hopfion.backend import xp
from hopfion.energy import EnergyParams
from hopfion.grid import Grid, curl, d_axis


def energy_value(m, grid: Grid, A_ex: float, D: float, Ku: float,
                 easy_axis: Tuple[float, float, float],
                 H_ext: Tuple[float, float, float],
                 D_interface: float = 0.0, Kc: float = 0.0):
    """Total *local* micromagnetic energy as a differentiable scalar (no ``float()``).

    Mirrors the discretizations in ``energy.py`` exactly, so ``jax.grad`` w.r.t.
    ``m`` reproduces ``-dV·effective_field`` and ``jax.grad`` w.r.t. the scalar
    coefficients gives true material sensitivities.
    """
    np = xp()
    # exchange — periodic forward-difference bond energy (adjoint = 3-point Laplacian)
    fx = (np.roll(m, -1, axis=1) - m) / grid.dx
    fy = (np.roll(m, -1, axis=2) - m) / grid.dy
    fz = (np.roll(m, -1, axis=3) - m) / grid.dz
    e = A_ex * (np.sum(fx * fx) + np.sum(fy * fy) + np.sum(fz * fz))
    # bulk (Bloch) DMI
    e = e + D * np.sum(m * curl(m, grid))
    # interfacial (Néel) DMI
    if D_interface != 0.0:
        dx_mx = d_axis(m[0], 0, grid.dx, grid)
        dy_my = d_axis(m[1], 1, grid.dy, grid)
        dx_mz = d_axis(m[2], 0, grid.dx, grid)
        dy_mz = d_axis(m[2], 1, grid.dy, grid)
        e = e + D_interface * np.sum(m[2] * (dx_mx + dy_my) - (m[0] * dx_mz + m[1] * dy_mz))
    # uniaxial anisotropy
    ex, ey, ez = easy_axis
    n = (ex * ex + ey * ey + ez * ez) ** 0.5
    ex, ey, ez = ex / n, ey / n, ez / n
    mde = m[0] * ex + m[1] * ey + m[2] * ez
    e = e - Ku * np.sum(mde * mde)
    # cubic anisotropy
    if Kc != 0.0:
        mx2, my2, mz2 = m[0] * m[0], m[1] * m[1], m[2] * m[2]
        e = e + Kc * np.sum(mx2 * my2 + my2 * mz2 + mz2 * mx2)
    # Zeeman
    Hx, Hy, Hz = H_ext
    e = e - np.sum(m[0] * Hx + m[1] * Hy + m[2] * Hz)
    return e * grid.dV


def energy_value_ep(m, grid: Grid, ep: EnergyParams):
    """``energy_value`` reading the scalars from an ``EnergyParams``."""
    Ku = ep.Ku  # scalar branch (Ku_field not differentiated here)
    return energy_value(m, grid, ep.A_ex, ep.D, Ku, ep.easy_axis, ep.H_ext,
                        ep.D_interface, ep.Kc)


def energy_gradient(m, grid: Grid, ep: EnergyParams):
    """``(1/dV)·∂E/∂m`` via ``jax.grad`` — equals ``-effective_field(m, grid, ep)``.

    The self-consistency with the hand-coded ``effective_field`` is the proof that
    the entire energy stack is correctly differentiable.
    """
    import jax
    g = jax.grad(lambda mm: energy_value_ep(mm, grid, ep))(m)
    return g / grid.dV


def denergy_dparam_through_relax(m0, grid: Grid, ep: EnergyParams, param: str,
                                 n_steps: int = 40, dt: float = 0.004) -> float:
    """Sensitivity of the relaxed-state energy to a material scalar, by
    differentiating *through* the compiled ``relax_scan`` (end-to-end
    differentiable simulation). ``param`` is e.g. ``"D"`` or ``"Ku"``."""
    import jax

    from hopfion.physics.integrators import relax_scan

    def final_energy(x):
        ep_x = replace(ep, **{param: x})
        m_final = relax_scan(m0, grid, ep_x, n_steps, dt)
        return energy_value_ep(m_final, grid, ep_x)

    return float(jax.grad(final_energy)(float(getattr(ep, param))))


def fit_scalar(loss: Callable[[float], float], x0: float, lr: float = 1e-3,
               n_iter: int = 200) -> float:
    """Minimal gradient descent on a scalar via ``jax.grad`` — the kernel of an
    inverse-design loop (e.g. tune a material coefficient to a target)."""
    import jax
    grad = jax.grad(loss)
    x = float(x0)
    for _ in range(n_iter):
        x = x - lr * float(grad(x))
    return x


__all__ = [
    "energy_value",
    "energy_value_ep",
    "energy_gradient",
    "denergy_dparam_through_relax",
    "fit_scalar",
]
