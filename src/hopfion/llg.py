"""Landau-Lifshitz-Gilbert dynamics.

Explicit form of LLG with Gilbert damping:

    dm/dt = -gamma (m x H_eff) - gamma alpha m x (m x H_eff)        (1)

(equivalent to the implicit form dm/dt = -gamma m x H + alpha m x dm/dt, after
solving for dm/dt). Units: gamma = 1 by default; calibrate against material in
post-processing.

For ground-state finding we use the purely dissipative limit (drop the
precession term in eq. (1)) -- this is "damped LLG" or "gradient descent on the
energy" in the same family as Lagrange relaxation.

The integrator stencils themselves live in ``hopfion.physics.integrators``;
this module keeps the Phase-A public surface stable while delegating the
per-step math to the new sub-package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from hopfion.energy import EnergyParams
from hopfion.grid import Grid
from hopfion.physics.integrators import (
    _cross,  # re-export for back-compat (some tests/scripts reach in)
    damped_step,
    heun_step,
    llg_rhs,
    rk4_step,
)


@dataclass
class LLGParams:
    gamma: float = 1.0    # gyromagnetic ratio (in normalized units)
    alpha: float = 0.1    # Gilbert damping
    dt: float = 0.01


def llg_step_heun(m, grid: Grid, ep: EnergyParams, lp: LLGParams, H_extra: Optional[Callable] = None):
    """One Heun (improved Euler) time step. Renormalizes |m| = 1 at the end.

    ``H_extra(m) -> H_field`` adds a transient contribution (e.g. a laser pulse).
    """
    return heun_step(m, grid, ep, lp.gamma, lp.alpha, lp.dt, H_extra=H_extra)


def llg_step_rk4(m, grid: Grid, ep: EnergyParams, lp: LLGParams, H_extra: Optional[Callable] = None):
    """RK4 alternative to Heun; ~2x cost per step but ~10^3x lower local truncation error."""
    return rk4_step(m, grid, ep, lp.gamma, lp.alpha, lp.dt, H_extra=H_extra)


def relax_step(m, grid: Grid, ep: EnergyParams, dt: float = 0.01):
    """One step of pure-dissipation gradient descent: dm/dt ~ m x (m x H_eff)."""
    return damped_step(m, grid, ep, dt=dt)


def relax(m, grid: Grid, ep: EnergyParams, n_steps: int = 1000, dt: float = 0.01,
          log_every: int = 0, step_callback: Optional[Callable] = None):
    """Run ``n_steps`` of damped-LLG relaxation.

    ``step_callback(m, k)`` -- if supplied, called after each step with the new
    field and the step index. Pipeline metric callbacks attach here.

    Under the JAX backend, when no per-step hook is requested (``step_callback``
    is None and ``log_every`` is 0), the whole loop is dispatched to a compiled
    ``jax.lax.scan`` path (``integrators.relax_scan``) for a large speedup.
    """
    from hopfion.backend import name as _backend_name
    if step_callback is None and not log_every and _backend_name() == "jax":
        from hopfion.physics.integrators import relax_scan
        return relax_scan(m, grid, ep, n_steps, dt)

    from hopfion.energy import total_energy
    for k in range(n_steps):
        m = relax_step(m, grid, ep, dt=dt)
        if step_callback is not None:
            step_callback(m, k)
        if log_every and k % log_every == 0:
            print(f"  relax step {k:5d}: E = {total_energy(m, grid, ep):.6f}")
    return m


def integrate(
    m,
    grid: Grid,
    ep: EnergyParams,
    lp: LLGParams,
    n_steps: int,
    H_extra: Optional[Callable] = None,
    snapshot_every: int = 0,
    step_callback: Optional[Callable] = None,
):
    """Run full LLG dynamics for ``n_steps``.

    ``H_extra(m, t)`` is a transient external field. ``step_callback(m, k, t)``
    is invoked after each step (pipeline metrics hook here). If
    ``snapshot_every > 0``, returns ``(m_final, snapshots, times)``.

    Under the JAX backend, a plain dynamics run (no transient field, snapshots, or
    callback) is dispatched to the compiled ``jax.lax.scan`` path
    ``integrators.integrate_scan``.
    """
    from hopfion.backend import name as _backend_name
    if (H_extra is None and step_callback is None and not snapshot_every
            and _backend_name() == "jax"):
        from hopfion.physics.integrators import integrate_scan
        return integrate_scan(m, grid, ep, lp.gamma, lp.alpha, n_steps, lp.dt)

    snapshots = [m.copy() if hasattr(m, "copy") else m] if snapshot_every else None
    times = [0.0] if snapshot_every else None
    t = 0.0
    for k in range(n_steps):
        if H_extra is not None:
            transient = lambda x, _t=t: H_extra(x, _t)
        else:
            transient = None
        m = llg_step_heun(m, grid, ep, lp, H_extra=transient)
        t += lp.dt
        if step_callback is not None:
            step_callback(m, k, t)
        if snapshot_every and (k + 1) % snapshot_every == 0:
            snapshots.append(m.copy() if hasattr(m, "copy") else m)
            times.append(t)
    if snapshot_every:
        return m, snapshots, times
    return m
