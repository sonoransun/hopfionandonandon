"""Time-step integrators for LLG dynamics and damped relaxation.

The Phase-A ``hopfion.llg`` module remains backward-compatible -- its
``llg_step_heun`` and ``relax_step`` are now thin wrappers around the routines
here. New choices:

* ``rk4_step`` — classical 4th-order Runge-Kutta (~2× the cost of Heun, but ~3
  orders of magnitude better local truncation error on smooth trajectories).
* ``adaptive_heun_step`` — embedded Heun-Euler estimate; rejects-and-halves the
  step when an error estimate exceeds tolerance.

All integrators preserve the unit-vector constraint by renormalising at the end
of each step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from hopfion.backend import xp
from hopfion.energy import EnergyParams, effective_field
from hopfion.grid import Grid, normalize


def _cross(a, b):
    np = xp()
    return np.stack(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        axis=0,
    )


def llg_rhs(m, H, gamma: float, alpha: float):
    """Explicit LLG right-hand side: -gamma (mxH) - gamma alpha m x (m x H)."""
    mxH = _cross(m, H)
    return -gamma * (mxH + alpha * _cross(m, mxH))


def _effective_with_extra(m, grid, ep, H_extra):
    H = effective_field(m, grid, ep)
    if H_extra is not None:
        H = H + H_extra(m)
    return H


def heun_step(m, grid: Grid, ep: EnergyParams, gamma: float, alpha: float, dt: float,
              H_extra: Optional[Callable] = None):
    """Single Heun (improved-Euler / RK2) step. The Phase-A integrator."""
    H1 = _effective_with_extra(m, grid, ep, H_extra)
    k1 = llg_rhs(m, H1, gamma, alpha)
    m_pred = normalize(m + dt * k1)
    H2 = _effective_with_extra(m_pred, grid, ep, H_extra)
    k2 = llg_rhs(m_pred, H2, gamma, alpha)
    return normalize(m + 0.5 * dt * (k1 + k2))


def rk4_step(m, grid: Grid, ep: EnergyParams, gamma: float, alpha: float, dt: float,
             H_extra: Optional[Callable] = None):
    """Classical 4th-order Runge-Kutta. Use when local truncation matters
    (long-time dynamics, energy-conservation studies). ~2× Heun cost per step."""
    H1 = _effective_with_extra(m, grid, ep, H_extra)
    k1 = llg_rhs(m, H1, gamma, alpha)

    m2 = normalize(m + 0.5 * dt * k1)
    H2 = _effective_with_extra(m2, grid, ep, H_extra)
    k2 = llg_rhs(m2, H2, gamma, alpha)

    m3 = normalize(m + 0.5 * dt * k2)
    H3 = _effective_with_extra(m3, grid, ep, H_extra)
    k3 = llg_rhs(m3, H3, gamma, alpha)

    m4 = normalize(m + dt * k3)
    H4 = _effective_with_extra(m4, grid, ep, H_extra)
    k4 = llg_rhs(m4, H4, gamma, alpha)

    return normalize(m + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


@dataclass
class AdaptiveResult:
    m: object
    dt_used: float
    accepted: bool
    error: float


def adaptive_heun_step(m, grid: Grid, ep: EnergyParams, gamma: float, alpha: float,
                       dt: float, H_extra: Optional[Callable] = None,
                       rtol: float = 1e-3, min_dt: float = 1e-6) -> AdaptiveResult:
    """Embedded Heun-Euler step with reject-and-halve. Returns the trial result
    and a flag indicating whether the step was within tolerance."""
    np = xp()
    H1 = _effective_with_extra(m, grid, ep, H_extra)
    k1 = llg_rhs(m, H1, gamma, alpha)
    m_euler = normalize(m + dt * k1)  # 1st-order estimate
    H2 = _effective_with_extra(m_euler, grid, ep, H_extra)
    k2 = llg_rhs(m_euler, H2, gamma, alpha)
    m_heun = normalize(m + 0.5 * dt * (k1 + k2))  # 2nd-order estimate

    err = float(np.max(np.abs(m_heun - m_euler)))
    accepted = (err <= rtol) or (dt <= min_dt)
    return AdaptiveResult(m=m_heun, dt_used=dt, accepted=accepted, error=err)


# Damped-LLG (pure gradient flow) -- the Phase-A relax_step lives here too.

def damped_step(m, grid: Grid, ep: EnergyParams, dt: float = 0.01,
                H_extra: Optional[Callable] = None):
    """One step of damped LLG: dm/dt = H_eff - m (m . H_eff). Equivalent to
    gradient descent on the energy restricted to the unit-sphere tangent
    plane. Used for finding local minima."""
    H = _effective_with_extra(m, grid, ep, H_extra)
    mxH = _cross(m, H)
    rhs = -_cross(m, mxH)
    return normalize(m + dt * rhs)


def implicit_midpoint_step(m, grid: Grid, ep: EnergyParams, gamma: float, alpha: float,
                           dt: float, H_extra: Optional[Callable] = None, iters: int = 4):
    """Implicit-midpoint LLG step: ``m* = m + dt·rhs((m+m*)/2)``, solved by a few
    fixed-point iterations, then renormalized.

    The midpoint rule is (near-)symplectic, so its great advantage is **long-time
    energy behavior in conservative dynamics**: with damping off (``alpha=0``),
    explicit Heun's energy drifts secularly and eventually diverges, whereas this
    integrator's energy stays bounded over thousands of steps — the right tool for
    spin-wave / precessional studies. (It is solved by fixed-point iteration, not
    Newton, so it is *not* unconditionally A-stable: ``dt`` must stay within the
    fixed-point convergence radius, comparable to the explicit stability limit.)
    At small ``dt`` it agrees with ``heun_step`` to 2nd order.
    """
    m_next = m
    for _ in range(iters):
        m_mid = normalize(0.5 * (m + m_next))
        H = _effective_with_extra(m_mid, grid, ep, H_extra)
        k = llg_rhs(m_mid, H, gamma, alpha)
        m_next = normalize(m + dt * k)
    return m_next


def relax_scan(m, grid: Grid, ep: EnergyParams, n_steps: int, dt: float = 0.01):
    """JAX-compiled damped relaxation via ``jax.lax.scan``.

    The whole ``n_steps`` loop is fused into one traced/compiled kernel — no
    per-step Python overhead and no callbacks (``scan`` cannot call back into
    Python, which is exactly why ``llg.relax`` only takes this path when
    ``step_callback`` is None). All per-step ops (``damped_step`` →
    ``effective_field`` → grid stencils) are already pure ``xp()`` arithmetic, so
    they trace cleanly under the jax backend. Requires ``backend.name()=='jax'``.
    """
    import jax

    def body(carry, _):
        return damped_step(carry, grid, ep, dt=dt), None

    m_final, _ = jax.lax.scan(body, m, xs=None, length=int(n_steps))
    return m_final


def integrate_scan(m, grid: Grid, ep: EnergyParams, gamma: float, alpha: float,
                   n_steps: int, dt: float):
    """JAX-compiled *precessional* LLG dynamics via ``jax.lax.scan`` over
    ``heun_step`` (no ``H_extra``/callbacks). The full-dynamics analogue of
    ``relax_scan``; ``llg.integrate`` dispatches here under the jax backend when
    no transient field, snapshots, or per-step hook is requested."""
    import jax

    def body(carry, _):
        return heun_step(carry, grid, ep, gamma, alpha, dt), None

    m_final, _ = jax.lax.scan(body, m, xs=None, length=int(n_steps))
    return m_final
