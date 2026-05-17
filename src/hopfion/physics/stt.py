"""Spin-transfer torque (Zhang-Li form) extension to LLG.

Adds two terms to the LLG equation:

    \\dot{\\mathbf{m}} = -\\gamma\\,\\mathbf{m}\\times\\mathbf{H}_{\\rm eff}
                       + \\alpha\\,\\mathbf{m}\\times\\dot{\\mathbf{m}}
                       - (\\mathbf{u}\\cdot\\nabla)\\mathbf{m}
                       + \\beta\\,\\mathbf{m}\\times(\\mathbf{u}\\cdot\\nabla)\\mathbf{m}.

``u`` is the spin-current-equivalent velocity (in normalized units the user
specifies it directly). ``beta`` is the non-adiabatic coefficient.

Implementation: at each LLG sub-step we add ``-(u·∇)m + β m×(u·∇)m`` to the
``llg_rhs``. We expose ``stt_step_heun`` that wraps the Heun integrator with
this extra term, so the stencil structure is identical to ``heun_step``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from hopfion.backend import xp
from hopfion.energy import EnergyParams, effective_field
from hopfion.grid import Grid, normalize, d_axis
from hopfion.physics.integrators import _cross, llg_rhs


@dataclass
class STTParams:
    """Spin-transfer-torque drive parameters."""
    u: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # spin-current velocity
    beta: float = 0.0                                 # non-adiabatic coefficient


def adv_grad_m(m, grid: Grid, u: Tuple[float, float, float]):
    """Advective term $(\\mathbf{u}\\cdot\\nabla)\\mathbf{m}$ via central FD."""
    np = xp()
    ux, uy, uz = u
    out = np.zeros_like(np.asarray(m))
    if ux != 0.0:
        # gradient along x of each component
        for i in range(3):
            out[i] = out[i] + ux * d_axis(m[i], 0, grid.dx, grid)
    if uy != 0.0:
        for i in range(3):
            out[i] = out[i] + uy * d_axis(m[i], 1, grid.dy, grid)
    if uz != 0.0:
        for i in range(3):
            out[i] = out[i] + uz * d_axis(m[i], 2, grid.dz, grid)
    return out


def stt_rhs(m, H, grid: Grid, gamma: float, alpha: float,
            stt: STTParams):
    """LLG RHS augmented with Zhang-Li STT terms."""
    base = llg_rhs(m, H, gamma, alpha)
    if stt.u == (0.0, 0.0, 0.0):
        return base
    adv = adv_grad_m(m, grid, stt.u)
    extra = -adv + stt.beta * _cross(m, adv)
    return base + extra


def stt_step_heun(m, grid: Grid, ep: EnergyParams, gamma: float, alpha: float,
                  dt: float, stt: STTParams,
                  H_extra: Optional[Callable] = None):
    """One Heun step of LLG + STT. Mirrors ``heun_step`` exactly except the
    RHS includes the advective + non-adiabatic STT terms.
    """
    H1 = effective_field(m, grid, ep)
    if H_extra is not None:
        H1 = H1 + H_extra(m)
    k1 = stt_rhs(m, H1, grid, gamma, alpha, stt)
    m_pred = normalize(m + dt * k1)
    H2 = effective_field(m_pred, grid, ep)
    if H_extra is not None:
        H2 = H2 + H_extra(m_pred)
    k2 = stt_rhs(m_pred, H2, grid, gamma, alpha, stt)
    return normalize(m + 0.5 * dt * (k1 + k2))


__all__ = ["STTParams", "adv_grad_m", "stt_rhs", "stt_step_heun"]
