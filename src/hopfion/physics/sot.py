"""Spin-orbit torque (SOT) extension to LLG.

For a fixed spin-polarization direction $\\hat{\\mathbf{p}}$ (set by the current
and the spin-Hall/Rashba geometry), SOT adds two torques:

    \\dot{\\mathbf{m}} = (\\text{LLG}) - \\tau_{\\rm DL}\\,\\mathbf{m}\\times(\\mathbf{m}\\times\\hat p)
                                       - \\tau_{\\rm FL}\\,\\mathbf{m}\\times\\hat p.

The *damping-like* term $-\\tau_{\\rm DL}\\,\\mathbf m\\times(\\mathbf m\\times\\hat p)
= \\tau_{\\rm DL}[\\hat p - (\\mathbf m\\cdot\\hat p)\\mathbf m]$ drives $\\mathbf m$
toward $\\hat p$ (the workhorse of SOT-MRAM switching); the *field-like* term acts
as an effective field along $\\hat p$. Unlike Zhang-Li STT (``stt.py``), SOT needs
no spatial gradient — it's a *local* torque, so it drives uniform switching and
domain/skyrmion motion via the boundary.

Mirrors ``stt.py``: wrap ``llg_rhs`` with the extra torque, Heun-integrate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from hopfion.backend import xp
from hopfion.energy import EnergyParams, effective_field
from hopfion.grid import Grid, normalize
from hopfion.physics.integrators import _cross, llg_rhs


@dataclass
class SOTParams:
    """Spin-orbit-torque drive parameters."""
    p: Tuple[float, float, float] = (0.0, 0.0, 1.0)   # spin-polarization direction
    dl: float = 0.0                                   # damping-like magnitude
    fl: float = 0.0                                   # field-like magnitude


def _p_field(m, p: Tuple[float, float, float]):
    """Broadcast a (normalized) polarization direction to a (3, nx, ny, nz) field."""
    np = xp()
    px, py, pz = p
    n = (px * px + py * py + pz * pz) ** 0.5 or 1.0
    px, py, pz = px / n, py / n, pz / n
    shape = np.asarray(m)[0].shape
    return np.stack([np.full(shape, px), np.full(shape, py), np.full(shape, pz)], axis=0)


def sot_rhs(m, H, grid: Grid, gamma: float, alpha: float, sot: SOTParams):
    """LLG RHS augmented with damping-like + field-like SOT terms."""
    base = llg_rhs(m, H, gamma, alpha)
    if sot.dl == 0.0 and sot.fl == 0.0:
        return base
    pf = _p_field(m, sot.p)
    extra = -sot.dl * _cross(m, _cross(m, pf)) - sot.fl * _cross(m, pf)
    return base + extra


def sot_step_heun(m, grid: Grid, ep: EnergyParams, gamma: float, alpha: float,
                  dt: float, sot: SOTParams, H_extra: Optional[Callable] = None):
    """One Heun step of LLG + SOT (mirrors ``stt.stt_step_heun``)."""
    H1 = effective_field(m, grid, ep)
    if H_extra is not None:
        H1 = H1 + H_extra(m)
    k1 = sot_rhs(m, H1, grid, gamma, alpha, sot)
    m_pred = normalize(m + dt * k1)
    H2 = effective_field(m_pred, grid, ep)
    if H_extra is not None:
        H2 = H2 + H_extra(m_pred)
    k2 = sot_rhs(m_pred, H2, grid, gamma, alpha, sot)
    return normalize(m + 0.5 * dt * (k1 + k2))


__all__ = ["SOTParams", "sot_rhs", "sot_step_heun"]
