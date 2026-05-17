"""Micromagnetic energy functional.

Units: ``mu_0 M_s == 1`` (so the effective field H_eff = -dE/dm has the same
units as the energy density coefficients). Energy terms:

* Heisenberg exchange:    E_ex = A_ex int |grad m|^2 d^3r
* Bulk (Bloch) DMI:       E_dmi = D int m . (curl m) d^3r
* Uniaxial anisotropy:    E_an = -K_u int (m . e_k)^2 d^3r
* External Zeeman:        E_Z = -int m . H_ext d^3r
* Optional moire potential (anisotropy modulation): handled via ``Ku_field``.

Each term provides ``_energy`` and ``_field`` (the negative variation).
``total_energy(m, ...)`` and ``effective_field(m, ...)`` sum them up.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from hopfion.backend import xp
from hopfion.grid import Grid, curl, grad_vector, laplacian_vector


@dataclass
class EnergyParams:
    """Coefficients for the micromagnetic energy."""

    A_ex: float = 1.0
    D: float = 0.0
    Ku: float = 0.0
    easy_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    H_ext: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Optional spatially varying anisotropy strength (e.g. moire modulation).
    # If provided, used instead of the scalar ``Ku``.
    Ku_field: Optional[object] = field(default=None, repr=False)
    # Phase B: long-range dipolar (magnetostatic) field via FFT convolution.
    # When True, ``effective_field`` includes -delta E_dip / delta m.
    dipolar: bool = False


def exchange_energy(m, grid: Grid, A_ex: float) -> float:
    """Bond-energy discretization: sum (m(r+e) - m(r))^2/dx^2 over forward bonds.

    Its discrete gradient is exactly the 3-point Laplacian used by
    ``exchange_field``, so total_energy / effective_field are an exact adjoint
    pair on a periodic grid -- the numerical FD-vs-analytic-H check passes to
    machine precision.
    """
    if grid.bc != "periodic":
        # Fall back to central-FD form (off by O(dx^2) from the 3-point laplacian).
        np = xp()
        gm = grad_vector(m, grid)
        return float(A_ex * np.sum(gm * gm) * grid.dV)
    np = xp()
    fwd_x = (np.roll(m, -1, axis=1) - m) / grid.dx
    fwd_y = (np.roll(m, -1, axis=2) - m) / grid.dy
    fwd_z = (np.roll(m, -1, axis=3) - m) / grid.dz
    return float(A_ex * (np.sum(fwd_x * fwd_x) + np.sum(fwd_y * fwd_y) + np.sum(fwd_z * fwd_z)) * grid.dV)


def exchange_field(m, grid: Grid, A_ex: float):
    return 2.0 * A_ex * laplacian_vector(m, grid)


def dmi_energy(m, grid: Grid, D: float) -> float:
    np = xp()
    cm = curl(m, grid)
    return float(D * np.sum(m * cm) * grid.dV)


def dmi_field(m, grid: Grid, D: float):
    return -2.0 * D * curl(m, grid)


def anisotropy_energy(m, grid: Grid, Ku, easy_axis: Tuple[float, float, float]) -> float:
    """``Ku`` may be a scalar or a spatial array."""
    np = xp()
    ex, ey, ez = easy_axis
    norm = (ex * ex + ey * ey + ez * ez) ** 0.5
    ex, ey, ez = ex / norm, ey / norm, ez / norm
    mdote = m[0] * ex + m[1] * ey + m[2] * ez
    return float(-np.sum(Ku * mdote * mdote) * grid.dV)


def anisotropy_field(m, grid: Grid, Ku, easy_axis: Tuple[float, float, float]):
    np = xp()
    ex, ey, ez = easy_axis
    norm = (ex * ex + ey * ey + ez * ez) ** 0.5
    ex, ey, ez = ex / norm, ey / norm, ez / norm
    mdote = m[0] * ex + m[1] * ey + m[2] * ez
    factor = 2.0 * Ku * mdote
    return np.stack([factor * ex, factor * ey, factor * ez], axis=0)


def zeeman_energy(m, grid: Grid, H_ext: Tuple[float, float, float]) -> float:
    np = xp()
    Hx, Hy, Hz = H_ext
    return float(-np.sum(m[0] * Hx + m[1] * Hy + m[2] * Hz) * grid.dV)


def zeeman_field(m, grid: Grid, H_ext: Tuple[float, float, float]):
    np = xp()
    Hx, Hy, Hz = H_ext
    shape = m[0].shape
    return np.stack(
        [np.full(shape, Hx), np.full(shape, Hy), np.full(shape, Hz)],
        axis=0,
    )


def total_energy(m, grid: Grid, p: EnergyParams) -> float:
    Ku = p.Ku_field if p.Ku_field is not None else p.Ku
    e = (
        exchange_energy(m, grid, p.A_ex)
        + dmi_energy(m, grid, p.D)
        + anisotropy_energy(m, grid, Ku, p.easy_axis)
        + zeeman_energy(m, grid, p.H_ext)
    )
    if p.dipolar:
        from hopfion.physics.dipolar import dipolar_energy
        e += dipolar_energy(m, grid)
    return e


def effective_field(m, grid: Grid, p: EnergyParams):
    """Sum H_eff = -dE/dm over all enabled terms."""
    Ku = p.Ku_field if p.Ku_field is not None else p.Ku
    H = (
        exchange_field(m, grid, p.A_ex)
        + dmi_field(m, grid, p.D)
        + anisotropy_field(m, grid, Ku, p.easy_axis)
        + zeeman_field(m, grid, p.H_ext)
    )
    if p.dipolar:
        from hopfion.physics.dipolar import dipolar_field
        H = H + dipolar_field(m, grid)
    return H
