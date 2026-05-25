"""Composite state constructors: Q± pair, hopfion-skyrmion hybrid, ...

These are the propagating "particles" of the flux pipeline. Each constructor
returns a single $\\mathbf{m}(\\mathbf{r})$ field of shape $(3, n_x, n_y, n_z)$;
the multiple topological objects are blended via the same Gaussian-weight
scheme used by ``lattice.array_hopfion``.

Available constructors:
    * ``q_pair`` — two hopfions of opposite Hopf charge.
    * ``skyrmion_tube`` — a 2D skyrmion extruded along z (a stand-alone tube).
    * ``hopfion_skyrmion_hybrid`` — a Q=1 hopfion linked with a skyrmion tube
      passing through its central ring.

The Q=−1 partner is built by spatial reflection of the standard Q=+1 ansatz
(parity flips the linking number). This avoids the De Moivre branch-cut
problems of raising the Hopf map's $v$ to a negative integer power.
"""
from __future__ import annotations

import math
from typing import Tuple

from hopfion.backend import xp
from hopfion.field import hopfion, uniform
from hopfion.grid import Grid, normalize


def _reflect_y(m):
    """Apply y -> -y to a (3, nx, ny, nz) field. Flips Hopf charge."""
    np = xp()
    return np.flip(m, axis=2)


def q_minus_one_hopfion(
    grid: Grid,
    R: float = 1.5,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: str = "z",
):
    """A Q_H = -1 hopfion. Built by reflecting a Q=+1 hopfion in y."""
    cx, cy, cz = center
    # Build a Q=+1 hopfion centered at (cx, -cy, cz) — when we flip y, its
    # center maps to (cx, cy, cz).
    m_plus = hopfion(grid, R=R, p=1, q=1, center=(cx, -cy, cz), axis=axis)
    m = _reflect_y(m_plus)
    # The reflection also flips the y-component direction, so account for that:
    np = xp()
    m_x, m_y, m_z = m[0], -m[1], m[2]
    return np.stack([m_x, m_y, m_z], axis=0)


def q_pair(
    grid: Grid,
    R: float = 1.5,
    separation: float = 4.0,
    axis_of_separation: str = "x",
    background: Tuple[float, float, float] = (0.0, 0.0, 1.0),
):
    """Two hopfions of opposite Hopf charge separated by ``separation``.

    The pair sits on the ``axis_of_separation`` axis (default x), centered on
    the origin. The +Q hopfion is at $-d/2$, the $-Q$ at $+d/2$.

    Total $Q_H = 0$. Useful as a clean test of flux conservation.
    """
    np = xp()
    half = separation / 2.0
    centers = {
        "x": [(-half, 0.0, 0.0), (+half, 0.0, 0.0)],
        "y": [(0.0, -half, 0.0), (0.0, +half, 0.0)],
        "z": [(0.0, 0.0, -half), (0.0, 0.0, +half)],
    }[axis_of_separation]
    m_plus_at = hopfion(grid, R=R, p=1, q=1, center=centers[0])
    m_minus_at = q_minus_one_hopfion(grid, R=R, center=centers[1])

    bg = uniform(grid, direction=background)
    # Gaussian-weighted blend around each center (same recipe as lattice.array_hopfion)
    sigma = 2.0 * R
    X, Y, Z = grid.coords()
    accum = bg
    for m_i, c in ((m_plus_at, centers[0]), (m_minus_at, centers[1])):
        dx, dy, dz = X - c[0], Y - c[1], Z - c[2]
        w = np.exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma))
        accum = accum + (m_i - bg) * w
    return normalize(accum)


# ---------------------------------------------------------------------------
# Skyrmion tube and hybrid
# ---------------------------------------------------------------------------


def skyrmion_tube(
    grid: Grid,
    radius: float = 1.0,
    helicity: float = 0.0,
    center_xy: Tuple[float, float] = (0.0, 0.0),
):
    """A 2D Bloch-type skyrmion of size ``radius``, extruded along z.

    Profile: standard 360° domain-wall in ρ. For ρ = 0, m = -ẑ; for ρ >> R,
    m = +ẑ.

    The ``helicity`` knob rotates the in-plane spin direction (Bloch vs Néel).

    Thin wrapper over :func:`hopfion.field.skyrmion` with ``vorticity=+1``; kept
    for the composite-state vocabulary (``q_pair``, ``hopfion_skyrmion_hybrid``).
    """
    from hopfion.field import skyrmion
    return skyrmion(grid, radius=radius, helicity=helicity, vorticity=1, center=center_xy)


def hopfion_skyrmion_hybrid(
    grid: Grid,
    R: float = 1.5,
    skyrmion_radius: float = 1.0,
    skyrmion_helicity: float = math.pi / 2,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """A Q_H=1 hopfion *linked* with a skyrmion tube passing through its
    central ring (the tube runs along z through the hopfion's z-axis ring).

    Construction: in the central tube region, the field is dominated by the
    skyrmion. Outside, it's the hopfion. The two are blended via a Gaussian
    along z. Total Hopf index ≈ 1; total skyrmion charge per z-slice ≈ 1 in
    the tube region, 0 elsewhere.
    """
    np = xp()
    cx, cy, cz = center
    m_h = hopfion(grid, R=R, p=1, q=1, center=center)
    m_s = skyrmion_tube(grid, radius=skyrmion_radius, helicity=skyrmion_helicity,
                        center_xy=(cx, cy))
    # Blend weight: skyrmion dominates near (x,y) = (cx, cy) with falloff in radial
    X, Y, _Z = grid.coords()
    rho2 = (X - cx) ** 2 + (Y - cy) ** 2
    w = np.exp(-rho2 / (2.0 * skyrmion_radius * skyrmion_radius))
    combined = (1.0 - w) * m_h + w * m_s
    return normalize(combined)


__all__ = [
    "q_minus_one_hopfion",
    "q_pair",
    "skyrmion_tube",
    "hopfion_skyrmion_hybrid",
]
