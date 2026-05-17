"""Hopfion-array initial conditions in 2D and 3D.

Given a list of site centers, superpose unit hopfion ansatze and renormalize.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from hopfion.backend import xp
from hopfion.field import hopfion, uniform
from hopfion.grid import Grid, normalize


def triangular_sites_2d(
    a: float,
    n_rings: int = 1,
    z: float = 0.0,
) -> List[Tuple[float, float, float]]:
    """Triangular-lattice sites within ``n_rings`` of the origin (in the xy-plane)."""
    sites = [(0.0, 0.0, z)]
    v1 = (a, 0.0)
    v2 = (a / 2.0, a * math.sqrt(3.0) / 2.0)
    for n in range(-n_rings, n_rings + 1):
        for mvar in range(-n_rings, n_rings + 1):
            if n == 0 and mvar == 0:
                continue
            # restrict to a roughly hexagonal region around origin
            x = n * v1[0] + mvar * v2[0]
            y = n * v1[1] + mvar * v2[1]
            if math.hypot(x, y) <= n_rings * a + 1e-6:
                sites.append((x, y, z))
    return sites


def square_sites_2d(a: float, nx: int, ny: int, z: float = 0.0):
    """``nx`` x ``ny`` square lattice centered on origin."""
    x0 = -0.5 * (nx - 1) * a
    y0 = -0.5 * (ny - 1) * a
    return [(x0 + ix * a, y0 + iy * a, z) for ix in range(nx) for iy in range(ny)]


def cubic_sites_3d(a: float, nx: int, ny: int, nz: int):
    x0 = -0.5 * (nx - 1) * a
    y0 = -0.5 * (ny - 1) * a
    z0 = -0.5 * (nz - 1) * a
    return [
        (x0 + ix * a, y0 + iy * a, z0 + iz * a)
        for ix in range(nx)
        for iy in range(ny)
        for iz in range(nz)
    ]


def array_hopfion(
    grid: Grid,
    sites: Iterable[Tuple[float, float, float]],
    R: float,
    background=(0.0, 0.0, 1.0),
    axis: str = "z",
):
    """Superpose hopfion ansatze at the given sites and project to S^2.

    The construction is heuristic (a smooth-superposition initializer that's
    then relaxed). Each site contributes a localized hopfion ``m_i``; far from
    all sites, the field approaches ``background``. We blend per cell:

        m(r) = normalize( background + sum_i (m_i(r) - background) * w_i(r) )

    where ``w_i`` is a Gaussian falloff localized to site ``i``.
    """
    np = xp()
    bg = uniform(grid, direction=background)
    accum = bg
    # Weight scale -- a few hopfion radii
    sigma = 2.0 * R
    X, Y, Z = grid.coords()
    for cx, cy, cz in sites:
        m_i = hopfion(grid, R=R, p=1, q=1, center=(cx, cy, cz), axis=axis)
        dx = X - cx
        dy = Y - cy
        dz = Z - cz
        w = np.exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma))
        accum = accum + (m_i - bg) * w
    return normalize(accum)
