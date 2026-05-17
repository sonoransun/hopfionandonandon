"""Moire-engineered modulations for hopfion-array stabilization.

Phase A (toy): a spatially periodic anisotropy modulation built from the sum
of plane waves whose wavevectors form the moire reciprocal-lattice basis. This
captures the *kinematic* effect of moire stacking (a periodic pinning
landscape) without bilayer mechanics, and is what notebooks 02/03 use.

    K(r) = K0 + V0 * sum_i cos(k_i . r + phi_i)

For a triangular 2D moire pattern in the (x, y) plane:

    k_1 = k * (1, 0, 0)
    k_2 = k * (-1/2,  sqrt(3)/2, 0)
    k_3 = k * (-1/2, -sqrt(3)/2, 0)

with k = 2 pi / a_moire.

Phase B will add a twisted-bilayer construction with a registry function
linking two field layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from hopfion.backend import xp
from hopfion.grid import Grid


SQRT3_OVER_2 = 0.8660254037844386


def triangular_2d_kvectors(a_moire: float) -> Tuple[Tuple[float, float, float], ...]:
    k = 2.0 * 3.141592653589793 / a_moire
    return (
        (k, 0.0, 0.0),
        (-0.5 * k, SQRT3_OVER_2 * k, 0.0),
        (-0.5 * k, -SQRT3_OVER_2 * k, 0.0),
    )


def square_2d_kvectors(a_moire: float) -> Tuple[Tuple[float, float, float], ...]:
    k = 2.0 * 3.141592653589793 / a_moire
    return ((k, 0.0, 0.0), (0.0, k, 0.0))


def honeycomb_2d_kvectors(a_moire: float) -> Tuple[Tuple[float, float, float], ...]:
    # Same star of three k's as triangular (honeycomb's BZ corners).
    return triangular_2d_kvectors(a_moire)


@dataclass
class MoirePotential:
    """Spatially-periodic anisotropy modulation."""

    K0: float = 0.05
    V0: float = 0.05
    a_moire: float = 5.0
    lattice: str = "triangular"  # 'triangular' | 'square' | 'honeycomb'

    def kvectors(self):
        if self.lattice == "triangular":
            return triangular_2d_kvectors(self.a_moire)
        if self.lattice == "square":
            return square_2d_kvectors(self.a_moire)
        if self.lattice == "honeycomb":
            return honeycomb_2d_kvectors(self.a_moire)
        raise ValueError(f"Unknown lattice {self.lattice!r}")

    def Ku_field(self, grid: Grid):
        """Return the spatial K_u(r) field of shape ``(nx, ny, nz)``."""
        np = xp()
        X, Y, Z = grid.coords()
        K = np.full(grid.shape, self.K0)
        for kx, ky, kz in self.kvectors():
            K = K + self.V0 * np.cos(kx * X + ky * Y + kz * Z)
        return K
