"""Cubic finite-difference operators on a 3D lattice.

Field convention: arrays of shape ``(3, nx, ny, nz)`` representing a vector
field, or ``(nx, ny, nz)`` for a scalar. Spacings ``dx``, ``dy``, ``dz`` (or a
single ``d`` if isotropic) and ``bc in {"periodic", "open"}``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from hopfion.backend import xp

BC = Literal["periodic", "open"]


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    nz: int
    dx: float = 1.0
    dy: float = 1.0
    dz: float = 1.0
    bc: BC = "periodic"

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def dV(self) -> float:
        return self.dx * self.dy * self.dz

    def coords(self):
        np = xp()
        x = (np.arange(self.nx) - self.nx / 2 + 0.5) * self.dx
        y = (np.arange(self.ny) - self.ny / 2 + 0.5) * self.dy
        z = (np.arange(self.nz) - self.nz / 2 + 0.5) * self.dz
        return np.meshgrid(x, y, z, indexing="ij")


def _roll(a, shift: int, axis: int):
    return xp().roll(a, shift, axis=axis)


def _shift(a, axis: int, direction: int, bc: BC):
    """Shift along ``axis`` by ``direction`` (+/-1). For open BC, edge values
    are replicated (Neumann)."""
    np = xp()
    if bc == "periodic":
        return _roll(a, -direction, axis=axis)
    # Open / Neumann: clamp the index
    n = a.shape[axis]
    idx = np.clip(np.arange(n) + direction, 0, n - 1)
    return np.take(a, idx, axis=axis)


def d_axis(a, axis: int, d: float, grid: Grid):
    """Central difference along ``axis``."""
    return (_shift(a, axis, +1, grid.bc) - _shift(a, axis, -1, grid.bc)) / (2.0 * d)


def grad_scalar(s, grid: Grid):
    """Gradient of a scalar field ``(nx, ny, nz)`` -> ``(3, nx, ny, nz)``."""
    np = xp()
    gx = d_axis(s, axis=0, d=grid.dx, grid=grid)
    gy = d_axis(s, axis=1, d=grid.dy, grid=grid)
    gz = d_axis(s, axis=2, d=grid.dz, grid=grid)
    return np.stack([gx, gy, gz], axis=0)


def grad_vector(v, grid: Grid):
    """Jacobian of a vector field ``(3, nx, ny, nz)`` -> ``(3, 3, nx, ny, nz)``.

    Index ``[i, j, ...]`` = d v_i / d x_j.
    """
    np = xp()
    rows = []
    for i in range(3):
        rows.append(grad_scalar(v[i], grid))
    return np.stack(rows, axis=0)


def divergence(v, grid: Grid):
    return d_axis(v[0], 0, grid.dx, grid) + d_axis(v[1], 1, grid.dy, grid) + d_axis(v[2], 2, grid.dz, grid)


def curl(v, grid: Grid):
    np = xp()
    dvz_dy = d_axis(v[2], 1, grid.dy, grid)
    dvy_dz = d_axis(v[1], 2, grid.dz, grid)
    dvx_dz = d_axis(v[0], 2, grid.dz, grid)
    dvz_dx = d_axis(v[2], 0, grid.dx, grid)
    dvy_dx = d_axis(v[1], 0, grid.dx, grid)
    dvx_dy = d_axis(v[0], 1, grid.dy, grid)
    return np.stack([dvz_dy - dvy_dz, dvx_dz - dvz_dx, dvy_dx - dvx_dy], axis=0)


def laplacian_scalar(s, grid: Grid):
    np = xp()
    lap = (
        (_shift(s, 0, +1, grid.bc) - 2 * s + _shift(s, 0, -1, grid.bc)) / grid.dx**2
        + (_shift(s, 1, +1, grid.bc) - 2 * s + _shift(s, 1, -1, grid.bc)) / grid.dy**2
        + (_shift(s, 2, +1, grid.bc) - 2 * s + _shift(s, 2, -1, grid.bc)) / grid.dz**2
    )
    return lap


def laplacian_vector(v, grid: Grid):
    np = xp()
    return np.stack([laplacian_scalar(v[i], grid) for i in range(3)], axis=0)


def normalize(v, eps: float = 1e-30):
    """Project a vector field to the unit sphere."""
    np = xp()
    n = np.sqrt(np.sum(v * v, axis=0, keepdims=True) + eps)
    return v / n
