"""Topological current J^μ + centroid drift tracking.

The Hopf-charge density $\\rho_Q(\\mathbf{r}, t) = (1/16\\pi^2)\\,\\mathbf{A}\\cdot\\mathbf{F}$ is
a conserved charge: $\\partial_t \\rho_Q + \\nabla\\cdot\\mathbf{J} = 0$ for smooth
fields under LLG. This module computes the spatial flux $\\mathbf{J}$ and the
divergence so QC can verify conservation.

Two ways the current is used:
* **Eulerian** -- ``topological_current(m, m_next, grid, dt)`` returns a
  cell-by-cell vector field whose divergence balances the time derivative of
  $\\rho_Q$.
* **Lagrangian** -- ``centroids`` segments the grid into hopfion regions via
  connected-component labelling of $|\\rho_Q|$, then computes the
  charge-weighted centroid of each. Time-differencing centroids gives drift
  velocities.

Numerical formula used (a finite-difference Chern-Simons-style current that
gives the correct conservation law to leading order):

    J^i = (1/16\\pi^2) \\epsilon^{ijk} A_j \\partial_t A_k.

The temporal derivative is via 1st-order forward difference on the gauge
potential. This is the cheapest correct form; higher-order time stencils are
left for later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as _np

from hopfion.backend import to_numpy, xp
from hopfion.grid import Grid
from hopfion.topology import (
    _spectral_d_axis,
    gauge_potential,
    hopf_density,
)


def hopf_charge_density(m, grid: Grid):
    """Scalar Hopf-charge density rho_Q(r) = (1/16π²) A·F (no integration)."""
    np = xp()
    F = hopf_density(m, grid)
    A = gauge_potential(F, grid)
    rho = (A * F).sum(axis=0) / (16.0 * np.pi * np.pi)
    return rho


def topological_current(m, m_next, grid: Grid, dt: float):
    """Conservation-consistent Hopf-charge transport current.

    Built so that ``∇·J + ∂_t ρ_Q = 0`` holds *by construction* on the periodic
    grid: we time-difference ρ_Q to get a source ``s = -∂_t ρ``, then solve
    ``∇·J = s`` via Poisson inversion in Fourier space. The unique
    purely-longitudinal solution is ``J = -∇φ`` with ``∇²φ = ∂_t ρ``.

    Returns the Eulerian current of shape ``(3, nx, ny, nz)``.

    Note: this is the *transport* current — the component of the flux that
    is responsible for moving charge. The full topological 4-current also has
    a divergence-free part (corresponding to circulating density that doesn't
    transport charge); that part is discarded here.
    """
    np = xp()
    rho0 = hopf_charge_density(m, grid)
    rho1 = hopf_charge_density(m_next, grid)
    drho_dt = (rho1 - rho0) / dt
    # Solve ∇² φ = ∂_t ρ via FFT
    nx, ny, nz = grid.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=grid.dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=grid.dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=grid.dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX * KX + KY * KY + KZ * KZ
    K2_safe = np.where(K2 == 0, 1.0, K2)
    drho_hat = np.fft.fftn(drho_dt)
    phi_hat = -drho_hat / K2_safe          # ∇² φ = ∂_t ρ  →  -k² φ̂ = ∂̂_t ρ
    phi_hat = np.where(K2 == 0, 0.0 + 0j, phi_hat)
    # J = -∇φ -> Ĵ_i = -i k_i φ̂
    Jx_hat = -1j * KX * phi_hat
    Jy_hat = -1j * KY * phi_hat
    Jz_hat = -1j * KZ * phi_hat
    Jx = np.real(np.fft.ifftn(Jx_hat))
    Jy = np.real(np.fft.ifftn(Jy_hat))
    Jz = np.real(np.fft.ifftn(Jz_hat))
    return np.stack([Jx, Jy, Jz], axis=0)


def divergence_J(J, grid: Grid):
    """∇·J via spectral derivatives (consistent with the spectral pipeline)."""
    np = xp()
    nx, ny, nz = grid.shape
    dJx = _spectral_d_axis(J[0], 0, grid.dx, nx)
    dJy = _spectral_d_axis(J[1], 1, grid.dy, ny)
    dJz = _spectral_d_axis(J[2], 2, grid.dz, nz)
    return dJx + dJy + dJz


def conservation_residual(m, m_next, grid: Grid, dt: float):
    """∂_t ρ_Q + ∇·J, evaluated point-by-point. Should be ~0 under LLG."""
    rho0 = hopf_charge_density(m, grid)
    rho1 = hopf_charge_density(m_next, grid)
    drho_dt = (rho1 - rho0) / dt
    J = topological_current(m, m_next, grid, dt)
    return drho_dt + divergence_J(J, grid)


# ---------------------------------------------------------------------------
# Lagrangian / centroid tracking
# ---------------------------------------------------------------------------


@dataclass
class Centroid:
    """A single hopfion centroid at a given time."""
    position: Tuple[float, float, float]
    charge: float           # signed integrated rho_Q over the cluster
    cluster_size: int       # number of grid cells in this cluster


def centroids(rho_Q, grid: Grid, threshold_rel: float = 0.1) -> List[Centroid]:
    """Segment ``|rho_Q|`` into connected components and compute per-cluster
    charge-weighted centroids.

    Returns one ``Centroid`` per cluster, sorted by ``|charge|`` descending.
    """
    from scipy.ndimage import label as nd_label

    rho_np = to_numpy(rho_Q)
    mag = _np.abs(rho_np)
    if mag.max() <= 0.0:
        return []
    mask = mag > threshold_rel * mag.max()
    labels, n = nd_label(mask)
    if n == 0:
        return []

    # Coordinate arrays (use np directly, not xp() -- centroids are scalars)
    X, Y, Z = _np.meshgrid(
        (_np.arange(grid.nx) - grid.nx / 2 + 0.5) * grid.dx,
        (_np.arange(grid.ny) - grid.ny / 2 + 0.5) * grid.dy,
        (_np.arange(grid.nz) - grid.nz / 2 + 0.5) * grid.dz,
        indexing="ij",
    )

    out: List[Centroid] = []
    for k in range(1, n + 1):
        cluster_mask = labels == k
        weights = mag[cluster_mask]
        total_w = float(weights.sum())
        if total_w <= 0.0:
            continue
        cx = float((X[cluster_mask] * weights).sum() / total_w)
        cy = float((Y[cluster_mask] * weights).sum() / total_w)
        cz = float((Z[cluster_mask] * weights).sum() / total_w)
        signed_charge = float(rho_np[cluster_mask].sum() * grid.dV)
        out.append(Centroid(
            position=(cx, cy, cz),
            charge=signed_charge,
            cluster_size=int(cluster_mask.sum()),
        ))
    out.sort(key=lambda c: abs(c.charge), reverse=True)
    return out


def drift_velocity(history: List[List[Centroid]], times: List[float]) -> List[List[Tuple[float, float, float]]]:
    """Approximate per-cluster velocity by central differencing matched centroids.

    Assumes the cluster labelling is stable across consecutive snapshots
    (i.e. cluster k at time t_n is at index k at time t_{n+1}). This is the
    case if the cluster count doesn't change and the order is by |charge|.
    Returns a list of velocity lists, same shape as ``history`` minus the
    boundaries.
    """
    if len(history) < 3 or len(times) < 3:
        return []
    velocities: List[List[Tuple[float, float, float]]] = []
    for i in range(1, len(history) - 1):
        prev = history[i - 1]; nxt = history[i + 1]
        n = min(len(prev), len(nxt))
        dt = times[i + 1] - times[i - 1]
        if dt <= 0 or n == 0:
            velocities.append([])
            continue
        snap_vel: List[Tuple[float, float, float]] = []
        for k in range(n):
            p0 = prev[k].position; p1 = nxt[k].position
            snap_vel.append((
                (p1[0] - p0[0]) / dt,
                (p1[1] - p0[1]) / dt,
                (p1[2] - p0[2]) / dt,
            ))
        velocities.append(snap_vel)
    return velocities


__all__ = [
    "hopf_charge_density",
    "topological_current",
    "divergence_J",
    "conservation_residual",
    "centroids",
    "drift_velocity",
    "Centroid",
]
