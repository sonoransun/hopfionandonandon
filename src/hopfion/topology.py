"""Hopf invariant via FFT on a periodic grid.

Convention:

    F_i = m . (d_j m x d_k m)     for (i, j, k) cyclic

    F = curl A   in Coulomb gauge (div A = 0), solved in Fourier space

    Q_H = (1 / (16 pi^2)) integral( A . F ) d^3 r

This factor follows from the S^2 area form's 1/(4 pi) normalization appearing
twice (once in F itself, once in A = curl^{-1} F).

For periodic boundary conditions, the curl-inversion is exact up to the k = 0
mode, which we set to zero (the Hopf charge is independent of any gauge-zero
mode).
"""
from __future__ import annotations

import numpy as _np

from hopfion.backend import to_numpy, xp
from hopfion.grid import Grid, d_axis


def _spectral_d_axis(s, axis: int, d: float, n: int):
    """Spectral derivative along ``axis`` using FFT (assumes periodic BC).

    Far more accurate than central differences for smooth, well-localized fields,
    and consistent with the FFT used for curl inversion below.
    """
    np = xp()
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=d)
    shape = [1, 1, 1]
    shape[axis] = n
    k = k.reshape(shape)
    s_hat = np.fft.fft(s, axis=axis)
    return np.real(np.fft.ifft(1j * k * s_hat, axis=axis))


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


def hopf_density(m, grid: Grid):
    """Compute F_i = m . (d_j m x d_k m), where (i, j, k) is cyclic.

    Uses spectral derivatives (FFT); requires ``grid.bc == 'periodic'``.
    Returns array of shape ``(3, nx, ny, nz)``.
    """
    if grid.bc != "periodic":
        raise ValueError("hopf_density / hopf_index require periodic BC for FFT inversion.")
    np = xp()
    nx, ny, nz = grid.shape
    dxm = np.stack([_spectral_d_axis(m[i], 0, grid.dx, nx) for i in range(3)], axis=0)
    dym = np.stack([_spectral_d_axis(m[i], 1, grid.dy, ny) for i in range(3)], axis=0)
    dzm = np.stack([_spectral_d_axis(m[i], 2, grid.dz, nz) for i in range(3)], axis=0)
    F_x = np.sum(m * _cross(dym, dzm), axis=0)
    F_y = np.sum(m * _cross(dzm, dxm), axis=0)
    F_z = np.sum(m * _cross(dxm, dym), axis=0)
    return np.stack([F_x, F_y, F_z], axis=0)


def gauge_potential(F, grid: Grid):
    """Solve curl A = F with div A = 0 by FFT on a periodic grid.

    A_hat = i (k x F_hat) / |k|^2, with the k=0 mode zeroed.
    """
    np = xp()
    F_hat = np.fft.fftn(F, axes=(1, 2, 3))
    nx, ny, nz = grid.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=grid.dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=grid.dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=grid.dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX * KX + KY * KY + KZ * KZ
    K2_safe = np.where(K2 == 0.0, 1.0, K2)
    Ax_hat = 1j * (KY * F_hat[2] - KZ * F_hat[1]) / K2_safe
    Ay_hat = 1j * (KZ * F_hat[0] - KX * F_hat[2]) / K2_safe
    Az_hat = 1j * (KX * F_hat[1] - KY * F_hat[0]) / K2_safe
    Ax_hat = np.where(K2 == 0.0, 0.0 + 0j, Ax_hat)
    Ay_hat = np.where(K2 == 0.0, 0.0 + 0j, Ay_hat)
    Az_hat = np.where(K2 == 0.0, 0.0 + 0j, Az_hat)
    A_hat = np.stack([Ax_hat, Ay_hat, Az_hat], axis=0)
    A = np.real(np.fft.ifftn(A_hat, axes=(1, 2, 3)))
    return A


def hopf_index(m, grid: Grid) -> float:
    """Compute the Hopf invariant Q_H of a unit-vector field on a periodic grid."""
    np = xp()
    F = hopf_density(m, grid)
    A = gauge_potential(F, grid)
    Q = np.sum(A * F) * grid.dV / (16.0 * np.pi * np.pi)
    return float(Q)


def skyrmion_charge_density(m, grid: Grid):
    """Pointwise 2D skyrmion-charge (Pontryagin) density.

        q(r) = (1 / 4 pi) m . (d_x m x d_y m)

    Uses central finite differences (``grid.d_axis``), consistent with
    ``skyrmion_density_xy`` and free of the periodic-BC requirement that
    ``hopf_density`` carries. Returns an array of shape ``(nx, ny, nz)``.
    """
    np = xp()
    dxm = np.stack([d_axis(m[i], 0, grid.dx, grid) for i in range(3)], axis=0)
    dym = np.stack([d_axis(m[i], 1, grid.dy, grid) for i in range(3)], axis=0)
    return np.sum(m * _cross(dxm, dym), axis=0) / (4.0 * np.pi)


def skyrmion_density_xy(m, grid: Grid):
    """2D skyrmion charge integrated over each (x, y) slice.

    Returns an array of shape ``(nz,)`` -- skyrmion charge per z-slice.
    """
    np = xp()
    q = skyrmion_charge_density(m, grid)
    return np.sum(q, axis=(0, 1)) * grid.dx * grid.dy


def skyrmion_number(m, grid: Grid, z_index: int | None = None) -> float:
    """Global 2D skyrmion number of a (z-extruded) texture.

    The skyrmion charge is a 2D invariant, so it is *not* summed over z. With
    ``z_index`` given, returns that slice's charge; otherwise returns the mean
    over all z-slices (the robust scalar for a tube that is z-invariant up to
    discretization).
    """
    np = xp()
    per_z = skyrmion_density_xy(m, grid)
    if z_index is not None:
        return float(per_z[z_index])
    return float(np.mean(per_z))


def _fd_gyro_field(m, grid: Grid):
    """Gyrovector / topological-charge density field F_i = m·(∂_j m × ∂_k m)
    via *central differences* (local, unlike the spectral ``hopf_density``).

    Finite differences are deliberate here: this feeds singularity detection,
    where the field is non-smooth and the FFT-spectral derivative rings.
    """
    np = xp()
    dxm = np.stack([d_axis(m[i], 0, grid.dx, grid) for i in range(3)], axis=0)
    dym = np.stack([d_axis(m[i], 1, grid.dy, grid) for i in range(3)], axis=0)
    dzm = np.stack([d_axis(m[i], 2, grid.dz, grid) for i in range(3)], axis=0)
    Fx = np.sum(m * _cross(dym, dzm), axis=0)
    Fy = np.sum(m * _cross(dzm, dxm), axis=0)
    Fz = np.sum(m * _cross(dxm, dym), axis=0)
    return np.stack([Fx, Fy, Fz], axis=0)


def monopole_density(m, grid: Grid):
    """Emergent-magnetic-charge (Bloch-point) density ρ = (1/4π) ∇·F.

    For a smooth field F = ∇×A so ∇·F ≡ 0; ρ is non-zero only at hedgehog
    singularities (Bloch points), where ∫ρ dV over an enclosing region is the
    integer monopole charge. Returns a scalar field of shape ``(nx, ny, nz)``.
    """
    np = xp()
    F = _fd_gyro_field(m, grid)
    div = (d_axis(F[0], 0, grid.dx, grid)
           + d_axis(F[1], 1, grid.dy, grid)
           + d_axis(F[2], 2, grid.dz, grid))
    return div / (4.0 * np.pi)


def bloch_points(m, grid: Grid, threshold_rel: float = 0.3, min_charge: float = 0.3):
    """Locate Bloch points (emergent monopoles): clusters of ``|monopole_density|``
    whose integrated charge exceeds ``min_charge``.

    Returns a list of ``(position, charge)`` tuples (sorted by |charge|). A smooth
    hopfion yields an empty list; a hedgehog yields one entry localized at the
    singularity with a significant sign-definite charge. (The magnitude reads
    below the continuum ±1 on a coarse periodic grid: a periodic box has zero net
    monopole charge, so the central peak is partially compensated by the wrap-around
    anti-hedgehog — detection of position and sign is robust, the integer less so.)
    """
    from scipy.ndimage import label as nd_label

    rho = to_numpy(monopole_density(m, grid))
    mag = _np.abs(rho)
    if mag.max() <= 0.0:
        return []
    mask = mag > threshold_rel * mag.max()
    labels, n = nd_label(mask)
    X, Y, Z = _np.meshgrid(
        (_np.arange(grid.nx) - grid.nx / 2 + 0.5) * grid.dx,
        (_np.arange(grid.ny) - grid.ny / 2 + 0.5) * grid.dy,
        (_np.arange(grid.nz) - grid.nz / 2 + 0.5) * grid.dz,
        indexing="ij",
    )
    out = []
    for k in range(1, n + 1):
        cl = labels == k
        charge = float(rho[cl].sum() * grid.dV)
        if abs(charge) < min_charge:
            continue
        w = mag[cl]
        tw = float(w.sum()) or 1.0
        pos = (float((X[cl] * w).sum() / tw),
               float((Y[cl] * w).sum() / tw),
               float((Z[cl] * w).sum() / tw))
        out.append((pos, charge))
    out.sort(key=lambda c: abs(c[1]), reverse=True)
    return out


def _preimage_loop_points(m, grid: Grid, target, tol: float = 0.05):
    """Ordered point cloud tracing a preimage loop (for linking-number Gauss
    integral). Voxels near ``target`` on S² are greedily chained nearest-neighbour
    into a closed curve."""
    mask = to_numpy(preimage_mask(m, target, tol=tol))
    idx = _np.argwhere(mask)
    if len(idx) < 3:
        return None
    pts = _np.stack([
        (idx[:, 0] - grid.nx / 2 + 0.5) * grid.dx,
        (idx[:, 1] - grid.ny / 2 + 0.5) * grid.dy,
        (idx[:, 2] - grid.nz / 2 + 0.5) * grid.dz,
    ], axis=1)
    # greedy nearest-neighbour ordering into a loop
    order = [0]
    remaining = set(range(1, len(pts)))
    while remaining:
        last = pts[order[-1]]
        j = min(remaining, key=lambda r: float(((pts[r] - last) ** 2).sum()))
        order.append(j)
        remaining.discard(j)
    return pts[order]


def linking_number(m, grid: Grid, target_a, target_b, tol: float = 0.05) -> float:
    """Gauss linking integral between the preimage loops of two S² targets.

    The Hopf invariant *is* the linking number of any two distinct preimages, so
    |linking_number| is an independent cross-check of ``hopf_index`` that uses no
    FFT/gauge-potential machinery — only the real-space preimage loops. The sign
    is orientation-dependent.

    Voxel-curve extraction is approximate and ``tol``-sensitive: too large a
    ``tol`` gives a thick shell the greedy ordering zig-zags through, too small
    fragments the loop. ``tol ≈ 0.05`` recovers |Lk| ≈ |Q_H| to ~10-20% for a
    well-resolved Q=1 hopfion (R/dx ≳ 4). Returns 0.0 if a preimage is empty.
    """
    A = _preimage_loop_points(m, grid, target_a, tol=tol)
    B = _preimage_loop_points(m, grid, target_b, tol=tol)
    if A is None or B is None:
        return 0.0
    dA = _np.roll(A, -1, axis=0) - A          # segment vectors
    dB = _np.roll(B, -1, axis=0) - B
    midA = A + 0.5 * dA
    midB = B + 0.5 * dB
    total = 0.0
    for i in range(len(A)):
        r = midA[i] - midB                     # (Nb, 3)
        rn = _np.linalg.norm(r, axis=1) ** 3 + 1e-12
        cross = _np.cross(_np.broadcast_to(dA[i], dB.shape), dB)
        total += float(_np.sum(_np.sum(cross * r, axis=1) / rn))
    return total / (4.0 * _np.pi)


def preimage_mask(m, target, tol: float = 0.15):
    """Return a boolean mask of cells where m is within ``tol`` of ``target`` on S^2.

    Useful for visualizing the closed-loop preimages that link to form a hopfion.
    """
    np = xp()
    tx, ty, tz = target
    norm = (tx * tx + ty * ty + tz * tz) ** 0.5
    tx, ty, tz = tx / norm, ty / norm, tz / norm
    dot = m[0] * tx + m[1] * ty + m[2] * tz
    return dot > (1.0 - tol)
