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

from hopfion.backend import xp
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


def skyrmion_density_xy(m, grid: Grid):
    """2D skyrmion-charge density on each (x, y) slice.

    Returns an array of shape ``(nz,)`` -- skyrmion charge per z-slice.
    """
    np = xp()
    dxm = np.stack([d_axis(m[i], 0, grid.dx, grid) for i in range(3)], axis=0)
    dym = np.stack([d_axis(m[i], 1, grid.dy, grid) for i in range(3)], axis=0)
    q = np.sum(m * _cross(dxm, dym), axis=0) / (4.0 * np.pi)
    # integrate over (x, y) per z
    Q_per_z = np.sum(q, axis=(0, 1)) * grid.dx * grid.dy
    return Q_per_z


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
