"""Magnetostatic (dipolar / demagnetizing) field via FFT convolution.

In a fully-periodic box, the magnetic scalar potential satisfies

    ∇²φ = ∇·m,        H_dip = -∇φ.

Solving in Fourier space gives the compact form

    Ĥ_dip,i(k) = -k_i k_j / |k|² · m̂_j(k),   k ≠ 0.

The k = 0 mode (uniform component) is ill-defined in a fully-periodic box --
its value depends on the boundary at infinity. We zero it out (a common
convention; equivalent to subtracting the spatial mean of m). This is the
right choice when the user means "long-wavelength dipolar coupling within a
periodic unit cell" rather than "shape-dependent demagnetization of a
finite-size sample."

The dipolar energy is

    E_dip = -(1/2) ∫ m · H_dip d³r = (1/2) Σ_k |k̂·m̂(k)|² / (volume),

always non-negative.

Units: μ₀ M_s² = 1 in the rest of the codebase. This term enters
``EnergyParams.dipolar = True`` and routes through ``effective_field``.
"""
from __future__ import annotations

import numpy as _np

from hopfion.backend import xp
from hopfion.grid import Grid


def _khat_kdotm(m, grid: Grid):
    """Return (k_hat ⊗ k_hat) · m̂ in Fourier space, i.e. the dipolar Ĥ_dip
    *without* the sign, divided by FFT normalization.

    Output shape: ``(3, nx, ny, nz)`` complex.
    """
    np = xp()
    nx, ny, nz = grid.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=grid.dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=grid.dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=grid.dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX * KX + KY * KY + KZ * KZ
    K2_safe = np.where(K2 == 0.0, 1.0, K2)

    m_hat = np.fft.fftn(m, axes=(1, 2, 3))
    kdotm = KX * m_hat[0] + KY * m_hat[1] + KZ * m_hat[2]
    # Ĥ_dip,i = -k_i (k·m̂) / k²
    Hx = -KX * kdotm / K2_safe
    Hy = -KY * kdotm / K2_safe
    Hz = -KZ * kdotm / K2_safe
    # zero the k=0 mode (uniform-mean handling)
    Hx = np.where(K2 == 0.0, 0.0 + 0j, Hx)
    Hy = np.where(K2 == 0.0, 0.0 + 0j, Hy)
    Hz = np.where(K2 == 0.0, 0.0 + 0j, Hz)
    return np.stack([Hx, Hy, Hz], axis=0)


def dipolar_field(m, grid: Grid):
    """Dipolar (magnetostatic) effective field; periodic BC; k=0 mode zeroed."""
    if grid.bc != "periodic":
        raise ValueError("dipolar_field requires periodic boundary conditions.")
    np = xp()
    H_hat = _khat_kdotm(m, grid)
    return np.real(np.fft.ifftn(H_hat, axes=(1, 2, 3)))


def dipolar_energy(m, grid: Grid) -> float:
    """E_dip = (1/2) Σ_k |k̂·m̂(k)|² / V, with the k=0 mode set to zero.

    Equivalent to -(1/2) ∫ m · H_dip d³r in real space.
    """
    if grid.bc != "periodic":
        raise ValueError("dipolar_energy requires periodic boundary conditions.")
    np = xp()
    nx, ny, nz = grid.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=grid.dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=grid.dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=grid.dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX * KX + KY * KY + KZ * KZ
    K2_safe = np.where(K2 == 0.0, 1.0, K2)
    m_hat = np.fft.fftn(m, axes=(1, 2, 3))
    kdotm = KX * m_hat[0] + KY * m_hat[1] + KZ * m_hat[2]
    integrand = np.where(K2 == 0.0, 0.0, (np.abs(kdotm) ** 2) / K2_safe)
    # Parseval: Σ_k |X̂|² = N · ∫|X|² ; we want (1/2) integral in real space
    N = nx * ny * nz
    total = 0.5 * float(np.sum(integrand)) * grid.dV / N
    return total


__all__ = ["dipolar_field", "dipolar_energy"]
