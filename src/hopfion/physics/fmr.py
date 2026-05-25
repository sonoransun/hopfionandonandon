"""Dynamical (precessional) eigenmodes of a relaxed configuration — the FMR /
spin-wave spectrum.

Unlike ``hessian.lowest_eigenmodes`` (the *static* curvature spectrum, real
eigenvalues), this linearizes the *conservative* LLG precession about a fixed
point. For a tangent perturbation ``v`` the linearized equation of motion is

    dv/dt = -γ · m × (Hess · v),

so the resonance frequencies are the (imaginary) eigenvalues of the non-symmetric
operator ``L = -γ (m×) ∘ Hess``: stable modes come in pairs ``±iω``. We reuse the
matrix-free Riemannian Hessian-vector product from ``hessian.py`` and ``scipy``'s
non-symmetric ``eigs``.

Sanity anchor: a uniform ferromagnet in a field ``H ẑ`` (no exchange/DMI/aniso)
gives the Larmor/Kittel frequency ``ω = γH`` — see ``tests/test_fmr.py``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as _np
from scipy.sparse.linalg import LinearOperator, eigs

from hopfion.energy import EnergyParams, effective_field
from hopfion.grid import Grid
from hopfion.physics.hessian import hessian_vector_product


def _cross(a, b):
    return _np.stack([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ], axis=0)


def dynamical_modes(m, grid: Grid, ep: EnergyParams, gamma: float = 1.0,
                    k: int = 6, eps: float = 1e-5,
                    freq_floor: float = 1e-6) -> Tuple[_np.ndarray, _np.ndarray]:
    """Resonance frequencies ``ω`` (and modes) of the linearized LLG at ``m``.

    Returns ``(freqs, modes)`` with ``freqs`` the physical (non-zero) frequencies
    ``|Im λ|`` sorted ascending and ``modes`` of shape ``(len(freqs), 3, nx, ny, nz)``.
    Near-zero eigenvalues (the radial / constraint directions) are filtered out via
    ``freq_floor``.
    """
    m_np = _np.asarray(m, dtype=float)
    shape = m_np.shape
    N = int(_np.prod(shape))
    H_at_m = _np.asarray(effective_field(m_np, grid, ep))

    def matvec(v_flat):
        v = v_flat.reshape(shape)
        Hv = _np.asarray(hessian_vector_product(m_np, v, grid, ep, eps=eps, H_at_m=H_at_m))
        return (-gamma * _cross(m_np, Hv)).ravel()

    L = LinearOperator((N, N), matvec=matvec, dtype=float)
    kk = min(k, N - 2)
    vals, vecs = eigs(L, k=kk, which="LM")
    freqs = _np.abs(vals.imag)
    keep = freqs > freq_floor
    freqs = freqs[keep]
    vecs = vecs[:, keep]
    order = _np.argsort(freqs)
    freqs = freqs[order]
    modes = _np.stack([vecs[:, i].real.reshape(shape) for i in range(len(order))], axis=0) \
        if len(order) else _np.empty((0,) + shape)
    return freqs, modes


__all__ = ["dynamical_modes"]
