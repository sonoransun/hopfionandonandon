"""Hessian eigenmode analysis around a relaxed configuration.

We don't form the Hessian explicitly -- it would be a $(3N \\times 3N)$ matrix
where $N$ is the cell count. Instead we compute Hessian-vector products via
central differences on the existing ``effective_field``:

    H · v  ≈  -[H_eff(m + eps v) - H_eff(m - eps v)] / (2 eps)

The magnetization is constrained to $S^2$ (|m| = 1), so true perturbations
live in the tangent space at each site: $v \\perp m$. We tangent-project the
input ``v`` and the output ``H v`` to keep the spectrum clean (constraint
zero-modes don't pollute the lowest few eigenvalues).

``lowest_eigenmodes`` uses ``scipy.sparse.linalg.eigsh`` with a matrix-free
``LinearOperator``. The returned ``SpectrumResult`` carries the eigenvalues,
eigenmode shapes (reshaped back to $(3, n_x, n_y, n_z)$), and a flag for
"marginal stability" if any eigenvalue is below ``-tol``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as _np
from scipy.sparse.linalg import LinearOperator, eigsh

from hopfion.energy import EnergyParams, effective_field
from hopfion.grid import Grid


def _tangent_project(v, m):
    """Remove the radial (parallel-to-m) component at each site."""
    dot = (v * m).sum(axis=0, keepdims=True)
    return v - m * dot


def hessian_vector_product(m, v, grid: Grid, ep: EnergyParams, eps: float = 1e-5,
                            H_at_m: Optional[_np.ndarray] = None):
    """Approximate the Riemannian Hessian of E on the |m|=1 manifold acting on v.

    The Riemannian Hessian on a sphere is

        Hess_M E (v) = P (∂²E/∂m²) P (v)  -  λ(r) v

    where λ(r) = m(r) · ∇E = -m(r) · H_eff(r) is the per-site Lagrange
    multiplier field. The first term is computed via central FD on
    ``effective_field``; the second is a per-site scalar multiplication.

    ``H_at_m`` may be supplied to avoid recomputing H_eff(m) inside a loop.
    """
    m_np = _np.asarray(m)
    v_t = _tangent_project(_np.asarray(v), m_np)
    H_p = _np.asarray(effective_field(m_np + eps * v_t, grid, ep))
    H_m = _np.asarray(effective_field(m_np - eps * v_t, grid, ep))
    Hv_ambient = -(H_p - H_m) / (2.0 * eps)
    Hv = _tangent_project(Hv_ambient, m_np)
    # Lagrange-multiplier correction
    if H_at_m is None:
        H_at_m = _np.asarray(effective_field(m_np, grid, ep))
    lam = -(m_np * H_at_m).sum(axis=0, keepdims=True)   # shape (1, nx, ny, nz)
    Hv = Hv - lam * v_t
    return Hv


@dataclass
class SpectrumResult:
    eigenvalues: _np.ndarray                       # shape (k,)
    eigenmodes: _np.ndarray                        # shape (k, 3, nx, ny, nz)
    min_eig: float = 0.0
    marginally_stable: bool = False

    def to_metrics_dict(self):
        return {
            "min_eig": float(self.min_eig),
            "all_eigs": [float(v) for v in self.eigenvalues],
            "marginally_stable": bool(self.marginally_stable),
        }


def lowest_eigenmodes(m, grid: Grid, ep: EnergyParams, k: int = 5,
                      eps: float = 1e-5, tol_marginal: float = 1e-3,
                      eigs_tol: float = 1e-4) -> SpectrumResult:
    """Lowest ``k`` algebraic eigenvalues of the constrained Hessian at ``m``.

    Use this after a relaxation has converged. A clean (metastable) hopfion has
    all eigenvalues ≥ 0 within ``tol_marginal``; any clearly negative
    eigenvalue indicates a saddle and the configuration will decay along the
    corresponding mode.
    """
    m_np = _np.asarray(m)
    shape = m_np.shape
    N = int(_np.prod(shape))
    H_at_m = _np.asarray(effective_field(m_np, grid, ep))   # cache

    def matvec(v_flat):
        v_field = v_flat.reshape(shape)
        Hv = hessian_vector_product(m_np, v_field, grid, ep, eps=eps, H_at_m=H_at_m)
        return _np.asarray(Hv).ravel()

    L = LinearOperator((N, N), matvec=matvec, dtype=float)
    k = min(k, N - 2)
    # 'SA' = smallest algebraic — what we want for stability analysis
    vals, vecs = eigsh(L, k=k, which="SA", tol=eigs_tol)
    order = _np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    modes = _np.stack([vecs[:, i].reshape(shape) for i in range(k)], axis=0)
    min_eig = float(vals[0])
    return SpectrumResult(
        eigenvalues=vals, eigenmodes=modes,
        min_eig=min_eig,
        marginally_stable=min_eig < -tol_marginal,
    )


__all__ = ["hessian_vector_product", "lowest_eigenmodes", "SpectrumResult"]
