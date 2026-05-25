# `hopfion.physics.fmr`

Dynamical (precessional) eigenmodes — the FMR / spin-wave spectrum. Unlike the
*static* Hessian spectrum ([physics_hessian.md](physics_hessian.md), real
eigenvalues), this linearizes the **conservative LLG precession** about a fixed
point: $\dot v = -\gamma\,\mathbf m\times(\text{Hess}\cdot v)$, whose eigenvalues
are $\pm i\omega$.

## Public API

| Symbol | Purpose |
|---|---|
| `dynamical_modes(m, grid, ep, gamma=1.0, k=6, eps=1e-5, freq_floor=1e-6)` | `(freqs, modes)` — physical resonance frequencies $\omega=\lvert\text{Im}\,\lambda\rvert$ (ascending) and their tangent modes |

Built matrix-free on the operator $L=-\gamma\,(\mathbf m\times)\circ\text{Hess}$,
reusing `hessian.hessian_vector_product`, solved with `scipy.sparse.linalg.eigs`
(non-symmetric). Near-zero (radial/constraint) eigenvalues are filtered by `freq_floor`.

## Anchor

A uniform ferromagnet in a field $H\hat z$ (no exchange/DMI/anisotropy) returns the
Larmor/Kittel frequency $\omega=\gamma H$ — see `tests/test_fmr.py`. Easy-axis
anisotropy stiffens the precession (raises $\omega$).

## See also
- [physics_hessian.md](physics_hessian.md) — the static stability spectrum it reuses.
