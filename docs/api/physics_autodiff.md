# `hopfion.physics.autodiff`

Differentiable energy + inverse design on the **JAX backend**. The pipeline's
`energy.total_energy` / `topology.hopf_index` cast to a Python `float`, which severs
`jax.grad`; this module re-expresses the (local) energy with identical
discretizations but no cast, making the simulator differentiable end-to-end.

Requires `backend.use("jax")`; set `jax_enable_x64` for double-precision parity.
Covers exchange / bulk + interfacial DMI / uniaxial + cubic anisotropy / Zeeman
(the local terms — not the FFT dipolar term).

## Public API

| Symbol | Purpose |
|---|---|
| `energy_value(m, grid, A_ex, D, Ku, easy_axis, H_ext, D_interface=0, Kc=0)` | differentiable total local energy (scalar) |
| `energy_value_ep(m, grid, ep)` | the same, reading scalars from an `EnergyParams` |
| `energy_gradient(m, grid, ep)` | `(1/dV)·jax.grad(energy)` — **equals `-effective_field`** (the differentiability proof) |
| `denergy_dparam_through_relax(m0, grid, ep, param, n_steps, dt)` | `dE_final/dparam` differentiated *through* the compiled `relax_scan` |
| `fit_scalar(loss, x0, lr, n_iter)` | minimal `jax.grad` gradient descent — the inverse-design kernel |

## Validated invariants (`tests/test_autodiff.py`, x64)

- `energy_gradient(m) == -effective_field(m)` to ~1e-14 — the autodiff path and the
  hand-coded field cannot silently diverge.
- `denergy_dparam_through_relax(..., "D")` matches a finite-difference `dE_final/dD`
  — genuine differentiable simulation through the relaxation.
- `fit_scalar` recovers the anisotropy `Ku` that hits a target energy.

## See also
- [backend.md](backend.md) — the numpy↔jax dispatch and the `relax_scan`/`integrate_scan` compiled loops.
