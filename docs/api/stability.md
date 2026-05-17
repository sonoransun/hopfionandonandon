# `hopfion.stability`

Perturbation + lifetime utilities. Phase A: perturb-and-relax test, white-noise thermal lifetime tracker. Phase B will add Hessian eigenmode analysis.

## Public API

| Function | Purpose |
|---|---|
| `perturb_and_relax(m, grid, ep, amplitude=0.05, n_steps=500, dt=0.01, seed=0)` | Jolt + relax; returns `StabilityResult` |
| `thermal_lifetime(m, grid, ep, lp, kT, n_steps=2000, seed=0, Q_tol=0.5)` | White-noise sLLG, returns `(Q_history, lifetime_steps)` |

`StabilityResult` is a frozen dataclass with `Q_initial`, `Q_final`, `E_initial`, `E_final`, `steps_run`.

## Usage

```python
from hopfion.stability import perturb_and_relax, thermal_lifetime
res = perturb_and_relax(m_relaxed, grid, ep, amplitude=0.05, n_steps=400)
print(f"Q drift: {res.Q_initial:+.3f} -> {res.Q_final:+.3f}")

history, t_life = thermal_lifetime(m_relaxed, grid, ep, lp, kT=0.5)
print(f"Hopfion survived {t_life} of {len(history)-1} steps")
```

## Caveats

- The thermal-lifetime sLLG is a **Phase A approximation**: noise is added as an extra field with variance $\sigma^2 = 2\alpha k_B T/(\Delta t\, dV)$, but the integrator is plain Heun (not Heun–Stratonovich). Use it for relative comparisons across $T$ values, not for absolute Arrhenius rates.
- The `Q_tol = 0.5` default means "$Q_H$ has drifted by half an integer" — a generous threshold; tighten to `0.1` for finer lifetime resolution.

## See also

- [llg.md](llg.md) — `relax_step` and `llg_step_heun` are the underlying primitives
- [topology.md](topology.md) — `hopf_index` is used to detect collapse
- [PHYSICS.md §5](../PHYSICS.md#5-llg-dynamics) — damped-LLG equation
