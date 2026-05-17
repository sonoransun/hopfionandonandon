# `hopfion.physics.integrators`

Per-step integrator stencils. `hopfion.llg` delegates to these — the Phase-A surface (`llg_step_heun`, `relax_step`, `relax`, `integrate`) is unchanged but the math lives here.

## Public API

| Function | Purpose |
|---|---|
| `heun_step(m, grid, ep, gamma, alpha, dt, H_extra=None)` | Improved-Euler (RK2). Default. |
| `rk4_step(m, grid, ep, gamma, alpha, dt, H_extra=None)` | Classical RK4 (~2× Heun cost, ~10³× better local truncation). |
| `adaptive_heun_step(m, grid, ep, gamma, alpha, dt, H_extra=None, rtol=1e-3)` | Embedded Heun-Euler with reject-and-halve. Returns `AdaptiveResult`. |
| `damped_step(m, grid, ep, dt=0.01, H_extra=None)` | Pure gradient flow (drop precession). |
| `llg_rhs(m, H, gamma, alpha)` | Explicit LLG right-hand side $-\gamma(\mathbf{m}\times\mathbf{H}) - \gamma\alpha\,\mathbf{m}\times(\mathbf{m}\times\mathbf{H})$. |

All integrators renormalize `|m| = 1` at the end of each step.

## Usage

```python
from hopfion.physics.integrators import rk4_step
m = rk4_step(m, grid, ep, gamma=1.0, alpha=0.05, dt=0.001)
```

## See also

- [llg.md](llg.md) — `LLGParams` + the back-compat surface.
- [physics_two_temp.md](physics_two_temp.md) — uses `heun_step` for Heun-Stratonovich.
- [PHYSICS.md §5](../PHYSICS.md#5-llg-dynamics) — the equations.
