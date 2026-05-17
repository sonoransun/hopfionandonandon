# `hopfion.physics.two_temp`

Two-temperature stochastic LLG: physically grounded laser-driven nucleation. Replaces the Phase-A white-noise stand-in.

## What it computes

1. Coupled ODEs for electron + lattice temperatures:
   $C_e \dot T_e = -G_{\rm el}(T_e - T_l) + P(t)$, $C_l \dot T_l = G_{\rm el}(T_e - T_l)$.
2. Stochastic field with FDT-consistent variance: $\langle \eta_i \eta_j \rangle = (2\alpha T_e / \Delta t\,\Delta V) \delta_{ij}$.
3. Heun-Stratonovich LLG step using the noise as `H_extra`.

## Public API

| Symbol | Purpose |
|---|---|
| `TwoTemperatureLLG(Te_peak, tau, t0, G_el, C_e, C_l, alpha, gamma)` | Configuration + engine |
| `.laser_power(t)` | Gaussian absorption profile |
| `.step_temperatures(T_e, T_l, dt, t)` | Forward-Euler ODE step |
| `.stochastic_field(T_e, dt, grid, rng)` | Sample thermal noise field |
| `.run(m, grid, ep, n_steps, dt, rng, step_callback=None, record_temperatures=False)` | Co-evolve m, T_e, T_l |

## Usage

Via the recipe `run-step kind: two_temperature` (see [PIPELINE.md](../PIPELINE.md)):

```yaml
run:
  - kind: two_temperature
    n_steps: 100
    dt: 0.002
    Te_peak: 20.0
    tau: 0.05
    t0: 0.1
    alpha: 0.1
  - kind: relax
    n_steps: 300
    dt: 0.002
```

Or directly:

```python
from hopfion.physics.two_temp import TwoTemperatureLLG
import numpy as np
engine = TwoTemperatureLLG(Te_peak=20, tau=0.05, t0=0.1, alpha=0.1)
rng = np.random.default_rng(0)
m_final = engine.run(m_init, grid, ep, n_steps=100, dt=0.002, rng=rng)
```

## Caveats

- Forward-Euler is fine for the bath ODE (T-timescale ≫ dt) but the spin dynamics must use the Heun step. The engine handles this for you.
- The noise sample is held constant within a Heun step (predictor + corrector) — that's the Stratonovich convention.

## See also

- [PHYSICS.md §6](../PHYSICS.md#6-laser-pulse-models) — derivation.
- [SOP-001](../sop/NUCLEATE_HOPFION.md) — recipe-driven nucleation workflow.
- [laser.md](laser.md) — Phase-A `GaussianPulse` and the `TwoTemperaturePulse` back-compat wrapper.
