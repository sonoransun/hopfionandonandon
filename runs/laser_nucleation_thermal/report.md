# Run report — `laser_nucleation_thermal`

**Verdict**: ✅ **ACCEPT**  ·  wall time: 7.4 s  ·  Q$_H$(final): +1.0544  ·  backend: `numpy`

> Phase-A laser nucleation via a white-noise thermal burst (the stand-in for the two-temperature model in Phase B2). The thermal step heats the spin field to chaotic, and the subsequent damped relax crystallizes a Q_H ≈ 1 texture from the cooldown. Stochastic — yield depends on seed. 

## Quality control

| status | criterion | message |
|---|---|---|
| ✅ pass | `norm_drift_below` | max |m|-1 drift = 3.33e-16 (tol 1e-10) |
| ✅ pass | `hopf_index_within` | Q_H final = +1.1390 vs target 1.0 (tol 0.5) |
| ✅ pass | `runtime_within` | total runtime = 7.3s (max 30.0s) |

## Metric summary

```json
{
  "energy": {
    "E_initial": -2090.1888,
    "E_final": 1156.074563052799,
    "max_increase": 68458.45297148444,
    "mean_step": 9.017398230702224,
    "n_samples": 361
  },
  "norm_drift": {
    "max_drift": 3.3306690738754696e-16,
    "final_drift": 3.3306690738754696e-16,
    "n_samples": 361
  },
  "q_hopf": {
    "Q_initial": 0.0,
    "Q_final": 1.1390120796378622,
    "max_drift": 2.8674629857283738,
    "n_samples": 16
  },
  "runtime": {
    "total_seconds": 7.283415896992665,
    "mean_step_ms": 20.288066565439177,
    "p95_step_ms": 35.05736038205214,
    "n_samples": 359
  }
}
```

## Final state

![final xy slice](figs/final_xy.png)

## Time series

### Energy time series

![Energy time series](figs/energy.png)

### Hopf invariant

![Hopf invariant](figs/q_hopf.png)

## Recipe

```yaml
name: laser_nucleation_thermal
backend: numpy
seed: 0
grid: {"nx": 48, "ny": 48, "nz": 48, "dx": 0.3, "dy": 0.3, "dz": 0.3, "bc": "periodic"}
material: {"A_ex": 1.0, "D": 1.5, "Ku": 0.7, "easy_axis": [0, 0, 1], "H_ext": [0, 0, 0], "dipolar": false, "moire": null}
initial: {"kind": "uniform", "direction": [0, 0, 1], "R": 1.5, "p": 1, "q": 1, "center": [0.0, 0.0, 0.0], "axis": "z", "amplitude": 0.0, "array_lattice": "triangular", "array_a": 8.0, "array_n_rings": 1, "array_nx": 3, "array_ny": 3, "file_path": null}
run:
  - {"kind": "thermal_burst", "n_steps": 60, "dt": 0.002, "gamma": 1.0, "alpha": 0.1, "kT": 15.0, "H0": [0.0, 0.0, 0.0], "t0": 0.0, "tau": 0.05, "profile": "point", "pulse_width": 1.0, "ring_radius": 1.5, "Te_peak": 0.0, "G_el": 1.0, "C_e": 1.0, "C_l": 1.0}
  - {"kind": "relax", "n_steps": 300, "dt": 0.002, "gamma": 1.0, "alpha": 0.1, "kT": 0.0, "H0": [0.0, 0.0, 0.0], "t0": 0.0, "tau": 0.05, "profile": "point", "pulse_width": 1.0, "ring_radius": 1.5, "Te_peak": 0.0, "G_el": 1.0, "C_e": 1.0, "C_l": 1.0}
```