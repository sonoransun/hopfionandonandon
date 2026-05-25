# Run report — `moire_lattice_7`

**Verdict**: ✅ **ACCEPT**  ·  wall time: 31.2 s  ·  Q$_H$(final): +6.9984  ·  backend: `numpy`

> 7-hopfion triangular array pinned by a moiré anisotropy modulation. Initial total Q_H ≈ 7; after relax, ≈ 6.99 (all sites survive). 

## Quality control

| status | criterion | message |
|---|---|---|
| ✅ pass | `energy_monotonicity` | max energy increase per step = -2.272e+00 (tol 1e-06) |
| ✅ pass | `hopf_index_within` | Q_H final = +6.9987 vs target 7.0 (tol 0.3) |
| ✅ pass | `norm_drift_below` | max |m|-1 drift = 3.33e-16 (tol 1e-10) |
| ✅ pass | `runtime_within` | total runtime = 30.8s (max 180.0s) |
| ✅ pass | `per_site_voronoi_health` | per-site Voronoi fidelity = 100.00% (7/7 sites, min 90%) |

## Metric summary

```json
{
  "energy": {
    "E_initial": -2032.9528153157962,
    "E_final": -3517.2038068335514,
    "max_increase": -2.2720713495336895,
    "mean_step": -5.937003966071021,
    "n_samples": 251
  },
  "norm_drift": {
    "max_drift": 3.3306690738754696e-16,
    "final_drift": 3.3306690738754696e-16,
    "n_samples": 251
  },
  "q_hopf": {
    "Q_initial": 7.000009624760607,
    "Q_final": 6.998745861700771,
    "max_drift": 0.0012637630598364424,
    "n_samples": 11
  },
  "q_skyrmion": {
    "N_initial": 0.14465780624507185,
    "N_final": 0.064938439059475,
    "max_drift": 0.07971936718559686,
    "n_samples": 11
  },
  "runtime": {
    "total_seconds": 30.84225799603155,
    "mean_step_ms": 123.8644899439018,
    "p95_step_ms": 397.90942560648546,
    "n_samples": 249
  },
  "flux": {
    "max_residual_rms": 0.0,
    "mean_residual_rms": 0.0,
    "n_samples": 0
  },
  "drift": {
    "n_snapshots": 11,
    "n_centroids_initial": 7,
    "n_centroids_final": 7,
    "max_drift_speed": 0.0
  },
  "per_site_q": {
    "n_snapshots": 11,
    "n_sites_initial": 7,
    "n_sites_final": 7,
    "Q_per_site_initial": [
      0.5927906261204082,
      0.5927906261204081,
      0.5904073731843981,
      0.5904073731843981,
      0.5904073731843981,
      0.5904073731843981,
      0.5576896137342534
    ],
    "Q_per_site_final": [
      0.7430261292108178,
      0.7430261292108177,
      0.7394386076762026,
      0.7394386076762025,
      0.7329372013663668,
      0.7329372013663666,
      0.7257579141950531
    ]
  },
  "per_site_q_voronoi": {
    "n_sites": 7,
    "n_snapshots": 11,
    "Q_per_site_initial": [
      0.9995170184886365,
      1.0001889681894343,
      1.0000286674732728,
      1.0000286674732741,
      1.000028667473285,
      1.0000286674732841,
      1.0001889681894398
    ],
    "Q_per_site_final": [
      0.9999964938631971,
      0.9998399278609718,
      0.9998054346083513,
      0.9997293214494797,
      0.9997293214494737,
      0.9998054346083396,
      0.9998399278609675
    ]
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
name: moire_lattice_7
backend: numpy
seed: 0
grid: {"nx": 96, "ny": 96, "nz": 32, "dx": 0.3, "dy": 0.3, "dz": 0.3, "bc": "periodic"}
material: {"A_ex": 1.0, "D": 1.5, "Ku": 0.0, "easy_axis": [0, 0, 1], "H_ext": [0, 0, 0], "dipolar": false, "moire": {"enabled": true, "K0": 0.7, "V0": 0.3, "a_moire": 8.0, "lattice": "triangular"}}
initial: {"kind": "hopfion_array", "direction": [0.0, 0.0, 1.0], "R": 1.5, "p": 1, "q": 1, "center": [0.0, 0.0, 0.0], "axis": "z", "amplitude": 0.0, "array_lattice": "triangular", "array_a": 8.0, "array_n_rings": 1, "array_nx": 3, "array_ny": 3, "file_path": null, "separation": 4.0, "axis_of_separation": "x", "skyrmion_radius": 1.0, "skyrmion_helicity": 1.5707963267948966, "skyrmion_vorticity": 1}
run:
  - {"kind": "relax", "n_steps": 250, "dt": 0.002, "gamma": 1.0, "alpha": 0.1, "kT": 0.0, "H0": [0.0, 0.0, 0.0], "t0": 0.0, "tau": 0.05, "profile": "point", "pulse_width": 1.0, "ring_radius": 1.5, "Te_peak": 0.0, "G_el": 1.0, "C_e": 1.0, "C_l": 1.0, "u": [0.0, 0.0, 0.0], "beta": 0.0, "integrator": "heun", "bilayer_theta": 0.1, "bilayer_a": 4.0, "bilayer_J0": 0.3, "bilayer_layer2": "uniform"}
```