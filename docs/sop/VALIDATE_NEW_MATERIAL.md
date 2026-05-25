# SOP-004: Validate a candidate material against the hopfion stability window

**Purpose**: given a new candidate `(A_ex, D, K_u)` triple, run a pre-flight battery and grade fitness for sustaining metastable hopfions. Output: a one-line verdict and a 3-recipe diagnostic.

**Recipes used**: temporary copies of `recipes/single_hopfion.yaml` and `recipes/moire_lattice.yaml` with the candidate material substituted in.

## Pre-flight grading

The hopfion stability window in normalized units is roughly:

| Parameter | Required range | Comment |
|---|---|---|
| `A_ex` | 1.0 (by normalization) | scale everything else relative to this |
| `D` | [1.2, 1.8] | bulk DMI strength |
| `K_u` | [0.5, 0.8] | easy-axis anisotropy |
| `L_D = A_ex / D` | (0.55, 0.85) | DMI helicity length should be ~R/2 |
| `Q = D² / (A_ex K_u)` | (2.0, 7.0) | dimensionless helicity index |

If the candidate falls outside these ranges, hopfion stabilization is unlikely on a typical grid.

## Standard battery

The `validate` subcommand automates the whole battery — substitute the candidate
material, run `single` / `nucleate` / `lattice`, and print the grade:

```bash
hopfion validate --material A_ex=1.0,D=1.5,Ku=0.7            # authoritative
hopfion validate --material A_ex=1.0,D=1.5,Ku=0.7 --quick   # fast smoke (caps steps)
hopfion validate --material A_ex=1.0,D=1.5,Ku=0.7 --hessian # + min-eigenvalue probe
```

`--quick` caps each run-step's `n_steps` for speed; it can over-grade an unstable
material because the topology hasn't had time to decay, so use the full (non-quick)
battery for an authoritative grade. The pre-flight `L_D` / `Q` metrics are printed
either way. The three recipes (`single_hopfion`, `laser_nucleation_2t`,
`moire_lattice`) are run with the substituted material; outputs land under
`runs/validation/<material>/`.

## Acceptance grades

`single` (an isolated hopfion holding) is the gate; if it fails, the material fails.

| Battery outcome | Grade | Recommendation |
|---|---|---|
| single + nucleate + lattice ACCEPT | A | sweet spot; proceed to lifetime work (SOP-003) |
| single + lattice ACCEPT (nucleate FAIL) | B | isolated + lattice work; nucleation marginal |
| single + nucleate ACCEPT (lattice FAIL) | C | metastable + nucleates, but the moiré lattice doesn't pin |
| single ACCEPT only | D | usable as a one-off; expect lifetime issues |
| single FAIL | F | not in the stability window; revisit (A_ex, D, K_u) |

## Quick diagnostic

For a non-A grade, `hopfion validate --hessian ...` runs a Hessian probe on the
relaxed single-hopfion seed and prints `min_eig` (a negative value flags a saddle,
not a minimum). To run it by hand on a saved run:

```bash
hopfion run recipes/single_hopfion.yaml --out runs/validation/{material}/single/
PYTHONPATH=src python3 -c "
import h5py
from hopfion.energy import EnergyParams
from hopfion.grid import Grid
from hopfion.physics.hessian import lowest_eigenmodes
# ... load grid, ep, m from runs/.../run.h5 ...
res = lowest_eigenmodes(m, grid, ep, k=5)
print('min eigenvalue:', res.min_eig)
print('eigenvalue spectrum:', res.eigenvalues)
"
```

If `min_eig < -1e-3`: the configuration is a saddle, not a minimum. Adjust D, K_u in the direction that flattens the unstable mode.

## References

- [docs/ARCHITECTURE.md §5](../ARCHITECTURE.md#5-hopfion-stability-window) — origin of the stability window numbers.
- [PHYSICS.md §3](../PHYSICS.md#3-micromagnetic-energy) — energy terms.
