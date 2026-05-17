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

Run these three recipes with the candidate material substituted:

```bash
# Substitute (A_ex, D, Ku) in the recipe via the CLI sweep:
hopfion run recipes/single_hopfion.yaml \
  --out runs/validation/{material}/single/
hopfion run recipes/laser_nucleation_2t.yaml \
  --out runs/validation/{material}/nucleate/
hopfion run recipes/moire_lattice.yaml \
  --out runs/validation/{material}/lattice/
```

(For a real workflow, write a YAML template and substitute via `jinja2` or a small Python wrapper. Phase C will add a `--material A_ex=X,D=Y,Ku=Z` shorthand.)

## Acceptance grades

| Outcome of all three | Grade | Recommendation |
|---|---|---|
| All ACCEPT | A | material is in the sweet spot; proceed to lifetime work (SOP-003) |
| single ACCEPT, others FAIL | B | usable for isolated hopfions but lattice geometry needs tuning |
| only single + nucleate ACCEPT | C | hopfion is metastable but the moiré lattice doesn't pin reliably |
| only single ACCEPT | D | usable only as a one-off; expect lifetime issues |
| none ACCEPT | F | not in the stability window; revisit (A_ex, D, K_u) |

## Quick diagnostic

For a non-A grade, run the Hessian probe on the relaxed single-hopfion seed:

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
