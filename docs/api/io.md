# `hopfion.io`

HDF5 snapshot I/O for runs. Field + run metadata + optional snapshot timeline + energy series in a single file.

## Public API

| Function | Purpose |
|---|---|
| `save_run(path, m, grid, ep, *, snapshots=None, times=None, energy_series=None, seed=None, note="")` | Write run to HDF5 |
| `load_run(path) -> (m, grid, ep, extras)` | Read back; `extras` is a dict |

## File layout

```
run.h5
├── m                       # (3, nx, ny, nz)  -- latest field
├── Ku_field?               # (nx, ny, nz)     -- if spatially-varying anisotropy was used
├── snapshots/
│   ├── m                   # (n_snap, 3, nx, ny, nz)
│   └── t                   # (n_snap,)
├── energy                  # (n_step,)
└── attrs:                  # grid params, EnergyParams, backend, git rev, seed, note
```

## Usage

```python
from hopfion.io import save_run, load_run
save_run("runs/relax_test.h5", m, grid, ep,
         snapshots=[s for s in trajectory],
         times=t_list,
         seed=0,
         note="N=48 relax of Q=1 hopfion")

m2, grid2, ep2, extras = load_run("runs/relax_test.h5")
print(extras["note"], extras["snapshots"].shape)
```

## Caveats

- The `git_rev` attribute is best-effort — it shells out to `git rev-parse` and writes `"unknown"` if that fails.
- All array data is stored as plain NumPy (backend-agnostic); on load, you may need to cast back to JAX manually if reusing under `HOPFION_BACKEND=jax`.

## See also

- [ARCHITECTURE.md](../ARCHITECTURE.md) — design rationale
- [energy.md](energy.md) — `EnergyParams` shape
