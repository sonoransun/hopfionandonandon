# `hopfion.energy`

Micromagnetic energy functional and effective field. Units: $\mu_0 M_s = 1$, so $\mathbf{H}_{\rm eff} = -\delta E/\delta \mathbf{m}$ has the same units as the energy density coefficients.

## `EnergyParams` dataclass

| Field | Default | Term |
|---|---|---|
| `A_ex` | 1.0 | Heisenberg exchange |
| `D` | 0.0 | Bulk (Bloch) DMI |
| `Ku` | 0.0 | Uniaxial anisotropy strength (scalar) |
| `easy_axis` | (0, 0, 1) | Anisotropy direction |
| `H_ext` | (0, 0, 0) | External Zeeman field |
| `Ku_field` | `None` | Spatial $K_u(\mathbf{r})$; overrides `Ku` if set |

## Per-term API

For each of `exchange`, `dmi`, `anisotropy`, `zeeman` there is an `*_energy(...)` and an `*_field(...)`:

```python
exchange_energy(m, grid, A_ex)    -> float
exchange_field(m, grid, A_ex)     -> (3, nx, ny, nz)
dmi_energy(m, grid, D)            -> float
dmi_field(m, grid, D)             -> (3, nx, ny, nz)
anisotropy_energy(m, grid, Ku, e) -> float
anisotropy_field(m, grid, Ku, e)  -> (3, nx, ny, nz)
zeeman_energy(m, grid, H)         -> float
zeeman_field(m, grid, H)          -> (3, nx, ny, nz)
```

And the aggregate:

```python
total_energy(m, grid, params)     -> float
effective_field(m, grid, params)  -> (3, nx, ny, nz)
```

## Discretization

**Exchange** uses a **bond-energy formulation** (forward differences) so its discrete gradient is exactly the 3-point Laplacian used by `exchange_field`. This makes `total_energy` and `effective_field` an exact adjoint pair on periodic grids. The other terms use central differences via `hopfion.grid`. See [PHYSICS.md §4](../PHYSICS.md#4-bond-energy-discretization) for the derivation.

## Usage

```python
from hopfion.energy import EnergyParams, total_energy, effective_field
params = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7, easy_axis=(0,0,1))
E = total_energy(m, grid, params)
H = effective_field(m, grid, params)         # -δE/δm
```

For moiré-modulated anisotropy:
```python
from hopfion.moire import MoirePotential
mp = MoirePotential(K0=0.7, V0=0.3, a_moire=8.0, lattice="triangular")
params = EnergyParams(A_ex=1.0, D=1.5, Ku_field=mp.Ku_field(grid))
```

## Verifying new terms

The FD consistency check in `tests/test_energy.py::PARAM_SETS` will catch sign or factor errors in any newly added term. Add a parameter set there and re-run — if numerical $\partial E/\partial m$ disagrees with analytic $-\mathbf{H}_{\rm eff}$ above $10^{-3}$, your term has a bug.

## See also

- [PHYSICS.md §3](../PHYSICS.md#3-micromagnetic-energy) — equations for each term
- [PHYSICS.md §4](../PHYSICS.md#4-bond-energy-discretization) — bond-form derivation
- [moire.md](moire.md) — how `Ku_field` enters
- [ARCHITECTURE.md §6A](../ARCHITECTURE.md#a-adding-a-new-energy-term) — recipe for new terms
