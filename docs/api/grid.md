# `hopfion.grid`

Cubic finite-difference operators on a 3D lattice plus the `Grid` dataclass that everything else passes around.

## `Grid` dataclass

| Field | Type | Default | Notes |
|---|---|---|---|
| `nx, ny, nz` | int | — | Cell counts |
| `dx, dy, dz` | float | 1.0 | Cell spacings |
| `bc` | `"periodic" \| "open"` | `"periodic"` | Boundary mode |

Methods: `shape`, `dV`, `coords()` (returns `(X, Y, Z)` meshgrid).

## Operators (all central FD)

| Function | Input shape | Output shape | Notes |
|---|---|---|---|
| `grad_scalar(s, grid)` | `(nx,ny,nz)` | `(3,nx,ny,nz)` | $\nabla s$ |
| `grad_vector(v, grid)` | `(3,nx,ny,nz)` | `(3,3,nx,ny,nz)` | Jacobian; `[i,j,…] = ∂v_i/∂x_j` |
| `divergence(v, grid)` | `(3,nx,ny,nz)` | `(nx,ny,nz)` | $\nabla\cdot\mathbf{v}$ |
| `curl(v, grid)` | `(3,nx,ny,nz)` | `(3,nx,ny,nz)` | $\nabla\times\mathbf{v}$ |
| `laplacian_scalar(s, grid)` | `(nx,ny,nz)` | `(nx,ny,nz)` | 3-point stencil |
| `laplacian_vector(v, grid)` | `(3,nx,ny,nz)` | `(3,nx,ny,nz)` | Componentwise |
| `normalize(v, eps=1e-30)` | `(3,nx,ny,nz)` | `(3,nx,ny,nz)` | Project to $S^2$ |

## Usage

```python
from hopfion.grid import Grid, grad_scalar, laplacian_vector, normalize
g = Grid(48, 48, 48, 0.3, 0.3, 0.3, "periodic")
m = ...  # (3, 48, 48, 48)
H_ex = 2.0 * laplacian_vector(m, g)   # exchange effective field
```

## Caveats

- `"periodic"` boundaries use `np.roll`; `"open"` clamps the index (Neumann). The `curl`/`laplacian`/`divergence` results near the boundary in open mode are not as accurate as for periodic — fine for visualization, but the Hopf-index FFT pipeline (`topology.py`) refuses non-periodic grids.
- Operators are second-order accurate; for spectral accuracy on smooth periodic fields, use the FFT-based derivatives in `topology._spectral_d_axis`.

## See also

- [topology.md](topology.md) — spectral derivatives + FFT curl-inversion
- [energy.md](energy.md) — uses the central-FD operators for DMI/anisotropy/Zeeman
- [PHYSICS.md §4](../PHYSICS.md#4-bond-energy-discretization) — why exchange uses a *different* discretization
