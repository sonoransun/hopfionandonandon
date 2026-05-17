# `hopfion.lattice`

2D and 3D hopfion-array initialization. Combines analytic ansätze at multiple sites via Gaussian blending + renormalization.

## Site generators

| Function | Returns | Notes |
|---|---|---|
| `triangular_sites_2d(a, n_rings=1, z=0)` | list of `(x,y,z)` | Center + first ring of 6 = 7 sites for `n_rings=1` |
| `square_sites_2d(a, nx, ny, z=0)` | list | `nx*ny` square array |
| `cubic_sites_3d(a, nx, ny, nz)` | list | `nx*ny*nz` cubic array |

## Array constructor

```python
array_hopfion(grid, sites, R, background=(0,0,1), axis="z") -> (3, nx, ny, nz)
```

For each site $i$, builds $\mathbf{m}_i$ from `hopfion.field.hopfion`, then blends:

$$\mathbf{m}(\mathbf{r}) = \mathrm{normalize}\!\Bigl(\mathbf{m}_{\rm bg} + \sum_i (\mathbf{m}_i(\mathbf{r}) - \mathbf{m}_{\rm bg})\, w_i(\mathbf{r})\Bigr)$$

with $w_i = \exp(-|\mathbf{r}-\mathbf{r}_i|^2/(2\sigma^2)), \sigma = 2R$. This is purely an *initial guess* — you then relax it under the moiré-modulated energy to land on the local minimum.

![7-hopfion moiré lattice after relax](../assets/moire_lattice.png)

## Usage

```python
from hopfion.lattice import triangular_sites_2d, array_hopfion
sites = triangular_sites_2d(a=8.0, n_rings=1)         # 7 sites
m = array_hopfion(grid, sites, R=1.5)
```

## Caveats

- **Initial $Q_H$ ≈ N if sites are well-separated.** If $a_{\rm moire} < \sim 4R$, neighboring hopfions interfere and the FFT-computed Hopf index drops below the site count even before relaxation.
- For asymmetric site configurations (e.g. defect-decorated lattice), pass any list of `(x, y, z)` tuples — the generator helpers are convenience only.

## See also

- [moire.md](moire.md) — the $K_u(\mathbf{r})$ pattern that pins these sites
- [field.md](field.md) — single-hopfion ansatz
- Notebook `notebooks/03_moire_lattice.ipynb`
