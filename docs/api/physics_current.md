# `hopfion.physics.current`

Topological current $\mathbf{J}(r, t)$ and Lagrangian centroid drift tracking.

## Public API

| Function | Purpose |
|---|---|
| `hopf_charge_density(m, grid)` | scalar $\rho_Q(\mathbf{r}) = (1/16\pi^2) \mathbf{A}\cdot\mathbf{F}$ |
| `topological_current(m, m_next, grid, dt)` | transport current $\mathbf{J}$; built via Poisson inversion so $\nabla\cdot\mathbf{J} = -\partial_t\rho$ by construction |
| `divergence_J(J, grid)` | spectral $\nabla\cdot\mathbf{J}$ |
| `conservation_residual(m, m_next, grid, dt)` | $\partial_t\rho + \nabla\cdot\mathbf{J}$ pointwise — should be ~0 |
| `centroids(rho_Q, grid, threshold_rel=0.1)` | cluster $\|\rho_Q\|$ via `scipy.ndimage.label`, return charge-weighted centroids |
| `drift_velocity(history, times)` | central-difference per-cluster velocities |
| `Centroid` | dataclass: `position`, `charge`, `cluster_size` |

## Method

Continuity $\partial_t \rho_Q + \nabla\cdot\mathbf{J} = 0$ + Coulomb gauge $\mathbf{J} = -\nabla\phi$ gives $\nabla^2\phi = \partial_t\rho$, solved in Fourier space. The result is the *transport-only* current; the divergence-free part of the topological 4-current is discarded.

## Usage

```python
from hopfion.physics.current import topological_current, centroids, hopf_charge_density
J = topological_current(m_now, m_next, grid, dt=0.01)
rho = hopf_charge_density(m_now, grid)
cs = centroids(rho, grid)
print([(c.position, c.charge) for c in cs])
```

## Pipeline integration

`io.extended_metrics: true` activates `FluxAccumulation` + `DriftVelocity` + `PerSiteQ` metrics that consume this module.

## See also

- [FLUX_AND_PROPAGATION.md §§1–2](../FLUX_AND_PROPAGATION.md#1-eulerian-current-jr-t)
- [pipeline_metrics.md](pipeline_metrics.md) — `FluxAccumulation`, `DriftVelocity`, `PerSiteQ`.
- [PHYSICS.md §2](../PHYSICS.md#2-the-hopf-invariant) — underlying $\mathbf{A}, \mathbf{F}$ machinery.
