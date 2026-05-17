# `hopfion.physics.composite`

Composite-state constructors: Q$\pm$ pair, skyrmion tube, hopfion-skyrmion hybrid.

## Public API

| Symbol | Purpose |
|---|---|
| `q_minus_one_hopfion(grid, R, center, axis)` | a Q$_H = -1$ hopfion (parity-reflection of a Q$=+1$ ansatz) |
| `q_pair(grid, R, separation, axis_of_separation, background)` | Q$\pm$ pair separated along the named axis |
| `skyrmion_tube(grid, radius, helicity, center_xy)` | 2D skyrmion extruded along z |
| `hopfion_skyrmion_hybrid(grid, R, skyrmion_radius, skyrmion_helicity, center)` | hopfion linked with a skyrmion tube |

## Construction notes

- **Q$\pm$ pair**: parity flip (y → −y reflection + m_y → −m_y) inverts the Hopf invariant. This avoids the De Moivre branch-cut problems of $v \to v^{-1}$.
- **Skyrmion tube**: profile $\theta(\rho) = \pi \exp(-\rho^2/2R^2)$ (cos-like). Tunable helicity (0 = Néel, π/2 = Bloch).
- **Hybrid**: Gaussian-blend $(1-w)\,\mathbf{m}_{\rm hopf} + w\,\mathbf{m}_{\rm sky}$ with $w = \exp(-\rho^2/2R_{\rm sky}^2)$, then renormalize.

## Usage

```python
from hopfion.physics.composite import q_pair, hopfion_skyrmion_hybrid
m_pair = q_pair(grid, R=1.0, separation=5.0, axis_of_separation="x")
m_hyb = hopfion_skyrmion_hybrid(grid, R=1.5, skyrmion_radius=0.8)
```

## Recipe integration

```yaml
initial:
  kind: q_pair                   # | hopfion_skyrmion_hybrid | skyrmion_tube
  R: 1.0
  separation: 5.0
  axis_of_separation: x
```

## See also

- [FLUX_AND_PROPAGATION.md §3](../FLUX_AND_PROPAGATION.md#3-composite-state-constructors)
- [sop/BUILD_COMPOSITE.md](../sop/BUILD_COMPOSITE.md)
- [field.md](field.md) — the single-hopfion ansatz underneath
