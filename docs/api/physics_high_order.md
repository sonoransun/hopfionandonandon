# `hopfion.physics.high_order`

Geometric integrators on the unit sphere. Currently: Crouch-Grossman RK4.

## Why

Classical RK4 produces an unconstrained update + projection. The projection introduces an $O(dt^4)$ drift in $|\mathbf{m}|$ that, accumulated over $10^4$ steps, also drifts $Q_H$. Crouch-Grossman uses Rodrigues rotations — an isometry that preserves $|\mathbf{m}|=1$ exactly.

## Public API

| Function | Purpose |
|---|---|
| `crouch_grossman_rk4_step(m, grid, ep, gamma, alpha, dt, H_extra=None)` | one CG-RK4 step on $S^2$ |
| `_rodrigues_rotate(m, omega, dt)` | apply rotation by axis $\hat\omega$, angle $\|\omega\|\,dt$ |
| `_angular_velocity(m, H, gamma, alpha)` | convert LLG RHS to instantaneous rotation rate |

## Method

```mermaid
flowchart LR
  M[m_n] --> H1[H_eff at m_n]
  H1 --> O1[ω₁]
  M --> M2[m₂ = R(ω₁, dt/2) m_n]
  M2 --> H2[H_eff at m₂]
  H2 --> O2[ω₂]
  M --> M3[m₃ = R(ω₂, dt/2) m_n]
  M3 --> H3[H_eff at m₃]
  H3 --> O3[ω₃]
  M --> M4[m₄ = R(ω₃, dt) m_n]
  M4 --> H4[H_eff at m₄]
  H4 --> O4[ω₄]
  O1 --> AVG["Ω = (ω₁+2ω₂+2ω₃+ω₄)/6"]
  O2 --> AVG
  O3 --> AVG
  O4 --> AVG
  AVG --> Mn[m_{n+1} = R(Ω, dt) m_n]
```

The result is 4th order in the angular velocity, exact in the manifold constraint.

## Usage

Direct:
```python
from hopfion.physics.high_order import crouch_grossman_rk4_step
m = crouch_grossman_rk4_step(m, grid, ep, gamma=1.0, alpha=0.05, dt=0.005)
```

Recipe:
```yaml
run:
  - kind: dynamics
    integrator: crouch_grossman_rk4
```

## Cost

~3-4× a single Heun step (4 effective-field calls + the Rodrigues). Justified when long-time conservation matters or when used inside `correction.kind: topological_gap`.

## See also

- [FLUX_AND_PROPAGATION.md §5](../FLUX_AND_PROPAGATION.md#5-high-order-geometric-integration)
- [physics_integrators.md](physics_integrators.md) — Heun and RK4 reference
- [ERROR_CORRECTION.md](../ERROR_CORRECTION.md) — pairing with topological-gap strategy
