# `hopfion.physics.stt`

Zhang-Li spin-transfer torque extension to LLG. Recipe kind: `dynamics_stt`.

## Equation

$$\dot{\mathbf{m}} = -\gamma\,\mathbf{m}\times\mathbf{H}_{\rm eff} + \alpha\,\mathbf{m}\times\dot{\mathbf{m}} - (\mathbf{u}\cdot\nabla)\mathbf{m} + \beta\,\mathbf{m}\times(\mathbf{u}\cdot\nabla)\mathbf{m}.$$

## Public API

| Symbol | Purpose |
|---|---|
| `STTParams(u, beta)` | drive parameters |
| `adv_grad_m(m, grid, u)` | $(\mathbf{u}\cdot\nabla)\mathbf{m}$ via central FD |
| `stt_rhs(m, H, grid, gamma, alpha, stt)` | LLG RHS + STT terms |
| `stt_step_heun(m, grid, ep, gamma, alpha, dt, stt, H_extra=None)` | one Heun step of STT-LLG |

## Drift velocity (analytic)

For an isolated hopfion under uniform $\mathbf{u}$, the steady-state drift is $\mathbf{v} \approx \frac{\beta}{\alpha}\,\mathbf{u}$ (Thiele-equation derivation, neglecting Magnus rotation). $\beta = \alpha$ → $v = u$; $\beta < \alpha$ → drift slower; $\beta > \alpha$ → faster.

## Usage

```python
from hopfion.physics.stt import STTParams, stt_step_heun
stt = STTParams(u=(0.0, 0.0, 0.08), beta=0.05)
m = stt_step_heun(m, grid, ep, gamma=1.0, alpha=0.2, dt=0.002, stt=stt)
```

Recipe equivalent:
```yaml
run:
  - kind: dynamics_stt
    n_steps: 200
    dt: 0.002
    alpha: 0.2
    gamma: 1.0
    u: [0.0, 0.0, 0.08]
    beta: 0.05
```

## See also

- [FLUX_AND_PROPAGATION.md §4](../FLUX_AND_PROPAGATION.md#4-spin-transfer-torque-drive)
- [sop/DRIVE_HOPFION.md](../sop/DRIVE_HOPFION.md)
- [physics_integrators.md](physics_integrators.md) — `heun_step` that `stt_step_heun` mirrors
