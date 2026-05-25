# `hopfion.physics.sot`

Spin-orbit torque (SOT) — a *local* current-induced torque (no spatial gradient,
unlike Zhang-Li STT in [stt.md](physics_stt.md)), for a fixed spin-polarization
direction $\hat{\mathbf p}$.

$$\dot{\mathbf m} = (\text{LLG}) - \tau_{\rm DL}\,\mathbf m\times(\mathbf m\times\hat p) - \tau_{\rm FL}\,\mathbf m\times\hat p.$$

The **damping-like** term drives $\mathbf m \to \hat p$ (SOT-MRAM switching); the
**field-like** term acts as an effective field along $\hat p$.

## Public API

| Symbol | Purpose |
|---|---|
| `SOTParams(p, dl, fl)` | polarization direction `p`, damping-like `dl`, field-like `fl` |
| `sot_rhs(m, H, grid, gamma, alpha, sot)` | LLG RHS + SOT terms |
| `sot_step_heun(m, grid, ep, gamma, alpha, dt, sot, H_extra=None)` | one Heun step (mirrors `stt.stt_step_heun`) |

## Pipeline

Run-step `kind: dynamics_sot` with fields `sot_p`, `sot_dl`, `sot_fl`. See
`recipes/sot_drive.yaml`. (The oscillatory `ac_drive` run-step — `ac_H0`, `ac_omega`
— is the microwave-excitation companion.)

## See also
- [physics_stt.md](physics_stt.md) — the gradient-coupled (Zhang-Li) torque.
- [physics_fmr.md](physics_fmr.md) — resonance modes excited by `ac_drive`.
