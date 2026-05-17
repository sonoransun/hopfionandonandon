# SOP-005: Drive a hopfion via spin-transfer torque

**Purpose**: design a recipe that drives a (single, paired, or arrayed) hopfion through a magnetic medium with current-induced motion; verify the drift velocity matches the prescribed $\mathbf{u}$.

**Recipe templates**: `recipes/flux_q_pair.yaml`, `recipes/stt_lattice.yaml`.

## Acceptance criteria

| Criterion | Pass condition | Severity |
|---|---|---|
| `norm_drift_below` | `< 1e-10` | fail |
| `flux_divergence_below` | `< 0.05` | warn |
| `drift_bound` | `< 1.0` (per-cluster speed) | warn |

## Standard procedure

1. **Build the seed**. Choose an initial composite (`hopfion`, `q_pair`, `hopfion_array`, `hybrid`). For pure drift tests, start with a single hopfion or a Q$\pm$ pair.
2. **Pre-relax**. Add a short `relax` step (50-100 steps) before the STT drive so the seed lands on a local minimum. Skip this if you specifically want to study transient dynamics.
3. **STT step**. Recipe step:
   ```yaml
   - kind: dynamics_stt
     n_steps: 200
     dt: 0.002
     alpha: 0.2
     gamma: 1.0
     u: [0.0, 0.0, 0.08]
     beta: 0.05
   ```
   $\mathbf{u}$ in normalized units; typical magnitude 0.01–0.1.
4. **Enable extended metrics**: `io.extended_metrics: true` so flux + drift are collected.
5. **Run + read**. `hopfion run recipes/your_recipe.yaml`. Inspect `report.md`:
   - `drift.max_drift_speed` should be ~$|\mathbf{u}|$ if $\beta = \alpha$, otherwise scaled.
   - `flux.max_residual_rms` should be small (continuity holds).

## Diagnostic actions

| Symptom | Likely cause | Fix |
|---|---|---|
| Hopfion drifts faster than $\mathbf{u}$ | $\beta \ne \alpha$; the non-adiabatic torque overcompensates | match $\beta$ to $\alpha$ for $v = u$ |
| Hopfion stalls | moiré pinning > STT force | increase $\mathbf{u}$ or reduce $V_0$ in moiré |
| Q drifts during STT | $Q$-changing perturbation; consider adding an error-correction controller | see `recipes/correction_*` and SOP-006 |
| Centroid count drops mid-run | drift wrapped through periodic BC and clusters merged | grow the box or reduce run duration |

## References

- [FLUX_AND_PROPAGATION.md §4](../FLUX_AND_PROPAGATION.md#4-spin-transfer-torque-drive)
- [api/physics_stt.md](../api/physics_stt.md)
