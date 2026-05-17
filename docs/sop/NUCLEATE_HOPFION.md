# SOP-001: Nucleate a single Q_H = 1 hopfion

**Purpose**: produce a Q_H = 1 hopfion from a uniformly magnetized starting state via a femtosecond-laser-style stimulus, characterize the yield as a function of pulse fluence.

**Recipe**: `recipes/laser_nucleation_2t.yaml` (Phase B, physically grounded) or `recipes/laser_nucleation.yaml` (Phase A white-noise stand-in).

## Acceptance criteria (per run)

| Criterion | Pass condition | Severity |
|---|---|---|
| `norm_drift_below` | `< 1e-10` | fail |
| `hopf_index_within` | `\|Q_H - 1\| < 0.5` (loose: nucleation is stochastic) | warn |
| `runtime_within` | `< 60 s` | warn |

For a tight production spec, tighten the Q_H tolerance and elevate it to `fail_on:`.

## Standard procedure

1. **Parameter check**. Confirm the material parameters are inside the hopfion stability window: `A_ex = 1.0`, `D ∈ [1.2, 1.8]`, `K_u ∈ [0.5, 0.8]`. Defaults: `D = 1.5`, `K_u = 0.7`.
2. **Grid sizing**. Use `dx ≤ R / 4` where R is the target hopfion size. For R = 1.5, choose `dx = 0.3` and `nx = ny = nz = 48` (box L = 14.4 > 4R = 6 ✓).
3. **Initial state**. `kind: uniform`, `direction: [0, 0, 1]`.
4. **Pulse design**. Start with `Te_peak = 20.0`, `tau = 0.05`, `t0 = 0.1`. Then a `relax` step of ≥ 250 sub-steps at `dt = 0.002`.
5. **Smoke run**: `hopfion run recipes/laser_nucleation_2t.yaml`. Read `Q_final` from the report.
6. **Yield characterization**: `hopfion batch recipes/laser_nucleation_2t.yaml --sweep "run[0].Te_peak=15,20,25,30,40" --seeds 0:8`. Read `batch_report.txt` for yield-by-Te.

## Expected outcomes

| Te_peak | Yield (~Q_H ≈ 1 ± 0.5) | Notes |
|---|---|---|
| < 10 | 0 % | bath stays too cool to disturb FM state |
| 15 – 20 | ~50–80 % | nucleation begins |
| 25 – 30 | ~70–90 % | sweet spot |
| > 40 | <60 % | over-driven, get higher-Q textures or chaos |

## Diagnostic actions on FAIL

| Failure | Action |
|---|---|
| Q_final → 0 reliably | increase `Te_peak`; check that material parameters are in the stability window |
| Q_final overshoot (≥ 2) | reduce `Te_peak` or lengthen the relax step |
| `energy_monotonicity` trips in relax phase | halve `dt`; the thermal phase left a stiff state |
| `norm_drift_below` trips | check backend (JAX float32 ≈ 1e-7; tighten to 1e-6 if running under JAX, or set `jax_enable_x64`) |

## References

- [PHYSICS.md §6](../PHYSICS.md#6-laser-pulse-models) — two-temperature derivation.
- [PIPELINE.md](../PIPELINE.md) — recipe schema.
- [QUALITY_CONTROL.md](../QUALITY_CONTROL.md) — criterion definitions.
