# `hopfion.physics.correction`

Three error-correction controllers for hopfion composite states.

## Public API

| Symbol | Purpose |
|---|---|
| `ActiveFeedbackController(target_Q, threshold, gain, cadence, correction_duration)` | closed-loop: detect Q excursion, apply Zeeman correction |
| `TopologicalGapController(min_gap)` | passive: pre-flight Hessian min-eigenvalue check |
| `StabilizerController(expected_sites, flip_threshold, cadence)` | code-style: per-site Q syndrome on lattice composites |
| `CorrectionResult` | dataclass: `kind`, `n_corrections`, `correction_events`, `final_fidelity` |
| `make_controller(kind, grid, ep, **params)` | factory used by the recipe runner |

## Each controller exposes

```python
controller.step_callback(m, k, t)    # invoked from LLG inner loop
controller.H_extra_factory()         # returns H_extra(m) callable (may be no-op)
controller.result                    # CorrectionResult accumulator
```

## Recipe integration

```yaml
correction:
  kind: active                       # | topological_gap | stabilizer | none
  target_Q: 1.0
  threshold: 0.1
  gain: 1.0
  cadence: 25
  correction_duration: 5
```

The runner instantiates the named controller and threads `step_callback` into the integrator loop, alongside the metric `step_callback`. `H_extra_factory` provides the corrective field (zero when no correction is active).

## When to use each

See [sop/PROTECT_COMPOSITE.md](../sop/PROTECT_COMPOSITE.md) for the decision tree. Quick summary:

- single hopfion / pair: active feedback
- pre-relaxed state with known gap: topological-gap + Crouch-Grossman
- lattice with ≥ 4 sites: stabilizer

## Limitations (Phase D wishlist)

- `StabilizerController` logs the syndrome event but doesn't yet trigger an actual localized two-temperature burst (the field-modification path is scaffolded; the burst-injection through the integrator isn't wired).
- `ActiveFeedbackController` uses a heuristic corrective vector ($\pm\hat z$). Richer feedback would target the soft mode from the Hessian.

## See also

- [ERROR_CORRECTION.md](../ERROR_CORRECTION.md) — strategy comparison + failure modes.
- [physics_hessian.md](physics_hessian.md) — gap engineering depends on Hessian.
- [physics_two_temp.md](physics_two_temp.md) — re-nucleation engine the stabilizer will eventually drive.
