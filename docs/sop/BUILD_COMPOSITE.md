# SOP-006: Build a composite topological state

**Purpose**: construct a multi-component initial state (Q$\pm$ pair, hopfion-skyrmion hybrid, lattice array) and verify its topological structure before propagating it.

**Recipes**: `recipes/flux_q_pair.yaml`, `recipes/hopfion_skyrmion_hybrid.yaml`, `recipes/moire_lattice.yaml`.

## Composite kinds at a glance

| Kind | Total Q_H | Topology | Use case |
|---|---|---|---|
| `q_pair` | 0 | Q$+$ at $-d/2$, Q$-$ at $+d/2$ | flux-conservation tests, annihilation studies |
| `hopfion_array` | N (lattice site count) | N hopfions on triangular / square / cubic lattice | stabilizer codes, density-driven dynamics |
| `hopfion_skyrmion_hybrid` | ~1 | Q=1 hopfion linked with skyrmion tube | meta-structure speculation, hybrid braiding |
| `skyrmion_tube` | 0 (3D) | per-slice 2D charge ±1 | comparison baseline |

## Standard procedure

1. **Pick the composite**. Set `initial.kind` accordingly. Recipe-level fields:
   - `q_pair`: `R`, `separation`, `axis_of_separation`
   - `hopfion_array`: `R`, `array_lattice`, `array_a`, `array_n_rings`
   - `hopfion_skyrmion_hybrid`: `R`, `skyrmion_radius`, `skyrmion_helicity`
2. **Verify Q via dry run**. Set `run: []` and `qc.fail_on: [hopf_index_within: {target: ..., tol: 0.1}]`. The runner builds the initial state, computes Q, prints the verdict — without running any dynamics.
3. **Centroid sanity**. With `io.extended_metrics: true`, the `per_site_q` metric reports the number of distinct topological clusters and their signed charges. Cross-check against expectations.
4. **Then propagate**. Add `relax` + `dynamics_stt` (or whatever drive) steps after the initial state is validated.

## Common construction pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Q$\pm$ pair too close | Hopf index is non-integer because the Hopf-fibration tail interferes | increase `separation` to ≥ 4·R |
| Lattice spacing too tight | sum of Q drops below the site count | `array_a >= 4·R` |
| Hybrid skyrmion radius too small | skyrmion fails to thread the hopfion ring properly | `skyrmion_radius` ~ 0.5·R |
| Box too small for hopfion + drift | pre-flight aborts with "image overlap" | grow the box |
| Centroid count differs from intent | clustering threshold mismatched to feature size | adjust `threshold_rel` in `DriftVelocity` (default 0.1) |

## References

- [FLUX_AND_PROPAGATION.md §3](../FLUX_AND_PROPAGATION.md#3-composite-state-constructors)
- [api/physics_composite.md](../api/physics_composite.md)
