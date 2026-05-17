# SOP-002: Characterize a moiré-stabilized hopfion lattice

**Purpose**: initialize an N-hopfion array on a moiré pinning landscape, relax under the moiré-modulated energy, verify all sites survive.

**Recipe**: `recipes/moire_lattice.yaml`.

## Acceptance criteria

| Criterion | Pass condition | Severity |
|---|---|---|
| `energy_monotonicity` | `< 1e-6` | fail |
| `hopf_index_within` | `\|Σ Q_H - N\| < 0.3` for N-site array | fail |
| `norm_drift_below` | `< 1e-10` | fail |
| `runtime_within` | `< 180 s` | warn |

## Standard procedure

1. **Site geometry**. Choose `array_lattice` ∈ {`triangular`, `square`} and the spacing `array_a`. For triangular with `n_rings: 1`, you get 7 sites.
2. **Inter-site spacing rule**. Set `array_a ≥ 4R`. Sub-rule `array_a` should match the moiré period `a_moire` so each hopfion sits on a moiré-K minimum.
3. **Material parameters**. Same hopfion stability window as SOP-001. Set `material.Ku: 0.0` and use a non-zero moiré modulation: `material.moire: {enabled: true, K0: 0.7, V0: 0.3, a_moire: 8.0}`.
4. **Run**: `hopfion run recipes/moire_lattice.yaml`.
5. **Verify** in `report.md`: `Q_final ≈ N` (e.g. 7.00 ± 0.05 for 7 sites).

## Per-site QC

Sum-Q gives a system-level pass/fail. For *per-site* QC, post-process:

```python
import h5py
from hopfion.topology import preimage_mask
# Identify site centers from recipe; for each site, mask a 4R cube and compute
# its skyrmion charge (z-slice integration). Each site should show Q_skyrm_z = ±1.
```

A scriptable wrapper for per-site Q is on the Phase-C wish-list; for now, eyeball `figs/final_xy.png`.

## Yield characterization

```bash
hopfion batch recipes/moire_lattice.yaml \
  --sweep "material.moire.V0=0.1,0.2,0.3,0.4;material.moire.a_moire=6,8,10" \
  --seeds 0:3
```

Interpretation:
- `V0` too small → pinning insufficient; hopfions diffuse and may merge.
- `a_moire` too small → sites overlap; merger / annihilation.
- `a_moire` too large → only the center site survives; outer sites drift to bg.

## Diagnostic actions on FAIL

| Failure | Action |
|---|---|
| `Q_final` significantly less than N | inter-site spacing too tight (try larger `array_a` or `a_moire`); modulation too weak (raise `V0`) |
| `Q_final ≈ 0` | confirm `material.moire.enabled: true`; check K0 is in stability window |
| `energy_monotonicity` trips | lower `dt` (moiré landscape is stiffer than uniform K) |

## References

- [PHYSICS.md §7](../PHYSICS.md#7-moiré-modulation) — moiré-potential equations.
- Notebook `notebooks/03_moire_lattice.ipynb` — interactive walk-through.
