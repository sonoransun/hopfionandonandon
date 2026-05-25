# SOP-003: Estimate hopfion thermal lifetime

**Purpose**: characterize the metastability of a relaxed hopfion against thermal fluctuations. Output: lifetime vs temperature curve.

**Recipes**: combine `recipes/single_hopfion.yaml` (to produce a relaxed seed state) with a custom thermal-burst recipe; or use the Phase-B `lowest_eigenmodes` Hessian pre-factor for Arrhenius estimates.

## Two complementary methods

### Method A — direct sLLG simulation

1. **Seed state**. Run `hopfion run recipes/single_hopfion.yaml`. Use its `runs/single_hopfion_q1/run.h5` as the input.
2. **Thermal-aged run**. Write a derived recipe with `initial.kind: file`, `file_path: runs/single_hopfion_q1/run.h5`, then a single `thermal_burst` step at the target `kT`. Set `n_steps` long enough to see decay statistics (≥ 5000 for low T).
3. **Lifetime measurement**. Either use `hopfion.stability.thermal_lifetime` programmatically, or post-process `summary.json`'s `histories.q_hopf.values` and find the first step where `|Q_H - 1| > 0.5`.
4. **Repeat for T sweep + multiple seeds** using the shipped `recipes/thermal_decay.yaml` (its `run[1]` is the sustained `thermal_burst`):
   ```bash
   hopfion batch recipes/thermal_decay.yaml \
     --sweep "run[1].kT=2,5,10,15,20" \
     --seeds 0:9
   ```
   Per-T mean lifetime should follow Arrhenius: `log τ = E_a/(k_B T) + const`.

### Method B — Hessian-based Arrhenius pre-factor

1. **Seed state** as in method A.
2. Load the relaxed `m`. Compute `lowest_eigenmodes(m, grid, ep, k=6)`.
3. **Pre-factor**:
   ```python
   from hopfion.physics.hessian import lowest_eigenmodes
   res = lowest_eigenmodes(m, grid, ep, k=6)
   prefactor = (np.prod(res.eigenvalues[res.eigenvalues > 0])) ** 0.5
   ```
4. **Activation barrier** $E_a$ — the saddle-to-minimum energy difference along the minimum-energy path from the hopfion to its decay product (e.g. the uniform state), via the shipped string method:
   ```python
   from hopfion.field import uniform
   from hopfion.physics.string_method import string_method, arrhenius_lifetime
   path = string_method(m, uniform(grid, (0, 0, 1)), grid, ep, n_images=11, n_iter=400)
   E_a = path.barrier                     # saddle - start; path.saddle_index locates the ridge
   ```
   Note the barrier is grid-resolution dependent: on coarse grids the lattice can unwind the topology cheaply, so refine `dx` for a quantitative $E_a$ (the method is validated against the analytic anisotropy double-well — see `tests/test_string_method.py`).
5. **Lifetime**:
   ```python
   tau = arrhenius_lifetime(E_a, kT=0.5, prefactor=prefactor, tau0=1.0)   # τ ≈ (τ0/prefactor)·exp(E_a/kT)
   ```

Method A gives a direct number; Method B (Hessian pre-factor + string-method barrier) is faster and now fully in-tree.

## Acceptance criteria

| Criterion | Pass condition |
|---|---|
| Decay timing is monotonic in T | higher T → shorter lifetime |
| `min_hessian_eigenvalue` at the seed | `≥ 0` (otherwise the seed isn't a real minimum and method A gives misleading numbers) |
| Mean lifetime at low T ≥ 100× mean lifetime at high T | yes, otherwise the temperature range is too narrow |

## Diagnostic actions

| Symptom | Cause |
|---|---|
| Lifetime decreases at *low* T | Q_H drift comes from numerical noise, not thermal; tighten `dt` or run longer to gather statistics |
| Negative Hessian eigenvalue at seed | the relaxation didn't reach a true minimum; relax longer; finer grid |
| Lifetime essentially infinite even at high T | sLLG noise variance under-estimated; check `kT` units (recipe uses normalized units, not Kelvin) |

## References

- [PHYSICS.md §5](../PHYSICS.md#5-llg-dynamics) — damped-LLG / sLLG.
- `src/hopfion/physics/hessian.py` — eigenmode analysis source.
- `src/hopfion/stability.py` — `thermal_lifetime` helper.
