# `hopfion.pipeline.batch`

Parameter sweeps and seed ensembles with yield statistics. Wraps `runner.run` over a cross-product of overrides.

## Public API

| Symbol | Purpose |
|---|---|
| `run_batch(rc, sweep, seeds, out_root, max_workers=1) -> BatchReport` | execute the sweep |
| `BatchReport` | runs, yield, failure modes, Cp index |
| `BatchReport.yield_overall` | fraction of runs that ACCEPTed |
| `BatchReport.yield_by(key)` | yield grouped by a dotted parameter key |
| `BatchReport.cp_index(key, target, half_width)` | Cp-style spec-vs-process-variation indicator |
| `BatchReport.failure_modes()` | Counter of failed-criterion names |
| `BatchReport.summary_text()` | human-readable summary block |

## Dotted-key syntax

Overrides target any field in the `RecipeConfig` tree via a dotted path:

- `material.D` — scalar
- `material.moire.V0` — nested dataclass
- `run[0].dt` — indexed list element

## CLI

```bash
hopfion batch RECIPE \
  --sweep "material.D=0.6,1.0,1.5;run[0].dt=0.001,0.002" \
  --seeds 0:5 \
  --workers 4 \
  --out batches/
```

The `--sweep` argument is a `;`-separated list of `KEY=v1,v2,...` clauses. The cross-product of all clauses × seeds is executed.

## Output

```
batches/<recipe-name>/
├── batch_report.json       # full ledger of runs and failures
├── batch_report.txt        # one-page summary
└── seed-N_KEY-V/           # per-run subdirectory (with run.h5, summary.json)
```

## Process-capability example

```python
from hopfion.pipeline.batch import run_batch
report = run_batch(rc, sweep={"material.D": [0.6, 1.0, 1.5]}, seeds=range(5))
print(report.yield_by("material.D"))
# {0.6: 0.0, 1.0: 1.0, 1.5: 1.0}
print(report.cp_index("material.D", target=1.0, half_width=0.05))
```

## See also

- [PIPELINE.md](../PIPELINE.md) — recipe schema.
- [SOP-001](../sop/NUCLEATE_HOPFION.md) — uses batch for yield characterization.
- [SOP-004](../sop/VALIDATE_NEW_MATERIAL.md) — validation battery.
