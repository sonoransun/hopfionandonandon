# `hopfion.pipeline.report`

Markdown report generation from an `HDF5 + summary.json` pair.

## Public API

| Function | Purpose |
|---|---|
| `write_report(run_dir) -> Path` | render `report.md` + figures under `run_dir/figs/` |

## What the report contains

1. **Header**: recipe name, verdict badge, wall time, `Q_final`, backend.
2. **Description**: pulled from the recipe.
3. **QC verdict table**: criterion-by-criterion with status icons and the human-readable message from each `CriterionResult`.
4. **Metric summary**: pretty-printed JSON of every metric summary dict.
5. **Final-state figure**: xy-slice quiver.
6. **Time series**: energy, Q_H, norm-drift (the last only if drift > 1e-12).
7. **Recipe**: collapsed YAML for reproducibility.

## Usage

```python
from hopfion.pipeline.report import write_report
path = write_report("runs/single_hopfion_q1")
```

Or via CLI:
```bash
hopfion report runs/single_hopfion_q1
```

## Figure backend

matplotlib only (PNG, ~110 dpi). No PyVista dependency in the report path so headless CI can regenerate reports.

## See also

- [PIPELINE.md](../PIPELINE.md) — what triggers report generation.
- [pipeline_runner.md](pipeline_runner.md) — what writes the `summary.json` and `run.h5` that this consumes.
