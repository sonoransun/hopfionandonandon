# `hopfion.pipeline.metrics`

Pluggable in-line metric callbacks. Each `Metric` collects a time-series during the LLG inner loop and produces a summary that QC criteria read.

## Public API

| Symbol | Purpose |
|---|---|
| `Metric` | Base class with `collect(ctx)` and `summary()` |
| `MetricContext` | dataclass: `m`, `grid`, `ep`, `step`, `t`, `phase` |
| `MetricCollector` | wraps a list of Metrics into one callback |
| `EnergyMonotonicity` | per-step energy time series |
| `NormDrift` | per-step `max\|m\|-1` |
| `HopfIndexDrift` | sub-sampled `Q_H` (default cadence 25) |
| `SkyrmionChargeDrift` | sub-sampled 2D skyrmion number `N_sk` (extended metrics) |
| `PerSiteQVoronoi` | per-lattice-site Hopf charge by Voronoi zone (auto-added for `hopfion_array` + extended metrics) |
| `BlochPointCount` | number of Bloch points (emergent monopoles) over time — flags topological transitions |
| `Runtime` | wall time per step |
| `default_metrics(hopf_cadence=25)` | factory returning the standard four |
| `extended_metrics(...)` | default four + `SkyrmionChargeDrift` + flux/drift/per-site |

## Cadence

`HopfIndexDrift` defaults to every-25-steps because computing `Q_H` is an FFT. Tune via `default_metrics(hopf_cadence=5)` if you need finer resolution; expect a ~3× slowdown.

## Usage

```python
from hopfion.pipeline.metrics import MetricCollector, default_metrics
from hopfion.llg import relax
collector = MetricCollector(metrics=default_metrics(), grid=g, ep=ep)
m = relax(m, g, ep, n_steps=200, dt=0.002, step_callback=collector.relax_callback)
print(collector.summarize())
```

## Authoring a new metric

```python
from hopfion.pipeline.metrics import Metric

class MyMetric(Metric):
    name = "my_metric"
    cadence = 1

    def __init__(self):
        self.history = []
        self.steps = []

    def collect(self, ctx):
        self.history.append(some_computation(ctx.m, ctx.grid))
        self.steps.append(ctx.step)

    def summary(self):
        return {"summary_field": float(self.history[-1]) if self.history else 0.0}
```

Append your instance to `MetricCollector.metrics` and it will be collected.

## See also

- [pipeline_qc.md](pipeline_qc.md) — criteria that read these summaries.
- [QUALITY_CONTROL.md](../QUALITY_CONTROL.md) — failure-mode catalog.
