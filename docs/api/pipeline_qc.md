# `hopfion.pipeline.qc`

Acceptance criteria + verdict aggregation. Criteria read metric summary dicts and return pass/fail with messages.

## Public API

| Symbol | Purpose |
|---|---|
| `Criterion` | base class |
| `CriterionResult` | dataclass: `name, passed, severity, message, measured, threshold` |
| `QCReport` | dataclass: `verdict, results, preflight_warnings, preflight_failures` |
| `evaluate(qc_spec, summaries, preflight_failures=None, preflight_warnings=None) -> QCReport` | the top-level entry |
| `CRITERIA: Dict[str, Type[Criterion]]` | registry mapping YAML key → class |

## Built-in criteria

| YAML key | Reads | Parameters |
|---|---|---|
| `energy_monotonicity` | `energy.max_increase` | `tol` (default 1e-6) |
| `hopf_index_within` | `q_hopf.Q_final` | `target` (1.0), `tol` (0.05) |
| `hopf_index_drift_below` | `q_hopf.max_drift` | `tol` (0.1) |
| `norm_drift_below` | `norm_drift.max_drift` | `tol` (1e-10) |
| `runtime_within` | `runtime.total_seconds` | `max_seconds` (60) |
| `min_hessian_eigenvalue` | `hessian.min_eig` | `tol` (-1e-6) |

## Severity

In a recipe:
```yaml
qc:
  fail_on: [{energy_monotonicity: {tol: 1e-6}}]    # tripwires
  warn_on: [{runtime_within: {max_seconds: 60}}]    # soft excursions
```

`fail_on` failures flip the verdict to `FAIL`. `warn_on` failures are recorded but don't change the verdict.

## Adding a criterion

```python
class MyCriterion(Criterion):
    name = "my_check"
    def evaluate(self, s):
        ...
        return CriterionResult(name=self.name, passed=..., severity=self.severity, ...)
CRITERIA[MyCriterion.name] = MyCriterion
```

## See also

- [QUALITY_CONTROL.md](../QUALITY_CONTROL.md) — full reference and failure-mode catalog.
- [pipeline_metrics.md](pipeline_metrics.md) — sources of the summary fields.
