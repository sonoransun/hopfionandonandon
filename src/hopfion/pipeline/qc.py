"""Quality-control framework: acceptance criteria + verdicts.

Each criterion reads the metric summary dict produced by ``MetricCollector``
and returns a ``CriterionResult``. The runner aggregates these into a
``QCReport`` with verdict ``ACCEPT`` or ``FAIL`` based on the recipe's
``qc.fail_on`` and ``qc.warn_on`` lists.

Criterion names (the YAML key) map to classes via ``CRITERIA``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


# ---------------------------------------------------------------------------
# Criterion result + base
# ---------------------------------------------------------------------------


@dataclass
class CriterionResult:
    name: str
    passed: bool
    severity: str             # 'fail' or 'warn'
    message: str
    measured: Optional[float] = None
    threshold: Optional[float] = None


class Criterion:
    """Override ``evaluate`` to read metrics and return a CriterionResult."""
    name: str = ""

    def __init__(self, severity: str = "fail", **params):
        self.severity = severity
        self.params = params

    def evaluate(self, summaries: Dict[str, Any]) -> CriterionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete criteria
# ---------------------------------------------------------------------------


class EnergyMonotonicityCriterion(Criterion):
    name = "energy_monotonicity"

    def evaluate(self, s):
        tol = float(self.params.get("tol", 1e-6))
        e = s.get("energy", {})
        max_inc = float(e.get("max_increase", 0.0))
        passed = max_inc < tol
        return CriterionResult(
            name=self.name, passed=passed, severity=self.severity,
            measured=max_inc, threshold=tol,
            message=f"max energy increase per step = {max_inc:.3e} (tol {tol:.0e})",
        )


class HopfIndexWithinCriterion(Criterion):
    name = "hopf_index_within"

    def evaluate(self, s):
        target = float(self.params.get("target", 1.0))
        tol = float(self.params.get("tol", 0.05))
        q = s.get("q_hopf", {})
        q_final = float(q.get("Q_final", float("nan")))
        passed = abs(q_final - target) < tol
        return CriterionResult(
            name=self.name, passed=passed, severity=self.severity,
            measured=q_final, threshold=tol,
            message=f"Q_H final = {q_final:+.4f} vs target {target} (tol {tol})",
        )


class HopfIndexDriftBelowCriterion(Criterion):
    name = "hopf_index_drift_below"

    def evaluate(self, s):
        tol = float(self.params.get("tol", 0.1))
        q = s.get("q_hopf", {})
        max_drift = float(q.get("max_drift", 0.0))
        passed = max_drift < tol
        return CriterionResult(
            name=self.name, passed=passed, severity=self.severity,
            measured=max_drift, threshold=tol,
            message=f"max Q_H drift = {max_drift:.4f} (tol {tol})",
        )


class NormDriftBelowCriterion(Criterion):
    name = "norm_drift_below"

    def evaluate(self, s):
        tol = float(self.params.get("tol", 1e-10))
        n = s.get("norm_drift", {})
        max_drift = float(n.get("max_drift", 0.0))
        passed = max_drift < tol
        return CriterionResult(
            name=self.name, passed=passed, severity=self.severity,
            measured=max_drift, threshold=tol,
            message=f"max |m|-1 drift = {max_drift:.2e} (tol {tol:.0e})",
        )


class RuntimeWithinCriterion(Criterion):
    name = "runtime_within"

    def evaluate(self, s):
        max_s = float(self.params.get("max_seconds", 60.0))
        rt = s.get("runtime", {})
        total = float(rt.get("total_seconds", 0.0))
        passed = total < max_s
        return CriterionResult(
            name=self.name, passed=passed, severity=self.severity,
            measured=total, threshold=max_s,
            message=f"total runtime = {total:.1f}s (max {max_s:.1f}s)",
        )


# Phase B criteria — registered now, evaluators land with the modules.

class MinHessianEigenvalueCriterion(Criterion):
    name = "min_hessian_eigenvalue"

    def evaluate(self, s):
        tol = float(self.params.get("tol", -1e-6))
        h = s.get("hessian", {})
        if "min_eig" not in h:
            return CriterionResult(
                name=self.name, passed=True, severity=self.severity,
                message="hessian metric not collected (skipped)")
        min_eig = float(h["min_eig"])
        passed = min_eig >= tol
        return CriterionResult(
            name=self.name, passed=passed, severity=self.severity,
            measured=min_eig, threshold=tol,
            message=f"min Hessian eigenvalue = {min_eig:+.3e} (tol {tol:+.0e})",
        )


class FluxDivergenceBelowCriterion(Criterion):
    name = "flux_divergence_below"

    def evaluate(self, s):
        tol = float(self.params.get("tol", 0.05))
        f = s.get("flux", {})
        if not f or "max_residual_rms" not in f:
            return CriterionResult(name=self.name, passed=True, severity=self.severity,
                                   message="flux metric not collected (skipped)")
        measured = float(f["max_residual_rms"])
        passed = measured < tol
        return CriterionResult(
            name=self.name, passed=passed, severity=self.severity,
            measured=measured, threshold=tol,
            message=f"max continuity residual RMS = {measured:.3e} (tol {tol:.0e})",
        )


class DriftBoundCriterion(Criterion):
    name = "drift_bound"

    def evaluate(self, s):
        max_v = float(self.params.get("max_v", 1.0))
        d = s.get("drift", {})
        if not d or "max_drift_speed" not in d:
            return CriterionResult(name=self.name, passed=True, severity=self.severity,
                                   message="drift metric not collected (skipped)")
        measured = float(d["max_drift_speed"])
        passed = measured < max_v
        return CriterionResult(
            name=self.name, passed=passed, severity=self.severity,
            measured=measured, threshold=max_v,
            message=f"max centroid drift speed = {measured:.3f} (bound {max_v:.3f})",
        )


class SyndromeHealthCriterion(Criterion):
    """Stabilizer-style code: per-site Q's should match an expected pattern."""
    name = "syndrome_health"

    def evaluate(self, s):
        min_fidelity = float(self.params.get("min_fidelity", 0.9))
        target_count = self.params.get("expected_sites", None)
        ps = s.get("per_site_q", {})
        if not ps or "Q_per_site_final" not in ps:
            return CriterionResult(name=self.name, passed=True, severity=self.severity,
                                   message="per_site_q metric not collected (skipped)")
        final = ps["Q_per_site_final"]
        initial = ps.get("Q_per_site_initial", [])
        if target_count is not None and len(final) != int(target_count):
            return CriterionResult(
                name=self.name, passed=False, severity=self.severity,
                measured=float(len(final)), threshold=float(target_count),
                message=f"site count drift: {len(initial)} -> {len(final)} (target {target_count})",
            )
        # fidelity = fraction of initial sites whose |Q| stayed within 0.5
        if not initial:
            return CriterionResult(name=self.name, passed=True, severity=self.severity,
                                   message="no initial sites to compare")
        n_kept = sum(
            1 for q0, q1 in zip(initial, final) if abs(abs(q0) - abs(q1)) < 0.5
        )
        fidelity = n_kept / len(initial)
        passed = fidelity >= min_fidelity
        return CriterionResult(
            name=self.name, passed=passed, severity=self.severity,
            measured=float(fidelity), threshold=float(min_fidelity),
            message=f"syndrome fidelity = {fidelity:.2%} (min {min_fidelity:.0%})",
        )


# Registry: YAML key -> class

CRITERIA: Dict[str, Type[Criterion]] = {
    EnergyMonotonicityCriterion.name: EnergyMonotonicityCriterion,
    HopfIndexWithinCriterion.name: HopfIndexWithinCriterion,
    HopfIndexDriftBelowCriterion.name: HopfIndexDriftBelowCriterion,
    NormDriftBelowCriterion.name: NormDriftBelowCriterion,
    RuntimeWithinCriterion.name: RuntimeWithinCriterion,
    MinHessianEigenvalueCriterion.name: MinHessianEigenvalueCriterion,
    FluxDivergenceBelowCriterion.name: FluxDivergenceBelowCriterion,
    DriftBoundCriterion.name: DriftBoundCriterion,
    SyndromeHealthCriterion.name: SyndromeHealthCriterion,
}


# ---------------------------------------------------------------------------
# Report aggregation
# ---------------------------------------------------------------------------


@dataclass
class QCReport:
    verdict: str                   # 'ACCEPT' or 'FAIL'
    results: List[CriterionResult] = field(default_factory=list)
    preflight_warnings: List[str] = field(default_factory=list)
    preflight_failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == "ACCEPT"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "results": [
                {
                    "name": r.name, "passed": r.passed, "severity": r.severity,
                    "message": r.message,
                    "measured": r.measured, "threshold": r.threshold,
                }
                for r in self.results
            ],
            "preflight_warnings": list(self.preflight_warnings),
            "preflight_failures": list(self.preflight_failures),
        }


def _instantiate(entry, severity: str) -> Criterion:
    """Build a Criterion from a YAML entry, which is either:
        - a string criterion name
        - a dict ``{name: {param: value, ...}}``
        - a dict ``{"name": "X", **params}``
    """
    if isinstance(entry, str):
        cls = CRITERIA.get(entry)
        if cls is None:
            raise ValueError(f"Unknown criterion: {entry!r}")
        return cls(severity=severity)
    if isinstance(entry, dict):
        if "name" in entry:
            name = entry["name"]
            params = {k: v for k, v in entry.items() if k != "name"}
        else:
            if len(entry) != 1:
                raise ValueError(f"Ambiguous criterion entry: {entry!r}")
            name, params = next(iter(entry.items()))
            params = dict(params) if isinstance(params, dict) else {}
        cls = CRITERIA.get(name)
        if cls is None:
            raise ValueError(f"Unknown criterion: {name!r}")
        return cls(severity=severity, **params)
    raise ValueError(f"Unrecognised criterion entry: {entry!r}")


def evaluate(qc_spec, summaries: Dict[str, Any],
             preflight_failures: Optional[List[str]] = None,
             preflight_warnings: Optional[List[str]] = None) -> QCReport:
    """Apply every criterion in ``qc_spec``, then pick a verdict.

    Verdict is FAIL if (a) any pre-flight failure was recorded, or (b) any
    ``fail_on`` criterion did not pass. ``warn_on`` criteria never flip the
    verdict but are recorded in the report.
    """
    preflight_failures = list(preflight_failures or [])
    preflight_warnings = list(preflight_warnings or [])
    results: List[CriterionResult] = []

    for entry in (qc_spec.fail_on or []):
        crit = _instantiate(entry, severity="fail")
        results.append(crit.evaluate(summaries))
    for entry in (qc_spec.warn_on or []):
        crit = _instantiate(entry, severity="warn")
        results.append(crit.evaluate(summaries))

    fail = any(r.severity == "fail" and not r.passed for r in results)
    if preflight_failures:
        fail = True
    verdict = "FAIL" if fail else "ACCEPT"
    return QCReport(
        verdict=verdict,
        results=results,
        preflight_warnings=preflight_warnings,
        preflight_failures=preflight_failures,
    )


__all__ = [
    "CriterionResult",
    "Criterion",
    "EnergyMonotonicityCriterion",
    "HopfIndexWithinCriterion",
    "HopfIndexDriftBelowCriterion",
    "NormDriftBelowCriterion",
    "RuntimeWithinCriterion",
    "MinHessianEigenvalueCriterion",
    "CRITERIA",
    "QCReport",
    "evaluate",
]
