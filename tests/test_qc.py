"""QC criteria + verdict aggregation."""
from hopfion.pipeline.qc import (
    EnergyMonotonicityCriterion,
    HopfIndexWithinCriterion,
    NormDriftBelowCriterion,
    RuntimeWithinCriterion,
    evaluate,
)
from hopfion.pipeline.recipe import QCSpec


GOOD = {
    "energy": {"max_increase": -1.0, "E_initial": 1.0, "E_final": -1.0, "n_samples": 10},
    "q_hopf": {"Q_initial": 0.99, "Q_final": 1.0001, "max_drift": 0.01, "n_samples": 3},
    "norm_drift": {"max_drift": 1e-15, "final_drift": 1e-15, "n_samples": 10},
    "runtime": {"total_seconds": 1.0, "mean_step_ms": 2, "n_samples": 5},
}


def test_energy_monotonicity_pass():
    c = EnergyMonotonicityCriterion(tol=1e-6)
    r = c.evaluate(GOOD)
    assert r.passed


def test_energy_monotonicity_fail():
    bad = dict(GOOD); bad["energy"] = dict(GOOD["energy"], max_increase=0.5)
    c = EnergyMonotonicityCriterion(tol=1e-6)
    r = c.evaluate(bad)
    assert not r.passed


def test_hopf_index_within_pass():
    c = HopfIndexWithinCriterion(target=1.0, tol=0.05)
    assert c.evaluate(GOOD).passed


def test_hopf_index_within_fail():
    bad = dict(GOOD); bad["q_hopf"] = dict(GOOD["q_hopf"], Q_final=0.5)
    c = HopfIndexWithinCriterion(target=1.0, tol=0.05)
    assert not c.evaluate(bad).passed


def test_evaluate_accept_when_no_failures():
    spec = QCSpec(fail_on=[
        {"energy_monotonicity": {"tol": 1e-6}},
        {"hopf_index_within": {"target": 1.0, "tol": 0.05}},
        {"norm_drift_below": {"tol": 1e-10}},
    ])
    rep = evaluate(spec, GOOD)
    assert rep.verdict == "ACCEPT"
    assert all(r.passed for r in rep.results)


def test_evaluate_fails_when_any_fail_on_fails():
    bad = dict(GOOD); bad["energy"] = dict(GOOD["energy"], max_increase=1.0)
    spec = QCSpec(fail_on=[{"energy_monotonicity": {"tol": 1e-6}}])
    rep = evaluate(spec, bad)
    assert rep.verdict == "FAIL"


def test_warn_does_not_flip_verdict():
    bad_warn = dict(GOOD); bad_warn["runtime"] = dict(GOOD["runtime"], total_seconds=1000)
    spec = QCSpec(
        fail_on=[{"hopf_index_within": {"target": 1.0, "tol": 0.05}}],
        warn_on=[{"runtime_within": {"max_seconds": 60}}],
    )
    rep = evaluate(spec, bad_warn)
    assert rep.verdict == "ACCEPT"
    warn = next(r for r in rep.results if r.severity == "warn")
    assert not warn.passed


def test_preflight_failures_flip_verdict_to_fail():
    spec = QCSpec(fail_on=[])
    rep = evaluate(spec, GOOD, preflight_failures=["some pre-flight problem"])
    assert rep.verdict == "FAIL"
