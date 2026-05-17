"""End-to-end recipe run tests."""
import os
from pathlib import Path

from hopfion.pipeline.recipe import RecipeConfig
from hopfion.pipeline.runner import run


SMALL_RECIPE = {
    "name": "smoke_single_q1",
    "grid": {"nx": 24, "ny": 24, "nz": 24, "dx": 0.4, "bc": "periodic"},
    "material": {"A_ex": 1.0, "D": 1.5, "Ku": 0.7,
                 "easy_axis": [0, 0, 1], "H_ext": [0, 0, 0]},
    "initial": {"kind": "hopfion", "R": 1.5, "p": 1, "q": 1},
    "run": [{"kind": "relax", "n_steps": 30, "dt": 0.003}],
    "qc": {"fail_on": [
        {"energy_monotonicity": {"tol": 1e-6}},
        {"hopf_index_within": {"target": 1.0, "tol": 0.2}},
        {"norm_drift_below": {"tol": 1e-10}},
    ]},
    "io": {"out": "runs/smoke_runner/"},
    "seed": 0,
}


def test_run_single_hopfion_accept(tmp_path):
    spec = dict(SMALL_RECIPE)
    spec["io"] = {"out": str(tmp_path / "out")}
    rc = RecipeConfig.from_dict(spec)
    result = run(rc, write=True)
    assert result.qc.verdict == "ACCEPT", result.qc.to_dict()
    assert result.metrics["q_hopf"]["Q_final"] > 0.85
    assert (tmp_path / "out" / "run.h5").exists()
    assert (tmp_path / "out" / "summary.json").exists()


def test_preflight_failure_aborts_without_running(tmp_path):
    spec = dict(SMALL_RECIPE)
    # box L = 12 * 0.4 = 4.8 < 4 * R(=1.5) = 6 -- periodic-image overlap
    spec["grid"] = {"nx": 12, "ny": 12, "nz": 12, "dx": 0.4, "bc": "periodic"}
    spec["io"] = {"out": str(tmp_path / "out")}
    rc = RecipeConfig.from_dict(spec)
    result = run(rc, write=False)
    assert result.qc.verdict == "FAIL"
    assert any("4R" in f or "image" in f for f in result.qc.preflight_failures)


def test_metric_summaries_have_expected_keys(tmp_path):
    spec = dict(SMALL_RECIPE)
    spec["io"] = {"out": str(tmp_path / "out")}
    rc = RecipeConfig.from_dict(spec)
    result = run(rc, write=False)
    assert set(result.metrics.keys()) == {"energy", "norm_drift", "q_hopf", "runtime"}
