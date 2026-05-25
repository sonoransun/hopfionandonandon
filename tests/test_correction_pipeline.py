"""End-to-end error-correction wiring: the shipped recipes run, active feedback
holds Q better than the baseline, and the gap controller records its pre-flight."""
from pathlib import Path

import pytest

from hopfion.pipeline.recipe import RecipeConfig
from hopfion.pipeline.runner import run

REPO = Path(__file__).resolve().parents[1]
_NAMES = [
    "correction_none",
    "correction_active",
    "correction_topological_gap",
    "correction_stabilizer",
    "thermal_decay",
]


@pytest.fixture(scope="module")
def corr_results():
    return {n: run(RecipeConfig.from_yaml(REPO / "recipes" / f"{n}.yaml"), write=False)
            for n in _NAMES}


def test_all_correction_recipes_accept(corr_results):
    for n, res in corr_results.items():
        assert res.qc.verdict == "ACCEPT", f"{n}: {res.qc.verdict}"


def test_active_correction_holds_q_better_than_none(corr_results):
    qn = corr_results["correction_none"].metrics["q_hopf"]["Q_final"]
    qa = corr_results["correction_active"].metrics["q_hopf"]["Q_final"]
    assert corr_results["correction_active"].metrics["correction"]["n_corrections"] > 0
    assert abs(qa - 1.0) < abs(qn - 1.0)


def test_topological_gap_records_preflight(corr_results):
    corr = corr_results["correction_topological_gap"].metrics["correction"]
    assert corr["kind"] == "topological_gap"
    assert "preflight" in corr and "min_eig" in corr["preflight"]


def test_no_correction_block_means_no_controller():
    # single_hopfion has no `correction:` block → no correction in metrics
    res = run(RecipeConfig.from_yaml(REPO / "recipes" / "single_hopfion.yaml"), write=False)
    assert "correction" not in res.metrics
