"""Golden regression: catch any unintended drift in committed recipes/regression/."""
import json
from pathlib import Path

import pytest

from hopfion.pipeline.recipe import RecipeConfig
from hopfion.pipeline.runner import run


REPO = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((REPO / "recipes" / "regression" / "golden.json").read_text())
TOL = GOLDEN["_tolerances"]


@pytest.mark.parametrize("key", [k for k in GOLDEN if not k.startswith("_")])
def test_regression_recipe_matches_golden(key, tmp_path):
    record = GOLDEN[key]
    rc = RecipeConfig.from_yaml(REPO / record["recipe"])
    rc.io.out = str(tmp_path)
    res = run(rc, write=False)
    assert res.qc.verdict == record["verdict"], (
        f"verdict drift: {res.qc.verdict} vs golden {record['verdict']}"
    )
    Q = float(res.metrics["q_hopf"]["Q_final"])
    E = float(res.metrics["energy"]["E_final"])
    nd = float(res.metrics["norm_drift"]["max_drift"])
    assert abs(Q - record["Q_final"]) < TOL["Q_final"], (
        f"Q_final drift: {Q} vs golden {record['Q_final']}"
    )
    assert abs(E - record["E_final"]) / max(abs(record["E_final"]), 1.0) < TOL["E_final_rel"], (
        f"E_final drift: {E} vs golden {record['E_final']}"
    )
    assert abs(nd - record["max_norm_drift"]) < TOL["max_norm_drift_abs"], (
        f"norm-drift drift: {nd} vs golden {record['max_norm_drift']}"
    )
