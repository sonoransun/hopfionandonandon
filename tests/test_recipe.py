"""RecipeConfig + preflight tests."""
import pytest

from hopfion.pipeline.recipe import RecipeConfig, preflight


VALID = {
    "name": "test",
    "grid": {"nx": 32, "ny": 32, "nz": 32, "dx": 0.3, "bc": "periodic"},
    "material": {"A_ex": 1.0, "D": 1.5, "Ku": 0.7,
                 "easy_axis": [0, 0, 1], "H_ext": [0, 0, 0]},
    "initial": {"kind": "hopfion", "R": 1.5, "p": 1, "q": 1},
    "run": [{"kind": "relax", "n_steps": 50, "dt": 0.002}],
    "qc": {"fail_on": [{"hopf_index_within": {"target": 1.0, "tol": 0.05}}]},
    "io": {"out": "runs/test/"},
    "seed": 0,
}


def test_from_dict_roundtrip():
    rc = RecipeConfig.from_dict(VALID)
    assert rc.name == "test"
    assert rc.material.D == 1.5
    assert rc.initial.kind == "hopfion"
    assert rc.run[0].dt == 0.002
    assert tuple(rc.material.easy_axis) == (0, 0, 1)


def test_preflight_accepts_valid_recipe():
    rc = RecipeConfig.from_dict(VALID)
    result = preflight(rc)
    assert result.passed, result.failures


def test_preflight_rejects_underresolved_hopfion():
    bad = dict(VALID)
    bad["grid"] = dict(VALID["grid"], dx=0.8)  # dx > R/2
    rc = RecipeConfig.from_dict(bad)
    result = preflight(rc)
    assert not result.passed
    assert any("under-resolved" in f for f in result.failures)


def test_preflight_rejects_unstable_dt():
    bad = dict(VALID)
    bad["run"] = [{"kind": "relax", "n_steps": 10, "dt": 1.0}]
    rc = RecipeConfig.from_dict(bad)
    result = preflight(rc)
    assert not result.passed
    assert any("dt" in f for f in result.failures)


def test_preflight_rejects_small_box():
    bad = dict(VALID)
    bad["grid"] = {"nx": 12, "ny": 12, "nz": 12, "dx": 0.3, "bc": "periodic"}
    bad["initial"] = {"kind": "hopfion", "R": 1.5, "p": 1, "q": 1}
    rc = RecipeConfig.from_dict(bad)
    result = preflight(rc)
    assert not result.passed
    assert any("4R" in f or "image" in f for f in result.failures)
