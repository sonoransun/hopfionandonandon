"""Material-validation CLI helpers: parsing + grade map (pure functions)."""
import pytest

from hopfion.pipeline.cli import _grade, _parse_material


def test_parse_material():
    assert _parse_material("A_ex=1.0,D=1.5,Ku=0.7") == {"A_ex": 1.0, "D": 1.5, "Ku": 0.7}


def test_parse_material_rejects_unknown_key():
    with pytest.raises(ValueError):
        _parse_material("foo=1")


def test_parse_material_rejects_empty():
    with pytest.raises(ValueError):
        _parse_material("")


def test_grade_map():
    assert _grade({"single": True, "nucleate": True, "lattice": True}) == "A"
    assert _grade({"single": True, "nucleate": False, "lattice": True}) == "B"
    assert _grade({"single": True, "nucleate": True, "lattice": False}) == "C"
    assert _grade({"single": True, "nucleate": False, "lattice": False}) == "D"
    assert _grade({"single": False, "nucleate": True, "lattice": True}) == "F"
