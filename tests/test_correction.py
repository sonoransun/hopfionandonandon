"""Error-correction controllers: instantiation, callback semantics."""
import numpy as np
import pytest

from hopfion.energy import EnergyParams
from hopfion.field import hopfion, uniform
from hopfion.grid import Grid
from hopfion.physics.correction import (
    ActiveFeedbackController,
    StabilizerController,
    TopologicalGapController,
    make_controller,
)


def _setup():
    g = Grid(20, 20, 20, 0.4, 0.4, 0.4, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7, easy_axis=(0, 0, 1))
    return g, ep


def test_active_feedback_triggers_on_q_deficit():
    g, ep = _setup()
    ctrl = ActiveFeedbackController(grid=g, ep=ep, target_Q=1.0,
                                    threshold=0.05, gain=0.5,
                                    cadence=1, correction_duration=2)
    # Use a uniform state (Q ≈ 0); expect detection of deficit
    m = uniform(g, direction=(0, 0, 1))
    ctrl.step_callback(m, k=0, t=0.0)
    assert ctrl.result.n_corrections > 0
    # Correction field is non-zero
    H_extra = ctrl.H_extra_factory()
    H = np.asarray(H_extra(m))
    assert np.max(np.abs(H)) > 0


def test_active_feedback_no_trigger_when_in_band():
    g, ep = _setup()
    ctrl = ActiveFeedbackController(grid=g, ep=ep, target_Q=1.0,
                                    threshold=0.5, gain=1.0, cadence=1)
    m = hopfion(g, R=1.2, p=1, q=1)
    ctrl.step_callback(m, k=0, t=0.0)
    # Q ≈ 1 is within threshold, no correction
    assert ctrl.result.n_corrections == 0


def test_topological_gap_preflight_runs():
    g, ep = _setup()
    ctrl = TopologicalGapController(grid=g, ep=ep, min_gap=-1e-3)
    m = uniform(g, direction=(0, 0, 1))
    ok, min_eig = ctrl.preflight(m)
    # Uniform-FM along easy axis should be stable
    assert ok
    assert min_eig > -1e-3


def test_stabilizer_detects_missing_site():
    g, ep = _setup()
    ctrl = StabilizerController(grid=g, ep=ep, expected_sites=2,
                                flip_threshold=0.5, cadence=1)
    # A single hopfion -- expected 2 sites, only got 1: should trigger
    m = hopfion(g, R=1.2, p=1, q=1)
    ctrl.step_callback(m, k=0, t=0.0)
    assert ctrl.result.n_corrections > 0


def test_make_controller_factory():
    g, ep = _setup()
    assert make_controller("none", g, ep) is None
    assert isinstance(make_controller("active", g, ep), ActiveFeedbackController)
    assert isinstance(make_controller("topological_gap", g, ep, min_gap=0.01),
                      TopologicalGapController)
    assert isinstance(make_controller("stabilizer", g, ep, expected_sites=3),
                      StabilizerController)
    with pytest.raises(ValueError):
        make_controller("unknown_kind", g, ep)
