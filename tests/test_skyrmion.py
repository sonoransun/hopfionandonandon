"""Skyrmion as a first-class tracked topological charge.

Companion to test_topology.py / test_composite.py: the 2D skyrmion (Pontryagin)
number, the skyrmion/antiskyrmion field constructor, the q_skyrmion metric, and
the skyrmion QC criteria.
"""
from pathlib import Path

import numpy as np

from hopfion.field import skyrmion
from hopfion.grid import Grid
from hopfion.physics.composite import skyrmion_tube
from hopfion.pipeline.metrics import MetricContext, SkyrmionChargeDrift
from hopfion.pipeline.qc import (
    SkyrmionNumberDriftBelowCriterion,
    SkyrmionNumberWithinCriterion,
)
from hopfion.pipeline.recipe import RecipeConfig
from hopfion.pipeline.runner import run
from hopfion.topology import (
    skyrmion_charge_density,
    skyrmion_density_xy,
    skyrmion_number,
)

REPO = Path(__file__).resolve().parents[1]


def make_grid(N=64, nz=16, d=0.3):
    return Grid(N, N, nz, d, d, d, "periodic")


# --- the invariant -------------------------------------------------------


def test_skyrmion_charge_is_minus_one():
    g = make_grid()
    m = skyrmion(g, radius=1.5, helicity=np.pi / 2, vorticity=1)
    N = skyrmion_number(m, g)
    assert abs(N - (-1.0)) < 0.05, f"N_sk = {N}"


def test_antiskyrmion_charge_flips_sign():
    g = make_grid()
    Ns = skyrmion_number(skyrmion(g, radius=1.5, vorticity=1), g)
    Na = skyrmion_number(skyrmion(g, radius=1.5, vorticity=-1), g)
    assert abs(Na - (+1.0)) < 0.05, f"N_anti = {Na}"
    assert np.sign(Na) == -np.sign(Ns), "vorticity should flip the charge sign"


def test_unit_norm_preserved():
    g = make_grid(N=32)
    for vort in (+1, -1):
        m = np.asarray(skyrmion(g, radius=1.0, vorticity=vort))
        n = np.sqrt((m * m).sum(axis=0))
        assert np.max(np.abs(n - 1.0)) < 1e-10


def test_z_invariant():
    g = make_grid(N=32, nz=16)
    m = np.asarray(skyrmion(g, radius=1.0))
    np.testing.assert_allclose(m[..., 0], m[..., 8], atol=1e-12)


def test_bloch_and_neel_have_same_charge():
    g = make_grid()
    n_bloch = skyrmion_number(skyrmion(g, radius=1.5, helicity=np.pi / 2), g)
    n_neel = skyrmion_number(skyrmion(g, radius=1.5, helicity=0.0), g)
    assert abs(n_bloch - n_neel) < 1e-9, f"helicity changed the charge: {n_bloch} vs {n_neel}"


def test_charge_density_integrates_to_number():
    g = make_grid()
    m = skyrmion(g, radius=1.5)
    q = np.asarray(skyrmion_charge_density(m, g))
    zc = g.nz // 2
    integrated = float(q[:, :, zc].sum()) * g.dx * g.dy
    assert abs(integrated - skyrmion_number(m, g, z_index=zc)) < 1e-10


def test_skyrmion_density_xy_backward_compat():
    g = make_grid(N=32, nz=12)
    m = skyrmion(g, radius=1.0)
    per_z = np.asarray(skyrmion_density_xy(m, g))
    assert per_z.shape == (g.nz,)
    # equals integrating the pointwise density over (x, y)
    q = np.asarray(skyrmion_charge_density(m, g))
    np.testing.assert_allclose(per_z, q.sum(axis=(0, 1)) * g.dx * g.dy, atol=1e-12)


def test_skyrmion_tube_delegates_to_field_skyrmion():
    g = make_grid(N=32, nz=12)
    a = np.asarray(skyrmion_tube(g, radius=1.0, helicity=np.pi / 2))
    b = np.asarray(skyrmion(g, radius=1.0, helicity=np.pi / 2, vorticity=1))
    np.testing.assert_allclose(a, b, atol=1e-14)


# --- metric --------------------------------------------------------------


def test_skyrmion_charge_drift_metric_summary():
    g = make_grid(N=32, nz=12)
    m = skyrmion(g, radius=1.0)
    met = SkyrmionChargeDrift(cadence=1)
    for step in range(3):
        met.collect(MetricContext(m=m, grid=g, ep=None, step=step, t=float(step)))
    s = met.summary()
    assert set(s) >= {"N_initial", "N_final", "max_drift", "n_samples"}
    assert s["n_samples"] == 3
    assert s["max_drift"] == 0.0  # static field → no drift


# --- QC criteria ---------------------------------------------------------


def test_skyrmion_number_within_criterion():
    crit = SkyrmionNumberWithinCriterion(severity="warn", target=-1.0, tol=0.1)
    assert crit.evaluate({"q_skyrmion": {"N_final": -0.97}}).passed
    assert not crit.evaluate({"q_skyrmion": {"N_final": -0.5}}).passed
    # skip-pass when the metric was never collected
    assert crit.evaluate({}).passed


def test_skyrmion_number_drift_below_criterion():
    crit = SkyrmionNumberDriftBelowCriterion(severity="fail", tol=0.1)
    assert crit.evaluate({"q_skyrmion": {"max_drift": 0.02}}).passed
    assert not crit.evaluate({"q_skyrmion": {"max_drift": 0.5}}).passed
    assert crit.evaluate({}).passed


# --- end-to-end recipe ---------------------------------------------------


def test_skyrmion_recipe_accepts():
    rc = RecipeConfig.from_yaml(REPO / "recipes" / "regression" / "skyrmion_tube.yaml")
    res = run(rc, write=False)
    assert res.qc.verdict == "ACCEPT"
    assert abs(res.metrics["q_skyrmion"]["N_final"] - (-1.0)) < 0.1
