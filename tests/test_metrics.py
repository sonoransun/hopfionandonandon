"""Metric callbacks: collection + summary shape."""
import numpy as np

from hopfion.energy import EnergyParams
from hopfion.field import hopfion
from hopfion.grid import Grid
from hopfion.pipeline.metrics import (
    EnergyMonotonicity,
    HopfIndexDrift,
    MetricCollector,
    NormDrift,
    Runtime,
    default_metrics,
)


def _ctx():
    g = Grid(24, 24, 24, 0.5, 0.5, 0.5, "periodic")
    m = hopfion(g, R=1.5, p=1, q=1)
    ep = EnergyParams(A_ex=1.0, D=0.4, Ku=0.1, easy_axis=(0, 0, 1))
    return g, m, ep


def test_energy_metric_records_each_call():
    g, m, ep = _ctx()
    col = MetricCollector(metrics=[EnergyMonotonicity()], grid=g, ep=ep)
    for k in range(5):
        col.relax_callback(m, k)
    s = col.summarize()
    assert s["energy"]["n_samples"] == 5
    assert "E_initial" in s["energy"]


def test_norm_drift_tracks_machine_precision():
    g, m, ep = _ctx()
    col = MetricCollector(metrics=[NormDrift()], grid=g, ep=ep)
    col.relax_callback(m, 0)
    s = col.summarize()
    assert s["norm_drift"]["max_drift"] < 1e-12


def test_hopf_index_drift_cadence_subsamples():
    g, m, ep = _ctx()
    col = MetricCollector(metrics=[HopfIndexDrift(cadence=10)], grid=g, ep=ep)
    for k in range(30):
        col.relax_callback(m, k)
    # collected at k = 0, 10, 20 → 3 samples
    assert col.summarize()["q_hopf"]["n_samples"] == 3


def test_default_metrics_returns_four():
    metrics = default_metrics()
    names = {m.name for m in metrics}
    assert names == {"energy", "norm_drift", "q_hopf", "runtime"}
