"""Bilayer run-step wiring: J0=0 decoupling + norm preservation through the runner."""
import numpy as np

from hopfion.energy import EnergyParams
from hopfion.field import hopfion
from hopfion.grid import Grid
from hopfion.llg import relax
from hopfion.pipeline.recipe import (
    GridSpec,
    InitialStateSpec,
    IOSpec,
    MaterialSpec,
    RecipeConfig,
    RunStep,
)
from hopfion.pipeline.runner import run


def _rc(J0):
    return RecipeConfig(
        name="bilayer_test",
        grid=GridSpec(nx=24, ny=24, nz=24, dx=0.4),
        material=MaterialSpec(A_ex=1.0, D=1.5, Ku=0.7),
        initial=InitialStateSpec(kind="hopfion", R=1.5),
        run=[RunStep(kind="bilayer", n_steps=30, dt=0.002, alpha=0.1,
                     bilayer_J0=J0, bilayer_layer2="uniform")],
        io=IOSpec(out="runs/_bilayer_test", write_report=False),
    )


def test_bilayer_runstep_preserves_norm():
    res = run(_rc(0.2), write=False)
    assert res.qc.verdict == "ACCEPT"
    assert res.metrics["norm_drift"]["max_drift"] < 1e-10


def test_bilayer_zero_coupling_matches_single_layer():
    g = Grid(24, 24, 24, 0.4, 0.4, 0.4, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7, easy_axis=(0, 0, 1))
    m_single = relax(hopfion(g, R=1.5), g, ep, n_steps=30, dt=0.002)
    res = run(_rc(0.0), write=False)
    # With J0=0 the layers decouple; layer-1 must reproduce a plain relax.
    assert np.allclose(np.asarray(res.m_final), np.asarray(m_single), atol=1e-10)
