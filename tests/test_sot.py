"""Spin-orbit torque + the dynamics_sot / ac_drive run-steps."""
import numpy as np

from hopfion.energy import EnergyParams
from hopfion.field import uniform
from hopfion.grid import Grid
from hopfion.physics.sot import SOTParams, sot_step_heun
from hopfion.pipeline.recipe import (
    GridSpec,
    InitialStateSpec,
    IOSpec,
    MaterialSpec,
    RecipeConfig,
    RunStep,
)
from hopfion.pipeline.runner import run


def test_damping_like_sot_drives_m_toward_polarization():
    g = Grid(16, 16, 16, 0.5, 0.5, 0.5, "periodic")
    ep = EnergyParams(A_ex=1.0, D=0.0, Ku=0.0)        # isolate the torque
    m = uniform(g, direction=(0.6, 0.0, 0.8))         # tilted off +z
    sot = SOTParams(p=(0, 0, 1), dl=0.5, fl=0.0)
    mz0 = float(np.asarray(m)[2].mean())
    for _ in range(300):
        m = sot_step_heun(m, g, ep, gamma=1.0, alpha=0.1, dt=0.01, sot=sot)
    mz1 = float(np.asarray(m)[2].mean())
    assert mz1 > mz0                                  # aligned further toward +z
    assert mz1 > 0.98                                 # essentially switched toward p
    n = np.sqrt((np.asarray(m) ** 2).sum(axis=0))
    assert np.abs(n - 1.0).max() < 1e-10


def _rc(kind, **stepkw):
    return RecipeConfig(
        name=f"{kind}_test",
        grid=GridSpec(nx=16, ny=16, nz=16, dx=0.5),
        material=MaterialSpec(A_ex=1.0, D=0.0, Ku=0.3),
        initial=InitialStateSpec(kind="uniform", direction=(0.6, 0.0, 0.8)),
        run=[RunStep(kind=kind, n_steps=40, dt=0.01, **stepkw)],
        io=IOSpec(out="runs/_sot_test", write_report=False),
    )


def test_dynamics_sot_runstep_runs():
    res = run(_rc("dynamics_sot", sot_p=(0, 0, 1), sot_dl=0.5), write=False)
    assert res.qc.verdict == "ACCEPT"
    assert res.metrics["norm_drift"]["max_drift"] < 1e-10


def test_ac_drive_runstep_runs():
    res = run(_rc("ac_drive", ac_H0=(0.2, 0.0, 0.0), ac_omega=2.0), write=False)
    assert res.qc.verdict == "ACCEPT"
    assert res.metrics["norm_drift"]["max_drift"] < 1e-10
