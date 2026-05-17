"""Heun / RK4 / damped step sanity tests."""
import numpy as np

from hopfion.energy import EnergyParams, total_energy
from hopfion.field import hopfion
from hopfion.grid import Grid
from hopfion.physics.integrators import damped_step, heun_step, rk4_step


def _setup():
    g = Grid(24, 24, 24, 0.5, 0.5, 0.5, "periodic")
    m = hopfion(g, R=1.5, p=1, q=1)
    ep = EnergyParams(A_ex=1.0, D=0.4, Ku=0.1, easy_axis=(0, 0, 1), H_ext=(0, 0, 0))
    return g, m, ep


def test_heun_step_preserves_norm():
    g, m, ep = _setup()
    for _ in range(20):
        m = heun_step(m, g, ep, gamma=1.0, alpha=0.05, dt=0.005)
    n = np.sqrt(np.sum(np.asarray(m) ** 2, axis=0))
    assert np.max(np.abs(n - 1.0)) < 1e-10


def test_rk4_step_preserves_norm_and_drops_energy():
    g, m, ep = _setup()
    E0 = total_energy(m, g, ep)
    for _ in range(20):
        m = rk4_step(m, g, ep, gamma=1.0, alpha=0.1, dt=0.005)
    E1 = total_energy(m, g, ep)
    n = np.sqrt(np.sum(np.asarray(m) ** 2, axis=0))
    assert np.max(np.abs(n - 1.0)) < 1e-10
    assert E1 < E0   # damped dynamics


def test_damped_step_monotonic_energy():
    g, m, ep = _setup()
    Es = [total_energy(m, g, ep)]
    for _ in range(40):
        m = damped_step(m, g, ep, dt=0.005)
        Es.append(total_energy(m, g, ep))
    diffs = np.diff(Es)
    assert np.max(diffs) < 1e-6
