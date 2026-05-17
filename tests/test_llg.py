"""LLG dynamics: |m|=1 preservation and energy monotonicity under damping."""
import numpy as np

from hopfion.energy import EnergyParams, total_energy
from hopfion.field import hopfion
from hopfion.grid import Grid
from hopfion.llg import LLGParams, llg_step_heun, relax_step
from hopfion.topology import hopf_index


def test_norm_preservation_under_full_llg():
    g = Grid(24, 24, 24, 0.5, 0.5, 0.5, "periodic")
    m = hopfion(g, R=1.5, p=1, q=1)
    ep = EnergyParams(A_ex=1.0, D=0.4, Ku=0.1, H_ext=(0, 0, 0.05))
    lp = LLGParams(gamma=1.0, alpha=0.05, dt=0.005)
    for _ in range(40):
        m = llg_step_heun(m, g, ep, lp)
    norms = np.sqrt(np.sum(m * m, axis=0))
    assert np.max(np.abs(norms - 1.0)) < 1e-10


def test_damped_relax_decreases_energy_monotonically():
    g = Grid(24, 24, 24, 0.5, 0.5, 0.5, "periodic")
    m = hopfion(g, R=1.5, p=1, q=1)
    ep = EnergyParams(A_ex=1.0, D=0.4, Ku=0.1, H_ext=(0, 0, 0.0))
    E_hist = [total_energy(m, g, ep)]
    for _ in range(80):
        m = relax_step(m, g, ep, dt=0.005)
        E_hist.append(total_energy(m, g, ep))
    diffs = np.diff(E_hist)
    assert np.max(diffs) < 1e-6, f"E not monotone: max-increase = {np.max(diffs):.3e}"


def test_hopf_index_stable_under_short_pure_precession():
    """With zero damping, Q_H should be exactly preserved by LLG precession
    (up to discretization). Run a short trajectory and check |dQ| stays small."""
    g = Grid(48, 48, 48, 0.5, 0.5, 0.5, "periodic")
    m = hopfion(g, R=2.0, p=1, q=1)
    ep = EnergyParams(A_ex=1.0, D=0.0, Ku=0.0, H_ext=(0, 0, 0))  # exchange only
    lp = LLGParams(gamma=1.0, alpha=0.0, dt=0.001)
    Q0 = hopf_index(m, g)
    for _ in range(20):
        m = llg_step_heun(m, g, ep, lp)
    Q = hopf_index(m, g)
    assert abs(Q - Q0) < 0.05, f"Q_H drifted from {Q0:.4f} to {Q:.4f}"
