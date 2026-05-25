"""FMR dynamical eigenspectrum: the linearized-LLG resonance frequencies."""
import pytest

from hopfion.energy import EnergyParams
from hopfion.field import uniform
from hopfion.grid import Grid
from hopfion.physics.fmr import dynamical_modes


@pytest.mark.parametrize("gamma,H", [(1.0, 0.5), (2.0, 0.3)])
def test_uniform_fm_kittel_frequency(gamma, H):
    """Uniform FM in a field H·ẑ (no exchange/DMI/anisotropy) precesses at the
    Larmor/Kittel frequency ω = γH — the analytic anchor for the solver."""
    g = Grid(6, 6, 6, 1.0, 1.0, 1.0, "periodic")
    ep = EnergyParams(A_ex=0.0, D=0.0, Ku=0.0, H_ext=(0, 0, H))
    m = uniform(g, (0, 0, 1))
    freqs, modes = dynamical_modes(m, g, ep, gamma=gamma, k=6)
    assert len(freqs) >= 1
    assert abs(freqs[0] - gamma * H) < 1e-2
    assert modes.shape[1:] == (3, g.nx, g.ny, g.nz)


def test_anisotropy_raises_resonance_frequency():
    """Adding easy-axis anisotropy stiffens the precession (higher ω) — a
    monotonic sanity check beyond the bare-field case."""
    g = Grid(6, 6, 6, 1.0, 1.0, 1.0, "periodic")
    m = uniform(g, (0, 0, 1))
    H = 0.5
    w0 = dynamical_modes(m, g, EnergyParams(A_ex=0.0, Ku=0.0, H_ext=(0, 0, H)), k=6)[0][0]
    w1 = dynamical_modes(m, g, EnergyParams(A_ex=0.0, Ku=0.3, easy_axis=(0, 0, 1),
                                            H_ext=(0, 0, H)), k=6)[0][0]
    assert w1 > w0
