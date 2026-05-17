"""Laser pulse model: profile shape and time-amplitude scaling."""
import numpy as np
import pytest

from hopfion.grid import Grid
from hopfion.laser import GaussianPulse, TwoTemperaturePulse


def test_gaussian_pulse_peaks_at_t0():
    g = Grid(16, 16, 16, 0.5, 0.5, 0.5, "periodic")
    pulse = GaussianPulse(H0=(0.0, 0.0, 1.0), t0=1.0, tau=0.2, profile="point", width=1.0)
    Hf = pulse.field_factory(g)
    # m argument is unused for the simple pulse; pass a placeholder.
    m_dummy = np.zeros((3,) + g.shape)
    H_peak = np.asarray(Hf(m_dummy, t=pulse.t0))
    H_off = np.asarray(Hf(m_dummy, t=pulse.t0 + 4 * pulse.tau))
    assert np.max(np.abs(H_peak)) > 0.9
    assert np.max(np.abs(H_off)) < 1e-5


def test_pulse_profiles_localized():
    g = Grid(32, 32, 32, 0.5, 0.5, 0.5, "periodic")
    for profile in ("point", "disk", "ring"):
        pulse = GaussianPulse(profile=profile, width=0.5, ring_radius=2.0)
        env = np.asarray(pulse.spatial_envelope(g))
        assert np.isfinite(env).all()
        assert env.max() > 0.0
        # peak should not be at the box boundary
        center_block = env[8:24, 8:24, 8:24]
        edge_max = max(env[0].max(), env[-1].max())
        assert center_block.max() > edge_max


def test_two_temperature_phase_b_stub():
    pulse = TwoTemperaturePulse()
    g = Grid(8, 8, 8)
    with pytest.raises(NotImplementedError):
        pulse.field_factory(g)
