"""Two-temperature sLLG engine: noise scaling + bath ODE behavior."""
import numpy as np

from hopfion.energy import EnergyParams
from hopfion.field import uniform
from hopfion.grid import Grid
from hopfion.physics.two_temp import TwoTemperatureLLG
from hopfion.topology import hopf_index


def test_noise_variance_scales_with_inverse_dV():
    """sigma^2 ∝ 1/dV. Halve dx -> dV shrinks 8x -> variance grows 8x."""
    g_coarse = Grid(10, 10, 10, 0.8, 0.8, 0.8, "periodic")
    g_fine = Grid(10, 10, 10, 0.4, 0.4, 0.4, "periodic")
    engine = TwoTemperatureLLG(Te_peak=0, alpha=0.1)
    rng = np.random.default_rng(0)
    n_coarse = engine.stochastic_field(T_e=4.0, dt=0.01, grid=g_coarse, rng=rng)
    n_fine = engine.stochastic_field(T_e=4.0, dt=0.01, grid=g_fine, rng=rng)
    var_coarse = float(np.var(n_coarse))
    var_fine = float(np.var(n_fine))
    ratio = var_fine / var_coarse
    # dV_coarse / dV_fine = (0.8/0.4)^3 = 8
    assert 5.0 < ratio < 12.0, f"variance ratio (fine/coarse) = {ratio:.2f}, expected ~8"


def test_noise_variance_linear_in_Te():
    g = Grid(10, 10, 10, 0.5, 0.5, 0.5, "periodic")
    engine = TwoTemperatureLLG(Te_peak=0, alpha=0.1)
    rng = np.random.default_rng(0)
    n1 = engine.stochastic_field(T_e=1.0, dt=0.01, grid=g, rng=rng)
    n4 = engine.stochastic_field(T_e=4.0, dt=0.01, grid=g, rng=rng)
    var1 = float(np.var(n1))
    var4 = float(np.var(n4))
    # var ∝ T_e, so var4/var1 should be ~4
    assert 3.0 < var4 / var1 < 5.0


def test_temperature_ode_relaxes_to_equilibrium():
    engine = TwoTemperatureLLG(Te_peak=0, tau=0.05, t0=0.0, G_el=1.0, C_e=1.0, C_l=1.0)
    Te, Tl = 10.0, 0.0
    # No laser power past t = 1 (well past t0=0)
    for _ in range(2000):
        Te, Tl = engine.step_temperatures(Te, Tl, dt=0.001, t=2.0)
    # With closed bath (no source), bath equilibrates to (T_e + T_l)/2 = 5
    assert abs(Te - 5.0) < 0.5 and abs(Tl - 5.0) < 0.5


def test_run_preserves_unit_norm_and_finite_field():
    g = Grid(16, 16, 16, 0.5, 0.5, 0.5, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7, easy_axis=(0, 0, 1))
    engine = TwoTemperatureLLG(Te_peak=5.0, tau=0.05, t0=0.05, alpha=0.1)
    m = uniform(g)
    rng = np.random.default_rng(0)
    m_final = engine.run(m, g, ep, n_steps=20, dt=0.002, rng=rng)
    n = np.sqrt(np.sum(np.asarray(m_final) ** 2, axis=0))
    assert np.max(np.abs(n - 1.0)) < 1e-10
    assert np.isfinite(m_final).all()


def test_two_temperature_pulse_legacy_class_works():
    """laser.TwoTemperaturePulse forwards to physics.two_temp."""
    from hopfion.laser import TwoTemperaturePulse
    pulse = TwoTemperaturePulse(Te_peak=5.0, tau=0.05, t0=0.05, alpha=0.1)
    # field_factory is the only thing that still raises (noise variance is time-varying)
    g = Grid(8, 8, 8, 0.5, 0.5, 0.5, "periodic")
    import pytest
    with pytest.raises(NotImplementedError):
        pulse.field_factory(g)
    # The .run() method works
    ep = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7)
    m = uniform(g)
    rng = np.random.default_rng(0)
    out = pulse.run(m, g, ep, n_steps=10, dt=0.002, rng=rng)
    assert np.isfinite(out).all()
