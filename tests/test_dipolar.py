"""Dipolar field: zero on uniform m (k=0 zeroed), FD-consistency with energy."""
import numpy as np

from hopfion.energy import EnergyParams, effective_field, total_energy
from hopfion.field import hopfion, uniform
from hopfion.grid import Grid
from hopfion.physics.dipolar import dipolar_energy, dipolar_field


def test_dipolar_field_uniform_is_zero():
    """With k=0 mode zeroed, a uniform field has zero dipolar response."""
    g = Grid(16, 16, 16, 0.5, 0.5, 0.5, "periodic")
    m = uniform(g, direction=(0, 0, 1))
    H = np.asarray(dipolar_field(m, g))
    assert np.max(np.abs(H)) < 1e-10
    assert dipolar_energy(m, g) < 1e-10


def test_dipolar_energy_nonnegative():
    g = Grid(20, 20, 20, 0.5, 0.5, 0.5, "periodic")
    m = hopfion(g, R=1.5, p=1, q=1)
    assert dipolar_energy(m, g) >= 0


def test_dipolar_grad_consistency_with_energy():
    """Numerical dE_dip/dm_i should equal -H_dip,i * dV (the negative of the field)."""
    rng = np.random.default_rng(0)
    g = Grid(10, 10, 10, 0.5, 0.5, 0.5, "periodic")
    m = rng.normal(size=(3, 10, 10, 10))
    m = m / np.sqrt((m * m).sum(0, keepdims=True))
    H = np.asarray(dipolar_field(m, g))
    eps = 1e-6
    errs = []
    for _ in range(6):
        i = rng.integers(3); ix, iy, iz = rng.integers(10, size=3)
        m_p = m.copy(); m_p[i, ix, iy, iz] += eps
        m_m = m.copy(); m_m[i, ix, iy, iz] -= eps
        dE_num = (dipolar_energy(m_p, g) - dipolar_energy(m_m, g)) / (2 * eps)
        dE_anal = -H[i, ix, iy, iz] * g.dV
        errs.append(abs(dE_num - dE_anal) / max(abs(dE_anal), 1e-10))
    # ~1e-6 expected from FD floor
    assert max(errs) < 1e-3


def test_energy_params_dipolar_flag_routes_through():
    g = Grid(10, 10, 10, 0.5, 0.5, 0.5, "periodic")
    m = hopfion(g, R=1.5, p=1, q=1)
    p_no = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7, dipolar=False)
    p_yes = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7, dipolar=True)
    # Adding the dipolar term changes the energy (non-zero for a hopfion)
    assert total_energy(m, g, p_yes) - total_energy(m, g, p_no) > 0
    H_no = np.asarray(effective_field(m, g, p_no))
    H_yes = np.asarray(effective_field(m, g, p_yes))
    assert not np.allclose(H_no, H_yes)


def test_dipolar_combined_energy_grad_consistency():
    """Full EnergyParams.total_energy / effective_field stay an adjoint pair
    even with the dipolar term enabled."""
    rng = np.random.default_rng(7)
    g = Grid(10, 10, 10, 0.5, 0.5, 0.5, "periodic")
    m = rng.normal(size=(3, 10, 10, 10))
    m = m / np.sqrt((m * m).sum(0, keepdims=True))
    ep = EnergyParams(A_ex=1.0, D=0.4, Ku=0.2, easy_axis=(0, 0, 1),
                     H_ext=(0.0, 0.0, 0.1), dipolar=True)
    H = np.asarray(effective_field(m, g, ep))
    eps = 1e-6
    errs = []
    for _ in range(8):
        i = rng.integers(3); ix, iy, iz = rng.integers(10, size=3)
        m_p = m.copy(); m_p[i, ix, iy, iz] += eps
        m_m = m.copy(); m_m[i, ix, iy, iz] -= eps
        dE_num = (total_energy(m_p, g, ep) - total_energy(m_m, g, ep)) / (2 * eps)
        dE_anal = -H[i, ix, iy, iz] * g.dV
        errs.append(abs(dE_num - dE_anal) / max(abs(dE_anal), 1e-10))
    assert max(errs) < 1e-2
