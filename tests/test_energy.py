"""Energy / effective-field FD consistency checks."""
import numpy as np
import pytest

from hopfion.energy import EnergyParams, total_energy, effective_field
from hopfion.grid import Grid


def random_unit_field(seed, shape):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(3,) + shape)
    return m / np.sqrt(np.sum(m * m, axis=0, keepdims=True))


PARAM_SETS = [
    ("exchange",  EnergyParams(A_ex=1.0, D=0.0, Ku=0.0, H_ext=(0, 0, 0))),
    ("dmi",       EnergyParams(A_ex=0.0, D=0.4, Ku=0.0, H_ext=(0, 0, 0))),
    ("anisotropy",EnergyParams(A_ex=0.0, D=0.0, Ku=0.5, easy_axis=(1, 0, 0), H_ext=(0, 0, 0))),
    ("zeeman",    EnergyParams(A_ex=0.0, D=0.0, Ku=0.0, H_ext=(0.1, 0.2, 0.05))),
    ("interfacial_dmi", EnergyParams(A_ex=0.0, D=0.0, Ku=0.0, D_interface=0.4, H_ext=(0, 0, 0))),
    ("cubic",     EnergyParams(A_ex=0.0, D=0.0, Ku=0.0, Kc=0.5, H_ext=(0, 0, 0))),
    ("combined",  EnergyParams(A_ex=1.0, D=0.4, Ku=0.3, easy_axis=(0, 0, 1), H_ext=(0.0, 0.0, 0.1))),
    ("combined_new", EnergyParams(A_ex=1.0, D_interface=0.3, Kc=0.4, easy_axis=(0, 0, 1), H_ext=(0.0, 0.0, 0.1))),
]


@pytest.mark.parametrize("label,ep", PARAM_SETS, ids=[p[0] for p in PARAM_SETS])
def test_energy_gradient_matches_effective_field(label, ep):
    g = Grid(12, 12, 12, 0.4, 0.4, 0.4, "periodic")
    m = random_unit_field(seed=0, shape=g.shape)
    H = effective_field(m, g, ep)
    rng = np.random.default_rng(7)
    eps = 1e-6
    errs = []
    for _ in range(8):
        i = rng.integers(3); ix, iy, iz = rng.integers(12, size=3)
        m_p = m.copy(); m_p[i, ix, iy, iz] += eps
        m_m = m.copy(); m_m[i, ix, iy, iz] -= eps
        dE_num = (total_energy(m_p, g, ep) - total_energy(m_m, g, ep)) / (2 * eps)
        dE_an = -H[i, ix, iy, iz] * g.dV
        denom = max(abs(dE_an), 1e-10)
        errs.append(abs(dE_num - dE_an) / denom)
    assert max(errs) < 1e-3, f"{label}: max rel err = {max(errs):.2e}"
