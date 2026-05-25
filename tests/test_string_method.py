"""String-method activation barriers + Arrhenius lifetimes."""
from hopfion.energy import EnergyParams, total_energy
from hopfion.field import hopfion, uniform
from hopfion.grid import Grid
from hopfion.llg import relax
from hopfion.physics.string_method import arrhenius_lifetime, string_method


def test_identical_endpoints_zero_barrier():
    g = Grid(16, 16, 16, 0.5, 0.5, 0.5, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7, easy_axis=(0, 0, 1))
    m = relax(hopfion(g, R=1.5), g, ep, n_steps=40, dt=0.002)
    r = string_method(m, m, g, ep, n_images=5, n_iter=30, dt=0.005)
    assert abs(r.barrier) < 1e-6


def test_anisotropy_double_well_barrier():
    """Uniform near +z -> near -z through the hard plane: a clean analytic
    barrier ≈ Ku * V with the saddle (in-plane) at the interior midpoint."""
    g = Grid(16, 16, 16, 0.5, 0.5, 0.5, "periodic")
    V = g.nx * g.ny * g.nz * g.dV
    Ku = 0.3
    ep = EnergyParams(A_ex=1.0, D=0.0, Ku=Ku, easy_axis=(0, 0, 1))
    a = uniform(g, direction=(0.05, 0.0, 1.0))
    b = uniform(g, direction=(0.05, 0.0, -1.0))
    r = string_method(a, b, g, ep, n_images=9, n_iter=200, dt=0.01)
    assert abs(r.barrier - Ku * V) < 0.15 * Ku * V       # within 15% of Ku·V
    assert 0 < r.saddle_index < 8                         # interior saddle
    assert abs(r.energies[0] - total_energy(a, g, ep)) < 1e-6


def test_arrhenius_lifetime_monotone_and_overflow():
    assert arrhenius_lifetime(20.0, 1.0) > arrhenius_lifetime(10.0, 1.0)
    assert arrhenius_lifetime(1e6, 1.0) == float("inf")     # overflow guard
    assert arrhenius_lifetime(5.0, 0.0) == float("inf")     # kT <= 0
