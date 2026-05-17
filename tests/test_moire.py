"""Moire potential and lattice array initialization smoke tests."""
import numpy as np

from hopfion.energy import EnergyParams, total_energy
from hopfion.grid import Grid
from hopfion.lattice import array_hopfion, square_sites_2d, triangular_sites_2d
from hopfion.moire import MoirePotential
from hopfion.topology import hopf_index


def test_moire_potential_structure():
    """Mean ~ K0 (the three cosines average out), and z-independent for 2D lattice."""
    g = Grid(64, 64, 8, 0.5, 0.5, 0.5, "periodic")
    moire = MoirePotential(K0=0.1, V0=1.0, a_moire=4.0, lattice="triangular")
    K = np.asarray(moire.Ku_field(g))
    assert abs(K.mean() - 0.1) < 0.02
    assert K.std() > 0.5  # significant modulation
    # z-independent (2D lattice in xy plane): K[..., 0] == K[..., 1] == ...
    for k in range(1, g.nz):
        np.testing.assert_allclose(K[..., 0], K[..., k], atol=1e-12)


def test_square_moire_periodic_along_x():
    """Square lattice IS periodic along x with period a_moire (when a_moire/dx is integer)."""
    g = Grid(32, 32, 16, 0.5, 0.5, 0.5, "periodic")
    moire = MoirePotential(K0=0.0, V0=1.0, a_moire=4.0, lattice="square")
    K = np.asarray(moire.Ku_field(g))
    cells_per_period = int(round(moire.a_moire / g.dx))  # 8
    np.testing.assert_allclose(K, np.roll(K, cells_per_period, axis=0), atol=1e-10)
    np.testing.assert_allclose(K, np.roll(K, cells_per_period, axis=1), atol=1e-10)


def test_lattice_array_initializer_unit_norm():
    g = Grid(48, 48, 24, 0.5, 0.5, 0.5, "periodic")
    sites = square_sites_2d(a=6.0, nx=3, ny=3)
    m = array_hopfion(g, sites, R=1.0)
    n = np.sqrt(np.sum(np.asarray(m) ** 2, axis=0))
    assert np.allclose(n, 1.0, atol=1e-12)


def test_triangular_lattice_sites_count():
    sites = triangular_sites_2d(a=2.0, n_rings=1)
    # 1 center + 6 first-ring neighbors
    assert len(sites) == 7
