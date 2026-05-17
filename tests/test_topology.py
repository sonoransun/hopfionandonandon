"""Hopf-invariant correctness on analytic hopfion fields."""
import pytest

from hopfion.field import hopfion, uniform
from hopfion.grid import Grid
from hopfion.topology import hopf_index


def make_grid(N=64, L=16.0):
    d = L / N
    return Grid(N, N, N, d, d, d, "periodic")


@pytest.mark.parametrize("p,q", [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)])
def test_hopf_index_integer(p, q):
    g = make_grid(N=96, L=16.0)
    m = hopfion(g, R=1.0, p=p, q=q)
    Q = hopf_index(m, g)
    assert abs(Q - p * q) < 0.05, f"Expected Q_H = {p*q}, got {Q:.4f}"


def test_uniform_hopf_zero():
    g = make_grid(N=48, L=8.0)
    m = uniform(g, direction=(0.0, 0.0, 1.0))
    Q = hopf_index(m, g)
    assert abs(Q) < 1e-6


def test_periodic_bc_required():
    g = Grid(32, 32, 32, 0.5, 0.5, 0.5, bc="open")
    m = uniform(g)
    with pytest.raises(ValueError):
        hopf_index(m, g)
