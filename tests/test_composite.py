"""Composite states: Q± pair, skyrmion tube, hopfion-skyrmion hybrid."""
import numpy as np

from hopfion.grid import Grid
from hopfion.physics.composite import (
    hopfion_skyrmion_hybrid,
    q_minus_one_hopfion,
    q_pair,
    skyrmion_tube,
)
from hopfion.topology import hopf_index


def test_q_minus_one_is_minus_one():
    g = Grid(48, 48, 48, 0.3, 0.3, 0.3, "periodic")
    m = q_minus_one_hopfion(g, R=1.5)
    Q = hopf_index(m, g)
    assert abs(Q - (-1.0)) < 0.05, f"Q_H = {Q}"


def test_q_pair_total_charge_zero():
    g = Grid(64, 64, 48, 0.3, 0.3, 0.3, "periodic")
    m = q_pair(g, R=1.0, separation=6.0)
    Q = hopf_index(m, g)
    assert abs(Q) < 0.1, f"q_pair Q_H = {Q}, expected ≈ 0"


def test_unit_norm_preserved_for_all_constructors():
    g = Grid(32, 32, 32, 0.4, 0.4, 0.4, "periodic")
    for builder in (
        lambda g: q_minus_one_hopfion(g, R=1.5),
        lambda g: q_pair(g, R=1.0, separation=4.0),
        lambda g: skyrmion_tube(g, radius=1.0),
        lambda g: hopfion_skyrmion_hybrid(g, R=1.5, skyrmion_radius=0.8),
    ):
        m = np.asarray(builder(g))
        n = np.sqrt((m * m).sum(axis=0))
        assert np.max(np.abs(n - 1.0)) < 1e-10


def test_skyrmion_tube_z_invariant():
    g = Grid(32, 32, 16, 0.4, 0.4, 0.4, "periodic")
    m = np.asarray(skyrmion_tube(g, radius=1.0))
    # Compare any two z-slices — should be identical for a z-invariant tube
    np.testing.assert_allclose(m[..., 0], m[..., 8], atol=1e-12)


def test_q_pair_axis_x():
    """Centroids of the pair should be at ±separation/2 along the x axis."""
    from hopfion.physics.current import centroids, hopf_charge_density
    g = Grid(48, 48, 32, 0.3, 0.3, 0.3, "periodic")
    sep = 5.0
    m = q_pair(g, R=1.0, separation=sep, axis_of_separation="x")
    rho = hopf_charge_density(m, g)
    cs = centroids(rho, g, threshold_rel=0.15)
    # Expect exactly two centroids on opposite sides of x=0
    assert len(cs) == 2
    xs = sorted(c.position[0] for c in cs)
    # One at negative-x, one at positive-x
    assert xs[0] < 0 and xs[1] > 0
    assert abs(abs(xs[0]) - sep / 2) < 0.5
    assert abs(abs(xs[1]) - sep / 2) < 0.5
