"""Esoteric topology read-outs: Bloch points, emergent field, linking number."""
import numpy as np

from hopfion.field import hopfion, uniform
from hopfion.grid import Grid
from hopfion.physics.current import emergent_field
from hopfion.topology import bloch_points, hopf_index, linking_number, monopole_density


def _hedgehog(g):
    X, Y, Z = (np.asarray(a) for a in g.coords())
    r = np.sqrt(X * X + Y * Y + Z * Z) + 1e-9
    return np.stack([X / r, Y / r, Z / r], axis=0)


def test_bloch_point_detected_in_hedgehog():
    g = Grid(24, 24, 24, 0.5, 0.5, 0.5, "periodic")
    bp = bloch_points(_hedgehog(g), g)
    assert len(bp) >= 1
    pos, charge = bp[0]
    assert abs(charge) > 0.3                       # significant, sign-definite
    assert all(abs(c) < 1.5 for c in pos)          # localized near the origin


def test_smooth_hopfion_has_no_bloch_points():
    g = Grid(48, 48, 48, 0.3, 0.3, 0.3, "periodic")
    m = hopfion(g, R=1.5)
    assert bloch_points(m, g) == []


def test_emergent_monopole_density_near_zero_for_smooth_field():
    g = Grid(48, 48, 48, 0.3, 0.3, 0.3, "periodic")
    m = hopfion(g, R=1.5)
    _, mono = emergent_field(m, g)
    hog = _hedgehog(Grid(24, 24, 24, 0.5, 0.5, 0.5, "periodic"))
    hog_g = Grid(24, 24, 24, 0.5, 0.5, 0.5, "periodic")
    # the smooth hopfion's monopole density is far weaker than the hedgehog's
    assert float(np.abs(np.asarray(mono)).max()) < \
        float(np.abs(np.asarray(monopole_density(hog, hog_g))).max())


def test_linking_number_matches_hopf_index():
    g = Grid(48, 48, 48, 0.3, 0.3, 0.3, "periodic")
    m = hopfion(g, R=1.5)
    lk = linking_number(m, g, (1, 0, 0), (-1, 0, 0))
    assert abs(abs(lk) - abs(hopf_index(m, g))) < 0.3   # |Lk| ≈ |Q_H| = 1
    assert abs(linking_number(uniform(g, (0, 0, 1)), g, (1, 0, 0), (-1, 0, 0))) < 1e-9
