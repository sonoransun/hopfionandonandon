"""HDF5 round-trip smoke test."""
import os
import tempfile

import numpy as np

from hopfion.energy import EnergyParams
from hopfion.field import hopfion
from hopfion.grid import Grid
from hopfion.io import load_run, save_run


def test_save_load_roundtrip():
    g = Grid(16, 16, 16, 0.5, 0.5, 0.5, "periodic")
    m = hopfion(g, R=1.0, p=1, q=1)
    ep = EnergyParams(A_ex=1.0, D=0.4, Ku=0.05, H_ext=(0, 0, 0.1))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "run.h5")
        save_run(path, m, g, ep, note="test", seed=0)
        m2, g2, ep2, extras = load_run(path)
    np.testing.assert_allclose(m2, np.asarray(m))
    assert g2.nx == g.nx and g2.dx == g.dx and g2.bc == g.bc
    assert ep2.A_ex == ep.A_ex and ep2.D == ep.D
    assert extras["note"] == "test"
