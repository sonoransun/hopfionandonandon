"""Hessian eigenmode analysis: distinguishes stable from saddle configurations."""
import numpy as np
import pytest

from hopfion.energy import EnergyParams
from hopfion.field import hopfion, uniform
from hopfion.grid import Grid
from hopfion.physics.hessian import (
    SpectrumResult,
    hessian_vector_product,
    lowest_eigenmodes,
)


def test_uniform_easy_axis_is_stable():
    """Ferromagnet aligned with the easy axis is a true energy minimum --
    Hessian must be positive-semi-definite."""
    g = Grid(16, 16, 16, 0.5, 0.5, 0.5, "periodic")
    ep = EnergyParams(A_ex=1.0, D=0.0, Ku=0.5, easy_axis=(0, 0, 1), H_ext=(0, 0, 0.1))
    m = uniform(g, direction=(0, 0, 1))
    res = lowest_eigenmodes(m, g, ep, k=4, eigs_tol=1e-3)
    assert res.min_eig > -1e-6, f"uniform-FM should be stable, got eigs {res.eigenvalues}"
    assert not res.marginally_stable


def test_off_easy_axis_uniform_is_saddle():
    """Ferromagnet pointed away from easy axis is not stable -- Hessian
    must show a clearly negative eigenvalue."""
    g = Grid(16, 16, 16, 0.5, 0.5, 0.5, "periodic")
    ep = EnergyParams(A_ex=1.0, D=0.0, Ku=0.5, easy_axis=(0, 0, 1), H_ext=(0, 0, 0.1))
    m = uniform(g, direction=(1, 0, 0))   # 90° from easy axis
    res = lowest_eigenmodes(m, g, ep, k=4, eigs_tol=1e-3)
    assert res.min_eig < -0.1, f"tilted state should be unstable, got {res.min_eig}"


def test_hessian_vector_product_tangent_preserved():
    """H @ v should land in the tangent bundle (v · m = 0 after the matvec)."""
    rng = np.random.default_rng(0)
    g = Grid(10, 10, 10, 0.5, 0.5, 0.5, "periodic")
    ep = EnergyParams(A_ex=1.0, D=0.4, Ku=0.5)
    m = uniform(g, direction=(0, 0, 1))
    v = rng.normal(size=m.shape)
    Hv = hessian_vector_product(m, v, g, ep)
    Hv_dot_m = float(np.abs((Hv * np.asarray(m)).sum(axis=0)).max())
    assert Hv_dot_m < 1e-8, f"Hv has radial component {Hv_dot_m}"


def test_spectrum_result_to_metrics_dict():
    g = Grid(10, 10, 10, 0.5, 0.5, 0.5, "periodic")
    ep = EnergyParams(A_ex=1.0, D=0.4, Ku=0.5)
    m = uniform(g, direction=(0, 0, 1))
    res = lowest_eigenmodes(m, g, ep, k=3)
    d = res.to_metrics_dict()
    assert "min_eig" in d and "all_eigs" in d and "marginally_stable" in d
    assert len(d["all_eigs"]) == 3
