"""Twisted-bilayer sanity tests."""
import numpy as np

from hopfion.energy import EnergyParams
from hopfion.field import hopfion, uniform
from hopfion.grid import Grid
from hopfion.physics.bilayer import BilayerConfig, BilayerLLG, registry_field


def _setup():
    g = Grid(24, 24, 4, 0.4, 0.4, 0.4, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.0, Ku=0.5, easy_axis=(0, 0, 1))
    return g, ep


def test_registry_field_periodic_in_xy_z_invariant():
    g, _ = _setup()
    cfg = BilayerConfig(theta=0.2, a=4.0, J0=0.3)
    chi = np.asarray(registry_field(g, cfg))
    # z-independent (sum of cos(qx*x + qy*y))
    assert np.allclose(chi[..., 0], chi[..., 2])
    # bounded between -1 and 1 (mean of 3 cosines)
    assert chi.max() <= 1.0 + 1e-12
    assert chi.min() >= -1.0 - 1e-12


def test_zero_coupling_decouples_layers():
    """With J0 = 0 the two layers evolve completely independently."""
    g, ep = _setup()
    cfg = BilayerConfig(theta=0.1, a=4.0, J0=0.0)
    engine = BilayerLLG(ep1=ep, ep2=ep, cfg=cfg, dt=0.002)

    m1_init = hopfion(g, R=1.5, p=1, q=1)
    m2_init = uniform(g, direction=(0, 0, 1))
    m1, m2 = engine.relax(m1_init, m2_init, g, n_steps=20)

    # Compare with single-layer relax of same initial m1
    from hopfion.llg import relax
    m1_solo = relax(m1_init, g, ep, n_steps=20, dt=0.002)

    assert np.allclose(np.asarray(m1), np.asarray(m1_solo), atol=1e-10)


def test_coupling_drives_perpendicular_layers_to_align():
    """Starting perpendicular, strong J0 should drive the layers toward
    alignment (mediated by chi(r), so per-cell direction varies)."""
    g, ep = _setup()
    cfg = BilayerConfig(theta=0.5, a=4.0, J0=2.0)
    engine = BilayerLLG(ep1=ep, ep2=ep, cfg=cfg, dt=0.001)
    m1 = uniform(g, direction=(1, 0, 0))
    m2 = uniform(g, direction=(0, 1, 0))     # perpendicular
    init_dot = float((np.asarray(m1) * np.asarray(m2)).sum())   # 0
    m1, m2 = engine.relax(m1, m2, g, n_steps=400)
    final_dot = float((np.asarray(m1) * np.asarray(m2)).sum())
    # final cell-wise dot products are no longer uniformly zero
    assert abs(final_dot) > 0.5, f"expected coupling-induced alignment, got dot={final_dot}"


def test_moire_period_formula():
    cfg = BilayerConfig(theta=0.1, a=4.0)
    expected = 4.0 / (2.0 * np.sin(0.05))
    assert abs(cfg.moire_period - expected) < 1e-12


def test_norm_preserved_under_step():
    g, ep = _setup()
    cfg = BilayerConfig(theta=0.1, a=4.0, J0=0.3)
    engine = BilayerLLG(ep1=ep, ep2=ep, cfg=cfg, dt=0.002)
    m1 = hopfion(g, R=1.5, p=1, q=1)
    m2 = uniform(g)
    for _ in range(10):
        m1, m2 = engine.step(m1, m2, g, registry_field(g, cfg))
    for layer in (m1, m2):
        n = np.sqrt((np.asarray(layer) ** 2).sum(axis=0))
        assert np.max(np.abs(n - 1.0)) < 1e-10
