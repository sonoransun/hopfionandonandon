"""Zhang-Li spin-transfer torque: drives a hopfion in the direction of u."""
import numpy as np

from hopfion.energy import EnergyParams
from hopfion.field import hopfion
from hopfion.grid import Grid
from hopfion.physics.stt import STTParams, adv_grad_m, stt_step_heun


def test_advective_term_zero_at_uniform():
    g = Grid(16, 16, 16, 0.5, 0.5, 0.5, "periodic")
    m = np.ones((3, 16, 16, 16))                 # spatially constant
    m[:] = np.array([0, 0, 1]).reshape(3, 1, 1, 1)
    adv = np.asarray(adv_grad_m(m, g, u=(0.1, 0.0, 0.0)))
    assert np.max(np.abs(adv)) < 1e-12


def test_advective_term_matches_gradient_for_hopfion():
    """For non-trivial m, (u·∇)m should agree with the directional derivative."""
    g = Grid(24, 24, 24, 0.4, 0.4, 0.4, "periodic")
    m = hopfion(g, R=1.5, p=1, q=1)
    adv = np.asarray(adv_grad_m(m, g, u=(0.1, 0.0, 0.0)))
    # Independent check: derivative of m_x along x times u_x
    from hopfion.grid import d_axis
    dmx = np.asarray(d_axis(m[0], 0, g.dx, g)) * 0.1
    assert np.allclose(adv[0], dmx, atol=1e-12)


def test_stt_step_preserves_unit_norm():
    g = Grid(20, 20, 20, 0.5, 0.5, 0.5, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.0, Ku=0.3, easy_axis=(0, 0, 1))
    m = hopfion(g, R=1.5, p=1, q=1)
    stt = STTParams(u=(0.0, 0.0, 0.05), beta=0.04)
    for _ in range(10):
        m = stt_step_heun(m, g, ep, gamma=1.0, alpha=0.1, dt=0.002, stt=stt)
    n = np.sqrt((np.asarray(m) ** 2).sum(axis=0))
    assert np.max(np.abs(n - 1.0)) < 1e-10


def test_stt_step_drives_hopfion_toward_u_direction():
    """With strong STT and damping, the hopfion centroid drifts along u."""
    from hopfion.physics.current import centroids, hopf_charge_density
    g = Grid(40, 40, 40, 0.4, 0.4, 0.4, "periodic")
    ep = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7, easy_axis=(0, 0, 1))
    m = hopfion(g, R=1.5, p=1, q=1)
    stt = STTParams(u=(0.0, 0.0, 0.1), beta=0.04)
    c0 = centroids(hop_density := hopf_charge_density(m, g), g)
    z0 = c0[0].position[2]
    for _ in range(80):
        m = stt_step_heun(m, g, ep, gamma=1.0, alpha=0.5, dt=0.002, stt=stt)
    cf = centroids(hopf_charge_density(m, g), g)
    zf = cf[0].position[2]
    # Should have drifted in +z (u points +z) by some discernible amount.
    # Threshold chosen loose -- exact drift velocity depends on alpha/beta.
    assert zf > z0 + 0.02, f"hopfion drifted only {zf - z0:.3f} (expected > 0.02)"
