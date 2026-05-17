"""Topological current + centroid drift tests."""
import numpy as np

from hopfion.field import hopfion
from hopfion.grid import Grid
from hopfion.physics.current import (
    Centroid,
    centroids,
    conservation_residual,
    drift_velocity,
    hopf_charge_density,
    topological_current,
)


def test_hopf_charge_density_integrates_to_hopf_index():
    g = Grid(48, 48, 48, 16.0 / 48, 16.0 / 48, 16.0 / 48, "periodic")
    m = hopfion(g, R=1.0, p=1, q=1)
    rho = hopf_charge_density(m, g)
    Q = float(np.asarray(rho).sum() * g.dV)
    assert abs(Q - 1.0) < 0.05


def test_rigid_translation_per_plane_flux_positive_along_motion():
    """For a +z-translating Q=+1 hopfion, the per-plane J_z flux at z=0
    is positive (charge crosses the plane in +z direction). On a periodic
    grid the volume-integrated mean of J_z is identically zero (the current
    is purely curl-free by construction); per-plane fluxes are the
    gauge-invariant transport diagnostic.
    """
    g = Grid(48, 48, 48, 0.4, 0.4, 0.4, "periodic")
    R = 1.5
    v = np.array([0.0, 0.0, 0.05])
    dt = 0.5
    m0 = hopfion(g, R=R, p=1, q=1, center=(0.0, 0.0, 0.0))
    m1 = hopfion(g, R=R, p=1, q=1, center=tuple(v * dt))

    J = np.asarray(topological_current(m0, m1, g, dt))
    z0 = g.nz // 2
    plane_flux = float(J[2, :, :, z0].sum() * g.dx * g.dy)
    assert plane_flux > 0, f"expected positive z-flux at z=0, got {plane_flux:.3e}"

    # Negative v_z should reverse the sign
    m2 = hopfion(g, R=R, p=1, q=1, center=tuple(-v * dt))
    J_rev = np.asarray(topological_current(m0, m2, g, dt))
    plane_flux_rev = float(J_rev[2, :, :, z0].sum() * g.dx * g.dy)
    assert plane_flux_rev < 0
    # Magnitudes should match (rigid-translation symmetry)
    assert abs(abs(plane_flux) - abs(plane_flux_rev)) < 0.1 * abs(plane_flux)


def test_conservation_residual_bounded():
    """For a rigid translation, ∂_t ρ + ∇·J should be ≈ 0 (continuity)."""
    g = Grid(48, 48, 48, 0.4, 0.4, 0.4, "periodic")
    R = 1.5; v = np.array([0.0, 0.0, 0.03]); dt = 0.3
    m0 = hopfion(g, R=R, p=1, q=1, center=(0.0, 0.0, 0.0))
    m1 = hopfion(g, R=R, p=1, q=1, center=tuple(v * dt))
    residual = np.asarray(conservation_residual(m0, m1, g, dt))
    # the residual magnitude should be small compared to typical |∂_t ρ| scale
    rms_residual = float(np.sqrt((residual ** 2).mean()))
    # finite-difference + finite-resolution noise floor -- not zero
    assert rms_residual < 0.5, f"conservation residual RMS = {rms_residual:.3e}"


def test_centroid_single_hopfion_at_offset():
    g = Grid(48, 48, 48, 0.4, 0.4, 0.4, "periodic")
    target = (1.2, -0.8, 0.4)
    m = hopfion(g, R=1.0, p=1, q=1, center=target)
    rho = hopf_charge_density(m, g)
    cs = centroids(rho, g, threshold_rel=0.2)
    assert len(cs) == 1
    pos = np.asarray(cs[0].position)
    assert np.linalg.norm(pos - np.asarray(target)) < 0.5, (
        f"centroid {pos} far from target {target}"
    )


def test_drift_velocity_recovers_known_motion():
    g = Grid(36, 36, 36, 0.4, 0.4, 0.4, "periodic")
    R = 1.0; v = np.array([0.0, 0.0, 0.06])
    history = []
    times = []
    for k in range(5):
        t = k * 0.5
        m = hopfion(g, R=R, p=1, q=1, center=tuple(v * t))
        history.append(centroids(hopf_charge_density(m, g), g))
        times.append(t)
    vels = drift_velocity(history, times)
    assert len(vels) > 0 and len(vels[0]) > 0
    recovered = np.asarray(vels[0][0])
    # Sub-grid centroid resolution at dx=0.4, R=1.0 limits accuracy
    assert abs(recovered[2] - v[2]) / v[2] < 0.4, f"recovered v_z = {recovered}"
    # Direction should be correct (positive +z)
    assert recovered[2] > 0
    assert abs(recovered[0]) < 0.01 and abs(recovered[1]) < 0.01
