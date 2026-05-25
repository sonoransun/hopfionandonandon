"""Per-site Hopf charge by Voronoi zones + the matching QC criterion."""
from hopfion.grid import Grid
from hopfion.lattice import array_hopfion, triangular_sites_2d
from hopfion.physics.current import hopf_charge_density, per_site_charges
from hopfion.pipeline.qc import PerSiteVoronoiHealthCriterion
from hopfion.topology import hopf_index


def test_per_site_charges_sum_matches_hopf_index():
    g = Grid(64, 64, 32, 0.4, 0.4, 0.4, "periodic")
    sites = triangular_sites_2d(a=6.0, n_rings=1)
    m = array_hopfion(g, sites, R=1.2)
    rho = hopf_charge_density(m, g)
    q = per_site_charges(rho, g, sites)
    assert len(q) == len(sites) == 7
    # each occupied site carries ~ +1
    assert all(abs(x - 1.0) < 0.1 for x in q)
    # Voronoi zones tile the box, so the per-site sum is the system charge
    assert abs(sum(q) - hopf_index(m, g)) < 1e-6


def test_per_site_voronoi_health_criterion():
    crit = PerSiteVoronoiHealthCriterion(severity="warn", min_fidelity=0.9)
    healthy = {"per_site_q_voronoi": {"Q_per_site_initial": [1, 1, 1],
                                      "Q_per_site_final": [1.0, 1.0, 0.9]}}
    assert crit.evaluate(healthy).passed
    degraded = {"per_site_q_voronoi": {"Q_per_site_initial": [1, 1, 1],
                                       "Q_per_site_final": [1.0, 0.1, 0.1]}}
    assert not crit.evaluate(degraded).passed
    # skip-pass when the metric was not collected
    assert crit.evaluate({}).passed
