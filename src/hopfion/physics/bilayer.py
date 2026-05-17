"""Twisted-bilayer micromagnetics with moiré-induced interlayer coupling.

Two spin fields $\\mathbf{m}^{(1)}(\\mathbf{r}), \\mathbf{m}^{(2)}(\\mathbf{r})$
on the same 3D grid (depth axis used for sub-lattice resolution within each
layer). The two layers are conceptually rotated by twist angle $\\theta$
relative to each other; the resulting *moiré registry function* enters as a
position-dependent inter-layer exchange:

    E_inter = -J_0 \\int \\chi(\\mathbf{r}; \\theta)\\, \\mathbf{m}^{(1)} \\cdot \\mathbf{m}^{(2)}\\, d^3 r

The simplest form of $\\chi$ for triangular layers is a 3-cosine sum on the
moiré reciprocal vectors $\\mathbf{q}_i$:

    \\chi(\\mathbf{r}; \\theta) = \\tfrac{1}{3} \\sum_{i=1,2,3} \\cos(\\mathbf{q}_i \\cdot \\mathbf{r})

where $|\\mathbf{q}_i| = (4\\pi)/(a\\sqrt{3}) \\cdot 2\\sin(\\theta/2)$.

This is the physically motivated successor to ``moire.MoirePotential``. The
Phase-A toy potential gave each cell a scalar anisotropy modulation; here we
get a fully *vector* interlayer coupling. Setting $J_0 = 0$ recovers two
independent single-layer runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as _np

from hopfion.backend import xp
from hopfion.energy import EnergyParams, effective_field
from hopfion.grid import Grid, normalize
from hopfion.physics.integrators import _cross, llg_rhs


@dataclass
class BilayerConfig:
    """Geometric + coupling parameters for the twisted bilayer."""
    theta: float = 0.1            # twist angle in radians
    a: float = 4.0                # intra-layer lattice constant
    J0: float = 0.3               # interlayer-exchange strength

    @property
    def moire_period(self) -> float:
        """a / (2 sin(theta/2))."""
        return self.a / (2.0 * math.sin(self.theta / 2.0))


def registry_field(grid: Grid, cfg: BilayerConfig):
    """Compute $\\chi(\\mathbf{r}; \\theta)$ on the simulation grid.

    Uses the triangular-star moiré reciprocal vectors derived from the twist.
    """
    np = xp()
    X, Y, _Z = grid.coords()
    # Magnitude of the moiré reciprocal vector
    q_mag = (4.0 * math.pi) / (cfg.a * math.sqrt(3.0)) * 2.0 * math.sin(cfg.theta / 2.0)
    # Three moiré q vectors at 0°, 120°, 240° in the xy plane
    chi = np.zeros_like(np.asarray(X))
    for k in range(3):
        ang = 2.0 * math.pi * k / 3.0
        qx = q_mag * math.cos(ang)
        qy = q_mag * math.sin(ang)
        chi = chi + np.cos(qx * X + qy * Y)
    return chi / 3.0


def interlayer_field(m_other, chi, J0: float):
    """Inter-layer effective field acting on layer 1 (call with m_other = m2).

    $H^{(1)}_{\\rm inter} = J_0\\, \\chi(\\mathbf{r})\\, \\mathbf{m}^{(2)}$
    (the field is along the *other* layer's magnetization, weighted by chi).
    """
    return J0 * chi * m_other   # broadcasts over the 3-vector axis


@dataclass
class BilayerLLG:
    """Two-layer LLG with interlayer-exchange coupling."""
    ep1: EnergyParams
    ep2: EnergyParams
    cfg: BilayerConfig
    gamma: float = 1.0
    alpha: float = 0.1
    dt: float = 0.002

    def step(self, m1, m2, grid: Grid, chi):
        """One Heun step on the joint (m1, m2) state."""
        # k1
        H1 = effective_field(m1, grid, self.ep1) + interlayer_field(m2, chi, self.cfg.J0)
        H2 = effective_field(m2, grid, self.ep2) + interlayer_field(m1, chi, self.cfg.J0)
        k1_1 = llg_rhs(m1, H1, self.gamma, self.alpha)
        k1_2 = llg_rhs(m2, H2, self.gamma, self.alpha)
        # predict
        m1_p = normalize(m1 + self.dt * k1_1)
        m2_p = normalize(m2 + self.dt * k1_2)
        # k2 at predicted state
        H1p = effective_field(m1_p, grid, self.ep1) + interlayer_field(m2_p, chi, self.cfg.J0)
        H2p = effective_field(m2_p, grid, self.ep2) + interlayer_field(m1_p, chi, self.cfg.J0)
        k2_1 = llg_rhs(m1_p, H1p, self.gamma, self.alpha)
        k2_2 = llg_rhs(m2_p, H2p, self.gamma, self.alpha)
        return (normalize(m1 + 0.5 * self.dt * (k1_1 + k2_1)),
                normalize(m2 + 0.5 * self.dt * (k1_2 + k2_2)))

    def relax_step(self, m1, m2, grid: Grid, chi):
        """One step of damped (no precession) bilayer descent."""
        H1 = effective_field(m1, grid, self.ep1) + interlayer_field(m2, chi, self.cfg.J0)
        H2 = effective_field(m2, grid, self.ep2) + interlayer_field(m1, chi, self.cfg.J0)
        rhs1 = -_cross(m1, _cross(m1, H1))
        rhs2 = -_cross(m2, _cross(m2, H2))
        return normalize(m1 + self.dt * rhs1), normalize(m2 + self.dt * rhs2)

    def relax(self, m1, m2, grid: Grid, n_steps: int = 200, chi=None):
        if chi is None:
            chi = registry_field(grid, self.cfg)
        for _ in range(n_steps):
            m1, m2 = self.relax_step(m1, m2, grid, chi)
        return m1, m2

    def integrate(self, m1, m2, grid: Grid, n_steps: int = 200, chi=None,
                  step_callback: Optional[Callable] = None):
        if chi is None:
            chi = registry_field(grid, self.cfg)
        t = 0.0
        for k in range(n_steps):
            m1, m2 = self.step(m1, m2, grid, chi)
            t += self.dt
            if step_callback is not None:
                step_callback(m1, m2, k, t)
        return m1, m2


__all__ = ["BilayerConfig", "BilayerLLG", "registry_field", "interlayer_field"]
