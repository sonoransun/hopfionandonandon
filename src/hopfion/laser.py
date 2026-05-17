"""Laser pulse models added as an extra field to the LLG effective field.

Phase A: classical Gaussian field pulse.

    H_laser(r, t) = H_0 * f(r) * exp(-(t - t0)^2 / tau^2)

with ``f(r)`` a spatial focal profile (point / disk / ring).

Phase B (later): two-temperature stochastic LLG noise driven by electron
temperature T_e(t). Stubbed below for the eventual interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

from hopfion.backend import xp
from hopfion.grid import Grid


@dataclass
class GaussianPulse:
    """Gaussian-in-time, spatially-shaped transient field."""

    H0: Tuple[float, float, float] = (0.0, 0.0, 1.0)   # peak field amplitude (vector)
    t0: float = 1.0                                    # pulse center time
    tau: float = 0.2                                   # 1/e half-width in time
    profile: str = "point"                             # 'point' | 'disk' | 'ring' | 'uniform'
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    width: float = 1.0                                 # spatial width of focal spot
    ring_radius: float = 1.0                           # for 'ring'

    def spatial_envelope(self, grid: Grid):
        np = xp()
        X, Y, Z = grid.coords()
        cx, cy, cz = self.center
        x, y, z = X - cx, Y - cy, Z - cz
        r2 = x * x + y * y + z * z
        rho2 = x * x + y * y
        w2 = self.width * self.width
        if self.profile == "point":
            return np.exp(-r2 / (2.0 * w2))
        if self.profile == "disk":
            return np.exp(-rho2 / (2.0 * w2)) * np.exp(-z * z / (2.0 * w2))
        if self.profile == "ring":
            rho = np.sqrt(rho2)
            return np.exp(-((rho - self.ring_radius) ** 2 + z * z) / (2.0 * w2))
        if self.profile == "uniform":
            return np.ones_like(r2)
        raise ValueError(f"Unknown focal profile {self.profile!r}")

    def field_factory(self, grid: Grid) -> Callable:
        """Return ``H(m, t) -> H_field`` ready to pass to ``llg.integrate``."""
        np = xp()
        envelope = self.spatial_envelope(grid)
        H0x, H0y, H0z = self.H0
        t0, tau = self.t0, self.tau

        def H_extra(m, t):
            time_amp = np.exp(-((t - t0) ** 2) / (tau * tau))
            field = envelope * time_amp
            return np.stack([H0x * field, H0y * field, H0z * field], axis=0)

        return H_extra


# Phase B implementation. Forwards to ``hopfion.physics.two_temp``; the legacy
# class is kept for back-compat (notebooks and callers that previously
# expected to import it from ``hopfion.laser``).
@dataclass
class TwoTemperaturePulse:
    """Two-temperature stochastic LLG laser pulse (Phase B).

    Solves coupled ODEs for electron and lattice temperatures and drives an
    sLLG noise field with FDT-consistent variance. Run via
    :meth:`run` rather than ``field_factory`` because the noise is sampled at
    each LLG step against the evolving electron temperature.
    """

    Te_peak: float = 15.0
    tau: float = 0.05
    t0: float = 0.1
    G_el: float = 1.0
    C_e: float = 1.0
    C_l: float = 1.0
    alpha: float = 0.1
    gamma: float = 1.0

    def engine(self):
        from hopfion.physics.two_temp import TwoTemperatureLLG
        return TwoTemperatureLLG(
            Te_peak=self.Te_peak, tau=self.tau, t0=self.t0,
            G_el=self.G_el, C_e=self.C_e, C_l=self.C_l,
            alpha=self.alpha, gamma=self.gamma,
        )

    def run(self, m, grid, ep, n_steps, dt, rng=None, step_callback=None,
            record_temperatures=False):
        import numpy as _np
        if rng is None:
            rng = _np.random.default_rng()
        return self.engine().run(m, grid, ep, n_steps=n_steps, dt=dt, rng=rng,
                                 step_callback=step_callback,
                                 record_temperatures=record_temperatures)

    def field_factory(self, grid: Grid, seed: int | None = None) -> Callable:
        """Not supported: the two-temperature pulse can't be a stateless H_extra
        because the noise variance evolves. Use :meth:`run` instead, or write a
        recipe with ``run: [{kind: two_temperature, ...}]``."""
        raise NotImplementedError(
            "TwoTemperaturePulse does not have a stateless field factory because "
            "the noise variance evolves with T_e(t). Call pulse.run(...) directly, "
            "or use the recipe schema's 'two_temperature' run-step kind."
        )
