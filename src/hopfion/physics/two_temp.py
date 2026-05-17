"""Two-temperature stochastic LLG (femtosecond-laser nucleation).

A femtosecond optical pulse deposits energy in the electron bath; electrons
transfer it to the lattice via electron-phonon coupling on a picosecond
timescale. The hot electron bath drives a thermal stochastic field in the LLG
equation, producing the chaotic-then-cooling trajectory that experimentally
nucleates magnetic hopfions.

Equations (in our normalized units, μ₀ M_s = γ = 1):

    C_e dT_e/dt = -G_el (T_e - T_l) + P(t)               (electron bath)
    C_l dT_l/dt =  G_el (T_e - T_l)                       (lattice bath)
    P(t) = (T_e^peak C_e / τ √π) exp(-(t - t₀)² / τ²)     (laser absorption)

The stochastic field for sLLG has variance set by fluctuation-dissipation
against the *electron* temperature:

    ⟨η_i(r, t) η_j(r', t')⟩ = (2 α T_e(t) / (Δt ΔV)) δ_ij δ_rr' δ_tt'

Integration uses Heun-Stratonovich (same noise sample in the predictor and
corrector half-steps), which is the canonical convention for thermal sLLG.

Phase-A's white-noise burst (``recipes/laser_nucleation.yaml``) was an
isothermal stand-in; this module replaces it with a temperature trajectory
driven by laser fluence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as _np

from hopfion.energy import EnergyParams
from hopfion.grid import Grid
from hopfion.physics.integrators import heun_step


_SQRT_PI = float(_np.sqrt(_np.pi))


@dataclass
class TwoTemperatureLLG:
    """Two-temperature ODE coupled to stochastic LLG.

    Parameters
    ----------
    Te_peak : float
        Approximate peak electron temperature (in the same energy units as the
        material parameters). With ``C_e = G_el = 1`` and a short pulse, the
        actual peak is close to ``Te_peak``.
    tau : float
        Pulse 1/e half-width (time).
    t0 : float
        Pulse center time.
    G_el, C_e, C_l : float
        Electron-lattice coupling, electron heat capacity, lattice heat capacity.
    alpha : float
        Gilbert damping (also sets noise amplitude via FDT).
    gamma : float
        Gyromagnetic ratio.
    """

    Te_peak: float = 15.0
    tau: float = 0.05
    t0: float = 0.1
    G_el: float = 1.0
    C_e: float = 1.0
    C_l: float = 1.0
    alpha: float = 0.1
    gamma: float = 1.0

    # ------------------------------------------------------------------
    # Bath dynamics
    # ------------------------------------------------------------------

    def laser_power(self, t: float) -> float:
        """Pulse power P(t). Calibrated so the peak T_e on a closed bath is
        ≈ Te_peak (heuristic; the actual peak depends on G_el and C_e)."""
        amp = self.Te_peak * self.C_e / (self.tau * _SQRT_PI)
        return float(amp * _np.exp(-((t - self.t0) ** 2) / (self.tau ** 2)))

    def step_temperatures(self, T_e: float, T_l: float, dt: float, t: float):
        """Forward Euler step. The bath ODE is well-conditioned at the
        timescales used here, so explicit is fine."""
        P = self.laser_power(t)
        dTe = (-self.G_el * (T_e - T_l) + P) / self.C_e
        dTl = self.G_el * (T_e - T_l) / self.C_l
        return T_e + dt * dTe, T_l + dt * dTl

    # ------------------------------------------------------------------
    # sLLG driver
    # ------------------------------------------------------------------

    def stochastic_field(self, T_e: float, dt: float, grid: Grid,
                         rng: _np.random.Generator):
        """Sample a white-noise stochastic field of variance set by T_e."""
        if T_e <= 0.0:
            return None
        sigma = float((2.0 * self.alpha * T_e / (dt * grid.dV)) ** 0.5)
        noise = rng.normal(size=(3, grid.nx, grid.ny, grid.nz)) * sigma
        return _np.asarray(noise)

    def run(self, m, grid: Grid, ep: EnergyParams, n_steps: int, dt: float,
            rng: _np.random.Generator,
            step_callback: Optional[Callable] = None,
            record_temperatures: bool = False):
        """Co-evolve T_e, T_l and m for ``n_steps`` of ``dt``.

        Returns the final field. If ``record_temperatures`` is true, also
        appends an ``(Te, Tl)`` series to the callback context as the ``t``
        argument when applicable (the callback receives ``(m, step, t)``).
        """
        T_e = 0.0
        T_l = 0.0
        t = 0.0
        Te_history: List[float] = []
        Tl_history: List[float] = []
        for k in range(n_steps):
            T_e, T_l = self.step_temperatures(T_e, T_l, dt, t)
            T_e = max(T_e, 0.0)
            T_l = max(T_l, 0.0)
            noise = self.stochastic_field(T_e, dt, grid, rng)
            if noise is None:
                H_extra = None
            else:
                # Heun-Stratonovich: same noise sample at both half-steps. The
                # closure captures the noise array, so heun_step's two calls
                # to H_extra return identical fields.
                H_extra = (lambda _m, _N=noise: _N)
            m = heun_step(m, grid, ep, self.gamma, self.alpha, dt, H_extra=H_extra)
            t += dt
            if record_temperatures:
                Te_history.append(T_e)
                Tl_history.append(T_l)
            if step_callback is not None:
                step_callback(m, k, t)
        if record_temperatures:
            self.last_Te = _np.asarray(Te_history)
            self.last_Tl = _np.asarray(Tl_history)
        return m


__all__ = ["TwoTemperatureLLG"]
