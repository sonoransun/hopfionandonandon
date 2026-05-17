"""Stability and lifetime analysis utilities.

* ``perturb_and_relax``: jolt a relaxed state and check that it returns to a
  similar configuration (same Q_H within tolerance).
* ``thermal_lifetime``: integrate damped LLG with white-noise field (a poor
  man's sLLG) and track Q_H drift over time. Returns a lifetime estimate (the
  number of steps before Q_H drifts by >= 0.5).
* ``hessian_eigenmode_spectrum``: stub for Phase B.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as _np

from hopfion.backend import to_numpy, xp
from hopfion.energy import EnergyParams, total_energy
from hopfion.field import add_perturbation
from hopfion.grid import Grid, normalize
from hopfion.llg import LLGParams, relax_step, llg_step_heun, _cross
from hopfion.topology import hopf_index


@dataclass
class StabilityResult:
    Q_initial: float
    Q_final: float
    E_initial: float
    E_final: float
    steps_run: int


def perturb_and_relax(
    m,
    grid: Grid,
    ep: EnergyParams,
    amplitude: float = 0.05,
    n_steps: int = 500,
    dt: float = 0.01,
    seed: int = 0,
) -> StabilityResult:
    Q0 = hopf_index(m, grid)
    E0 = total_energy(m, grid, ep)
    m_perturbed = add_perturbation(m, amplitude=amplitude, seed=seed)
    for _ in range(n_steps):
        m_perturbed = relax_step(m_perturbed, grid, ep, dt=dt)
    return StabilityResult(
        Q_initial=Q0,
        Q_final=hopf_index(m_perturbed, grid),
        E_initial=E0,
        E_final=total_energy(m_perturbed, grid, ep),
        steps_run=n_steps,
    )


def thermal_lifetime(
    m,
    grid: Grid,
    ep: EnergyParams,
    lp: LLGParams,
    kT: float,
    n_steps: int = 2000,
    seed: int = 0,
    Q_tol: float = 0.5,
):
    """Run LLG with white-noise stochastic field of strength ``sqrt(kT/dV)``.

    Returns ``(history_Q, lifetime_steps)``. ``lifetime_steps`` is the first
    step at which ``|Q_H - Q0| > Q_tol``, or ``n_steps`` if the topology
    survives the whole run.

    This is a Phase A approximation -- proper sLLG uses Heun-Stratonovich with
    fluctuation-dissipation-tied noise. See Phase B for the upgraded version.
    """
    np = xp()
    rng = _np.random.default_rng(seed)
    Q0 = hopf_index(m, grid)
    history = [Q0]
    lifetime = n_steps
    sigma = (2.0 * lp.alpha * kT / (lp.dt * grid.dV)) ** 0.5
    for step in range(1, n_steps + 1):
        noise = rng.normal(size=tuple([3, *grid.shape])) * sigma
        H_noise = np.asarray(noise)
        # add noise as an extra field via H_extra callback
        def extra(_m, _H=H_noise):
            return _H
        m = llg_step_heun(m, grid, ep, lp, H_extra=lambda x: H_noise)
        Q = hopf_index(m, grid)
        history.append(Q)
        if lifetime == n_steps and abs(Q - Q0) > Q_tol:
            lifetime = step
    return history, lifetime
