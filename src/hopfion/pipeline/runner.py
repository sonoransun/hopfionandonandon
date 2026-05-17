"""Single-run orchestrator.

``run(recipe)`` does:

    1. Pre-flight checks  → fail fast on misconfigured recipe.
    2. Build state        → Grid, EnergyParams (incl. moire Ku_field), initial m.
    3. Execute run steps  → relax / dynamics / thermal_burst / pulse, with
                            in-line metric callbacks.
    4. Post-flight QC     → evaluate acceptance criteria against summary.
    5. Write outputs      → HDF5 (m + snapshots), JSON sidecar (metrics + QC),
                            and (if requested) a Markdown report.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as _np

from hopfion.backend import to_numpy, use as use_backend
from hopfion.energy import EnergyParams
from hopfion.field import add_perturbation, hopfion, uniform
from hopfion.grid import Grid
from hopfion.laser import GaussianPulse
from hopfion.lattice import (
    array_hopfion,
    cubic_sites_3d,
    square_sites_2d,
    triangular_sites_2d,
)
from hopfion.llg import LLGParams, integrate, llg_step_heun, relax
from hopfion.moire import MoirePotential
from hopfion.pipeline import qc as qc_mod
from hopfion.pipeline.metrics import MetricCollector, default_metrics, extended_metrics
from hopfion.pipeline.recipe import RecipeConfig, RunStep, preflight
from hopfion.topology import hopf_index


@dataclass
class RunResult:
    recipe: RecipeConfig
    qc: qc_mod.QCReport
    metrics: Dict[str, Any]
    m_final: Any
    out_dir: str
    seconds: float
    histories: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# State builders
# ---------------------------------------------------------------------------


def build_grid(rc: RecipeConfig) -> Grid:
    g = rc.grid
    return Grid(g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.bc)


def build_energy_params(rc: RecipeConfig, grid: Grid) -> EnergyParams:
    mat = rc.material
    ku_field = None
    if mat.moire is not None and mat.moire.enabled:
        mp = MoirePotential(K0=mat.moire.K0, V0=mat.moire.V0,
                            a_moire=mat.moire.a_moire, lattice=mat.moire.lattice)
        ku_field = mp.Ku_field(grid)
    return EnergyParams(
        A_ex=mat.A_ex, D=mat.D, Ku=mat.Ku,
        easy_axis=mat.easy_axis, H_ext=mat.H_ext, Ku_field=ku_field,
    )


def build_initial_state(rc: RecipeConfig, grid: Grid):
    spec = rc.initial
    if spec.kind == "uniform":
        return uniform(grid, direction=spec.direction)
    if spec.kind == "perturbed_uniform":
        return add_perturbation(uniform(grid, direction=spec.direction),
                                amplitude=spec.amplitude, seed=rc.seed)
    if spec.kind == "hopfion":
        return hopfion(grid, R=spec.R, p=spec.p, q=spec.q,
                       center=spec.center, axis=spec.axis)
    if spec.kind == "hopfion_array":
        if spec.array_lattice == "triangular":
            sites = triangular_sites_2d(a=spec.array_a, n_rings=spec.array_n_rings)
        elif spec.array_lattice == "square":
            sites = square_sites_2d(a=spec.array_a, nx=spec.array_nx, ny=spec.array_ny)
        elif spec.array_lattice == "cubic":
            sites = cubic_sites_3d(a=spec.array_a,
                                   nx=spec.array_nx, ny=spec.array_ny, nz=spec.array_nx)
        else:
            raise ValueError(f"Unknown array_lattice {spec.array_lattice!r}")
        return array_hopfion(grid, sites, R=spec.R, axis=spec.axis)
    if spec.kind == "file":
        if not spec.file_path:
            raise ValueError("initial.kind='file' requires file_path")
        import h5py
        with h5py.File(spec.file_path, "r") as f:
            return _np.asarray(f["m"])
    if spec.kind == "q_pair":
        from hopfion.physics.composite import q_pair
        return q_pair(grid, R=spec.R, separation=spec.separation,
                      axis_of_separation=spec.axis_of_separation,
                      background=spec.direction)
    if spec.kind == "hopfion_skyrmion_hybrid":
        from hopfion.physics.composite import hopfion_skyrmion_hybrid
        return hopfion_skyrmion_hybrid(grid, R=spec.R,
                                       skyrmion_radius=spec.skyrmion_radius,
                                       skyrmion_helicity=spec.skyrmion_helicity,
                                       center=spec.center)
    if spec.kind == "skyrmion_tube":
        from hopfion.physics.composite import skyrmion_tube
        return skyrmion_tube(grid, radius=spec.skyrmion_radius,
                             helicity=spec.skyrmion_helicity)
    raise ValueError(f"Unknown initial.kind: {spec.kind!r}")


# ---------------------------------------------------------------------------
# Run-step execution
# ---------------------------------------------------------------------------


def _execute_step(m, grid: Grid, ep: EnergyParams, step: RunStep,
                  rc: RecipeConfig, collector: MetricCollector, rng: _np.random.Generator):
    """Execute one run step in place; returns the new m."""
    collector.phase = step.kind

    if step.kind == "relax":
        return relax(m, grid, ep, n_steps=step.n_steps, dt=step.dt,
                     step_callback=collector.relax_callback)

    if step.kind == "dynamics":
        lp = LLGParams(gamma=step.gamma, alpha=step.alpha, dt=step.dt)
        return integrate(m, grid, ep, lp, n_steps=step.n_steps,
                         step_callback=collector.integrate_callback)

    if step.kind == "pulse":
        lp = LLGParams(gamma=step.gamma, alpha=step.alpha, dt=step.dt)
        pulse = GaussianPulse(H0=step.H0, t0=step.t0, tau=step.tau,
                              profile=step.profile, width=step.pulse_width,
                              ring_radius=step.ring_radius)
        H_extra = pulse.field_factory(grid)
        return integrate(m, grid, ep, lp, n_steps=step.n_steps, H_extra=H_extra,
                         step_callback=collector.integrate_callback)

    if step.kind == "thermal_burst":
        # Phase-A white-noise burst stand-in for the Phase-B two-temperature engine.
        lp = LLGParams(gamma=step.gamma, alpha=step.alpha, dt=step.dt)
        sigma = (2.0 * lp.alpha * step.kT / (lp.dt * grid.dV)) ** 0.5
        for k in range(step.n_steps):
            noise = rng.normal(size=(3, grid.nx, grid.ny, grid.nz)) * sigma
            H_extra = (lambda _m, _N=noise: _N)
            m = llg_step_heun(m, grid, ep, lp, H_extra=H_extra)
            collector.integrate_callback(m, k, t=(k + 1) * lp.dt)
        return m

    if step.kind == "two_temperature":
        # Phase B2 hook -- routed via physics.two_temp once available.
        from hopfion.physics.two_temp import TwoTemperatureLLG
        engine = TwoTemperatureLLG(
            Te_peak=step.Te_peak, tau=step.tau, t0=step.t0,
            G_el=step.G_el, C_e=step.C_e, C_l=step.C_l,
            alpha=step.alpha, gamma=step.gamma,
        )
        return engine.run(m, grid, ep, n_steps=step.n_steps, dt=step.dt,
                          rng=rng, step_callback=collector.integrate_callback)

    if step.kind == "dynamics_stt":
        # C2: Zhang-Li spin-transfer-torque driven LLG
        from hopfion.physics.stt import STTParams, stt_step_heun
        stt = STTParams(u=step.u, beta=step.beta)
        t = 0.0
        for k in range(step.n_steps):
            m = stt_step_heun(m, grid, ep, step.gamma, step.alpha, step.dt, stt)
            t += step.dt
            collector.integrate_callback(m, k, t=t)
        return m

    raise ValueError(f"Unknown run-step kind: {step.kind!r}")


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _write_outputs(out_dir: Path, m_final, grid: Grid, ep: EnergyParams,
                   rc: RecipeConfig, qc_report: qc_mod.QCReport,
                   summaries: Dict[str, Any], histories: Dict[str, Any],
                   seconds: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # HDF5: final field + Ku_field if spatially varying
    import h5py
    h5_path = out_dir / "run.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("m", data=to_numpy(m_final))
        if ep.Ku_field is not None:
            f.create_dataset("Ku_field", data=to_numpy(ep.Ku_field))
        # grid + flat material attrs
        f.attrs.update(dict(
            nx=grid.nx, ny=grid.ny, nz=grid.nz,
            dx=grid.dx, dy=grid.dy, dz=grid.dz, bc=grid.bc,
            A_ex=ep.A_ex, D=ep.D, Ku=ep.Ku,
            easy_axis=_np.asarray(ep.easy_axis),
            H_ext=_np.asarray(ep.H_ext),
            recipe=rc.name, seed=rc.seed, backend=rc.backend,
        ))

    # JSON sidecar: recipe (round-tripable summary), metrics, QC, histories
    json_path = out_dir / "summary.json"
    payload = {
        "recipe": {
            "name": rc.name,
            "description": rc.description,
            "seed": rc.seed,
            "backend": rc.backend,
            "grid": rc.grid.__dict__,
            "material": {**rc.material.__dict__,
                         "moire": rc.material.moire.__dict__ if rc.material.moire else None},
            "initial": rc.initial.__dict__,
            "run": [s.__dict__ for s in rc.run],
        },
        "metrics": summaries,
        "histories": histories,
        "qc": qc_report.to_dict(),
        "Q_final": float(hopf_index(m_final, grid)),
        "wall_seconds": seconds,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def run(rc: RecipeConfig, write: bool = True, hopf_cadence: int = 25) -> RunResult:
    """Execute ``rc`` end-to-end. Returns a ``RunResult``.

    ``write=False`` skips HDF5/JSON output (useful for tests and batch runs that
    aggregate results without per-run files).
    """
    use_backend(rc.backend)

    # 1. Pre-flight
    pre = preflight(rc)
    if not pre.passed:
        qc_report = qc_mod.QCReport(
            verdict="FAIL", results=[],
            preflight_failures=pre.failures,
            preflight_warnings=pre.warnings,
        )
        return RunResult(recipe=rc, qc=qc_report, metrics={},
                         m_final=None, out_dir=rc.io.out, seconds=0.0)

    # 2. Build state
    grid = build_grid(rc)
    ep = build_energy_params(rc, grid)
    m = build_initial_state(rc, grid)

    # 3. Execute
    rng = _np.random.default_rng(rc.seed)
    metrics_list = (extended_metrics(hopf_cadence=hopf_cadence, drift_cadence=hopf_cadence)
                    if rc.io.extended_metrics
                    else default_metrics(hopf_cadence=hopf_cadence))
    collector = MetricCollector(metrics=metrics_list, grid=grid, ep=ep)
    # seed metrics with the initial state
    from hopfion.pipeline.metrics import MetricContext
    init_ctx = MetricContext(m=m, grid=grid, ep=ep, step=-1, t=0.0, phase="init")
    for met in collector.metrics:
        if hasattr(met, "history") or hasattr(met, "residual_rms") or hasattr(met, "snapshots") or hasattr(met, "q_per_site_history"):
            met.collect(init_ctx)

    t_start = time.perf_counter()
    for step in rc.run:
        m = _execute_step(m, grid, ep, step, rc, collector, rng)
    seconds = time.perf_counter() - t_start

    # 4. QC
    summaries = collector.summarize()
    qc_report = qc_mod.evaluate(rc.qc, summaries,
                                preflight_failures=pre.failures,
                                preflight_warnings=pre.warnings)

    # 5. Write outputs
    out_dir = Path(rc.io.out)
    if write:
        _write_outputs(out_dir, m, grid, ep, rc, qc_report, summaries,
                       collector.histories(), seconds)

    return RunResult(recipe=rc, qc=qc_report, metrics=summaries,
                     m_final=m, out_dir=str(out_dir), seconds=seconds,
                     histories=collector.histories())


__all__ = [
    "RunResult",
    "run",
    "build_grid",
    "build_energy_params",
    "build_initial_state",
]
