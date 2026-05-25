"""In-line metric callbacks.

Pluggable collectors attached to the LLG inner loop via ``step_callback``.
Each metric has a ``cadence`` (collect every N steps) and a ``summary()`` that
the runner serializes to JSON.

The pattern: ``runner`` builds a list of metrics, wraps them in a single
``MetricCollector`` callback, passes that to ``llg.integrate`` / ``llg.relax``.
After the run, ``collector.summarize()`` yields a dict that the QC framework
inspects.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as _np

from hopfion.backend import to_numpy
from hopfion.energy import EnergyParams, total_energy
from hopfion.grid import Grid
from hopfion.topology import hopf_index, skyrmion_number


@dataclass
class MetricContext:
    """Everything a metric might want at collection time."""
    m: Any            # current field
    grid: Grid
    ep: EnergyParams
    step: int
    t: float
    phase: str = ""   # which run-step kind is active


# ---------------------------------------------------------------------------
# Base + concrete metrics
# ---------------------------------------------------------------------------


class Metric:
    """Override ``collect`` and ``summary``. ``cadence`` lets you sub-sample."""
    name: str = ""
    cadence: int = 1

    def collect(self, ctx: MetricContext) -> None:
        raise NotImplementedError

    def summary(self) -> Dict[str, Any]:
        raise NotImplementedError


class EnergyMonotonicity(Metric):
    name = "energy"
    cadence = 1

    def __init__(self):
        self.history: List[float] = []
        self.steps: List[int] = []

    def collect(self, ctx):
        self.history.append(total_energy(ctx.m, ctx.grid, ctx.ep))
        self.steps.append(ctx.step)

    def summary(self):
        arr = _np.asarray(self.history)
        if arr.size < 2:
            return {"E_initial": float(arr[0]) if arr.size else 0.0,
                    "E_final": float(arr[-1]) if arr.size else 0.0,
                    "max_increase": 0.0, "mean_step": 0.0, "n_samples": int(arr.size)}
        diffs = _np.diff(arr)
        return {
            "E_initial": float(arr[0]),
            "E_final": float(arr[-1]),
            "max_increase": float(diffs.max()),
            "mean_step": float(diffs.mean()),
            "n_samples": int(arr.size),
        }


class NormDrift(Metric):
    name = "norm_drift"
    cadence = 1

    def __init__(self):
        self.history: List[float] = []
        self.steps: List[int] = []

    def collect(self, ctx):
        m = to_numpy(ctx.m)
        n = _np.sqrt((m * m).sum(axis=0))
        self.history.append(float(_np.max(_np.abs(n - 1.0))))
        self.steps.append(ctx.step)

    def summary(self):
        arr = _np.asarray(self.history)
        return {"max_drift": float(arr.max()) if arr.size else 0.0,
                "final_drift": float(arr[-1]) if arr.size else 0.0,
                "n_samples": int(arr.size)}


class HopfIndexDrift(Metric):
    """Sub-sampled because each call is an FFT."""
    name = "q_hopf"

    def __init__(self, cadence: int = 25):
        self.cadence = cadence
        self.history: List[float] = []
        self.steps: List[int] = []

    def collect(self, ctx):
        self.history.append(hopf_index(ctx.m, ctx.grid))
        self.steps.append(ctx.step)

    def summary(self):
        arr = _np.asarray(self.history)
        if arr.size == 0:
            return {"Q_initial": 0.0, "Q_final": 0.0, "max_drift": 0.0, "n_samples": 0}
        return {
            "Q_initial": float(arr[0]),
            "Q_final": float(arr[-1]),
            "max_drift": float(_np.max(_np.abs(arr - arr[0]))),
            "n_samples": int(arr.size),
        }


class SkyrmionChargeDrift(Metric):
    """Track the 2D skyrmion (Pontryagin) number over time.

    The skyrmion analogue of ``HopfIndexDrift``. Cheap (finite-difference,
    no FFT), so it could run every step, but defaults to the same sub-sampling
    cadence for symmetry with the Hopf tracker.
    """
    name = "q_skyrmion"

    def __init__(self, cadence: int = 25):
        self.cadence = cadence
        self.history: List[float] = []
        self.steps: List[int] = []

    def collect(self, ctx):
        self.history.append(skyrmion_number(ctx.m, ctx.grid))
        self.steps.append(ctx.step)

    def summary(self):
        arr = _np.asarray(self.history)
        if arr.size == 0:
            return {"N_initial": 0.0, "N_final": 0.0, "max_drift": 0.0, "n_samples": 0}
        return {
            "N_initial": float(arr[0]),
            "N_final": float(arr[-1]),
            "max_drift": float(_np.max(_np.abs(arr - arr[0]))),
            "n_samples": int(arr.size),
        }


class BlochPointCount(Metric):
    """Track the number of Bloch points (emergent monopoles) over time — the
    singularities through which topology changes. A clean hopfion has zero;
    a count spike flags a topological transition."""
    name = "bloch_points"

    def __init__(self, cadence: int = 25):
        self.cadence = cadence
        self.history: List[int] = []
        self.steps: List[int] = []

    def collect(self, ctx):
        from hopfion.topology import bloch_points
        self.history.append(len(bloch_points(ctx.m, ctx.grid)))
        self.steps.append(ctx.step)

    def summary(self):
        if not self.history:
            return {"n_initial": 0, "n_final": 0, "max_count": 0, "n_samples": 0}
        return {
            "n_initial": int(self.history[0]),
            "n_final": int(self.history[-1]),
            "max_count": int(max(self.history)),
            "n_samples": len(self.history),
        }


class Runtime(Metric):
    name = "runtime"
    cadence = 1

    def __init__(self):
        self.t0: Optional[float] = None
        self.per_step: List[float] = []
        self._last: Optional[float] = None

    def collect(self, ctx):
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now
        if self._last is not None:
            self.per_step.append(now - self._last)
        self._last = now

    def summary(self):
        if not self.per_step:
            return {"total_seconds": 0.0, "mean_step_ms": 0.0, "n_samples": 0}
        arr = _np.asarray(self.per_step)
        return {
            "total_seconds": float(arr.sum()),
            "mean_step_ms": float(arr.mean() * 1000),
            "p95_step_ms": float(_np.percentile(arr, 95) * 1000),
            "n_samples": int(arr.size),
        }


class FluxAccumulation(Metric):
    """Track integrated continuity-equation residual over time.

    Samples J^μ and ∇·J between adjacent snapshots; records the RMS continuity
    residual ``∂_t ρ + ∇·J``. For a well-resolved smooth field this should
    stay near floating-point precision; spurious drift indicates either a
    discretization issue or a topological event (Bloch point passing).
    """
    name = "flux"

    def __init__(self, cadence: int = 25):
        self.cadence = cadence
        self.residual_rms: List[float] = []
        self.steps: List[int] = []
        self._prev_m = None
        self._prev_t = None

    def collect(self, ctx):
        if self._prev_m is not None and ctx.t is not None and ctx.t > self._prev_t:
            from hopfion.physics.current import conservation_residual
            dt = ctx.t - self._prev_t
            r = _np.asarray(conservation_residual(self._prev_m, ctx.m, ctx.grid, dt))
            self.residual_rms.append(float(_np.sqrt((r * r).mean())))
            self.steps.append(ctx.step)
        # snapshot for next call
        self._prev_m = _np.asarray(ctx.m).copy()
        self._prev_t = ctx.t if ctx.t is not None else 0.0

    def summary(self):
        if not self.residual_rms:
            return {"max_residual_rms": 0.0, "mean_residual_rms": 0.0, "n_samples": 0}
        arr = _np.asarray(self.residual_rms)
        return {
            "max_residual_rms": float(arr.max()),
            "mean_residual_rms": float(arr.mean()),
            "n_samples": int(arr.size),
        }


class DriftVelocity(Metric):
    """Track per-hopfion centroid positions and (later) velocities.

    Stores the list-of-centroids snapshot every ``cadence`` steps. Summary
    reports the centroid count time-series and the most-recent positions.
    """
    name = "drift"

    def __init__(self, cadence: int = 25, threshold_rel: float = 0.1):
        self.cadence = cadence
        self.threshold_rel = threshold_rel
        self.snapshots: List[list] = []         # list of list of Centroid
        self.times: List[float] = []
        self.steps: List[int] = []

    def collect(self, ctx):
        from hopfion.physics.current import centroids, hopf_charge_density
        rho = hopf_charge_density(ctx.m, ctx.grid)
        cs = centroids(rho, ctx.grid, threshold_rel=self.threshold_rel)
        self.snapshots.append(cs)
        self.times.append(ctx.t if ctx.t is not None else float(ctx.step))
        self.steps.append(ctx.step)

    def summary(self):
        if not self.snapshots:
            return {"n_snapshots": 0, "n_centroids_initial": 0, "n_centroids_final": 0,
                    "max_drift_speed": 0.0}
        from hopfion.physics.current import drift_velocity
        n0 = len(self.snapshots[0])
        nf = len(self.snapshots[-1])
        max_speed = 0.0
        if len(self.snapshots) >= 3:
            try:
                vels = drift_velocity(self.snapshots, self.times)
                for snap in vels:
                    for v in snap:
                        s = float(_np.sqrt(sum(x * x for x in v)))
                        if s > max_speed:
                            max_speed = s
            except Exception:
                pass
        return {
            "n_snapshots": len(self.snapshots),
            "n_centroids_initial": int(n0),
            "n_centroids_final": int(nf),
            "max_drift_speed": float(max_speed),
        }


class PerSiteQ(Metric):
    """For lattice initial states: track the signed charge of each segmented
    cluster (assumed to correspond to a per-site hopfion). Useful for
    stabilizer-style error correction."""
    name = "per_site_q"

    def __init__(self, cadence: int = 25, threshold_rel: float = 0.1):
        self.cadence = cadence
        self.threshold_rel = threshold_rel
        self.q_per_site_history: List[List[float]] = []
        self.steps: List[int] = []

    def collect(self, ctx):
        from hopfion.physics.current import centroids, hopf_charge_density
        rho = hopf_charge_density(ctx.m, ctx.grid)
        cs = centroids(rho, ctx.grid, threshold_rel=self.threshold_rel)
        self.q_per_site_history.append([float(c.charge) for c in cs])
        self.steps.append(ctx.step)

    def summary(self):
        if not self.q_per_site_history:
            return {"n_snapshots": 0}
        initial = self.q_per_site_history[0]
        final = self.q_per_site_history[-1]
        return {
            "n_snapshots": len(self.q_per_site_history),
            "n_sites_initial": len(initial),
            "n_sites_final": len(final),
            "Q_per_site_initial": initial,
            "Q_per_site_final": final,
        }


class PerSiteQVoronoi(Metric):
    """Per-lattice-site Hopf charge by Voronoi-zone integration.

    Unlike ``PerSiteQ`` (which segments charge blobs without reference to the
    intended lattice), this assigns every cell to its nearest lattice ``site``
    and integrates ``rho_Q`` per zone. The result is aligned to the fixed site
    list, so a site that loses its hopfion shows up as a specific zone going to
    ~0 — the natural signal for stabilizer-style QC.
    """
    name = "per_site_q_voronoi"

    def __init__(self, sites, cadence: int = 25):
        self.sites = list(sites)
        self.cadence = cadence
        self.history: List[List[float]] = []
        self.steps: List[int] = []

    def collect(self, ctx):
        from hopfion.physics.current import hopf_charge_density, per_site_charges
        rho = hopf_charge_density(ctx.m, ctx.grid)
        self.history.append(per_site_charges(rho, ctx.grid, self.sites))
        self.steps.append(ctx.step)

    def summary(self):
        if not self.history:
            return {"n_sites": len(self.sites), "n_snapshots": 0}
        initial = self.history[0]
        final = self.history[-1]
        return {
            "n_sites": len(self.sites),
            "n_snapshots": len(self.history),
            "Q_per_site_initial": initial,
            "Q_per_site_final": final,
        }


# ---------------------------------------------------------------------------
# Collector — wraps a list of metrics into one callback
# ---------------------------------------------------------------------------


@dataclass
class MetricCollector:
    metrics: List[Metric] = field(default_factory=list)
    grid: Optional[Grid] = None
    ep: Optional[EnergyParams] = None
    phase: str = ""

    def callback(self, m, step, t=None):
        ctx = MetricContext(m=m, grid=self.grid, ep=self.ep,
                            step=step, t=t if t is not None else 0.0, phase=self.phase)
        for met in self.metrics:
            if step % met.cadence == 0:
                met.collect(ctx)

    def relax_callback(self, m, step):
        """Adapter for relax(step_callback=...) which omits t."""
        self.callback(m, step, t=None)

    def integrate_callback(self, m, step, t):
        """Adapter for integrate(step_callback=...)."""
        self.callback(m, step, t=t)

    def summarize(self) -> Dict[str, Any]:
        return {m.name: m.summary() for m in self.metrics}

    def histories(self) -> Dict[str, Dict[str, List]]:
        """Return time-series for plotting in the report. Skips metrics that
        don't have a ``history`` list (e.g. Runtime, which has ``per_step``)."""
        out: Dict[str, Dict[str, List]] = {}
        for met in self.metrics:
            if hasattr(met, "history") and hasattr(met, "steps"):
                out[met.name] = {"steps": list(met.steps), "values": list(met.history)}
        return out


def default_metrics(hopf_cadence: int = 25) -> List[Metric]:
    return [EnergyMonotonicity(), NormDrift(), HopfIndexDrift(cadence=hopf_cadence), Runtime()]


def extended_metrics(hopf_cadence: int = 25, drift_cadence: int = 25) -> List[Metric]:
    """default + FluxAccumulation + DriftVelocity + PerSiteQ.
    Use for composite-state recipes where flux/drift are meaningful."""
    return [
        EnergyMonotonicity(), NormDrift(), HopfIndexDrift(cadence=hopf_cadence),
        SkyrmionChargeDrift(cadence=hopf_cadence),
        Runtime(),
        FluxAccumulation(cadence=drift_cadence),
        DriftVelocity(cadence=drift_cadence),
        PerSiteQ(cadence=drift_cadence),
    ]


__all__ = [
    "Metric",
    "MetricContext",
    "MetricCollector",
    "EnergyMonotonicity",
    "NormDrift",
    "HopfIndexDrift",
    "SkyrmionChargeDrift",
    "BlochPointCount",
    "Runtime",
    "FluxAccumulation",
    "DriftVelocity",
    "PerSiteQ",
    "PerSiteQVoronoi",
    "default_metrics",
    "extended_metrics",
]
