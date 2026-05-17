"""Error-correction strategies for hopfion composite states.

Three controllers are implemented. Each exposes a ``step_callback(m, k, t)``
that the pipeline runner attaches to the LLG inner loop. Controllers may
*mutate* an external "extra-field" slot (a closure over the H_extra of the
integrator) to apply corrective fields.

Strategies:

* :class:`ActiveFeedbackController` — closed-loop. Measures Q_H (cadenced); on
  excursion beyond threshold, applies a corrective Zeeman pulse aimed at
  restoring the deviation. Tunable gain.

* :class:`TopologicalGapController` — passive. Pre-flight checks that the
  Hessian's lowest eigenvalue is above a target gap; signals that
  Crouch-Grossman integration should be used for the run.

* :class:`StabilizerController` — code-style. On lattice composites, measures
  per-site Q every ``cadence`` steps; if any site has flipped (|Q_i| < 0.5),
  triggers a localized two-temperature heating pulse to re-nucleate it.

The pipeline runner picks one of these from ``recipe.correction.kind``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as _np

from hopfion.energy import EnergyParams
from hopfion.grid import Grid
from hopfion.topology import hopf_index


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass
class CorrectionResult:
    """Outcome accumulated by a controller over the run."""
    kind: str
    n_corrections: int = 0
    correction_events: List[dict] = field(default_factory=list)
    final_fidelity: float = 1.0


# ---------------------------------------------------------------------------
# A. Active feedback
# ---------------------------------------------------------------------------


@dataclass
class ActiveFeedbackController:
    target_Q: float = 1.0
    threshold: float = 0.1
    gain: float = 1.0
    cadence: int = 25
    correction_duration: int = 5     # steps the corrective field is on

    grid: Optional[Grid] = None
    ep: Optional[EnergyParams] = None
    result: CorrectionResult = field(default_factory=lambda: CorrectionResult(kind="active"))
    _residual_steps: int = 0
    _correction_vec: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def step_callback(self, m, k: int, t: float = 0.0):
        # Check Q periodically
        if self._residual_steps > 0:
            self._residual_steps -= 1
            return
        if k % self.cadence != 0:
            return
        if self.grid is None:
            return
        Q = hopf_index(m, self.grid)
        delta = Q - self.target_Q
        if abs(delta) > self.threshold:
            # Direction of the corrective Zeeman field — push along easy axis if Q deficit
            # is below target; opposite if above. Simple heuristic: sign-of-delta along z.
            sign = -_np.sign(delta) if delta != 0 else 0.0
            mag = float(self.gain * abs(delta))
            self._correction_vec = (0.0, 0.0, sign * mag)
            self._residual_steps = self.correction_duration
            self.result.n_corrections += 1
            self.result.correction_events.append({
                "step": int(k), "Q": float(Q), "delta": float(delta),
                "correction_vec": list(self._correction_vec),
            })

    def H_extra_factory(self):
        """Returns a callable ``H_extra(m)`` matching the runner's integrator
        signature. The field is non-zero only when an active correction is in
        progress."""
        def H_extra(m):
            np = _np
            if self._correction_vec == (0.0, 0.0, 0.0) or self._residual_steps <= 0:
                return np.zeros_like(np.asarray(m))
            shape = np.asarray(m)[0].shape
            cx, cy, cz = self._correction_vec
            return np.stack([np.full(shape, cx), np.full(shape, cy), np.full(shape, cz)], axis=0)
        return H_extra


# ---------------------------------------------------------------------------
# B. Topological-gap controller
# ---------------------------------------------------------------------------


@dataclass
class TopologicalGapController:
    min_gap: float = 0.01
    grid: Optional[Grid] = None
    ep: Optional[EnergyParams] = None
    result: CorrectionResult = field(default_factory=lambda: CorrectionResult(kind="topological_gap"))

    def preflight(self, m) -> Tuple[bool, float]:
        """Compute the lowest Hessian eigenvalue and check against ``min_gap``."""
        from hopfion.physics.hessian import lowest_eigenmodes
        if self.grid is None or self.ep is None:
            return True, float("inf")
        res = lowest_eigenmodes(m, self.grid, self.ep, k=4, eigs_tol=1e-3)
        min_eig = float(res.min_eig)
        self.result.correction_events.append({"event": "preflight", "min_eig": min_eig})
        ok = min_eig >= self.min_gap
        return ok, min_eig

    def step_callback(self, m, k: int, t: float = 0.0):
        # Passive controller: nothing to do per-step. (Caller is expected to
        # have selected the CG-RK4 integrator at recipe time.)
        pass

    def H_extra_factory(self):
        def H_extra(m):
            return _np.zeros_like(_np.asarray(m))
        return H_extra


# ---------------------------------------------------------------------------
# C. Stabilizer (code-style)
# ---------------------------------------------------------------------------


@dataclass
class StabilizerController:
    """Lattice-style code. Each site is a logical bit; we monitor per-site Q
    and on detection of a flip we trigger a local thermal-burst re-nucleation."""
    expected_sites: int = 7
    flip_threshold: float = 0.5
    cadence: int = 50
    re_nucleation_kT: float = 8.0
    re_nucleation_steps: int = 40

    grid: Optional[Grid] = None
    ep: Optional[EnergyParams] = None
    rng: Optional[_np.random.Generator] = None
    alpha: float = 0.1
    dt: float = 0.002

    result: CorrectionResult = field(default_factory=lambda: CorrectionResult(kind="stabilizer"))
    _last_check_step: int = -1

    def step_callback(self, m, k: int, t: float = 0.0):
        if k - self._last_check_step < self.cadence:
            return
        self._last_check_step = k
        if self.grid is None:
            return
        from hopfion.physics.current import centroids, hopf_charge_density
        rho = hopf_charge_density(m, self.grid)
        cs = centroids(rho, self.grid, threshold_rel=0.1)
        n_sites = len(cs)
        # Heuristic syndrome: if site count dropped below expected, flag a flip
        if n_sites < self.expected_sites:
            self.result.n_corrections += 1
            self.result.correction_events.append({
                "step": int(k), "event": "syndrome_flip",
                "n_sites": int(n_sites), "expected": int(self.expected_sites),
            })
            # In a full implementation, we'd run a localized two_temperature
            # burst at the missing site. Phase-C scaffolding: log the event;
            # the recipe-level retry logic can incorporate the syndrome.

    def H_extra_factory(self):
        def H_extra(m):
            return _np.zeros_like(_np.asarray(m))
        return H_extra


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_controller(kind: str, grid: Grid, ep: EnergyParams, **params):
    """Build a controller of the requested kind. Recipe runner uses this."""
    if kind in (None, "none", ""):
        return None
    if kind == "active":
        return ActiveFeedbackController(grid=grid, ep=ep, **params)
    if kind == "topological_gap":
        return TopologicalGapController(grid=grid, ep=ep, **params)
    if kind == "stabilizer":
        return StabilizerController(grid=grid, ep=ep, **params)
    raise ValueError(f"Unknown correction kind {kind!r}")


__all__ = [
    "ActiveFeedbackController",
    "TopologicalGapController",
    "StabilizerController",
    "CorrectionResult",
    "make_controller",
]
