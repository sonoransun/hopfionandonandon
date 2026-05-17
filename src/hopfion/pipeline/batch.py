"""Ensemble + parameter-sweep runner with yield statistics.

Inputs: a base ``RecipeConfig`` + a dict mapping dotted keys to lists of
values + a list of seeds. Outputs: a ``BatchReport`` with per-run verdicts,
yield per slice of the sweep, failure-mode histogram, and Cp-style indices.

Parallelism is opt-in via ``max_workers``; the default of 1 keeps things
deterministic.
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as _np

from hopfion.pipeline.recipe import RecipeConfig
from hopfion.pipeline.runner import run as run_single


# ---------------------------------------------------------------------------
# Apply a dotted key onto a RecipeConfig
# ---------------------------------------------------------------------------


_BRACKET = re.compile(r"^(.+?)\[(\d+)\]$")


def _set_dotted(rc: RecipeConfig, key: str, value: Any) -> None:
    parts = key.split(".")
    target = rc
    for i, part in enumerate(parts):
        m = _BRACKET.match(part)
        if m:
            name, idx = m.group(1), int(m.group(2))
            container = getattr(target, name)
            if i == len(parts) - 1:
                container[idx] = value
            else:
                target = container[idx]
        else:
            if i == len(parts) - 1:
                setattr(target, part, value)
            else:
                target = getattr(target, part)


# ---------------------------------------------------------------------------
# Batch report
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    seed: int
    overrides: Dict[str, Any]
    verdict: str
    q_final: float
    wall_seconds: float
    out_dir: str
    failures: List[str] = field(default_factory=list)


@dataclass
class BatchReport:
    base_recipe_name: str
    runs: List[RunRecord]
    out_root: str

    @property
    def all_passed(self) -> bool:
        return all(r.verdict == "ACCEPT" for r in self.runs)

    @property
    def yield_overall(self) -> float:
        if not self.runs:
            return 0.0
        return sum(1 for r in self.runs if r.verdict == "ACCEPT") / len(self.runs)

    def yield_by(self, key: str) -> Dict[Any, float]:
        groups: Dict[Any, List[bool]] = {}
        for r in self.runs:
            k = r.overrides.get(key, "<base>")
            groups.setdefault(k, []).append(r.verdict == "ACCEPT")
        return {k: sum(v) / len(v) for k, v in groups.items()}

    def failure_modes(self) -> Counter:
        c: Counter = Counter()
        for r in self.runs:
            for f in r.failures:
                c[f] += 1
        return c

    def cp_index(self, key: str, target: float, half_width: float) -> Dict[Any, float]:
        """Cp-style spec-vs-process-variation index per slice of ``key``.

        Cp = (USL - LSL) / (6 sigma) where sigma is the std of Q_H within the slice.
        """
        slices: Dict[Any, List[float]] = {}
        for r in self.runs:
            slices.setdefault(r.overrides.get(key, "<base>"), []).append(r.q_final)
        out: Dict[Any, float] = {}
        for k, vals in slices.items():
            sigma = float(_np.std(vals)) if len(vals) > 1 else float("nan")
            if sigma == 0 or _np.isnan(sigma):
                out[k] = float("inf")
            else:
                out[k] = (2.0 * half_width) / (6.0 * sigma)
        return out

    def summary_text(self) -> str:
        lines = []
        lines.append(f"RECIPE: {self.base_recipe_name}")
        lines.append(f"TOTAL RUNS: {len(self.runs)}")
        passed = sum(1 for r in self.runs if r.verdict == "ACCEPT")
        lines.append(f"PASS: {passed}  FAIL: {len(self.runs) - passed}  "
                     f"YIELD: {self.yield_overall:.0%}")
        modes = self.failure_modes()
        if modes:
            lines.append("FAILURE MODES:")
            for name, count in modes.most_common():
                lines.append(f"  {name:36s} {count}")
        sweep_keys = sorted({k for r in self.runs for k in r.overrides})
        for key in sweep_keys:
            yields = self.yield_by(key)
            if len(yields) > 1:
                lines.append(f"YIELD BY {key}:")
                for k, y in sorted(yields.items(), key=lambda x: (str(type(x[0])), x[0])):
                    lines.append(f"  {k!r:20s}  {y:.0%}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recipe": self.base_recipe_name,
            "out_root": self.out_root,
            "yield_overall": self.yield_overall,
            "runs": [r.__dict__ for r in self.runs],
            "failure_modes": dict(self.failure_modes()),
        }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def _run_one(args):
    rc, out_dir, seed, overrides = args
    rc = copy.deepcopy(rc)
    rc.seed = int(seed)
    for k, v in overrides.items():
        _set_dotted(rc, k, v)
    rc.io.out = out_dir
    result = run_single(rc, write=True)
    failures = [r.name for r in result.qc.results if not r.passed and r.severity == "fail"]
    failures.extend(f"preflight:{f}" for f in result.qc.preflight_failures)
    return RunRecord(
        seed=seed, overrides=overrides,
        verdict=result.qc.verdict,
        q_final=float(result.metrics.get("q_hopf", {}).get("Q_final", float("nan"))),
        wall_seconds=result.seconds, out_dir=out_dir,
        failures=failures,
    )


def run_batch(rc: RecipeConfig, sweep: Dict[str, List[Any]], seeds: List[int],
              out_root: str = "batches", max_workers: int = 1) -> BatchReport:
    """Execute the cross product of ``sweep`` × ``seeds``. Returns a BatchReport."""
    out_root_path = Path(out_root) / rc.name
    out_root_path.mkdir(parents=True, exist_ok=True)

    keys = list(sweep.keys())
    value_lists = [sweep[k] for k in keys] if keys else [[None]]

    jobs = []
    for combo in itertools.product(*value_lists):
        overrides = {keys[i]: combo[i] for i in range(len(keys))} if keys else {}
        for seed in seeds:
            tag = "_".join([f"{k.replace('.', '-')}-{v}" for k, v in overrides.items()] +
                           [f"seed-{seed}"])
            out_dir = str(out_root_path / (tag if tag else f"seed-{seed}"))
            jobs.append((rc, out_dir, seed, overrides))

    if max_workers > 1:
        from multiprocessing import Pool
        with Pool(processes=max_workers) as pool:
            records = pool.map(_run_one, jobs)
    else:
        records = [_run_one(j) for j in jobs]

    report = BatchReport(base_recipe_name=rc.name, runs=records, out_root=str(out_root_path))
    (out_root_path / "batch_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, default=str))
    (out_root_path / "batch_report.txt").write_text(report.summary_text())
    return report


__all__ = ["BatchReport", "RunRecord", "run_batch"]
