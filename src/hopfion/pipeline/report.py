"""Markdown report generation from an HDF5 + JSON sidecar.

A report is written to ``<out>/report.md`` and references PNG figures saved
alongside it. The Markdown is self-contained so a reader can open it in any
markdown viewer (GitHub, VS Code, ``grip``) and see the run at a glance.

Sections:
    * Header (recipe name, verdict badge, wall time)
    * Pre-flight (any warnings / failures)
    * QC verdict table (criterion-by-criterion)
    * Metric histories (energy + Q_H plots; norm drift if non-trivial)
    * Final-state figures (xy slice, optionally preimage scatter)
    * Recipe (collapsed YAML for reproducibility)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as _np

from hopfion.energy import EnergyParams
from hopfion.grid import Grid
from hopfion.viz import slice_quiver


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load(run_dir: Path):
    summary = json.loads((run_dir / "summary.json").read_text())
    with h5py.File(run_dir / "run.h5", "r") as f:
        m = f["m"][...]
        attrs = {k: f.attrs[k] for k in f.attrs}
    grid = Grid(
        nx=int(attrs["nx"]), ny=int(attrs["ny"]), nz=int(attrs["nz"]),
        dx=float(attrs["dx"]), dy=float(attrs["dy"]), dz=float(attrs["dz"]),
        bc=str(attrs["bc"]),
    )
    return summary, m, grid, attrs


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_energy(summary, out_path):
    hist = summary.get("histories", {}).get("energy")
    if not hist or not hist["values"]:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(hist["steps"], hist["values"], color="#1f77b4")
    ax.set_xlabel("step"); ax.set_ylabel("energy"); ax.grid(True, alpha=0.3)
    ax.set_title("Energy over run")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_q_hopf(summary, out_path):
    hist = summary.get("histories", {}).get("q_hopf")
    if not hist or not hist["values"]:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(hist["steps"], hist["values"], "o-", color="#d62728")
    ax.set_xlabel("step"); ax.set_ylabel("Q$_H$"); ax.grid(True, alpha=0.3)
    ax.set_title("Hopf invariant over run")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_norm_drift(summary, out_path):
    hist = summary.get("histories", {}).get("norm_drift")
    if not hist or not hist["values"]:
        return False
    vals = hist["values"]
    if max(vals) < 1e-12:
        return False  # trivial — skip
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.semilogy(hist["steps"], vals, color="#2ca02c")
    ax.set_xlabel("step"); ax.set_ylabel("max ||m|-1|"); ax.grid(True, alpha=0.3)
    ax.set_title("Unit-norm drift")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_xy_slice(m, grid, out_path):
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    slice_quiver(m, grid, plane="xy", stride=max(1, grid.nx // 24), ax=ax)
    ax.set_title("final state — xy slice")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _badge(verdict: str) -> str:
    if verdict == "ACCEPT":
        return "✅ **ACCEPT**"
    return "❌ **FAIL**"


def _render_qc_table(qc: Dict[str, Any]) -> str:
    rows = ["| status | criterion | message |", "|---|---|---|"]
    for r in qc.get("results", []):
        mark = "✅ pass" if r["passed"] else ("❌ fail" if r["severity"] == "fail" else "⚠️ warn")
        rows.append(f"| {mark} | `{r['name']}` | {r['message']} |")
    return "\n".join(rows)


def write_report(run_dir: str | Path) -> Path:
    """Build figures + Markdown report under ``run_dir``."""
    run_dir = Path(run_dir)
    summary, m, grid, attrs = _load(run_dir)
    fig_dir = run_dir / "figs"
    fig_dir.mkdir(exist_ok=True)

    plots = []
    if _plot_energy(summary, fig_dir / "energy.png"):
        plots.append(("Energy time series", "figs/energy.png"))
    if _plot_q_hopf(summary, fig_dir / "q_hopf.png"):
        plots.append(("Hopf invariant", "figs/q_hopf.png"))
    if _plot_norm_drift(summary, fig_dir / "norm_drift.png"):
        plots.append(("Norm drift (log)", "figs/norm_drift.png"))
    _plot_xy_slice(m, grid, fig_dir / "final_xy.png")

    qc_table = _render_qc_table(summary["qc"])
    recipe = summary["recipe"]
    metrics = summary["metrics"]

    lines = []
    lines.append(f"# Run report — `{recipe['name']}`")
    lines.append("")
    lines.append(f"**Verdict**: {_badge(summary['qc']['verdict'])}  ·  "
                 f"wall time: {summary['wall_seconds']:.1f} s  ·  "
                 f"Q$_H$(final): {summary['Q_final']:+.4f}  ·  "
                 f"backend: `{recipe['backend']}`")
    if recipe.get("description"):
        lines.append("")
        lines.append("> " + recipe["description"].replace("\n", " "))

    lines.append("")
    lines.append("## Quality control")
    if summary["qc"].get("preflight_failures"):
        lines.append("")
        lines.append("**Pre-flight failures:**")
        for f in summary["qc"]["preflight_failures"]:
            lines.append(f"- ❌ {f}")
    if summary["qc"].get("preflight_warnings"):
        lines.append("")
        lines.append("**Pre-flight warnings:**")
        for w in summary["qc"]["preflight_warnings"]:
            lines.append(f"- ⚠️ {w}")
    lines.append("")
    lines.append(qc_table)

    lines.append("")
    lines.append("## Metric summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2, default=str))
    lines.append("```")

    lines.append("")
    lines.append("## Final state")
    lines.append("")
    lines.append("![final xy slice](figs/final_xy.png)")

    if plots:
        lines.append("")
        lines.append("## Time series")
        for title, rel in plots:
            lines.append("")
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{title}]({rel})")

    lines.append("")
    lines.append("## Recipe")
    lines.append("")
    lines.append("```yaml")
    lines.append("name: " + recipe["name"])
    lines.append("backend: " + recipe["backend"])
    lines.append(f"seed: {recipe['seed']}")
    lines.append("grid: " + json.dumps(recipe["grid"]))
    lines.append("material: " + json.dumps(recipe["material"], default=str))
    lines.append("initial: " + json.dumps(recipe["initial"], default=str))
    lines.append("run:")
    for s in recipe["run"]:
        lines.append("  - " + json.dumps(s, default=str))
    lines.append("```")

    out_path = run_dir / "report.md"
    out_path.write_text("\n".join(lines))
    return out_path


__all__ = ["write_report"]
