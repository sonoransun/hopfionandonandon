"""``hopfion`` command-line entry point.

Sub-commands:

    hopfion run RECIPE          # single run
    hopfion batch RECIPE        # parameter sweep / ensemble
    hopfion qc RUN_DIR          # re-evaluate QC from existing summary.json
    hopfion report RUN_DIR      # (re)generate report.md
    hopfion ls [RUNS_DIR]       # list runs and their verdicts

Returns exit code 0 for ACCEPT runs, 2 for FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


def cmd_run(args) -> int:
    from hopfion.pipeline.recipe import RecipeConfig
    from hopfion.pipeline.report import write_report
    from hopfion.pipeline.runner import run

    rc = RecipeConfig.from_yaml(args.recipe)
    if args.out is not None:
        rc.io.out = args.out
    if args.no_report:
        rc.io.write_report = False
    if args.backend:
        rc.backend = args.backend

    result = run(rc, write=True)
    verdict = result.qc.verdict
    print(f"[{verdict}] {rc.name}  ·  wall {result.seconds:.1f}s  ·  out={result.out_dir}")
    for r in result.qc.results:
        flag = "PASS" if r.passed else ("FAIL" if r.severity == "fail" else "WARN")
        print(f"  [{flag:4}] {r.name}: {r.message}")
    if rc.io.write_report and verdict != "FAIL" or args.report_on_fail:
        try:
            p = write_report(result.out_dir)
            print(f"  report: {p}")
        except Exception as e:
            print(f"  report failed: {e}", file=sys.stderr)
    return 0 if verdict == "ACCEPT" else 2


def cmd_batch(args) -> int:
    from hopfion.pipeline.batch import run_batch
    from hopfion.pipeline.recipe import RecipeConfig

    rc = RecipeConfig.from_yaml(args.recipe)
    seeds = _parse_seeds(args.seeds) if args.seeds else [rc.seed]
    sweep = _parse_sweep(args.sweep) if args.sweep else {}
    report = run_batch(rc, sweep=sweep, seeds=seeds, out_root=args.out or "batches",
                       max_workers=args.workers)
    print(report.summary_text())
    return 0 if report.all_passed else 2


def cmd_report(args) -> int:
    from hopfion.pipeline.report import write_report
    p = write_report(args.run_dir)
    print(f"wrote {p}")
    return 0


def cmd_qc(args) -> int:
    from hopfion.pipeline.qc import evaluate
    from hopfion.pipeline.recipe import QCSpec

    summary = json.loads(Path(args.run_dir, "summary.json").read_text())
    qc_spec = QCSpec(**summary["recipe"].get("qc_override", summary["recipe"].get("qc", {}))) \
        if isinstance(summary["recipe"].get("qc"), dict) \
        else QCSpec()
    # If the recipe stored fail_on/warn_on, reuse them
    raw_qc = summary["recipe"].get("qc")
    if isinstance(raw_qc, dict):
        qc_spec = QCSpec(fail_on=raw_qc.get("fail_on", []), warn_on=raw_qc.get("warn_on", []))
    qc = evaluate(qc_spec, summary["metrics"],
                  preflight_failures=summary["qc"].get("preflight_failures"),
                  preflight_warnings=summary["qc"].get("preflight_warnings"))
    print(f"[{qc.verdict}] {summary['recipe']['name']}")
    for r in qc.results:
        flag = "PASS" if r.passed else ("FAIL" if r.severity == "fail" else "WARN")
        print(f"  [{flag:4}] {r.name}: {r.message}")
    return 0 if qc.verdict == "ACCEPT" else 2


def cmd_ls(args) -> int:
    root = Path(args.runs_dir)
    if not root.is_dir():
        print(f"{root} is not a directory", file=sys.stderr)
        return 1
    rows = []
    for d in sorted(root.iterdir()):
        sj = d / "summary.json"
        if not sj.exists():
            continue
        s = json.loads(sj.read_text())
        rows.append((d.name, s["qc"]["verdict"], s["recipe"]["name"],
                     s.get("Q_final", float("nan")), s.get("wall_seconds", 0.0)))
    if not rows:
        print(f"no runs under {root}")
        return 0
    print(f"{'dir':<32} {'verdict':<8} {'recipe':<28} {'Q_final':>10} {'wall(s)':>10}")
    print("-" * 92)
    for d, v, name, q, t in rows:
        print(f"{d:<32} {v:<8} {name:<28} {q:>+10.4f} {t:>10.2f}")
    return 0


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------


def _parse_sweep(s: str) -> dict:
    """Parse 'KEY=v1,v2,v3' or repeated --sweep flags.

    KEY is dotted: 'material.D', 'run[0].dt'. Returns ``{key: [parsed_values]}``.
    """
    sweep = {}
    for item in s.split(";") if ";" in s else [s]:
        key, _, values = item.partition("=")
        sweep[key.strip()] = [_coerce(v) for v in values.split(",") if v]
    return sweep


def _coerce(v: str):
    v = v.strip()
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _parse_seeds(s: str) -> List[int]:
    """Accept 'a:b', 'a:b:c', or 'a,b,c'."""
    if ":" in s:
        parts = [int(x) for x in s.split(":")]
        if len(parts) == 2:
            a, b = parts
            return list(range(a, b))
        if len(parts) == 3:
            a, b, step = parts
            return list(range(a, b, step))
    return [int(x) for x in s.split(",") if x]


# ---------------------------------------------------------------------------
# argparse setup
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("hopfion", description="hopfion simulation pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="execute a single recipe")
    run.add_argument("recipe", help="path to a YAML recipe")
    run.add_argument("--out", help="override io.out", default=None)
    run.add_argument("--backend", choices=["numpy", "jax"], default=None,
                     help="override backend")
    run.add_argument("--no-report", action="store_true", help="skip report.md generation")
    run.add_argument("--report-on-fail", action="store_true",
                     help="generate report even when QC fails")
    run.set_defaults(func=cmd_run)

    batch = sub.add_parser("batch", help="run an ensemble or parameter sweep")
    batch.add_argument("recipe")
    batch.add_argument("--sweep", default=None,
                       help='e.g. "material.D=0.6,0.9,1.2,1.5;run[0].dt=0.002,0.005"')
    batch.add_argument("--seeds", default=None,
                       help="seed range 'a:b' or list 'a,b,c'")
    batch.add_argument("--workers", type=int, default=1,
                       help="parallel workers (multiprocessing)")
    batch.add_argument("--out", default=None, help="batch output root (default: batches/)")
    batch.set_defaults(func=cmd_batch)

    qc = sub.add_parser("qc", help="re-evaluate QC from a run's summary.json")
    qc.add_argument("run_dir")
    qc.set_defaults(func=cmd_qc)

    rep = sub.add_parser("report", help="(re)generate report.md for a run")
    rep.add_argument("run_dir")
    rep.set_defaults(func=cmd_report)

    ls = sub.add_parser("ls", help="list runs and verdicts under a directory")
    ls.add_argument("runs_dir", nargs="?", default="runs")
    ls.set_defaults(func=cmd_ls)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
