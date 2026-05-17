"""Time hopf_index at several grid sizes (dominated by FFT)."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hopfion.field import hopfion
from hopfion.grid import Grid
from hopfion.topology import hopf_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_hopf_index.csv")
    ap.add_argument("--reps", type=int, default=10)
    args = ap.parse_args()

    rows = [("N", "reps", "mean_ms", "Q")]
    for N in (32, 64, 96, 128):
        g = Grid(N, N, N, 16.0 / N, 16.0 / N, 16.0 / N, "periodic")
        m = hopfion(g, R=1.0, p=1, q=1)
        # warm-up
        _ = hopf_index(m, g)
        t0 = time.perf_counter()
        for _ in range(args.reps):
            Q = hopf_index(m, g)
        dt = time.perf_counter() - t0
        rows.append((N, args.reps, f"{1000*dt/args.reps:.2f}", f"{Q:+.5f}"))
        print(f"N={N:3d}: {1000*dt/args.reps:.1f} ms/call,  Q={Q:+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    main()
