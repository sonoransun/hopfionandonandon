"""Time damped-relax step at several grid sizes. Writes CSV to --out."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hopfion.energy import EnergyParams
from hopfion.field import hopfion
from hopfion.grid import Grid
from hopfion.llg import relax_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_relax.csv")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    rows = [("N", "dx", "n_steps", "wall_s", "ms_per_step")]
    for N in (32, 48, 64):
        dx = 0.4
        g = Grid(N, N, N, dx, dx, dx, "periodic")
        m = hopfion(g, R=1.5, p=1, q=1)
        ep = EnergyParams(A_ex=1.0, D=1.5, Ku=0.7, easy_axis=(0, 0, 1))
        t0 = time.perf_counter()
        for _ in range(args.steps):
            m = relax_step(m, g, ep, dt=0.002)
        dt = time.perf_counter() - t0
        rows.append((N, dx, args.steps, f"{dt:.3f}", f"{1000*dt/args.steps:.2f}"))
        print(f"N={N:3d}: {dt:.2f}s  ({1000*dt/args.steps:.1f} ms/step)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    main()
