"""Time the dipolar FFT field. Placeholder until physics/dipolar.py lands.
Currently times a single FFT round-trip as a proxy."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from hopfion.field import hopfion
from hopfion.grid import Grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_dipolar.csv")
    args = ap.parse_args()

    rows = [("N", "fft_ms", "ifft_ms")]
    for N in (32, 64, 96):
        g = Grid(N, N, N, 0.5, 0.5, 0.5, "periodic")
        m = hopfion(g, R=1.5, p=1, q=1)
        t0 = time.perf_counter()
        m_hat = np.fft.fftn(np.asarray(m), axes=(1, 2, 3))
        dt_fft = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = np.fft.ifftn(m_hat, axes=(1, 2, 3))
        dt_ifft = time.perf_counter() - t0
        rows.append((N, f"{1000*dt_fft:.2f}", f"{1000*dt_ifft:.2f}"))
        print(f"N={N:3d}: fft={1000*dt_fft:.1f} ms, ifft={1000*dt_ifft:.1f} ms")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerows(rows)


if __name__ == "__main__":
    main()
