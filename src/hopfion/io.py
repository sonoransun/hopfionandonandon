"""HDF5 snapshot I/O.

Stores the magnetization field plus run metadata in a single file:

    /m            -- shape (3, nx, ny, nz), the field at the latest snapshot
    /snapshots/t  -- dataset (n_snap,)  -- simulation times
    /snapshots/m  -- dataset (n_snap, 3, nx, ny, nz) if recorded
    /energy       -- dataset (n_step,) -- energy time series, if recorded
    attrs:        -- grid params, energy params, backend, seed, git rev
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import asdict
from typing import Iterable, Optional

import h5py
import numpy as _np

from hopfion.backend import name as backend_name
from hopfion.backend import to_numpy
from hopfion.energy import EnergyParams
from hopfion.grid import Grid


def _git_rev() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def save_run(
    path: str,
    m,
    grid: Grid,
    ep: EnergyParams,
    *,
    snapshots: Optional[Iterable] = None,
    times: Optional[Iterable[float]] = None,
    energy_series: Optional[Iterable[float]] = None,
    seed: Optional[int] = None,
    note: str = "",
) -> None:
    """Write a run to HDF5."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("m", data=to_numpy(m))
        f.attrs["nx"] = grid.nx
        f.attrs["ny"] = grid.ny
        f.attrs["nz"] = grid.nz
        f.attrs["dx"] = grid.dx
        f.attrs["dy"] = grid.dy
        f.attrs["dz"] = grid.dz
        f.attrs["bc"] = grid.bc
        for k, v in asdict(ep).items():
            if k == "Ku_field":
                if v is not None:
                    f.create_dataset("Ku_field", data=to_numpy(v))
                continue
            f.attrs[f"ep_{k}"] = _np.asarray(v) if isinstance(v, (tuple, list)) else v
        f.attrs["backend"] = backend_name()
        f.attrs["git_rev"] = _git_rev()
        f.attrs["note"] = note
        if seed is not None:
            f.attrs["seed"] = seed
        if snapshots is not None:
            arr = _np.stack([to_numpy(s) for s in snapshots], axis=0)
            f.create_dataset("snapshots/m", data=arr)
            if times is not None:
                f.create_dataset("snapshots/t", data=_np.asarray(list(times)))
        if energy_series is not None:
            f.create_dataset("energy", data=_np.asarray(list(energy_series)))


def load_run(path: str):
    """Read an HDF5 snapshot. Returns ``(m, grid, ep, extras)``."""
    with h5py.File(path, "r") as f:
        m = f["m"][...]
        grid = Grid(
            nx=int(f.attrs["nx"]),
            ny=int(f.attrs["ny"]),
            nz=int(f.attrs["nz"]),
            dx=float(f.attrs["dx"]),
            dy=float(f.attrs["dy"]),
            dz=float(f.attrs["dz"]),
            bc=str(f.attrs["bc"]),
        )
        ku_field = None
        if "Ku_field" in f:
            ku_field = f["Ku_field"][...]
        ep = EnergyParams(
            A_ex=float(f.attrs["ep_A_ex"]),
            D=float(f.attrs["ep_D"]),
            Ku=float(f.attrs["ep_Ku"]),
            easy_axis=tuple(f.attrs["ep_easy_axis"]),
            H_ext=tuple(f.attrs["ep_H_ext"]),
            Ku_field=ku_field,
        )
        extras = {"backend": str(f.attrs.get("backend", "")), "note": str(f.attrs.get("note", ""))}
        if "snapshots/m" in f:
            extras["snapshots"] = f["snapshots/m"][...]
            extras["times"] = f["snapshots/t"][...] if "snapshots/t" in f else None
        if "energy" in f:
            extras["energy"] = f["energy"][...]
    return m, grid, ep, extras
