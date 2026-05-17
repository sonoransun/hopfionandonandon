"""Recipe schema: declarative spec for a hopfion-simulation run.

Recipes are YAML files. Internally everything is dataclasses; the YAML loader
is a thin ``yaml.safe_load`` + ``from_dict`` shim with no Pydantic.

A recipe declares: grid + boundary, material parameters, initial state
construction, an ordered list of run steps (relax / dynamics / thermal_burst /
pulse), QC criteria, and output specification.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import yaml


# ---------------------------------------------------------------------------
# Sub-schemas
# ---------------------------------------------------------------------------

@dataclass
class GridSpec:
    nx: int
    ny: int
    nz: int
    dx: float = 1.0
    dy: Optional[float] = None
    dz: Optional[float] = None
    bc: str = "periodic"

    def __post_init__(self):
        if self.dy is None:
            self.dy = self.dx
        if self.dz is None:
            self.dz = self.dx


@dataclass
class MoireSpec:
    enabled: bool = False
    K0: float = 0.5
    V0: float = 0.2
    a_moire: float = 8.0
    lattice: str = "triangular"


@dataclass
class MaterialSpec:
    A_ex: float = 1.0
    D: float = 0.0
    Ku: float = 0.0
    easy_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    H_ext: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dipolar: bool = False
    moire: Optional[MoireSpec] = None

    def __post_init__(self):
        self.easy_axis = tuple(self.easy_axis)
        self.H_ext = tuple(self.H_ext)


@dataclass
class InitialStateSpec:
    """How to build the starting field.

    Supported kinds:
        - "hopfion": single ansatz at (R, p, q, center, axis)
        - "uniform": constant field along `direction`
        - "perturbed_uniform": uniform + small Gaussian noise (`amplitude`)
        - "hopfion_array": triangular/square sites with R per-site
        - "file": load m from an HDF5 path
    """
    kind: str = "uniform"
    direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    R: float = 1.5
    p: int = 1
    q: int = 1
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: str = "z"
    amplitude: float = 0.0           # for perturbed_uniform
    array_lattice: str = "triangular"
    array_a: float = 8.0
    array_n_rings: int = 1
    array_nx: int = 3
    array_ny: int = 3
    file_path: Optional[str] = None
    # Composite-state-only (C2)
    separation: float = 4.0          # for q_pair
    axis_of_separation: str = "x"
    skyrmion_radius: float = 1.0     # for hopfion_skyrmion_hybrid
    skyrmion_helicity: float = 1.5707963267948966  # π/2 (Bloch)


@dataclass
class RunStep:
    """One step in the run sequence.

    Kinds:
        - "relax":           damped LLG for `n_steps` at `dt`
        - "dynamics":        full LLG (precession + damping)
        - "thermal_burst":   white-noise sLLG at `kT` for `n_steps`
        - "pulse":           add a Gaussian field pulse for the duration
        - "two_temperature": two-temperature stochastic LLG (B2)
        - "dynamics_stt":    LLG augmented with Zhang-Li spin-transfer torque (C2)
    """
    kind: str
    n_steps: int = 0
    dt: float = 0.002
    gamma: float = 1.0
    alpha: float = 0.1
    kT: float = 0.0
    # Pulse-only fields
    H0: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    t0: float = 0.0
    tau: float = 0.05
    profile: str = "point"
    pulse_width: float = 1.0
    ring_radius: float = 1.5
    # Two-temperature-only fields (B2)
    Te_peak: float = 0.0
    G_el: float = 1.0
    C_e: float = 1.0
    C_l: float = 1.0
    # STT-only fields (C2)
    u: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    beta: float = 0.0
    # Integrator selection (C3)
    integrator: str = "heun"   # "heun" | "rk4" | "crouch_grossman_rk4"

    def __post_init__(self):
        self.H0 = tuple(self.H0)
        self.u = tuple(self.u)


@dataclass
class QCSpec:
    """Acceptance criteria configuration.

    Each entry is a dict like ``{"name": "energy_monotonicity", "tol": 1e-6}``;
    the runner maps the name to a class in ``hopfion.pipeline.qc``.
    """
    fail_on: List[dict] = field(default_factory=list)
    warn_on: List[dict] = field(default_factory=list)


@dataclass
class IOSpec:
    out: str = "runs/run/"
    snapshots_every: int = 0
    write_report: bool = True
    extended_metrics: bool = False   # collect Flux + Drift + PerSiteQ in addition to defaults


@dataclass
class RecipeConfig:
    name: str
    grid: GridSpec
    material: MaterialSpec
    initial: InitialStateSpec
    run: List[RunStep]
    qc: QCSpec = field(default_factory=QCSpec)
    io: IOSpec = field(default_factory=IOSpec)
    description: str = ""
    seed: int = 0
    backend: str = "numpy"

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RecipeConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data, source=str(path))

    @classmethod
    def from_dict(cls, data: dict, source: str = "") -> "RecipeConfig":
        try:
            grid = GridSpec(**data["grid"])
            mat_raw = dict(data["material"])
            moire_raw = mat_raw.pop("moire", None)
            material = MaterialSpec(**mat_raw)
            if moire_raw is not None:
                material.moire = MoireSpec(**moire_raw)
            initial = InitialStateSpec(**data.get("initial", {}))
            run = [RunStep(**rs) for rs in data["run"]]
            qc = QCSpec(**data.get("qc", {}))
            io_spec = IOSpec(**data.get("io", {}))
            return cls(
                name=data["name"],
                grid=grid,
                material=material,
                initial=initial,
                run=run,
                qc=qc,
                io=io_spec,
                description=data.get("description", ""),
                seed=int(data.get("seed", 0)),
                backend=data.get("backend", "numpy"),
            )
        except KeyError as e:
            raise ValueError(f"Recipe {source!r}: missing required field {e!s}") from e


# ---------------------------------------------------------------------------
# Pre-flight checks (refuse to start the run if violated)
# ---------------------------------------------------------------------------

@dataclass
class PreflightResult:
    passed: bool
    failures: List[str]
    warnings: List[str]


def preflight(rc: RecipeConfig) -> PreflightResult:
    """Cheap dimensional / sanity checks. Returns failures + warnings."""
    failures: List[str] = []
    warnings: List[str] = []

    g = rc.grid

    # Resolution sanity vs hopfion size
    if rc.initial.kind in ("hopfion", "hopfion_array"):
        R = rc.initial.R
        if max(g.dx, g.dy, g.dz) > R / 2.0:
            failures.append(
                f"grid spacing max({g.dx},{g.dy},{g.dz}) > R/2 ({R/2:.3f}) -- "
                f"hopfion of size {R} is under-resolved")
        if max(g.dx, g.dy, g.dz) > R / 3.0:
            warnings.append(
                f"grid spacing > R/3 -- accuracy will be marginal; recommend dx <= R/4")

        # Box must accommodate the hopfion (and its periodic images)
        L = [g.nx * g.dx, g.ny * g.dy, g.nz * g.dz]
        if g.bc == "periodic" and min(L) < 4 * R:
            failures.append(
                f"box {L} too small for R={R} under periodic BC; need L > 4R "
                f"in each direction to avoid image overlap")

    # Time-step stability (rough, conservative)
    for i, step in enumerate(rc.run):
        if step.kind in ("relax", "dynamics", "two_temperature"):
            d_min = min(g.dx, g.dy, g.dz)
            # explicit-Heun stability for Laplacian
            dt_max_exch = d_min ** 2 / (4.0 * rc.material.A_ex)
            if step.dt > dt_max_exch:
                failures.append(
                    f"run[{i}] dt={step.dt} exceeds exchange-stability bound "
                    f"{dt_max_exch:.4f} (dx^2 / 4 A_ex)")

    # Backend availability
    if rc.backend == "jax":
        try:
            import jax  # noqa: F401
        except ImportError:
            failures.append("backend='jax' requested but JAX is not installed")

    # DMI helicity vs hopfion size (warning only)
    if rc.material.D > 0 and rc.initial.R > 0:
        L_D = rc.material.A_ex / rc.material.D
        if rc.initial.R > 5 * L_D or rc.initial.R < 0.2 * L_D:
            warnings.append(
                f"hopfion R={rc.initial.R:.2f} far from DMI length L_D={L_D:.2f}; "
                f"stability window may be tight")

    return PreflightResult(passed=not failures, failures=failures, warnings=warnings)


__all__ = [
    "GridSpec",
    "MoireSpec",
    "MaterialSpec",
    "InitialStateSpec",
    "RunStep",
    "QCSpec",
    "IOSpec",
    "RecipeConfig",
    "PreflightResult",
    "preflight",
]
