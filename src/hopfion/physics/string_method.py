"""Minimum-energy path between two spin textures via the string method.

The activation barrier of a hopfion (the energy it must climb to decay) is the
saddle-to-minimum energy difference along the minimum-energy path (MEP) that
connects the metastable state to its decay product (e.g. the uniform
ferromagnet). We find that path with the *simplified string method* of E, Ren &
Vanden-Eijnden:

    1. Discretize the path into ``n_images`` configurations, the two endpoints
       fixed.
    2. Evolve every interior image one step of damped (gradient-descent) LLG so
       it falls toward lower energy -- ``llg.damped_step`` reused verbatim.
    3. Reparameterize: redistribute the images to equal arc length along the
       string. This is what keeps them spread out along the path instead of all
       sliding into the nearest minimum; it replaces the spring forces of a
       plain NEB (no spring constant to tune).

The barrier is then ``max(E_image) - E_start``, and the highest-energy image
approximates the saddle. With ``climbing=True`` the peak image is frozen during
descent so it is not pulled down off the ridge, sharpening the barrier estimate.

Pair the barrier with a Hessian Arrhenius pre-factor (``hessian.lowest_eigenmodes``)
to get a lifetime via :func:`arrhenius_lifetime`. See SOP-003.

This is an offline analysis tool and runs on the NumPy backend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import numpy as _np

from hopfion.energy import EnergyParams, total_energy
from hopfion.grid import Grid
from hopfion.llg import relax_step


def _normalize(m):
    n = _np.sqrt((m * m).sum(axis=0, keepdims=True))
    return m / _np.where(n == 0.0, 1.0, n)


def _slerp(a, b, t):
    """Per-site geodesic interpolation on S^2. Linear+renormalize would inject
    spurious exchange energy at the blend; great-circle interpolation keeps the
    intermediate texture smooth."""
    dot = _np.clip((a * b).sum(axis=0, keepdims=True), -1.0, 1.0)
    omega = _np.arccos(dot)
    so = _np.sin(omega)
    small = so < 1e-7
    with _np.errstate(invalid="ignore", divide="ignore"):
        geo = (_np.sin((1.0 - t) * omega) * a + _np.sin(t * omega) * b) / _np.where(small, 1.0, so)
    lin = (1.0 - t) * a + t * b
    return _normalize(_np.where(small, lin, geo))


def _reparameterize(images: List[_np.ndarray]) -> List[_np.ndarray]:
    """Redistribute images to equal arc length along the string (endpoints fixed),
    interpolating geodesically on S^2."""
    n = len(images)
    seg = [0.0]
    for i in range(1, n):
        seg.append(seg[-1] + float(_np.sqrt(((images[i] - images[i - 1]) ** 2).sum())))
    total = seg[-1]
    if total <= 0.0:
        return images
    seg = _np.asarray(seg)
    targets = _np.linspace(0.0, total, n)
    out = [images[0]]
    for j in range(1, n - 1):
        s = targets[j]
        k = min(max(int(_np.searchsorted(seg, s)), 1), n - 1)
        s0, s1 = seg[k - 1], seg[k]
        w = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)
        out.append(_slerp(images[k - 1], images[k], w))
    out.append(images[-1])
    return out


@dataclass
class StringMethodResult:
    barrier: float               # E_saddle - E_start
    saddle_index: int            # index of the highest-energy image
    energies: List[float]        # energy of each image along the converged path
    images: List[_np.ndarray]    # the converged path (list of (3, nx, ny, nz))


def string_method(m_start, m_end, grid: Grid, ep: EnergyParams,
                  n_images: int = 7, n_iter: int = 200, dt: float = 0.01,
                  climbing: bool = False, reparam_every: int = 1) -> StringMethodResult:
    """Minimum-energy path and activation barrier between two states.

    ``m_start`` is the metastable state (its energy is the barrier reference);
    ``m_end`` is the decay product. Returns a :class:`StringMethodResult`.

    ``dt`` drives the per-image damped descent and is bounded by the same
    exchange-stability limit as relaxation, ``dt < dx^2 / (4 A_ex)``; the default
    is conservative. Returns a :class:`StringMethodResult`.
    """
    a = _normalize(_np.asarray(m_start, dtype=float))
    b = _normalize(_np.asarray(m_end, dtype=float))
    # Initial path: geodesic (great-circle) interpolation on S^2.
    ts = _np.linspace(0.0, 1.0, n_images)
    images = [a] + [_slerp(a, b, float(t)) for t in ts[1:-1]] + [b]

    for it in range(n_iter):
        energies = [total_energy(im, grid, ep) for im in images]
        peak = int(_np.argmax(energies))
        new = [images[0]]
        for i in range(1, n_images - 1):
            if climbing and i == peak:
                new.append(images[i])     # freeze the ridge image (climbing surrogate)
            else:
                new.append(relax_step(images[i], grid, ep, dt=dt))
        new.append(images[-1])
        images = new
        if reparam_every and (it + 1) % reparam_every == 0:
            images = _reparameterize(images)

    energies = [float(total_energy(im, grid, ep)) for im in images]
    saddle_index = int(_np.argmax(energies))
    barrier = float(energies[saddle_index] - energies[0])
    return StringMethodResult(barrier=barrier, saddle_index=saddle_index,
                              energies=energies, images=images)


def arrhenius_lifetime(barrier: float, kT: float, prefactor: float = 1.0,
                       tau0: float = 1.0) -> float:
    """Arrhenius lifetime ``tau = (tau0 / prefactor) * exp(barrier / kT)``.

    ``prefactor`` is an attempt-frequency scale; a common choice is the square
    root of the product of the positive Hessian eigenvalues at the minimum
    (``hessian.lowest_eigenmodes``), capturing the curvature of the energy basin.
    ``kT`` is in the same energy units as ``barrier`` (normalized units here).
    """
    if kT <= 0.0:
        return float("inf")
    exponent = barrier / kT
    if exponent > 700.0:   # math.exp overflows ~709
        return float("inf")
    return (tau0 / prefactor) * math.exp(exponent)


__all__ = ["StringMethodResult", "string_method", "arrhenius_lifetime"]
