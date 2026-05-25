"""Analytic hopfion field constructors via the Hopf fibration.

The Hopf map S^3 -> S^2 in coordinates (u, v) in C^2 with |u|^2 + |v|^2 = 1:

    m = (2 Re(u v*), 2 Im(u v*), |u|^2 - |v|^2)

Inverse stereographic projection R^3 -> S^3 at scale ``R``:

    u = (R^2 - r^2 + 2 i R z) / (R^2 + r^2)
    v = 2 R (x + i y) / (R^2 + r^2)

The (p, q) generalization u -> u^p / N, v -> v^q / N (with N normalizing on S^3)
yields a hopfion with Hopf index Q_H = p * q.
"""
from __future__ import annotations

from typing import Tuple

from hopfion.backend import xp
from hopfion.grid import Grid


def hopfion(
    grid: Grid,
    R: float,
    p: int = 1,
    q: int = 1,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    axis: str = "z",
):
    """Construct a hopfion ansatz with Hopf index ``p * q``.

    Parameters
    ----------
    grid : Grid
        Spatial grid.
    R : float
        Hopfion size (radius of the central torus).
    p, q : int
        Hopf-map exponents. Q_H = p * q. Default (1, 1).
    center : tuple of float
        Hopfion center in spatial coordinates.
    axis : {"x", "y", "z"}
        Axis of the central ring.
    """
    np = xp()
    X, Y, Z = grid.coords()
    cx, cy, cz = center
    if axis == "z":
        x, y, z = X - cx, Y - cy, Z - cz
    elif axis == "y":
        x, y, z = X - cx, Z - cz, Y - cy
    elif axis == "x":
        x, y, z = Y - cy, Z - cz, X - cx
    else:
        raise ValueError(f"axis must be one of x, y, z; got {axis!r}")

    r2 = x * x + y * y + z * z
    denom = R * R + r2

    # On-S^3 complex coordinates (|u|^2 + |v|^2 == 1)
    u_re = (R * R - r2) / denom
    u_im = 2 * R * z / denom
    v_re = 2 * R * x / denom
    v_im = 2 * R * y / denom

    # Raise to integer powers (p, q) keeping |u|^2 + |v|^2 normalized.
    if p != 1:
        u_re, u_im = _complex_pow(u_re, u_im, p)
    if q != 1:
        v_re, v_im = _complex_pow(v_re, v_im, q)

    norm = np.sqrt(u_re * u_re + u_im * u_im + v_re * v_re + v_im * v_im + 1e-30)
    u_re, u_im, v_re, v_im = u_re / norm, u_im / norm, v_re / norm, v_im / norm

    # Hopf map: m_x + i m_y = 2 u v*, m_z = |u|^2 - |v|^2
    mx = 2 * (u_re * v_re + u_im * v_im)
    my = 2 * (u_im * v_re - u_re * v_im)
    mz = u_re * u_re + u_im * u_im - v_re * v_re - v_im * v_im

    return np.stack([mx, my, mz], axis=0)


def _complex_pow(re, im, n: int):
    """Compute (re + i im)^n via De Moivre on the principal branch."""
    np = xp()
    r = np.sqrt(re * re + im * im + 1e-30)
    theta = np.arctan2(im, re)
    rn = r**n
    return rn * np.cos(n * theta), rn * np.sin(n * theta)


def skyrmion(
    grid: Grid,
    radius: float = 1.0,
    helicity: float = 0.0,
    vorticity: int = 1,
    center: Tuple[float, float] = (0.0, 0.0),
):
    """A 2D skyrmion texture extruded along z.

    Parameters
    ----------
    radius : float
        Size of the 360-degree domain wall in the in-plane radius ``rho``.
    helicity : float
        Rotates the in-plane spin direction: ``0`` is Néel, ``pi/2`` is Bloch.
    vorticity : int
        Winding of the in-plane angle. ``+1`` is a skyrmion; ``-1`` is an
        antiskyrmion (its topological charge has the opposite sign).
    center : tuple of float
        In-plane center ``(cx, cy)``.

    Profile: ``theta(rho) = pi * exp(-rho^2 / 2 radius^2)`` so ``m = -z`` at the
    core and ``m = +z`` in the background. Returns shape ``(3, nx, ny, nz)``.
    """
    np = xp()
    X, Y, _Z = grid.coords()
    cx, cy = center
    rho = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    phi = np.arctan2(Y - cy, X - cx)
    theta = np.pi * np.exp(-rho * rho / (2.0 * radius * radius))
    m_rho = np.sin(theta)
    m_z = np.cos(theta)
    angle = vorticity * phi + helicity
    m_x = m_rho * np.cos(angle)
    m_y = m_rho * np.sin(angle)
    return np.stack([m_x, m_y, m_z], axis=0)


def uniform(grid: Grid, direction=(0.0, 0.0, 1.0)):
    """Uniform magnetization (e.g. ferromagnetic ground state)."""
    np = xp()
    dx, dy, dz = direction
    norm = (dx * dx + dy * dy + dz * dz) ** 0.5
    dx, dy, dz = dx / norm, dy / norm, dz / norm
    shape = grid.shape
    return np.stack(
        [
            np.full(shape, dx),
            np.full(shape, dy),
            np.full(shape, dz),
        ],
        axis=0,
    )


def add_perturbation(m, amplitude: float = 0.01, seed: int | None = None):
    """Add a small random tangential perturbation, then renormalize."""
    import numpy as _np
    from hopfion.grid import normalize

    rng = _np.random.default_rng(seed)
    noise = rng.normal(size=m.shape) * amplitude
    np = xp()
    perturbed = m + np.asarray(noise)
    return normalize(perturbed)
