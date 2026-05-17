"""Visualization helpers.

Two backends:

* ``pyvista`` (optional dep): 3D preimage-tube rendering of hopfions. Strongly
  recommended for hopfion topology (the linked rings are striking).
* matplotlib fallback: 2D slice quiver plots and 3D scatter of preimage cells.

All functions take a precomputed ``m`` field; they don't run any dynamics.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as _np

from hopfion.backend import to_numpy
from hopfion.grid import Grid
from hopfion.topology import preimage_mask


def slice_quiver(m, grid: Grid, plane: str = "xy", index: Optional[int] = None, stride: int = 2, ax=None):
    """Plot a 2D slice quiver of the magnetization."""
    import matplotlib.pyplot as plt
    m_np = to_numpy(m)
    if plane == "xy":
        if index is None:
            index = grid.nz // 2
        U = m_np[0, ::stride, ::stride, index]
        V = m_np[1, ::stride, ::stride, index]
        C = m_np[2, ::stride, ::stride, index]
        x = (_np.arange(grid.nx) - grid.nx / 2 + 0.5)[::stride] * grid.dx
        y = (_np.arange(grid.ny) - grid.ny / 2 + 0.5)[::stride] * grid.dy
        X, Y = _np.meshgrid(x, y, indexing="ij")
    elif plane == "xz":
        if index is None:
            index = grid.ny // 2
        U = m_np[0, ::stride, index, ::stride]
        V = m_np[2, ::stride, index, ::stride]
        C = m_np[1, ::stride, index, ::stride]
        x = (_np.arange(grid.nx) - grid.nx / 2 + 0.5)[::stride] * grid.dx
        z = (_np.arange(grid.nz) - grid.nz / 2 + 0.5)[::stride] * grid.dz
        X, Y = _np.meshgrid(x, z, indexing="ij")
    elif plane == "yz":
        if index is None:
            index = grid.nx // 2
        U = m_np[1, index, ::stride, ::stride]
        V = m_np[2, index, ::stride, ::stride]
        C = m_np[0, index, ::stride, ::stride]
        y = (_np.arange(grid.ny) - grid.ny / 2 + 0.5)[::stride] * grid.dy
        z = (_np.arange(grid.nz) - grid.nz / 2 + 0.5)[::stride] * grid.dz
        X, Y = _np.meshgrid(y, z, indexing="ij")
    else:
        raise ValueError(plane)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    q = ax.quiver(X, Y, U, V, C, cmap="coolwarm", scale=30)
    ax.set_aspect("equal")
    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])
    return ax, q


def preimage_scatter(m, grid: Grid, targets=None, tol: float = 0.1, ax=None):
    """3D scatter plot of preimage cells -- the linked-loop signature of a hopfion."""
    import matplotlib.pyplot as plt
    if targets is None:
        # +x and -x preimages (linked rings characteristic of Q_H=1)
        targets = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    X, Y, Z = grid.coords()
    Xnp, Ynp, Znp = to_numpy(X), to_numpy(Y), to_numpy(Z)
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
    for k, target in enumerate(targets):
        mask = to_numpy(preimage_mask(m, target, tol=tol))
        ax.scatter(Xnp[mask], Ynp[mask], Znp[mask], s=4, color=colors[k % len(colors)], alpha=0.4, label=f"m={target}")
    ax.legend()
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return ax


def preimage_pyvista(m, grid: Grid, targets=None, tol: float = 0.15):
    """Render preimage tubes via PyVista's marching-cubes isosurface.

    Returns the ``pyvista.Plotter``. Call ``.show()`` interactively, or
    ``.screenshot(fname)`` for headless rendering.
    """
    try:
        import pyvista as pv
    except ImportError as e:
        raise ImportError(
            "preimage_pyvista requires pyvista. Install with `pip install hopfion[viz]`."
        ) from e
    m_np = to_numpy(m)
    if targets is None:
        targets = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]
    spacing = (grid.dx, grid.dy, grid.dz)
    origin = (
        -0.5 * (grid.nx - 1) * grid.dx,
        -0.5 * (grid.ny - 1) * grid.dy,
        -0.5 * (grid.nz - 1) * grid.dz,
    )
    grid_pv = pv.ImageData(dimensions=grid.shape, spacing=spacing, origin=origin)
    plotter = pv.Plotter()
    colors = ["red", "blue", "green", "orange"]
    for k, target in enumerate(targets):
        tx, ty, tz = target
        norm = (tx * tx + ty * ty + tz * tz) ** 0.5
        tx, ty, tz = tx / norm, ty / norm, tz / norm
        f = ((m_np[0] - tx) ** 2 + (m_np[1] - ty) ** 2 + (m_np[2] - tz) ** 2).ravel(order="F")
        grid_pv[f"f_{k}"] = f
        iso = grid_pv.contour([tol * tol], scalars=f"f_{k}")
        plotter.add_mesh(iso, color=colors[k % len(colors)], opacity=0.7)
    return plotter
