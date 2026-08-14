"""Show the boundary conditions and loads the solver actually applies.

This is not a CAD mock-up of the joint — it reads ``ArmGeometry3D.fixed_nodes()``
and ``load_dofs_for_force()`` directly, so what appears on screen is exactly the
constraint and load set the FEM uses. A drawing made in CAD would show a
*reconstruction* of the boundary conditions; the point of this figure is to
check the real ones.

Renders the design domain semi-transparent with:
  * every constrained node marked (all 3 DoF fixed) — the six screw bearing
    annuli, four in the top mounting pad and two in the root-side leg;
  * the loaded nodes at the motor mount, with an arrow per load case scaled
    to the applied force.

Usage::

    ./venv/bin/python -m sims.render_bc_loads
"""
from __future__ import annotations

import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from sims.render_figures_v5 import (DOMAIN_MM, GREY, INK, PA12, _aim, plotter,
                                    shot, trim, geom_fem)

pv.OFF_SCREEN = True
OUT = "figures"
FIX_COLOR = "#2f6ba8"
LOAD_COLOR = "#a83232"

LOAD_CASES = [
    ("LC1 Hover",       (0.0, 0.0, 2.94)),
    ("LC2 Maneuver",    (0.0, 0.0, 5.88)),
    ("LC3 Hard landing", (0.0, 0.0, -14.7)),
    ("LC4 Banked",      (0.0, 2.94, 5.0922)),
]


def node_xyz(geom, idx):
    """Node indices -> mm coordinates."""
    nx1, ny1 = geom.nx + 1, geom.ny + 1
    iz = idx // (nx1 * ny1)
    iy = (idx % (nx1 * ny1)) // nx1
    ix = idx % nx1
    return np.column_stack([ix * geom.dx, iy * geom.dy, iz * geom.dz]) * 1e3


def build(view, size, show_loads=True):
    geom, fem = geom_fem()
    surf = pv.Box(bounds=DOMAIN_MM).triangulate()

    fixed = node_xyz(geom, geom.fixed_nodes())
    load_nodes, _ = geom.load_dofs_for_force((0.0, 0.0, 1.0))
    loaded = node_xyz(geom, np.unique(load_nodes // 3))

    pl = plotter(size)
    pl.add_mesh(surf, color="#c8d4e0", opacity=0.16, show_edges=True,
                edge_color="#9fb3c8", line_width=1)
    pl.add_mesh(pv.PolyData(fixed), color=FIX_COLOR, point_size=9,
                render_points_as_spheres=True)
    pl.add_mesh(pv.PolyData(loaded), color=LOAD_COLOR, point_size=9,
                render_points_as_spheres=True)

    if show_loads:
        c = loaded.mean(axis=0)
        for _, F in LOAD_CASES:
            F = np.array(F, float)
            n = np.linalg.norm(F)
            d = F / n
            L = 16 + 26 * n / 14.7                     # length scales with |F|
            start = c - d * L if F[2] > 0 else c
            pl.add_mesh(pv.Arrow(start=start, direction=d, scale=L,
                                 tip_length=0.28, tip_radius=0.075,
                                 shaft_radius=0.028),
                        color=LOAD_COLOR, ambient=0.3)
    _aim(pl, DOMAIN_MM, size, direction=view, margin=1.10)
    return trim(shot(pl)), len(fixed), len(loaded)


def main():
    os.makedirs(OUT, exist_ok=True)
    size = (1700, 950)
    img_iso, n_fix, n_load = build((0.55, -1.0, 0.42), size)
    img_root, _, _ = build((0.30, -1.0, 0.75), (1300, 1000), show_loads=False)

    fig = plt.figure(figsize=(8.0, 4.9), dpi=300)
    a1 = fig.add_axes([0.015, 0.30, 0.635, 0.60])
    a1.imshow(img_iso); a1.set_xticks([]); a1.set_yticks([])
    a2 = fig.add_axes([0.665, 0.30, 0.325, 0.60])
    a2.imshow(img_root); a2.set_xticks([]); a2.set_yticks([])
    for ax in (a1, a2):
        for s in ax.spines.values():
            s.set_visible(False)

    fig.text(0.015, 0.975, "Boundary conditions and applied loads "
             "(as implemented in the solver)", ha="left", va="top",
             fontsize=12.5, color=INK)
    fig.text(0.015, 0.93,
             f"Blue: {n_fix} nodes fixed in all 3 DoF — the six screw bearing "
             f"annuli.   Red: {n_load} loaded nodes at the motor mount.",
             ha="left", va="top", fontsize=9, color=GREY)
    fig.text(0.20, 0.245, "(a) design domain, loads shown", ha="center",
             fontsize=9.5, color=INK)
    fig.text(0.828, 0.245, "(b) root, looking down", ha="center",
             fontsize=9.5, color=INK)
    rows = "    ".join(f"{n}: {np.linalg.norm(F):.2f} N" for n, F in LOAD_CASES)
    fig.text(0.015, 0.175, rows, ha="left", va="top", fontsize=8.5,
             color=LOAD_COLOR)
    note = textwrap.fill(
        "Four screws bear on the top mounting pad (z = 47.5-52.6 mm); two bear "
        "on the root-side leg (z = 0-4 mm), 38 mm below, matching the measured "
        "clear gap between the carbon plates. LC5 (propeller torque, "
        "0.10 N\u00b7m about z) is applied as a couple on the same mount and "
        "is not drawn.", 118)
    fig.text(0.015, 0.125, note, ha="left", va="top", fontsize=8.5, color=GREY)
    path = f"{OUT}/fig05_bc_loads.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}  ({n_fix} fixed nodes, {n_load} loaded nodes)")


if __name__ == "__main__":
    main()
