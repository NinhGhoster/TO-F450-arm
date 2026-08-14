"""Look at the OEM arm root: render it from every side and list its holes.

Written because the fastener joint could not be settled from numbers alone.
The part is tessellated once from ``F450_Arm.stp`` and then rendered from
top, bottom, side and iso with the root region isolated, so the underside can
actually be seen rather than inferred.

Coordinates are mapped from the CAD (Y-up) into the FEM frame used by
``ArmGeometry3D``:  x = X - Xmin (along the arm from the root),
y = Z - Zmin (across the width), z = Y - Ymin (up).

Usage::

    ./venv/bin/python -m sims.inspect_arm_root
"""
from __future__ import annotations

import os

import numpy as np
import pyvista as pv

from sims.render_figures_v5 import plotter, shot, trim, add_part, PA12, _aim

pv.OFF_SCREEN = True

ARM_STEP = "figure_pack_for_artist/cad/F450_Arm.stp"
OUT = "figures/inspect"
ROOT_X = 62.0          # mm from the root — covers the whole joint region


CACHE = "figures/inspect/_arm_fem_frame.vtp"


def arm_in_fem_frame():
    """Tessellated arm, mapped CAD (Y-up) -> FEM frame (Z-up), origin at the
    inboard-bottom-near corner."""
    from sims.render_figures_v5 import _step_solids
    if os.path.exists(CACHE):
        return pv.read(CACHE)
    solids = [s for s in _step_solids(ARM_STEP) if s is not None]
    arm = solids[0] if len(solids) == 1 else \
        max(solids, key=lambda m: m.n_cells)
    p = arm.points.copy()
    arm.points = np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])
    b = np.array(arm.bounds)
    arm.points = arm.points - np.array([b[0], b[2], b[4]])
    os.makedirs(OUT, exist_ok=True)
    arm.save(CACHE)
    return arm


def views(arm):
    b = np.array(arm.bounds)
    print(f"arm bbox in FEM frame: x 0..{b[1]:.1f}  y 0..{b[3]:.1f}  "
          f"z 0..{b[5]:.1f} mm")
    # Keep x < ROOT_X. A plane clip is unambiguous; clip_box's invert
    # convention kept the wrong side and left only slivers.
    root = arm.clip("x", origin=(ROOT_X, 0, 0), invert=True)
    print(f"root region: {root.n_cells} cells "
          f"(whole arm {arm.n_cells})")
    os.makedirs(OUT, exist_ok=True)
    box = (0, ROOT_X, 0, b[3], 0, b[5])
    specs = [
        ("root_top",    (0.0, 0.001, 1.0),  "TOP  (looking down)"),
        ("root_bottom", (0.0, 0.001, -1.0), "BOTTOM (looking up) "
                                            "<- bottom screw holes appear here"),
        ("root_side",   (0.0, -1.0, 0.05),  "SIDE"),
        ("root_iso",    (0.5, -1.0, 0.45),  "ISO"),
        ("root_inboard", (-1.0, 0.05, 0.05), "INBOARD END-ON"),
    ]
    for name, direction, label in specs:
        size = (1500, 1100)
        pl = plotter(size)
        add_part(pl, root, PA12, ambient=0.30, diffuse=0.85)
        _aim(pl, box, size, direction=direction, margin=1.05)
        img = trim(shot(pl))
        path = f"{OUT}/{name}.png"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(7, 7 * img.shape[0] / img.shape[1] + 0.35),
                         dpi=200)
        ax = fig.add_axes([0.01, 0.01, 0.98, 0.90])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        fig.text(0.02, 0.975, f"OEM F450 arm root — {label}",
                 ha="left", va="top", fontsize=11)
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
        print(f"  wrote {path}")


if __name__ == "__main__":
    arm = arm_in_fem_frame()
    views(arm)
