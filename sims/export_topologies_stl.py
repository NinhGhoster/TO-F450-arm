"""Export SIMP iso-surfaces from each V_f result pkl as binary STL.

These STLs go in figure_pack_for_artist/topologies/ and are handed to the
CAD artist for use as the source geometry in Figures 2, 8, and 12 of the
v4 manuscript.

Reuses the same _extract_smooth_surface helper that the PyVista renderer
uses, so the STL the artist receives matches exactly what shows up in
the current PNGs.

The PyVista mesh is in metres; STL output is in millimetres to match
SolidWorks / Fusion / KeyShot conventions.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pyvista as pv

from src.mesh3d import ArmGeometry3D
from src.fem import FEM3D, Material
from src.plotting_3d import _extract_smooth_surface


VOL_FRACS = (0.10, 0.20, 0.30, 0.50, 0.70)
NOMINAL_OMEGA_Hz = 500

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_3d")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..",
                        "figure_pack_for_artist", "topologies")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    geom = ArmGeometry3D()
    fem = FEM3D(geom, Material())

    for vf in VOL_FRACS:
        pkl_path = os.path.join(
            RESULTS_DIR,
            f"freq_vf{int(vf*100):02d}_omega{NOMINAL_OMEGA_Hz:04d}.pkl",
        )
        if not os.path.exists(pkl_path):
            print(f"  [skip] {pkl_path} missing")
            continue
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)

        surface = _extract_smooth_surface(geom, fem, data["rho"], iso=0.5)
        tri = surface.triangulate()
        # PyVista uses metres; scale to millimetres for CAD interop
        tri.points = tri.points * 1000.0

        out = os.path.join(
            OUT_DIR, f"modal_vf{int(vf*100):02d}_omega{NOMINAL_OMEGA_Hz:04d}.stl",
        )
        tri.save(out, binary=True)
        n_tri = tri.n_cells
        n_pts = tri.n_points
        print(f"  V_f = {vf:.2f}  →  {out}  ({n_tri} tris, {n_pts} verts)")

    print("\nAll topology STLs written.")


if __name__ == "__main__":
    main()
