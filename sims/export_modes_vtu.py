"""Export the first four mode shapes of the V_f = 0.50 modal-optimised arm.

Produces, in figure_pack_for_artist/modes/:

  modal_vf50_mode1.vtp  — iso-surface with u_mag scalar (open in ParaView)
  modal_vf50_mode2.vtp
  modal_vf50_mode3.vtp
  modal_vf50_mode4.vtp
  mode_data.csv          — frequency table + mode-type tags

Reuses the per-element-magnitude pattern from
src.plotting_3d.plot_mode_shape_3d, so the VTU surface the artist sees in
ParaView matches exactly what the manuscript PNG shows.

Surfaces are exported in millimetres.
"""
from __future__ import annotations

import csv
import os
import pickle

import numpy as np

from src.mesh3d import ArmGeometry3D
from src.fem import FEM3D, Material
from src.plotting_3d import _extract_smooth_surface


NOMINAL_VF = 0.50
NOMINAL_OMEGA_Hz = 500
N_MODES = 4
MODE_TYPES = (
    "vertical bending",
    "lateral bending",
    "first torsion",
    "higher bending",
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_3d")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..",
                        "figure_pack_for_artist", "modes")
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> None:
    pkl = os.path.join(
        RESULTS_DIR,
        f"freq_vf{int(NOMINAL_VF*100):02d}_omega{NOMINAL_OMEGA_Hz:04d}.pkl",
    )
    with open(pkl, "rb") as fh:
        data = pickle.load(fh)

    geom = ArmGeometry3D()
    fem = FEM3D(geom, Material())

    freqs = np.asarray(data["modal"]["frequencies_Hz"])
    modes = np.asarray(data["modal"]["mode_shapes"])  # (n_dofs, n_modes_stored)
    rho = data["rho"]

    csv_path = os.path.join(OUT_DIR, "mode_data.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mode_index", "frequency_Hz", "mode_type"])

        for k in range(min(N_MODES, modes.shape[1])):
            mode_shape = modes[:, k]
            u_node = mode_shape.reshape(-1, 3)
            u_mag_node = np.linalg.norm(u_node, axis=1)
            u_mag_elem = u_mag_node[fem.elem_node_ids].mean(axis=1)
            u_max = float(u_mag_elem.max()) if u_mag_elem.max() > 0 else 1.0
            u_mag_norm = u_mag_elem / u_max

            surface = _extract_smooth_surface(
                geom, fem, rho,
                extra_fields={"u_mag": u_mag_norm},
                iso=0.5,
            )
            surface.points = surface.points * 1000.0  # to mm

            out = os.path.join(OUT_DIR, f"modal_vf50_mode{k+1}.vtp")
            surface.save(out)

            mode_type = MODE_TYPES[k] if k < len(MODE_TYPES) else "—"
            writer.writerow([k + 1, f"{freqs[k]:.1f}", mode_type])
            print(f"  Mode {k+1}: {freqs[k]:.1f} Hz ({mode_type}) → {out}")

    print(f"\nCSV summary: {csv_path}")


if __name__ == "__main__":
    main()
