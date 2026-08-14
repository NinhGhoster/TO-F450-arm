"""Cross-validate the voxel pipeline against CalculiX on the OEM arm.

The topology results come from the in-house voxel FEM; the OEM baseline comes
from CalculiX on a tet mesh of the real CAD solid. Comparing the two is only
meaningful if the voxel pipeline reproduces the CAD answer for the *same*
geometry. That is what this script checks.

The OEM solid is voxelised onto the production 216 x 42 x 54 mm / 2 mm grid,
given rho = 1 inside and the SIMP floor outside, and run through
``FEM3D.modal_analysis`` with the same 60 g tip mass. If the first mode lands
near CalculiX's 16.8 Hz the pipeline is trustworthy for the sweep; if it does
not, nothing downstream is.

Usage::

    ./venv/bin/python -m sims.validate_voxel_vs_cad
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from src.fem import FEM3D, Material
from src.mesh3d import ArmGeometry3D

ARM_VTP = "figures/inspect/_arm_fem_frame.vtp"
TIP_MASS_KG = 0.060
RHO_MIN = 1e-3


def voxelise(geom):
    """rho = 1 for element centres inside the OEM solid, RHO_MIN outside."""
    arm = pv.read(ARM_VTP)
    b = np.array(arm.bounds)
    # The cache has x = 0 at the MOTOR end; the FEM domain has x = 0 at the
    # FRAME end, so mirror along x. A mirrored solid has identical mass and
    # stiffness properties, so this does not affect the comparison.
    p = arm.points.copy()
    p[:, 0] = b[1] - p[:, 0]
    arm.points = p

    cx = (np.arange(geom.nx) + 0.5) * geom.dx * 1e3
    cy = (np.arange(geom.ny) + 0.5) * geom.dy * 1e3
    cz = (np.arange(geom.nz) + 0.5) * geom.dz * 1e3
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    pts = pv.PolyData(np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]))
    sel = pts.select_enclosed_points(arm.extract_surface(), tolerance=1e-6,
                                     check_surface=False)
    inside = sel["SelectedPoints"].astype(bool).reshape(X.shape)
    return inside


def main():
    geom = ArmGeometry3D()
    mat = Material()
    fem = FEM3D(geom, mat)

    inside = voxelise(geom)
    ex, ey, ez = fem.elem_grid_idx.T
    rho = np.where(inside[ex, ey, ez], 1.0, RHO_MIN)

    vox_mm3 = geom.dx * geom.dy * geom.dz * 1e9
    mass_g = rho[rho > 0.5].sum() * vox_mm3 * mat.rho * 1e-6
    print(f"voxelised OEM arm: {int((rho > 0.5).sum())} solid voxels")
    print(f"   mass {mass_g:.1f} g   (CAD solid is 34.5 g)")

    freqs, _, _ = fem.modal_analysis(
        rho, n_modes=6, tip_mass_kg=TIP_MASS_KG,
        tip_mass_nodes=geom.motor_mount_top_nodes())
    print(f"   voxel omega_1 = {freqs[0]:.1f} Hz   (CalculiX on the CAD: 16.8 Hz)")
    print("   first six:", np.round(freqs, 1))
    d = 100.0 * (freqs[0] - 16.8) / 16.8
    print(f"   difference on the first mode: {d:+.1f} %")


if __name__ == "__main__":
    main()
