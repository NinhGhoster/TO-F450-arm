"""Render the five-load-case response of both the reference and the design.

The manuscript reports five load cases in its tables but was showing only one
of them as a picture, and only for the OEM arm. That makes it impossible to see
*where* each case loads the part, or how the optimised design redistributes
stress relative to the arm it replaces.

Two combined figures are produced, each with five sub-panels sharing one colour
scale so the cases can be compared against one another:

``fig05_oem_loadcases``
    The OEM arm, from the CalculiX tet solution.

``fig12_opt_loadcases``
    The equal-mass optimised design (V_f = 0.08), from the in-house voxel
    solution.

Each panel shows the deformed shape with the undeformed geometry ghosted
behind, so bending is visible alongside stress. The exaggeration factor is
common to every panel in both figures, which keeps relative deflections honest
and lets the two figures be compared with each other.

The two figures deliberately do **not** share a stress scale: the OEM arm peaks
at 22.4 MPa and the optimised design two orders of magnitude lower, so a common
scale would render one of them blank. The peak and safety factor are printed on
each panel instead.

Usage::

    QARM_RESULTS_DIR=results_3d_v6 ./venv/bin/python -m sims.render_loadcases
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from src.mesh3d import ArmGeometry3D
from sims.read_frd import von_mises
from sims.render_figures_v5 import (DOMAIN_MM, RESULTS, _aim, panel_size,
                                    plotter, save_column, shot,
                                    solid_surface_mm, trim)
from sims.render_oem_figures import read_frd_blocks, read_mesh

FIGURES = "figures"
SIGMA_Y = 38.0
WARP_MM = 6.0          # governing case is drawn with this much visible deflection

CASES = [
    ("LC1_hover",      "LC1 hover",            "$F_z$ = 2.94 N"),
    ("LC2_maneuver",   "LC2 maneuver (2 g)",   "$F_z$ = 5.88 N"),
    ("LC3_landing",    "LC3 hard landing",     "$F_z$ = $-$14.7 N"),
    ("LC4_banked",     "LC4 banked turn",      "$F$ = (0, 2.94, 5.09) N"),
    ("LC5_proptorque", "LC5 propeller torque", "$M_z$ = 0.10 N$\\cdot$m"),
]


def _panel(box, mesh, scal, clim, warp=None, cmap="inferno"):
    """One panel with the shared camera, optionally drawn deformed.

    ``warp`` is a per-point displacement in mm. When given, the deformed mesh
    is drawn over a ghost of the undeformed one -- the same treatment the
    mode-shape figure uses.
    """
    size = panel_size(box, width_px=1500)
    pl = plotter(size)
    if warp is not None:
        pl.add_mesh(mesh.copy(), color="#cfd6de", opacity=0.20,
                    show_scalar_bar=False, smooth_shading=True)
        m = mesh.copy()
        m.point_data["_w"] = warp
        mesh = m.warp_by_vector("_w")
    pl.add_mesh(mesh, scalars=scal, cmap=cmap, clim=clim,
                show_scalar_bar=False, smooth_shading=True,
                specular=0.22, specular_power=18)
    _aim(pl, box, size)
    return trim(shot(pl))


def oem_load_cases(dpi=300):
    """One combined figure: the OEM arm under all five cases."""
    grid, ids = read_mesh()
    vm_f, u_f = {}, {}
    for tag, _, _ in CASES:
        st = read_frd_blocks(f"fea_oem/{tag}/ArmMesh.frd", "STRESS", 6)[-1]
        vm_f[tag] = von_mises(np.array([st[n] for n in ids]))
        dp = read_frd_blocks(f"fea_oem/{tag}/ArmMesh.frd", "DISP", 3)[-1]
        u_f[tag] = np.array([dp[n] for n in ids])
    vmax = float(np.ceil(max(v.max() for v in vm_f.values())))
    scale = WARP_MM / max(np.abs(u_f["LC3_landing"]).max(), 1e-9)

    images, labels = [], []
    for tag, name, load in CASES:
        g = grid.copy()
        g.point_data["vm"] = vm_f[tag]
        g.point_data["_w"] = u_f[tag] * scale
        surf = g.extract_surface()
        images.append(_panel(grid.bounds, surf, "vm", (0.0, vmax),
                             warp=surf.point_data["_w"]))
        tip = np.linalg.norm(u_f[tag], axis=1).max()
        labels.append(f"{name}, {load} — peak {vm_f[tag].max():.2f} MPa, "
                      f"SF$_y$ = {SIGMA_Y / vm_f[tag].max():.2f}, "
                      f"max deflection {tip:.2f} mm")
        print(f"  {name:<22} peak {vm_f[tag].max():7.3f} MPa   "
              f"SF {SIGMA_Y / vm_f[tag].max():7.2f}   u {tip:6.2f} mm")

    save_column(images, labels, f"{FIGURES}/fig05_oem_loadcases.png",
                title="OEM F450 arm in SLS-PA12 under the five flight load cases",
                cbar=(0.0, vmax, "inferno", "von Mises stress (MPa)"),
                dpi=dpi)


def optimised_load_cases(vf=0.08, dpi=300):
    """The same five cases on the equal-mass optimised design."""
    geom = ArmGeometry3D()
    with open(os.path.join(RESULTS,
              f"freq_vf{int(vf * 100):02d}_omega0500.pkl"), "rb") as fh:
        d = pickle.load(fh)
    rho = d["rho"]
    solid = rho > 0.5
    vms = {t: d["static"][t]["vm"] / 1e6 for t, _, _ in CASES}
    us = {t: d["static"][t]["u"].reshape(-1, 3) * 1e3 for t, _, _ in CASES}
    vmax = float(np.ceil(max(v[solid].max() for v in vms.values()) * 100) / 100)
    scale = WARP_MM / max(np.abs(us["LC3_landing"]).max(), 1e-9)

    nx1, ny1 = geom.nx + 1, geom.ny + 1
    dx, dy, dz = geom.dx * 1e3, geom.dy * 1e3, geom.dz * 1e3

    images, labels = [], []
    for tag, name, load in CASES:
        vm = np.where(solid, vms[tag], 0.0)
        surf = solid_surface_mm(rho, extra_fields={"vm": vm}, min_body_frac=0)
        # Sample the nodal displacement field onto the surface points by
        # nearest grid node; the surface is a smoothed cell boundary, so it
        # carries no node indices of its own.
        p = surf.points
        ix = np.clip(np.rint(p[:, 0] / dx).astype(int), 0, geom.nx)
        iy = np.clip(np.rint(p[:, 1] / dy).astype(int), 0, geom.ny)
        iz = np.clip(np.rint(p[:, 2] / dz).astype(int), 0, geom.nz)
        w = us[tag][iz * nx1 * ny1 + iy * nx1 + ix] * scale
        images.append(_panel(DOMAIN_MM, surf, "vm", (0.0, vmax), warp=w))
        peak = vms[tag][solid].max()
        tip = np.linalg.norm(us[tag], axis=1).max()
        labels.append(f"{name}, {load} — peak {peak:.3f} MPa, "
                      f"SF$_y$ = {SIGMA_Y / peak:.0f}, "
                      f"max deflection {tip:.3f} mm")
        print(f"  {name:<22} peak {peak:7.4f} MPa   SF {SIGMA_Y / peak:7.1f}   "
              f"u {tip:6.3f} mm")

    # Split across two figures: five stacked panels overruns a page.
    save_column(images[:3], labels[:3],
                f"{FIGURES}/fig11a_opt_loadcases_thrust.png",
                title=f"Optimised design ($V_f$ = {vf:.2f}, 44.7 g): "
                      f"thrust and landing cases",
                cbar=(0.0, vmax, "inferno", "von Mises stress (MPa)"), dpi=dpi)
    save_column(images[3:], labels[3:],
                f"{FIGURES}/fig11b_opt_loadcases_banked_torque.png",
                title=f"Optimised design ($V_f$ = {vf:.2f}, 44.7 g): "
                      f"banked turn and propeller torque",
                cbar=(0.0, vmax, "inferno", "von Mises stress (MPa)"), dpi=dpi)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    print("OEM arm (CalculiX tets):")
    oem_load_cases()
    print("\nOptimised design V_f = 0.08 (in-house voxels):")
    optimised_load_cases()
    # The per-case singles are superseded by the combined figures.
    for f in [f"{FIGURES}/{p}{k}.png" for p in ("fig05_oem_vm_LC",
              "fig12_opt_vm_LC") for k in range(1, 6)] + \
             [f"{FIGURES}/fig12_opt_loadcases.png"]:
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
