"""Render the OEM-arm reference figures from the CalculiX results.

The OEM arm is the design every optimised topology is judged against, so it
needs the same visual treatment as the optimised ones: identical camera,
identical colour map, identical typography.  The 3D pixels come from PyVista
and every piece of text from Matplotlib, which is the same split the topology
figures use.

Two figures are produced:

``fig03a_oem_vm_LC3``
    von Mises stress under the governing hard-landing case, on the OEM
    geometry.  This is the figure that shows where the safety factor of 1.70
    is spent.

``fig03b_oem_mode1``
    The 16.8 Hz first bending mode, warped for legibility.  This is the figure
    that shows the arm is sub-critical.

Usage::

    ./venv/bin/python -m sims.render_oem_figures
"""
from __future__ import annotations

import os

import numpy as np
import pyvista as pv

from sims.read_frd import von_mises
from sims.render_figures_v5 import (INK, _aim, panel_size, plotter, save_single,
                                    shot, trim)

FEA = "fea_oem"
FIGURES = "figures"
SIGMA_Y = 38.0


def read_mesh(path=f"{FEA}/LC2_maneuver/ArmMesh.inp"):
    """Build a PyVista tet grid from the CalculiX deck.

    Read from the deck rather than from the CAD so the picture shows the mesh
    that was actually solved.
    """
    lines = open(path, errors="ignore").read().splitlines()
    ids, xyz, conn, block = [], [], [], None
    for l in lines:
        if l.startswith("**"):
            continue
        if l.startswith("*"):
            u = l.upper()
            block = ("node" if u.startswith("*NODE") else
                     "elem" if u.startswith("*ELEMENT") and "C3D4" in u
                     else None)
            continue
        if not l.strip() or block is None:
            continue
        p = [x.strip() for x in l.split(",") if x.strip()]
        if block == "node" and len(p) >= 4:
            ids.append(int(p[0]))
            xyz.append([float(p[1]), float(p[2]), float(p[3])])
        elif block == "elem" and len(p) >= 5:
            conn.append([int(x) for x in p[1:5]])

    ids = np.asarray(ids)
    pts = np.asarray(xyz)
    # The OEM CAD frame puts x = 0 at the MOTOR end; the design domain and
    # every topology figure put x = 0 at the FRAME end.  Turn the arm end for
    # end so it faces the same way as everything it is compared against —
    # otherwise the reader sees the root on opposite sides of adjacent
    # figures.
    #
    # Use a 180 deg rotation about z (x -> -x, y -> -y) rather than a bare
    # x-mirror: a mirror is improper, so it would invert every tet and leave
    # the surface normals pointing inwards, which wrecks the shading.  The arm
    # is symmetric about its y centreline, so the rotation looks identical to
    # the mirror.  Then shift each axis back to start at zero.
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]
    pts -= pts.min(axis=0)
    lut = np.zeros(ids.max() + 1, dtype=np.int64)
    lut[ids] = np.arange(len(ids))
    c = lut[np.asarray(conn)]
    cells = np.hstack([np.full((len(c), 1), 4, dtype=np.int64), c]).ravel()
    grid = pv.UnstructuredGrid(cells,
                               np.full(len(c), pv.CellType.TETRA, dtype=np.uint8),
                               pts)
    return grid, ids


def read_frd_blocks(path, want="DISP", ncomp=3):
    """Every block of one result type, in file order.

    ``sims.read_frd`` keeps only the last block, which is right for a static
    run but loses five of the six modes in a frequency run.
    """
    out, cur, grabbing = [], {}, False
    for line in open(path, errors="ignore"):
        if line.startswith(" -4"):
            name = line.split()[1].upper()
            if name.startswith(want):
                if cur:
                    out.append(cur)
                cur, grabbing = {}, True
            else:
                grabbing = False
            continue
        if line.startswith(" -3"):
            grabbing = False
            continue
        if grabbing and line.startswith(" -1"):
            body = line[3:]
            try:
                node = int(body[:10])
                v = [float(body[10 + 12 * i: 22 + 12 * i]) for i in range(ncomp)]
            except ValueError:
                p = line.split()
                node, v = int(p[1]), [float(x) for x in p[2:2 + ncomp]]
            cur[node] = v
    if cur:
        out.append(cur)
    return out


def _render(grid, scalars, clim, cmap, label, panel_w=1800):
    box = grid.bounds
    size = panel_size(box, width_px=panel_w)
    pl = plotter(size)
    # Attach to the volume grid before extracting: extract_surface() drops
    # interior points, so scalars indexed by the volume node list have to be
    # carried through the filter rather than applied after it.
    grid.point_data[label] = scalars
    surf = grid.extract_surface()
    pl.add_mesh(surf, scalars=label, cmap=cmap, clim=clim,
                show_scalar_bar=False, smooth_shading=True,
                specular=0.25, specular_power=18)
    _aim(pl, box, size)
    return trim(shot(pl))


def fig_oem_vm(dpi=300):
    grid, ids = read_mesh()
    blocks = read_frd_blocks(f"{FEA}/LC3_landing/ArmMesh.frd", "STRESS", 6)
    s = blocks[-1]
    vm = von_mises(np.array([s[n] for n in ids]))
    img = _render(grid, vm, (0.0, float(np.ceil(vm.max()))), "inferno", "vm")
    out = f"{FIGURES}/fig03a_oem_vm_LC3.png"
    save_single(img, out, title="OEM arm — von Mises stress, hard landing",
                cbar=(0.0, float(np.ceil(vm.max())), "inferno",
                      "von Mises stress (MPa)"), dpi=dpi)
    print(f"{out}   peak {vm.max():.2f} MPa   SF_y = {SIGMA_Y/vm.max():.2f}")


def fig_oem_mode1(dpi=300, warp_frac=0.12):
    grid, ids = read_mesh()
    modes = read_frd_blocks(f"{FEA}/MODAL/ArmMesh.frd", "DISP", 3)
    d = np.array([modes[0][n] for n in ids])
    mag = np.linalg.norm(d, axis=1)
    mag /= mag.max()

    # Warp by a fixed fraction of the arm length so the mode reads at a glance.
    # Mode shapes carry no absolute scale, so the exponent is presentational.
    span = max(grid.bounds[1] - grid.bounds[0], 1.0)
    g = grid.copy()
    g.point_data["u"] = d / np.abs(d).max() * span * warp_frac
    g.point_data["mag"] = mag
    warped = g.warp_by_vector("u")

    box = grid.bounds
    size = panel_size(box, width_px=1800)
    pl = plotter(size)
    pl.add_mesh(grid.extract_surface(), color="#cfd6de", opacity=0.22,
                show_scalar_bar=False, smooth_shading=True)
    pl.add_mesh(warped.extract_surface(), scalars="mag", cmap="viridis",
                clim=(0, 1), show_scalar_bar=False, smooth_shading=True,
                specular=0.25, specular_power=18)
    _aim(pl, box, size)
    img = trim(shot(pl))
    out = f"{FIGURES}/fig03b_oem_mode1.png"
    save_single(img, out, title="OEM arm — first bending mode, 16.8 Hz",
                cbar=(0.0, 1.0, "viridis",
                      "normalised displacement |u| / |u|max"), dpi=dpi)
    print(f"{out}   (undeformed outline shown at 22 % opacity)")


def main():
    os.makedirs(FIGURES, exist_ok=True)
    fig_oem_vm()
    fig_oem_mode1()


if __name__ == "__main__":
    main()
