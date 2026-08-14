"""Export the modal problem as an Abaqus input deck for independent checking.

The point of this module is *not* to produce another figure. It is to let a
solver that neither this project nor its author wrote reproduce the headline
modal result, so the number in the manuscript does not rest on the in-house
implementation alone. The emitted ``.inp`` is plain Abaqus keyword format, so
the identical file runs in CalculiX (open source) and in Abaqus/Standard.

Two decks are written, because they answer two different questions.

``simp``
    Every active element, with the SIMP-interpolated stiffness and the
    Pedersen mass penalty baked into per-element materials, binned by density.
    This mirrors the in-house model as closely as an input deck can, so a
    frequency mismatch points at the *solver*, not the model.

``solid``
    Only the elements the optimiser actually kept (rho > 0.5), as solid PA12.
    This is the design as it would be built. It carries no low-density ersatz
    material at all, so it is the decisive test of whether modes 3-6 of the
    V_f = 0.50 design are genuine structural modes or artifacts of SIMP's
    minimum density.

Usage::

    ./venv/bin/python -m sims.export_abaqus_inp --vf 0.50
    ccx -i verification/vf50_solid          # writes .frd / .dat
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from src.fem import FEM3D, Material
from src.mesh3d import ArmGeometry3D

OUT_DIR = "verification"
TIP_MASS_KG = 0.06        # motor + propeller, as used in the in-house modal run
N_MODES = 6
N_BINS = 40               # density bins for the SIMP deck


def _simp_E(rho, E, p=FEM3D.SIMP_PENALTY, e_min=FEM3D.E_MIN_FRAC):
    """Effective Young's modulus, matching FEM3D.assemble_K exactly."""
    return E * (e_min + (1.0 - e_min) * rho ** p)


def _simp_rho(rho, rho_solid,
              thr=FEM3D.MASS_THRESHOLD, q=FEM3D.MASS_PENALTY_Q):
    """Effective mass density, matching FEM3D._mass_simp_factor exactly."""
    m = np.where(rho >= thr, rho, (np.maximum(rho, 0.0) / thr) ** q * thr)
    return rho_solid * m


def write_inp(path, geom, fem, rho_active, mode, mat: Material):
    """Emit one Abaqus deck. ``mode`` is 'simp' or 'solid'."""
    ex, ey, ez = fem.elem_grid_idx.T

    if mode == "solid":
        keep = rho_active > 0.5
        # Keep only the largest connected body. The V_f = 0.50 design leaves a
        # detached 51-voxel fragment; with the low-density ersatz material gone
        # it is genuinely floating, and CalculiX duly returns its six
        # rigid-body modes as the first six eigenvalues, burying the
        # structural ones.
        from scipy import ndimage
        g = np.zeros((geom.nx, geom.ny, geom.nz), dtype=bool)
        g[ex[keep], ey[keep], ez[keep]] = True
        lab, n = ndimage.label(g)
        if n > 1:
            main = int(np.argmax(np.bincount(lab.ravel())[1:])) + 1
            keep &= (lab[ex, ey, ez] == main)
        E_el = np.full(keep.sum(), mat.E)
        D_el = np.full(keep.sum(), mat.rho)
    elif mode == "simp":
        keep = np.ones(len(rho_active), dtype=bool)
        E_el = _simp_E(rho_active, mat.E)
        D_el = _simp_rho(rho_active, mat.rho)
    else:
        raise ValueError(mode)

    e_idx = np.where(keep)[0]
    # Element connectivity, in the same node ordering the in-house code uses.
    # That ordering (bottom face CCW, then top face) is already the Abaqus
    # C3D8 convention, so no permutation is needed.
    conn = np.array([geom.element_nodes(ex[i], ey[i], ez[i]) for i in e_idx])

    # Only emit nodes that some retained element actually references —
    # free-floating nodes make the stiffness matrix singular.
    used = np.unique(conn)
    remap = np.zeros(geom.n_nodes if hasattr(geom, "n_nodes")
                     else used.max() + 1, dtype=np.int64)
    remap[used] = np.arange(1, len(used) + 1)          # 1-based for Abaqus

    nx1, ny1 = geom.nx + 1, geom.ny + 1
    iz = used // (nx1 * ny1)
    iy = (used % (nx1 * ny1)) // nx1
    ix = used % nx1
    xyz = np.column_stack([ix * geom.dx, iy * geom.dy, iz * geom.dz])

    fixed = np.intersect1d(geom.fixed_nodes(), used)
    tip = np.intersect1d(geom.motor_mount_top_nodes(), used)
    if len(tip) == 0:
        # motor_mount_top_nodes() looks at the very top face of the design
        # domain. A geometry that does not reach it — the OEM arm, whose motor
        # pad sits ~2 mm below — would otherwise have nowhere to hang the tip
        # mass. Fall back to the highest surviving nodes within the motor pad
        # radius, which is the same bearing surface physically.
        nx1, ny1 = geom.nx + 1, geom.ny + 1
        iz_u = used // (nx1 * ny1)
        iy_u = (used % (nx1 * ny1)) // nx1
        ix_u = used % nx1
        r = np.hypot(ix_u * geom.dx - geom.motor_centre[0],
                     iy_u * geom.dy - geom.motor_centre[1])
        near = used[r <= geom.motor_dia / 2.0 + geom.motor_keep_ring]
        if len(near) == 0:
            raise RuntimeError("no nodes near the motor mount; cannot place "
                               "the tip mass")
        zt = (near // (nx1 * ny1))
        tip = near[zt >= zt.max() - 1]
        print(f"  [tip mass] top-face nodes absent; using {len(tip)} nodes at "
              f"the highest material within the motor pad")

    # Bin elements by material so the deck carries tens of materials, not tens
    # of thousands of one-element sections.
    if mode == "solid":
        bins = np.zeros(len(e_idx), dtype=int)
        bin_E, bin_D = np.array([mat.E]), np.array([mat.rho])
    else:
        r = rho_active[e_idx]
        edges = np.linspace(r.min(), r.max() + 1e-12, N_BINS + 1)
        bins = np.clip(np.digitize(r, edges) - 1, 0, N_BINS - 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        bin_E, bin_D = _simp_E(centres, mat.E), _simp_rho(centres, mat.rho)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        w = f.write
        w(f"** F450 quadcopter arm - modal verification deck ({mode})\n")
        w(f"** {len(used)} nodes, {len(e_idx)} C3D8 elements, "
          f"{3*len(used)} DoF\n")
        w("** Units: m, kg, N, Pa. Generated by sims/export_abaqus_inp.py\n")
        w("*HEADING\nF450 arm modal analysis\n")

        w("*NODE\n")
        for n, (x, y, z) in zip(remap[used], xyz):
            w(f"{n}, {x:.6e}, {y:.6e}, {z:.6e}\n")

        w("*ELEMENT, TYPE=C3D8, ELSET=EALL\n")
        for k, c in enumerate(conn, start=1):
            w(f"{k}, " + ", ".join(str(v) for v in remap[c]) + "\n")

        for b in range(len(bin_E)):
            members = np.where(bins == b)[0] + 1
            if len(members) == 0:
                continue
            w(f"*ELSET, ELSET=EB{b:03d}\n")
            for i in range(0, len(members), 16):
                w(", ".join(str(v) for v in members[i:i + 16]) + "\n")
            w(f"*MATERIAL, NAME=M{b:03d}\n*ELASTIC\n")
            w(f"{bin_E[b]:.6e}, {mat.nu}\n*DENSITY\n{bin_D[b]:.6e}\n")
            w(f"*SOLID SECTION, ELSET=EB{b:03d}, MATERIAL=M{b:03d}\n")

        # Tip mass: 60 g split equally over the motor-mount pad nodes, which
        # is exactly how FEM3D.assemble_M places it.
        w("*NSET, NSET=NTIP\n")
        for i in range(0, len(tip), 16):
            w(", ".join(str(v) for v in remap[tip][i:i + 16]) + "\n")
        w("*ELEMENT, TYPE=MASS, ELSET=ETIP\n")
        for j, n in enumerate(remap[tip], start=len(conn) + 1):
            w(f"{j}, {n}\n")
        w(f"*MASS, ELSET=ETIP\n{TIP_MASS_KG / len(tip):.6e}\n")

        w("*NSET, NSET=NFIX\n")
        for i in range(0, len(fixed), 16):
            w(", ".join(str(v) for v in remap[fixed][i:i + 16]) + "\n")
        w("*BOUNDARY\nNFIX, 1, 3\n")

        w(f"*STEP\n*FREQUENCY, STORAGE=YES\n{N_MODES}\n")
        w("*NODE FILE\nU\n*END STEP\n")

    return dict(nodes=len(used), elements=len(e_idx), fixed=len(fixed),
                tip_nodes=len(tip), path=path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vf", type=float, default=0.50)
    ap.add_argument("--modes", nargs="*", default=["solid", "simp"],
                    choices=["solid", "simp"])
    ap.add_argument("--baseline", action="store_true",
                    help="fully solid design domain (V_f = 1.00) — the "
                         "cross-check with no SIMP interpolation at all")
    a = ap.parse_args()

    import pickle
    geom = ArmGeometry3D()
    mat = Material()
    fem = FEM3D(geom, mat)
    if a.baseline:
        tag, rho = "baseline", np.ones(fem.n_active)
        a.modes = ["solid"]
    else:
        tag = f"vf{int(a.vf*100):02d}"
        with open(f"results_3d/freq_{tag}_omega0500.pkl", "rb") as fh:
            rho = pickle.load(fh)["rho"]

    for mode in a.modes:
        info = write_inp(f"{OUT_DIR}/{tag}_{mode}.inp", geom, fem, rho, mode,
                         mat)
        print(f"{mode:6s} -> {info['path']}  "
              f"{info['nodes']} nodes, {info['elements']} elements, "
              f"{info['fixed']} fixed nodes, {info['tip_nodes']} tip nodes")


if __name__ == "__main__":
    main()
