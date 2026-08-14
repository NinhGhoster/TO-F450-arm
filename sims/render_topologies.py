"""Render the per-V_f topology panels with the corrected surface extraction.

The v4 renderer builds these with marching cubes on the zero-padded density
field.  That puts the nodes on the design-domain faces at exactly rho/2, which
is the 0.5 iso-value, so the isosurface clips the domain box and leaves four
hairlines tracing its edges — geometry that is not in the design at all.

This module uses the same threshold-then-extract-surface path as
``render_figures_v5``: take the cells above the iso-value, pull the external
surface of that cell set, and apply volume-preserving Taubin smoothing.  The
box edges never enter the surface, so the hairlines cannot appear.

Detached fragments are deliberately **kept** (``min_body_frac=0``) rather than
cleaned up, and their count is reported so the manuscript caption can disclose
them.  Hiding them would misrepresent what SIMP produced.

Frequencies are reported as the first *thrust-excitable* mode (§3.3), so the
panels agree with Table 6 rather than with the raw eigenvalue index.

Usage::

    QARM_RESULTS_DIR=results_3d_v6 ./venv/bin/python -m sims.render_topologies
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np
from scipy import ndimage

from sims.render_figures_v5 import (DOMAIN_MM, PA12, RESULTS, _aim, add_part,
                                    geom_fem, panel_size, plotter, save_column,
                                    save_single, shot, solid_surface_mm, trim)

FIGURES = "figures"


def fragment_count(rho, iso=0.5):
    """Solid bodies in the design, and the share of solid volume off the main one."""
    from src.mesh3d import ArmGeometry3D
    from src.fem import FEM3D, Material
    geom = ArmGeometry3D()
    fem = FEM3D(geom, Material())
    ex, ey, ez = fem.elem_grid_idx.T
    keep = rho > iso
    g = np.zeros((geom.nx, geom.ny, geom.nz), dtype=bool)
    g[ex[keep], ey[keep], ez[keep]] = True
    lab, n = ndimage.label(g)
    if n <= 1:
        return n, 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return n, float(sizes.sum() - sizes.max()) / float(sizes.sum())


def render_modes(vf=0.50, n_modes=4, dpi=300):
    """The first mode shapes of one design, drawn exactly like the topologies.

    Rendering these through the v4 path made them look like a different part
    from the topology panels of the same design: marching cubes produces a
    blobbier surface than the thresholded cell set, and it carries the
    domain-box hairlines. Same surface extraction, camera and lighting here,
    so a reader can compare the two figures directly.

    Displacement is masked to the solid region and normalised over it. The
    near-void ersatz material (rho ~ 1e-3) moves orders of magnitude more and
    means nothing physically; letting it set the colour scale renders the
    genuine modes as featureless dark solids.
    """
    from sims.modal_participation import effective_mass_voxel
    geom, fem = geom_fem()
    with open(os.path.join(RESULTS,
              f"freq_vf{int(vf*100):02d}_omega0500.pkl"), "rb") as fh:
        d = pickle.load(fh)
    freqs = np.asarray(d["modal"]["frequencies_Hz"])
    eff = effective_mass_voxel(fem, geom, d["rho"], d["modal"]["mode_shapes"],
                               tip_mass_kg=d["modal"].get("tip_mass_kg", 0.060))
    solid = d["rho"] > 0.5
    panel = panel_size(DOMAIN_MM, 1500)
    images, labels = [], []

    for k in range(n_modes):
        u = np.linalg.norm(d["modal"]["mode_shapes"][:, k].reshape(-1, 3), axis=1)
        ue = u[fem.elem_node_ids].mean(axis=1)
        ue = np.where(solid, ue, 0.0)
        ue = ue / (ue[solid].max() or 1.0)
        surf = solid_surface_mm(d["rho"], extra_fields={"u_mag": ue},
                                min_body_frac=0)
        pl = plotter(panel)
        pl.add_mesh(surf, scalars="u_mag", cmap="viridis", clim=(0, 1),
                    smooth_shading=True, specular=0.18, specular_power=20,
                    ambient=0.22, diffuse=0.92, show_scalar_bar=False)
        _aim(pl, DOMAIN_MM, panel)
        images.append(trim(shot(pl)))
        kind = ("thrust-driven bending" if eff[k] >= 0.05
                else "not excited by thrust")
        labels.append(f"Mode {k+1} — {freqs[k]:.1f} Hz, "
                      f"{eff[k]*100:.1f} % effective mass in z ({kind})")
        print(f"  mode {k+1}: {freqs[k]:7.1f} Hz, effective mass in z "
              f"{eff[k]*100:5.2f} %")

    save_column(images, labels, f"{FIGURES}/fig08_modes.png",
                title=f"First four mode shapes of the $V_f$ = {vf:.2f} design",
                cbar=(0.0, 1.0, "viridis", "|u| (normalised)"),
                dpi=dpi)


MAIN_VFS = (0.08, 0.10, 0.20)      # where there is enough void to shape
APPX_VFS = (0.30, 0.50)            # near-solid; the design space is closed


def combined(dpi=300):
    """Two figures: the designs that are shaped, and the ones that cannot be."""
    panel = panel_size(DOMAIN_MM, 1500)

    def build(vfs):
        imgs, labs = [], []
        for vf in vfs:
            with open(os.path.join(RESULTS,
                      f"freq_vf{int(vf*100):02d}_omega0500.pkl"), "rb") as fh:
                d = pickle.load(fh)
            rho = d["rho"]
            surf = solid_surface_mm(rho, min_body_frac=0)
            pl = plotter(panel)
            add_part(pl, surf, PA12)
            _aim(pl, DOMAIN_MM, panel)
            imgs.append(trim(shot(pl)))
            solid = 100.0 * (rho > 0.5).sum() / len(rho)
            labs.append(f"$V_f$ = {vf:.2f} — {d['record']['mass_g']:.1f} g, "
                        f"{solid:.1f} % of the envelope solid, "
                        f"{100 - solid:.1f} % void left to shape")
        return imgs, labs

    imgs, labs = build(MAIN_VFS)
    save_column(imgs, labs, f"{FIGURES}/fig09_topologies.png",
                title="Converged topologies where the optimiser has room to act",
                dpi=dpi)

    imgs, labs = build(APPX_VFS)
    save_column(imgs, labs, f"{FIGURES}/figA1_topologies_high_vf.png",
                title="Converged topologies at high volume fraction",
                dpi=dpi)


def main(dpi=300):
    os.makedirs(FIGURES, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RESULTS, "freq_vf*_omega*.pkl")))
    if not files:
        raise SystemExit(f"no sweep results in {RESULTS!r}")

    combined(dpi=dpi)
    panel = panel_size(DOMAIN_MM, 1500)
    for f in files:
        vf = int(os.path.basename(f).split("_")[1][2:]) / 100.0
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        rho = d["rho"]

        # min_body_frac=0 keeps every detached body; the caption discloses them.
        surf = solid_surface_mm(rho, min_body_frac=0)
        pl = plotter(panel)
        add_part(pl, surf, PA12)
        _aim(pl, DOMAIN_MM, panel)
        img = trim(shot(pl))

        out = f"{FIGURES}/fig08_topo_vf{int(vf*100):02d}.png"
        save_single(img, out, title=f"Optimised topology, $V_f$ = {vf:.2f}",
                    dpi=dpi)
        n, frac = fragment_count(rho)
        print(f"{out}   {n} solid bod{'y' if n == 1 else 'ies'}"
              + (f", {frac*100:.2f} % of solid volume detached" if n > 1 else ""))

    render_modes(dpi=dpi)

    # A V_f = 0.70 panel survives from a sweep that is no longer run; leaving it
    # in the figures directory invites it back into the manuscript.
    stale = f"{FIGURES}/fig08_topo_vf70.png"
    if os.path.exists(stale):
        os.remove(stale)
        print(f"removed stale {stale}")


if __name__ == "__main__":
    main()
