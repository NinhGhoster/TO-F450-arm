"""v5 figure renderer — publication remakes of the seven engineering figures
specified in ``figure_pack_for_artist/README.md``.

Everything here is rendered from the *real* saved FEM fields in
``results_3d/*.pkl``; nothing is stylised or approximated.  No simulation is
re-run.

Rendering is deliberately split in two:

* **PyVista / VTK** produces 3D pixels only — geometry, shading, contours.
* **Matplotlib** does every piece of typography: titles, panel labels,
  colourbars, annotation, panel layout.

That split is the fix for the v4 figures.  VTK's ``add_text`` and scalar-bar
actors are what produced the clipped ``0.08`` colourbar tick, the crowded
upper-left annotation block, and the inconsistent font sizes.  Matplotlib
composites over the rendered RGB buffer with real margins, so nothing can
be clipped.

Camera framing is analytic (``_aim``): every panel in a grid is aimed at the
*same* design-domain box rather than at its own mesh bounds, so the four
topologies and the four mode shapes are directly comparable at identical
scale.

Usage::

    ./venv/bin/python -m sims.render_figures_v5              # all seven
    ./venv/bin/python -m sims.render_figures_v5 --only fig12 # one figure
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import cm, colors as mcolors
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from src.fem import FEM3D, Material
from src.mesh3d import ArmGeometry3D
from src.plotting_3d import _extract_smooth_surface

pv.OFF_SCREEN = True

# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------
PA12 = "#3a6fa5"        # PA12 polyamide blue
CARBON = "#2c2c2c"      # carbon-fibre plate dark grey
MOTOR = "#883030"       # 2212-class motor red
INK = "#1a1a1a"
GREY = "#6b6b6b"

RESULTS = os.environ.get("QARM_RESULTS_DIR", "results_3d")
FIGURES = "figures"

# Canonical three-quarter view.  Mostly side-on so the 220 mm arm reads at
# length, with just enough elevation and yaw to show the bridge depth.
VIEW = (0.45, -1.0, 0.42)

# The design domain every 3D view is framed against (mm).  Framing against a
# fixed box rather than per-mesh bounds is what keeps panels comparable.
DOMAIN_MM = (0.0, 216.0, 0.0, 42.0, 0.0, 54.0)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "mathtext.default": "regular",
    "savefig.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def geom_fem():
    """Build (and cache) the production 3D-bridge geometry + FEM."""
    if "gf" not in _CACHE:
        geom = ArmGeometry3D()
        _CACHE["gf"] = (geom, FEM3D(geom, Material()))
    return _CACHE["gf"]


def load_pkl(name: str) -> dict:
    if name not in _CACHE:
        with open(os.path.join(RESULTS, name), "rb") as f:
            _CACHE[name] = pickle.load(f)
    return _CACHE[name]


def summary() -> list:
    if "summary" not in _CACHE:
        with open(os.path.join(RESULTS, "summary_freq.json")) as f:
            _CACHE["summary"] = json.load(f)
    return _CACHE["summary"]


def summary_for(vf: float) -> dict:
    for row in summary():
        if row.get("Vf") == vf and row.get("omega_target_Hz"):
            return row
    raise KeyError(f"no summary row for V_f = {vf}")


def surface_mm(rho, extra_fields=None, iso=0.5, n_smooth=30):
    """Iso-surface of a density field, in millimetres.

    Delegates to the tested v3/v4 extractor (which works in metres and
    max-pools element fields onto the surface so peak stresses survive), then
    scales to mm because the whole figure pack is specified in mm.

    Used for the fully-solid baseline only; graded density fields go through
    ``solid_surface_mm`` instead — see the note there.
    """
    geom, fem = geom_fem()
    surf = _extract_smooth_surface(geom, fem, rho, extra_fields=extra_fields,
                                   iso=iso, n_smooth=n_smooth)
    surf.points = surf.points * 1000.0
    return surf


def solid_surface_mm(rho, extra_fields=None, iso=0.5, n_smooth=28,
                     min_body_frac=0.01):
    """Watertight boundary of the solid (rho > iso) voxel set, in millimetres.

    Marching cubes cannot be used on the graded density field here.  The
    node values are the 8-cell average of a zero-padded cell array, so a node
    lying on a face of the design domain averages 4 real cells with 4 zeros
    and lands on exactly rho/2.  Wherever the material reaches the domain
    boundary at rho = 1 that is exactly the 0.5 iso-value, and marching cubes
    emits degenerate slivers along the domain edges.  Those slivers are what
    drew the four hairlines around the bounding box of every v4/v5 topology
    render — they are not material.

    Thresholding the voxels and taking the external surface has no iso
    crossing at the boundary at all, so the artifact cannot arise.  The
    result is a staircase, which volume-preserving Taubin smoothing then
    relaxes (plain Laplacian smoothing would shrink thin members).

    ``min_body_frac`` drops detached bodies smaller than that fraction of the
    solid volume — the loose specks near the motor end, which read as dirt.
    The threshold is deliberately small: the isolated bolt columns at low
    volume fractions are 2.6-6.4 % of the solid and are real structure, so
    they are kept.
    """
    geom, fem = geom_fem()
    ex, ey, ez = fem.elem_grid_idx.T

    if min_body_frac:
        from scipy import ndimage
        g = np.zeros((geom.nx, geom.ny, geom.nz))
        g[ex, ey, ez] = rho
        lab, n = ndimage.label(g > iso)
        if n > 1:
            sizes = np.bincount(lab.ravel())[1:]
            drop = np.where(sizes < min_body_frac * sizes.sum())[0] + 1
            if len(drop):
                g[np.isin(lab, drop)] = 0.0
                rho = g[ex, ey, ez]
    grid = pv.ImageData(
        dimensions=(geom.nx + 1, geom.ny + 1, geom.nz + 1),
        spacing=(geom.dx * 1e3, geom.dy * 1e3, geom.dz * 1e3),
        origin=(0.0, 0.0, 0.0),
    )

    def _cells(values, fill=0.0):
        g = np.full((geom.nx, geom.ny, geom.nz), fill, dtype=float)
        g[ex, ey, ez] = values
        return g.transpose(2, 1, 0).ravel(order="C")

    grid.cell_data["density"] = _cells(rho)
    for name, vals in (extra_fields or {}).items():
        grid.cell_data[name] = _cells(vals)

    solid = grid.threshold(iso, scalars="density")
    surf = solid.extract_surface(algorithm="dataset_surface")
    surf = surf.cell_data_to_point_data()
    if n_smooth:
        surf = surf.smooth_taubin(n_iter=n_smooth, pass_band=0.05)
    return surf.compute_normals(auto_orient_normals=True, feature_angle=60.0)


# ---------------------------------------------------------------------------
# Render rig
# ---------------------------------------------------------------------------
def plotter(size):
    pl = pv.Plotter(off_screen=True, window_size=list(size))
    pl.set_background("white")
    return pl


def _basis(direction=VIEW, up0=(0, 0, 1)):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    right = np.cross(d, np.asarray(up0, float))
    right /= np.linalg.norm(right)
    up = np.cross(right, d)
    return d, right, up / np.linalg.norm(up)


def projected_extent(box, direction=VIEW):
    """Half-width and half-height of ``box`` as seen along ``direction``."""
    _, right, up = _basis(direction)
    c = np.array([(box[0] + box[1]) / 2, (box[2] + box[3]) / 2,
                  (box[4] + box[5]) / 2])
    rel = np.array(list(itertools.product(box[0:2], box[2:4], box[4:6]))) - c
    return np.abs(rel @ right).max(), np.abs(rel @ up).max()


def panel_size(box, width_px=1500, direction=VIEW):
    """Window size whose aspect matches the projected geometry.

    Sizing the window to the part instead of to a fixed 16:10 is what removes
    the wide empty bands above and below a 220 x 56 x 44 mm arm.
    """
    hw, hh = projected_extent(box, direction)
    return (int(width_px), max(1, int(round(width_px * hh / hw))))


def _aim(pl, box, size, direction=VIEW, margin=1.06, ortho=True, up0=(0, 0, 1)):
    """Aim the camera at ``box`` along ``direction``, fitted with ``margin``.

    With ``ortho`` the parallel scale is computed analytically from the box
    corners, so two plotters given the same box render at exactly the same
    scale — the property that makes a 2x2 grid readable as a comparison.
    """
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    c = np.array([(box[0] + box[1]) / 2, (box[2] + box[3]) / 2,
                  (box[4] + box[5]) / 2])
    right = np.cross(d, np.asarray(up0, float))
    right /= np.linalg.norm(right)
    up = np.cross(right, d)
    up /= np.linalg.norm(up)

    rel = np.array(list(itertools.product(box[0:2], box[2:4], box[4:6]))) - c
    half_w = np.abs(rel @ right).max()
    half_h = np.abs(rel @ up).max()
    aspect = size[0] / size[1]
    scale = max(half_h, half_w / aspect) * margin

    span = max(box[1] - box[0], box[3] - box[2], box[5] - box[4])
    pl.camera_position = [tuple(c + d * span * 4.0), tuple(c), tuple(up)]
    if ortho:
        pl.enable_parallel_projection()
        pl.camera.parallel_scale = scale
    else:
        pl.camera.view_angle = 24.0
        pl.camera.position = tuple(c + d * scale / np.tan(np.radians(12.0)))
    pl.reset_camera_clipping_range()
    return pl


def shot(pl) -> np.ndarray:
    """Anti-aliased RGB buffer.  Closes the plotter."""
    pl.enable_anti_aliasing("ssaa")
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def trim(img: np.ndarray, pad: int = 10) -> np.ndarray:
    """Crop uniform white margin, leaving ``pad`` px.

    Only ever applied to standalone renders — never to grid panels, where
    per-panel cropping would destroy the shared scale.
    """
    mask = (img[:, :, :3] < 250).any(axis=2)
    if not mask.any():
        return img
    rows, cols = np.where(mask)
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, img.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]


def add_part(pl, mesh, color=PA12, opacity=1.0, flat=False, **kw):
    """Add a solid part with the house material response."""
    style = dict(specular=0.30, specular_power=26, diffuse=0.92, ambient=0.20)
    style.update(kw)
    return pl.add_mesh(mesh, color=color, opacity=opacity,
                       smooth_shading=not flat, show_edges=False, **style)


# ---------------------------------------------------------------------------
# OEM CAD (STEP) — read through the OpenCascade kernel
# ---------------------------------------------------------------------------
# The F450 STEP files are modelled **Y-up**: the plates are 1.5 mm thick in Y
# and every arm instance in the frame assembly has a 55.0 mm Y extent.  Reading
# them as Z-up rolls the arm 90 degrees about its own long axis, which is what
# put the v5 fig02a arm on its side.  CAD_TO_WORLD is the rotation that fixes
# it: (X, Y, Z)_cad -> (x, y, z) = (X, -Z, Y), a proper right-handed rotation
# about X (a plain Y/Z swap would mirror the part).
CAD_DIR = "figure_pack_for_artist/cad"
ASSEMBLY = f"{CAD_DIR}/F450_Frame_Assembly.stp"
ASM_BTM_PLATE, ASM_TOP_PLATE, ASM_ARM = 0, 4, 2   # solid indices, by bbox


def cad_to_world(mesh):
    p = mesh.points.copy()
    mesh.points = np.column_stack([p[:, 0], -p[:, 2], p[:, 1]])
    return mesh


def _step_solids(path):
    """Tessellated solids from a STEP file, in file order."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location

    r = STEPControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError(f"cannot read {path}")
    r.TransferRoots()
    shape = r.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.25, False, 0.3, True)

    out = []
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        solid = TopoDS.Solid_s(ex.Current())
        pts, tris, off = [], [], 0
        fe = TopExp_Explorer(solid, TopAbs_FACE)
        while fe.More():
            face = TopoDS.Face_s(fe.Current())
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, loc)
            if tri is not None:
                T = loc.Transformation()
                for i in range(1, tri.NbNodes() + 1):
                    p = tri.Node(i).Transformed(T)
                    pts.append([p.X(), p.Y(), p.Z()])
                for i in range(1, tri.NbTriangles() + 1):
                    a, b, c = tri.Triangle(i).Get()
                    tris.append([3, off + a - 1, off + b - 1, off + c - 1])
                off += tri.NbNodes()
            fe.Next()
        out.append(pv.PolyData(np.array(pts), np.array(tris).ravel()).clean()
                   if tris else None)
        ex.Next()
    return out


def oem_assembly_world():
    """One arm plus both central plates, in true relative position, Z-up.

    Everything comes from ``F450_Frame_Assembly.stp`` rather than being placed
    by hand, so the plate gap, the arm's seating between the plates and the
    arm's roll orientation are the CAD's, not mine.  The scene is then spun
    about the vertical axis so the chosen arm runs along +x.
    """
    if "oem" not in _CACHE:
        S = _step_solids(ASSEMBLY)
        # The assembly carries the real fasteners: 16 top-side screws
        # (Ø5.5 x 15 mm) and 8 bottom-side (Ø5.5 x 8 mm), i.e. 4 top and
        # 2 bottom per arm. Pick out the set belonging to the arm we render.
        def _size(m):
            b = np.array(m.bounds)
            return b[1::2] - b[0::2]

        cand_top = [i for i, m in enumerate(S)
                    if m is not None and 5 < _size(m)[0] < 7 and 14 < _size(m)[1] < 16]
        cand_bot = [i for i, m in enumerate(S)
                    if m is not None and 5 < _size(m)[0] < 7 and 7 < _size(m)[1] < 9]
        pb = np.array(S[ASM_TOP_PLATE].bounds)
        pc = np.array([(pb[0] + pb[1]) / 2, (pb[4] + pb[5]) / 2])   # X, Z
        ab0 = np.array(S[ASM_ARM].bounds)
        v = np.array([(ab0[0] + ab0[1]) / 2, (ab0[4] + ab0[5]) / 2]) - pc
        v /= np.linalg.norm(v)

        def _mine(i):
            b = np.array(S[i].bounds)
            d = np.array([(b[0] + b[1]) / 2, (b[4] + b[5]) / 2]) - pc
            return 0 < d @ v < 75 and abs(-d[0] * v[1] + d[1] * v[0]) < 28

        screws = [cad_to_world(S[i])
                  for i in cand_top + cand_bot if _mine(i)]
        arm = cad_to_world(S[ASM_ARM])
        top = cad_to_world(S[ASM_TOP_PLATE])
        bot = cad_to_world(S[ASM_BTM_PLATE])
        # Record true plate sizes BEFORE the spin — afterwards the
        # axis-aligned bbox of a rotated plate is not its dimensions.
        tb0, bb0 = np.array(top.bounds), np.array(bot.bounds)
        _CACHE["oem_dims"] = dict(
            top=(tb0[1] - tb0[0], tb0[3] - tb0[2]),
            bot=(bb0[1] - bb0[0], bb0[3] - bb0[2]),
            gap=tb0[4] - bb0[5],
        )
        tb = np.array(top.bounds)
        cx, cy = (tb[0] + tb[1]) / 2, (tb[2] + tb[3]) / 2
        ab = np.array(arm.bounds)
        ang = -np.arctan2((ab[2] + ab[3]) / 2 - cy, (ab[0] + ab[1]) / 2 - cx)
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for m in (arm, top, bot):
            m.points = (m.points - np.array([cx, cy, 0.0])) @ R.T
        # Drop the plate stack so the bottom plate's top face is z = 0.
        dz = np.array(bot.bounds)[5]
        for m in (arm, top, bot):
            m.points = m.points - np.array([0.0, 0.0, dz])
        _CACHE["oem"] = (arm, top, bot)
    return tuple(m.copy() for m in _CACHE["oem"])


def oem_plates_flat(gap):
    """The two real central plates, centred on the origin in plan, with the
    bottom plate's top face at z = 0 and the top plate's underside at z = gap.

    Used for the optimised-arm assembly, where the arm height comes from the
    simulation domain rather than from the OEM arm.
    """
    S = _step_solids(ASSEMBLY)
    out = []
    for idx, z0 in ((ASM_BTM_PLATE, -PLATE_T), (ASM_TOP_PLATE, gap)):
        m = cad_to_world(S[idx])
        b = np.array(m.bounds)
        m.points = m.points - np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2,
                                        b[4] - z0])
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# Assembly primitives (mm)
# ---------------------------------------------------------------------------
MOTOR_DIA, MOTOR_H, BELL_H = 28.0, 30.0, 6.0
PLATE_T, PLATE_GAP = 1.6, 44.0


def motor_at(cx, cy, z_base):
    """2212-class motor: body + bell + shaft, sitting on z_base."""
    body = pv.Cylinder(center=(cx, cy, z_base + MOTOR_H / 2), direction=(0, 0, 1),
                       radius=MOTOR_DIA / 2, height=MOTOR_H, resolution=64)
    bell = pv.Cylinder(center=(cx, cy, z_base + MOTOR_H + BELL_H / 2),
                       direction=(0, 0, 1), radius=MOTOR_DIA / 2 + 1.0,
                       height=BELL_H, resolution=64)
    shaft = pv.Cylinder(center=(cx, cy, z_base + MOTOR_H + BELL_H + 3.0),
                        direction=(0, 0, 1), radius=2.5, height=6.0, resolution=32)
    return pv.MultiBlock({"body": body, "bell": bell, "shaft": shaft})


def plate(cx, cy, half, z0, thickness=PLATE_T):
    return pv.Box(bounds=(cx - half, cx + half, cy - half, cy + half,
                          z0, z0 + thickness))


# ---------------------------------------------------------------------------
# Matplotlib composition
# ---------------------------------------------------------------------------
def _place(ax, img):
    ax.imshow(img, interpolation="lanczos")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def save_single(img, path, title=None, subtitle=None, cbar=None,
                width_in=8.0, dpi=300, note=None):
    """One render + optional right-hand colourbar, titles laid out in mpl."""
    h, w = img.shape[:2]
    fig_h = width_in * h / w
    # Every text block is wrapped to the figure width before the canvas is
    # sized, then the header/footer bands are grown to fit the wrapped line
    # count.  Nothing can run past the edge regardless of string length.
    title = textwrap.fill(title, int(width_in * 9.9)) if title else None
    subtitle = textwrap.fill(subtitle, int(width_in * 13.6)) if subtitle else None
    note = textwrap.fill(note, int(width_in * 14.2)) if note else None
    n_t = title.count("\n") + 1 if title else 0
    n_s = subtitle.count("\n") + 1 if subtitle else 0
    top = 0.10 + 0.24 * n_t + 0.19 * n_s
    bot = (0.20 + 0.16 * (note.count("\n") + 1)) if note else 0.10
    H = fig_h + top + bot
    fig = plt.figure(figsize=(width_in, H), dpi=dpi)

    left, right = 0.02, 0.86 if cbar else 0.98
    ax = fig.add_axes([left, bot / H, right - left, fig_h / H])
    _place(ax, img)

    if title:
        fig.text(0.02, 1 - 0.06 / H, title, ha="left", va="top",
                 fontsize=12.5, color=INK)
    if subtitle:
        fig.text(0.02, 1 - (0.10 + 0.24 * n_t) / H, subtitle, ha="left",
                 va="top", fontsize=9.5, color=GREY, style="italic")
    if note:
        fig.text(0.02, 0.06 / H, note, ha="left", va="bottom",
                 fontsize=9.0, color=GREY)

    if cbar:
        vmin, vmax, cmap, label = cbar
        cax = fig.add_axes([right + 0.035, bot / H + 0.10 * fig_h / H,
                            0.026, fig_h / H * 0.74])
        sm = cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax),
                               cmap=cmap)
        cb = fig.colorbar(sm, cax=cax)
        # Label on the LEFT of the bar: with ticks on the right of a narrow
        # bar near the figure edge, a right-hand label falls off the canvas.
        cax.yaxis.set_label_position("left")
        cb.set_label(label, fontsize=10, color=INK, labelpad=10)
        cb.ax.tick_params(labelsize=9, color=GREY, labelcolor=INK)
        # Both end values are pinned explicitly — the v4 figures lost the top
        # tick to the figure edge.
        ticks = np.linspace(vmin, vmax, 5)
        cb.set_ticks(ticks)
        cb.ax.set_yticklabels([f"{t:.3g}" for t in ticks])
        cb.outline.set_linewidth(0.6)
        cb.outline.set_edgecolor(GREY)

    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}  ({int(width_in*dpi)}x{int((fig_h+top+bot)*dpi)} px)")


def save_grid(images, labels, path, title=None, cbar=None,
              width_in=8.0, dpi=300, note=None):
    """2x2 grid with per-panel captions and one shared horizontal colourbar."""
    h, w = images[0].shape[:2]
    panel_h_in = (width_in / 2) * h / w
    lab_in = 0.30
    title = textwrap.fill(title, int(width_in * 9.9)) if title else None
    top = (0.10 + 0.26 * (title.count("\n") + 1)) if title else 0.06
    cbar_in = 0.85 if cbar else 0.0
    note = textwrap.fill(note, int(width_in * 14.8)) if note else None
    bot = (0.34 + 0.15 * note.count("\n")) if note else 0.06
    fig_h = 2 * (panel_h_in + lab_in) + top + cbar_in + bot
    fig = plt.figure(figsize=(width_in, fig_h), dpi=dpi)

    for k, (img, lab) in enumerate(zip(images, labels)):
        r, c = k // 2, k % 2
        x0 = 0.015 + c * 0.4925
        y0 = 1 - (top + (r + 1) * (panel_h_in + lab_in)) / fig_h
        ax = fig.add_axes([x0, y0 + lab_in / fig_h, 0.4725,
                           panel_h_in / fig_h])
        _place(ax, img)
        fig.text(x0 + 0.4725 / 2, y0 + 0.38 * lab_in / fig_h, lab,
                 ha="center", va="center", fontsize=9.5, color=INK)

    if title:
        fig.text(0.015, 1 - 0.06 / fig_h, title, ha="left", va="top",
                 fontsize=12.5, color=INK)
    if cbar:
        vmin, vmax, cmap, label = cbar
        cax = fig.add_axes([0.28, (bot + 0.55 * cbar_in) / fig_h, 0.44,
                            0.18 * cbar_in / fig_h])
        sm = cm.ScalarMappable(norm=mcolors.Normalize(vmin, vmax), cmap=cmap)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=10, color=INK, labelpad=6)
        cb.ax.tick_params(labelsize=9, color=GREY, labelcolor=INK)
        cb.set_ticks(np.linspace(vmin, vmax, 5))
        cb.outline.set_linewidth(0.6)
        cb.outline.set_edgecolor(GREY)
    if note:
        fig.text(0.015, 0.30 * bot / fig_h, note, ha="left", va="bottom",
                 fontsize=8.5, color=GREY)

    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}  ({int(width_in*dpi)}x{int(fig_h*dpi)} px)")


# ---------------------------------------------------------------------------
# Figure 12 (a-d) — modal-optimised topologies at four V_f
# ---------------------------------------------------------------------------
def save_column(images, labels, path, title=None, cbar=None,
                width_in=6.5, dpi=300, note=None):
    """Stack N panels in one column with (a), (b), ... sub-labels.

    ``save_grid`` is fixed at 2x2, which does not suit this study: the arm is
    long and thin, so side-by-side panels shrink it to illegibility, and the
    figures that need combining have three or five parts rather than four. A
    single column keeps every panel at full width, which is what makes the
    stress and mode patterns readable.
    """
    n = len(images)
    heights = [(width_in * im.shape[0] / im.shape[1]) for im in images]
    lab_in = 0.26
    title_wrapped = textwrap.fill(title, int(width_in * 11)) if title else None
    top = (0.10 + 0.24 * (title_wrapped.count("\n") + 1)) if title_wrapped else 0.05
    cbar_in = 0.62 if cbar else 0.0
    note_wrapped = textwrap.fill(note, int(width_in * 16)) if note else None
    # Colourbar and note each need their own band; sharing one made the
    # colourbar label land on top of the note text.
    bot = (0.46 + 0.15 * (note_wrapped.count("\n") + 1)) if note_wrapped else 0.05
    fig_h = sum(heights) + n * lab_in + top + cbar_in + bot
    fig = plt.figure(figsize=(width_in, fig_h), dpi=dpi)

    y = 1 - top / fig_h
    for k, (img, lab) in enumerate(zip(images, labels)):
        h = heights[k] / fig_h
        ax = fig.add_axes([0.02, y - h, 0.96, h])
        ax.imshow(img)
        ax.axis("off")
        fig.text(0.02, y - h - lab_in / fig_h * 0.62,
                 f"({'abcdefgh'[k]}) {lab}", ha="left", va="center",
                 fontsize=9, color=INK)
        y -= h + lab_in / fig_h

    if title_wrapped:
        fig.text(0.5, 1 - 0.16 / fig_h, title_wrapped, ha="center", va="top",
                 fontsize=11, color=INK)
    if cbar:
        vmin, vmax, cmap, clabel = cbar
        cax = fig.add_axes([0.22, (bot + 0.10) / fig_h, 0.56, 0.15 / fig_h])
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(clabel, fontsize=9)
        cb.ax.tick_params(labelsize=8)
    if note_wrapped:
        fig.text(0.5, 0.07 / fig_h, note_wrapped, ha="center", va="bottom",
                 fontsize=8.5, color=GREY)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}  ({n} panels)")


def fig12_topology(dpi=300):
    print("fig12_topology_abcd")
    panel = panel_size(DOMAIN_MM, 1500)
    vfs = [0.10, 0.20, 0.30, 0.50]
    images, labels = [], []
    for k, vf in enumerate(vfs):
        d = load_pkl(f"freq_vf{int(vf*100):02d}_omega0500.pkl")
        surf = solid_surface_mm(d["rho"])
        pl = plotter(panel)
        add_part(pl, surf, PA12)
        _aim(pl, DOMAIN_MM, panel)
        images.append(shot(pl))
        row = summary_for(vf)
        tag = " (headline)" if vf == 0.50 else ""
        labels.append(
            f"({'abcd'[k]}) $V_f$ = {vf:.2f}{tag} — mass {row['mass_g']:.0f} g, "
            f"$\\omega_1$ = {row['omega_1_Hz']:.1f} Hz")
    save_grid(
        images, labels, f"{FIGURES}/fig12_topology_abcd.png",
        title="Modal-frequency-constrained TO topologies "
              "($\\omega_{target}$ = 500 Hz)",
        width_in=8.0, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 8 (a-d) — first four mode shapes of the V_f = 0.50 design
# ---------------------------------------------------------------------------
# Mode identification for the V_f = 0.50 design, verified against the saved
# eigenvectors rather than assumed.  Modes 3 and 4 are NOT the "first torsion"
# and "higher bending" modes the v4 draft claimed: a connected-component
# analysis of the density field shows the main body's peak displacement is
# 0.23 % and 0.78 % of the modal peak, with the motion localized on a detached
# 51-voxel fragment and the near-void ersatz material.  They are numerical
# artifacts of SIMP's low-density regions, not structural modes of the arm.
# Modes 1 and 2 are genuine (main-body peaks at 50 % and 70 %).
MODE_TYPES = ["vertical bending", "lateral bending",
              "localized fragment mode", "localized fragment mode"]
MODE_NOTE = (
    "Undeformed geometry coloured by mode-shape displacement magnitude; "
    "identical camera, lighting and scale in all four panels. Modes 1 and 2 "
    "are structural modes of the arm. In modes 3 and 4 the main body is "
    "effectively stationary (peak displacement 0.23 % and 0.78 % of the modal "
    "peak) and the motion is localized on a detached 51-voxel material "
    "fragment — a SIMP low-density artifact, not a structural mode.")


def fig08_modes(dpi=300):
    print("fig08_modes_abcd")
    panel = panel_size(DOMAIN_MM, 1500)
    geom, fem = geom_fem()
    d = load_pkl("freq_vf50_omega0500.pkl")
    freqs = d["modal"]["frequencies_Hz"]
    # Only the solid (rho > iso) region is drawn, so the displacement field is
    # masked and normalised over that region alone.  Normalising over *all*
    # active elements — as the v4 renderer did — lets the near-void ersatz
    # material (rho ~ 1e-3), whose displacements are orders of magnitude
    # larger and physically meaningless, set the colour scale.  That is why
    # modes 3 and 4 rendered as featureless dark solids.
    solid = d["rho"] > 0.5
    images, labels = [], []
    for k in range(4):
        u_node = d["modal"]["mode_shapes"][:, k].reshape(-1, 3)
        u_mag = np.linalg.norm(u_node, axis=1)
        u_elem = u_mag[fem.elem_node_ids].mean(axis=1)
        u_elem = np.where(solid, u_elem, 0.0)
        u_elem = u_elem / (u_elem[solid].max() or 1.0)
        surf = solid_surface_mm(d["rho"], extra_fields={"u_mag": u_elem})
        pl = plotter(panel)
        pl.add_mesh(surf, scalars="u_mag", cmap="viridis", clim=(0, 1),
                    smooth_shading=True, specular=0.18, specular_power=20,
                    ambient=0.22, diffuse=0.92, show_scalar_bar=False)
        _aim(pl, DOMAIN_MM, panel)
        images.append(shot(pl))
        labels.append(f"({'abcd'[k]}) Mode {k+1} — {freqs[k]:.1f} Hz "
                      f"({MODE_TYPES[k]})")
    save_grid(
        images, labels, f"{FIGURES}/fig08_modes_abcd.png",
        title="First four natural mode shapes of the $V_f$ = 0.50 "
              "modal-optimised design",
        cbar=(0.0, 1.0, "viridis", "|u| (normalised)"),
        width_in=8.0, dpi=dpi)


# ---------------------------------------------------------------------------
# Figures 5 and 6 — baseline solid block, stress and fatigue fields
# ---------------------------------------------------------------------------
def _baseline_surface(field_name, values):
    geom, fem = geom_fem()
    rho = np.ones(fem.n_active)
    # n_smooth=0: the baseline IS a cuboid, and the v4 render's rounded-off
    # corners are exactly why readers could not tell what they were looking at.
    return surface_mm(rho, extra_fields={field_name: values}, n_smooth=0)


def _baseline_render(surf, scalars, clim, cmap, panel=(1800, 1000)):
    pl = plotter(panel)
    pl.add_mesh(surf, scalars=scalars, cmap=cmap, clim=clim,
                smooth_shading=False, specular=0.12, ambient=0.28,
                diffuse=0.90, show_scalar_bar=False)
    _aim(pl, DOMAIN_MM, panel)
    return trim(shot(pl))


def fig05_baseline_vm(dpi=300):
    print("fig05_baseline_vm_LC2")
    b = load_pkl("raw_baseline_modal.pkl")
    vm = b["static"]["LC2_maneuver"]["vm"] / 1e6
    vmax = float(np.nanmax(vm))
    surf = _baseline_surface("vm_MPa", vm)
    img = _baseline_render(surf, "vm_MPa", (0.0, vmax), "jet")
    save_single(
        img, f"{FIGURES}/fig05_baseline_vm_LC2.png",
        title="Baseline arm — von Mises stress under LC2 (Maneuver)",
        cbar=(0.0, vmax, "jet", "von Mises (MPa)"),
        width_in=8.0, dpi=dpi)
    return vmax


def fig06_baseline_life(dpi=300, endurance=13.0):
    print("fig06_baseline_life")
    b = load_pkl("raw_baseline_modal.pkl")
    alt = b["fatigue"]["sigma_alt_eq"] / 1e6
    peak = float(np.nanmax(alt))
    surf = _baseline_surface("alt_MPa", alt)
    img = _baseline_render(surf, "alt_MPa", (0.0, endurance), "jet")
    save_single(  # noqa: E128
        img, f"{FIGURES}/fig06_baseline_life.png",
        title="Baseline arm — $\\sigma_{alt,eq}$ under "
              "LC1$\\leftrightarrow$LC2 (endurance limit 13 MPa)",
        cbar=(0.0, endurance, "jet", "$\\sigma_{alt,eq}$ (MPa)"),
        width_in=8.0, dpi=dpi)
    return peak


# ---------------------------------------------------------------------------
# Figure 2a — OEM arm clamped in the twin-plate central frame
# ---------------------------------------------------------------------------
def _oem_arm_mm(path="data/F450_arm.stl"):
    """OEM arm STL, placed inboard-end at x=0, centred on y=0, resting z>=0."""
    m = pv.read(path)
    b = m.bounds
    m.points = m.points - np.array([b[0], (b[2] + b[3]) / 2, b[4]])
    return m


def _union_bounds(meshes):
    a = np.array([m.bounds for m in meshes])
    return (a[:, 0].min(), a[:, 1].max(), a[:, 2].min(), a[:, 3].max(),
            a[:, 4].min(), a[:, 5].max())


def fig02a_oem_assembly(dpi=300):
    print("fig02a_OEM_assembly")
    arm, top, bot = oem_assembly_world()
    ab = np.array(arm.bounds)
    # Motor centre from the CAD, not from the silhouette: the four M2.5 motor
    # screw holes in F450_Arm.stp sit at local x = 169.8 and 189.8, i.e. their
    # centroid is 201.9 mm out from the arm's root at local x = -22.08 —
    # 92.57 % of the 218.1 mm span. Sample the mount face at that station.
    xm = ab[0] + 0.9257 * (ab[1] - ab[0])
    band = arm.points[np.abs(arm.points[:, 0] - xm) < 7.0]
    mot = motor_at(xm, band[:, 1].mean(), band[:, 2].max())

    size = (2000, 1150)
    pl = plotter(size)
    add_part(pl, arm, PA12)
    add_part(pl, bot, CARBON, flat=True, ambient=0.58, diffuse=0.62)
    add_part(pl, top, CARBON, opacity=0.42, flat=True, ambient=0.58,
             diffuse=0.62)
    add_part(pl, mot, MOTOR, ambient=0.26)
    box = _union_bounds([arm, bot, top, mot["body"], mot["bell"]])
    _aim(pl, box, size, direction=(0.50, -1.0, 0.46), ortho=False, margin=1.02)
    img = trim(shot(pl))
    dims = _CACHE["oem_dims"]
    save_single(
        img, f"{FIGURES}/fig02a_OEM_assembly.png",
        title="OEM F450 arm in the twin-plate central frame "
              "(top plate semi-transparent)",
        subtitle=None,
        width_in=8.0, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 2b — F450 with four modal-optimised arms
# ---------------------------------------------------------------------------
def fig02b_quadcopter(dpi=300, vf=0.50):
    print("fig02b_quadcopter_overview")
    d = load_pkl(f"freq_vf{int(vf*100):02d}_omega0500.pkl")
    one = solid_surface_mm(d["rho"], min_body_frac=0)
    one.points = one.points - np.array([0.0, 21.0, 0.0])   # centre on y = 0 (L_y/2)
    # t is the root offset from the hub. The motor centre sits 195.6 mm out
    # along the arm, so the true 450 mm motor-to-motor diagonal the F450 is
    # named for needs t = 225 - 195.6 = 29.4 mm.
    #
    # Four arms at 90 degrees interpenetrate at the hub unless t exceeds half
    # the arm width. On the corrected geometry the arm is 42 mm wide, so the
    # limit is 21 mm and t = 29.4 clears it — the true diagonal is drawable.
    # (It was not on the earlier 56 mm-wide domain, which forced an inflated
    # offset and a 468 mm diagonal.)
    t = 225.0 - 195.6
    one.points = one.points + np.array([t, 0.0, 0.0])
    r_motor = t + 195.6
    diag_mm = 2 * r_motor

    size = (1700, 1500)
    pl = plotter(size)
    meshes = []
    for ang in (45, 135, 225, 315):
        a = one.rotate_z(ang, point=(0, 0, 0), inplace=False)
        add_part(pl, a, PA12)
        mx = r_motor * np.cos(np.radians(ang))
        my = r_motor * np.sin(np.radians(ang))
        mot = motor_at(mx, my, 54.0)
        add_part(pl, mot, MOTOR, ambient=0.26)
        meshes += [a, mot["bell"]]
    # Real OEM plates rather than invented 80 x 80 mm squares. The top plate
    # is lifted to the simulated arm height (54 mm) instead of the OEM's
    # 38 mm clear gap, because the arm shown here is the simulation's.
    bot, top = oem_plates_flat(54.0)
    add_part(pl, bot, CARBON, flat=True, ambient=0.58, diffuse=0.62)
    add_part(pl, top, CARBON, opacity=0.50, flat=True, ambient=0.58,
             diffuse=0.62)
    meshes += [bot, top]

    # Look along +x, i.e. down the bisector between two arms rather than
    # straight down one of them, so all four arms stay distinguishable.
    _aim(pl, _union_bounds(meshes), size, direction=(1.0, -0.30, 0.40),
         ortho=False, margin=1.20)
    img = trim(shot(pl))
    save_single(
        img, f"{FIGURES}/fig02b_quadcopter_overview.png",
        title=f"F450 with optimised arms ($V_f$ = {vf:.2f})",
        subtitle=None,
        width_in=8.0, dpi=dpi)


# ---------------------------------------------------------------------------
# Figure 3 (fig04) — dimensioned drawing of the design domain
# ---------------------------------------------------------------------------
# Fastener pattern measured from the OEM assembly: four screws into the top
# flange, two into the bottom flange. (x, y, z0, z1) in mm.
SCREWS_TOP = [(5.8, 13.0), (5.8, 29.0), (25.3, 9.5), (25.3, 32.5)]
SCREWS_BOT = [(24.9, 10.5), (24.9, 31.5)]
BOLT_D, MOUNT_D, MOUNT_C = 3.1, 10.0, (202.0, 21.0)
LX, LY, LZ = 216.0, 42.0, 54.0


def _dim(ax, p0, p1, offset, text, vertical=False, tick=3.0, fs=9):
    """One dimension: extension lines, double arrow, centred text."""
    (x0, y0), (x1, y1) = p0, p1
    if vertical:
        xd = offset
        ax.plot([x0, xd + np.sign(xd - x0) * tick], [y0, y0], lw=0.6, color=INK)
        ax.plot([x1, xd + np.sign(xd - x1) * tick], [y1, y1], lw=0.6, color=INK)
        ax.add_patch(FancyArrowPatch((xd, y0), (xd, y1), arrowstyle="<|-|>",
                                     mutation_scale=8, lw=0.7, color=INK,
                                     shrinkA=0, shrinkB=0))
        ax.text(xd - 3, (y0 + y1) / 2, text, ha="right", va="center",
                fontsize=fs, color=INK, rotation=90)
    else:
        yd = offset
        ax.plot([x0, x0], [y0, yd + np.sign(yd - y0) * tick], lw=0.6, color=INK)
        ax.plot([x1, x1], [y1, yd + np.sign(yd - y1) * tick], lw=0.6, color=INK)
        ax.add_patch(FancyArrowPatch((x0, yd), (x1, yd), arrowstyle="<|-|>",
                                     mutation_scale=8, lw=0.7, color=INK,
                                     shrinkA=0, shrinkB=0))
        ax.text((x0 + x1) / 2, yd + 2.5, text, ha="center", va="bottom",
                fontsize=fs, color=INK)


def _leader(ax, xy_to, xy_text, text, color, ha="left", fs=9.5):
    ax.annotate(text, xy=xy_to, xytext=xy_text, ha=ha, va="center",
                fontsize=fs, color=color,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=color,
                                shrinkA=0, shrinkB=2,
                                connectionstyle="arc3,rad=0.0"))


def fig04_geometry_bc(dpi=300):
    print("fig04_geometry_BC")
    fig = plt.figure(figsize=(8.0, 4.5), dpi=dpi)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_aspect("equal")
    ax.set_xlim(-54, 272)
    ax.set_ylim(-62, 106)
    ax.axis("off")

    # Design-domain footprint
    ax.add_patch(Rectangle((0, 0), LX, LY, fill=False, lw=1.6, color=INK,
                           zorder=3))
    # Centreline
    ax.plot([-8, LX + 8], [LY / 2, LY / 2], lw=0.5, color=GREY,
            dashes=(12, 4, 2, 4), zorder=1)

    # Top-flange screws solid, bottom-flange screws dashed — in a top view
    # they would otherwise be indistinguishable.
    for (cx, cy) in SCREWS_TOP:
        ax.add_patch(Circle((cx, cy), BOLT_D / 2, facecolor="#cfe0f0",
                            edgecolor="#2f6ba8", lw=1.0, zorder=4))
    for (cx, cy) in SCREWS_BOT:
        ax.add_patch(Circle((cx, cy), BOLT_D / 2, facecolor="none",
                            edgecolor="#2f6ba8", lw=1.0, ls=(0, (3, 2)),
                            zorder=4))
    ax.add_patch(Circle(MOUNT_C, MOUNT_D / 2, facecolor="#f6d6d6",
                        edgecolor="#a83232", lw=1.2, zorder=4))
    for (cx, cy) in SCREWS_TOP + SCREWS_BOT + [MOUNT_C]:
        r = 4.0 if (cx, cy) != MOUNT_C else 8.0
        ax.plot([cx - r, cx + r], [cy, cy], lw=0.5, color=GREY, zorder=5)
        ax.plot([cx, cx], [cy - r, cy + r], lw=0.5, color=GREY, zorder=5)

    # Dimensions
    _dim(ax, (0, 0), (LX, 0), -30, "216")
    _dim(ax, (0, 0), (0, LY), -34, "42", vertical=True)
    # Screw spacing dimensioned off the screw centres themselves, not off a
    # far edge — extension lines have to touch the feature they measure.
    _dim(ax, (5.8, 13.0), (5.8, 29.0), -14, "16", vertical=True)
    _dim(ax, (0, 0), (5.8, 0), -14, "5.8")
    _dim(ax, (5.8, 0), (25.3, 0), -14, "19.5")
    _dim(ax, (25.3, 0), (MOUNT_C[0], 0), -14, "176.7")

    # Callouts, placed clear of the part with leaders pointing in
    _leader(ax, (5.8, 29.0), (-40, 92),
            "Fixed support — Ø3.1 mm screw holes\n"
            "4 into top flange (solid), z = 42.8–54\n"
            "2 into bottom flange (dashed), z = 10.7–18.7",
            "#2f6ba8")
    _leader(ax, (MOUNT_C[0], MOUNT_C[1] + MOUNT_D / 2), (196, 88),
            "Bearing load\nmotor mount Ø10 mm", "#a83232", ha="right")

    # Title block
    bx, by, bw, bh = 132.0, -60.0, 130.0, 26.0
    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, lw=0.9, color=INK))
    ax.plot([bx, bx + bw], [by + bh * 0.55, by + bh * 0.55], lw=0.6, color=INK)
    ax.plot([bx + bw * 0.62, bx + bw * 0.62], [by, by + bh * 0.55], lw=0.6,
            color=INK)
    ax.text(bx + 4, by + bh * 0.78, "F450 simulated arm — design domain",
            fontsize=9.5, va="center", color=INK)
    ax.text(bx + 4, by + bh * 0.27, "Overall height (z) = 54 mm",
            fontsize=8.5, va="center", color=INK)
    ax.text(bx + bw * 0.66, by + bh * 0.27, "Units: mm\nTop view",
            fontsize=8, va="center", color=GREY, linespacing=1.5)

    fig.savefig(f"{FIGURES}/fig04_geometry_BC.png", dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"  wrote {FIGURES}/fig04_geometry_BC.png  ({8*dpi}x{int(4.5*dpi)} px)")


# ---------------------------------------------------------------------------
FIGS = {
    "fig02a": fig02a_oem_assembly,
    "fig02b": fig02b_quadcopter,
    "fig04": fig04_geometry_bc,
    "fig05": fig05_baseline_vm,
    "fig06": fig06_baseline_life,
    "fig08": fig08_modes,
    "fig12": fig12_topology,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS),
                    help="render only these figures (default: all)")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    for name in (a.only or sorted(FIGS)):
        FIGS[name](dpi=a.dpi)


if __name__ == "__main__":
    main()
