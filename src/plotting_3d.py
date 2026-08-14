"""
3D rendering of arm topologies, stress fields, and fatigue-life fields
using PyVista / VTK off-screen rendering.

Surfaces are extracted from the voxel density field by marching cubes
(`pv.ImageData.contour`) so they are smooth, not pixellated.  Per-element
stress fields are mapped to the smooth surface via point sampling from
the parent ImageData.

The optional assembly geometry (motor, bolts, central frame plate)
provides the reader with the F450 context: where does this arm sit, what
attaches to it, how big is the motor relative to the arm?  All assembly
parts are scaled to F450 OEM specifications.
"""
from __future__ import annotations

import os
import numpy as np
import pyvista as pv

from .mesh import ArmGeometry
from .fem import FEM3D


# Force off-screen rendering (no DISPLAY needed)
pv.OFF_SCREEN = True
pv.set_plot_theme("document")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _active_to_grid(geom: ArmGeometry, fem: FEM3D, values: np.ndarray,
                     fill_outside: float = 0.0) -> np.ndarray:
    g = np.full((geom.nx, geom.ny, geom.nz), fill_outside, dtype=float)
    ex, ey, ez = fem.elem_grid_idx.T
    g[ex, ey, ez] = values
    return g


def _make_point_image_data(geom: ArmGeometry,
                            cell_arrays: dict) -> pv.ImageData:
    """Build a pv.ImageData with cell-centred fields, used for contouring.

    For marching cubes we need point-data, not cell-data.  We convert by
    averaging each voxel value to its 8 corner nodes (volume-averaged).
    cell_arrays is a dict of name -> (nx, ny, nz) array.
    """
    # The contour filter expects point arrays, so build a point-data ImageData
    # whose values are an upsampled (nx+1, ny+1, nz+1) grid obtained by
    # 8-way averaging of the surrounding cell values.  Faster path: pad the
    # cell array, then convolve with a (2,2,2)/8 kernel.
    img = pv.ImageData(
        dimensions=(geom.nx + 1, geom.ny + 1, geom.nz + 1),
        spacing=(geom.dx, geom.dy, geom.dz),
        origin=(0.0, 0.0, 0.0),
    )
    for name, cell in cell_arrays.items():
        # Pad with 0 outside the volume so the contour closes at the boundary
        padded = np.pad(cell, 1, mode="constant", constant_values=0.0)
        # Each point at (i,j,k) sees the 8 surrounding cells: padded[i:i+2, ...]
        # We slice and sum
        pt = (
            padded[0:-1, 0:-1, 0:-1] + padded[1:, 0:-1, 0:-1] +
            padded[0:-1, 1:,   0:-1] + padded[1:, 1:,   0:-1] +
            padded[0:-1, 0:-1, 1:  ] + padded[1:, 0:-1, 1:  ] +
            padded[0:-1, 1:,   1:  ] + padded[1:, 1:,   1:  ]
        ) / 8.0
        # pt shape: (nx+1, ny+1, nz+1)
        img.point_data[name] = pt.transpose(2, 1, 0).ravel(order="C")
    return img


def _make_max_pool_image_data(geom: ArmGeometry,
                                cell_arrays: dict) -> pv.ImageData:
    """Like _make_point_image_data, but at each node use the MAXIMUM of the
    8 surrounding cell values rather than the average.  This preserves peak
    stress values when stress is later sampled at surface points by the
    contour / probe filters.
    """
    img = pv.ImageData(
        dimensions=(geom.nx + 1, geom.ny + 1, geom.nz + 1),
        spacing=(geom.dx, geom.dy, geom.dz),
        origin=(0.0, 0.0, 0.0),
    )
    for name, cell in cell_arrays.items():
        padded = np.pad(cell, 1, mode="constant", constant_values=0.0)
        # Stack the 8 neighbour slices and take max
        stack = np.stack([
            padded[0:-1, 0:-1, 0:-1], padded[1:, 0:-1, 0:-1],
            padded[0:-1, 1:,   0:-1], padded[1:, 1:,   0:-1],
            padded[0:-1, 0:-1, 1:  ], padded[1:, 0:-1, 1:  ],
            padded[0:-1, 1:,   1:  ], padded[1:, 1:,   1:  ],
        ], axis=0)
        pt = stack.max(axis=0)
        img.point_data[name] = pt.transpose(2, 1, 0).ravel(order="C")
    return img


def _extract_smooth_surface(geom: ArmGeometry, fem: FEM3D,
                             rho_active: np.ndarray,
                             extra_fields: dict = None,
                             iso: float = 0.5,
                             n_smooth: int = 30) -> pv.PolyData:
    """Return the closed surface of the material domain.

    For graded densities (optimized topology with the density filter), we
    use marching cubes on the smoothed point-data field — this gives a
    smooth iso-surface.

    For sharp densities (baseline, ρ is exactly 0 or 1), marching cubes
    only catches the very thin skin between two adjacent voxels and the
    result looks like a wireframe.  We detect this case by checking how
    much of the density mass is in the interior {ρ ≈ 1} class; if it is
    >95 % we instead threshold the volume and extract its boundary as a
    proper closed surface, then lightly smooth.

    Extra fields (stress, life, …) are mapped onto the surface via
    max-pool sampling from the original cell-data image, so peak interior
    values are preserved near the visible surface instead of being
    averaged away.
    """
    rho_grid = _active_to_grid(geom, fem, rho_active, fill_outside=0.0)
    # Heuristic: fraction of in-arm voxels that are nearly solid
    in_arm = rho_grid > 1e-3
    if in_arm.sum() > 0:
        solid_fraction = float((rho_grid[in_arm] > 0.95).sum() / in_arm.sum())
    else:
        solid_fraction = 0.0

    if solid_fraction > 0.95:
        # Sharp density — threshold the cell-data volume and extract its surface
        img_cell = pv.ImageData(
            dimensions=(geom.nx + 1, geom.ny + 1, geom.nz + 1),
            spacing=(geom.dx, geom.dy, geom.dz),
            origin=(0.0, 0.0, 0.0),
        )
        img_cell.cell_data["density"] = rho_grid.transpose(2, 1, 0).ravel(order="C")
        if extra_fields:
            for k, vals in extra_fields.items():
                grid = _active_to_grid(geom, fem, vals, fill_outside=0.0)
                img_cell.cell_data[k] = grid.transpose(2, 1, 0).ravel(order="C")
        solid_volume = img_cell.threshold(iso, scalars="density",
                                            invert=False)
        surface = solid_volume.extract_geometry()
        # Push cell data to point data on the surface so colouring is smooth
        surface = surface.cell_data_to_point_data()
        if n_smooth > 0:
            surface = surface.smooth(n_iter=max(5, n_smooth // 3),
                                       relaxation_factor=0.10,
                                       feature_smoothing=False,
                                       boundary_smoothing=False)
        return surface

    # Graded density — marching cubes for a smooth iso-surface
    img_density = _make_point_image_data(geom, {"density": rho_grid})
    surface = img_density.contour([iso], scalars="density",
                                     compute_normals=True,
                                     compute_scalars=False)
    if n_smooth > 0:
        surface = surface.smooth(n_iter=n_smooth, relaxation_factor=0.20,
                                   feature_smoothing=False,
                                   boundary_smoothing=False)
    if extra_fields:
        max_fields = {k: _active_to_grid(geom, fem, vals, fill_outside=0.0)
                      for k, vals in extra_fields.items()}
        img_max = _make_max_pool_image_data(geom, max_fields)
        sampled = surface.sample(img_max)
        for name in extra_fields:
            surface[name] = sampled[name]
    return surface


def _camera_for(view: str, bounds: tuple) -> tuple:
    """Return (position, focal_point, view_up) for canonical views."""
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    Lx = bounds[1] - bounds[0]
    Ly = bounds[3] - bounds[2]
    Lz = bounds[5] - bounds[4]
    L = max(Lx, Ly, Lz)
    if view == "iso":
        return ((cx + L*0.8, cy - L*0.8, cz + L*0.7),
                (cx, cy, cz), (0, 0, 1))
    if view == "top":
        return ((cx, cy, cz + L*1.2), (cx, cy, cz), (0, 1, 0))
    if view == "side":
        return ((cx, cy + L*1.2, cz), (cx, cy, cz), (0, 0, 1))
    if view == "front":
        return ((cx - L*1.2, cy, cz), (cx, cy, cz), (0, 0, 1))
    if view == "iso2":
        return ((cx - L*0.8, cy + L*0.8, cz + L*0.7),
                (cx, cy, cz), (0, 0, 1))
    raise ValueError(f"unknown view: {view}")


# -----------------------------------------------------------------------------
# F450 assembly bits — motor, bolts, body plate, propeller
# -----------------------------------------------------------------------------
F450_MOTOR_DIA = 28e-3        # 2212 motor outer diameter
F450_MOTOR_HEIGHT = 19e-3     # 2212 motor body height
F450_BELL_HEIGHT = 6e-3
F450_BOLT_DIA = 2.5e-3
F450_BOLT_HEAD_DIA = 5e-3
F450_BOLT_HEAD_H = 2e-3
F450_BOLT_SHAFT_H = 8e-3
F450_PLATE_THICKNESS = 1.5e-3
F450_PLATE_SIZE = 75e-3
F450_GAP = 44e-3              # vertical clearance between top and bottom plates
                              # (= the height of the OEM F450 arm bridge),
                              # which houses the flight controller and battery
F450_PROP_LENGTH = 0.254
F450_PROP_WIDTH = 25e-3
F450_PROP_THICKNESS = 4e-3


def _make_motor(centre_xy: tuple, z_top_of_arm: float) -> pv.MultiBlock:
    """Return a small assembly of cylinders representing a 2212 motor
    sitting on top of the arm at motor mount location."""
    cx, cy = centre_xy
    motor_body = pv.Cylinder(
        center=(cx, cy, z_top_of_arm + F450_MOTOR_HEIGHT / 2),
        direction=(0, 0, 1),
        radius=F450_MOTOR_DIA / 2,
        height=F450_MOTOR_HEIGHT, resolution=40,
    )
    motor_bell = pv.Cylinder(
        center=(cx, cy, z_top_of_arm + F450_MOTOR_HEIGHT + F450_BELL_HEIGHT / 2),
        direction=(0, 0, 1),
        radius=F450_MOTOR_DIA / 2 + 1e-3,
        height=F450_BELL_HEIGHT, resolution=40,
    )
    shaft = pv.Cylinder(
        center=(cx, cy, z_top_of_arm + F450_MOTOR_HEIGHT + F450_BELL_HEIGHT + 3e-3),
        direction=(0, 0, 1),
        radius=2.5e-3, height=6e-3, resolution=20,
    )
    return pv.MultiBlock({"body": motor_body, "bell": motor_bell, "shaft": shaft})


def _make_propeller(centre_xy: tuple, z_above_motor: float,
                     length: float = F450_PROP_LENGTH) -> pv.PolyData:
    """A simple two-blade propeller flat plate at the given height."""
    cx, cy = centre_xy
    blade = pv.Box(bounds=(-length / 2, length / 2,
                              -F450_PROP_WIDTH / 2, F450_PROP_WIDTH / 2,
                              -F450_PROP_THICKNESS / 2, F450_PROP_THICKNESS / 2))
    blade = blade.translate((cx, cy, z_above_motor), inplace=False)
    return blade


def _make_bolt(centre_xy: tuple, z_top_of_arm: float,
                head_above_arm: bool = True,
                shaft_height: float = None) -> pv.MultiBlock:
    """Bolt = head cylinder + shaft cylinder."""
    cx, cy = centre_xy
    sh = shaft_height if shaft_height is not None else F450_BOLT_SHAFT_H
    if head_above_arm:
        head_z = z_top_of_arm + F450_BOLT_HEAD_H / 2
        shaft_z = z_top_of_arm - sh / 2
    else:
        head_z = z_top_of_arm - F450_BOLT_HEAD_H / 2
        shaft_z = z_top_of_arm + sh / 2
    head = pv.Cylinder(center=(cx, cy, head_z), direction=(0, 0, 1),
                        radius=F450_BOLT_HEAD_DIA / 2,
                        height=F450_BOLT_HEAD_H, resolution=24)
    shaft = pv.Cylinder(center=(cx, cy, shaft_z), direction=(0, 0, 1),
                         radius=F450_BOLT_DIA / 2,
                         height=sh, resolution=20)
    return pv.MultiBlock({"head": head, "shaft": shaft})


# -----------------------------------------------------------------------------
# OEM F450 arm rendering from the supplied STP/STL file
# -----------------------------------------------------------------------------
_OEM_ARM_STL = None


def _load_oem_arm(stl_path: str = "data/F450_arm.stl") -> pv.PolyData:
    """Lazily load the OEM F450 arm STL. Coordinates in the file are in
    millimeters; we scale to meters here so it lives in the same coordinate
    system as the rest of the geometry."""
    global _OEM_ARM_STL
    if _OEM_ARM_STL is None:
        m = pv.read(stl_path)
        # Scale mm → m
        m.points = m.points * 1e-3
        _OEM_ARM_STL = m
    return _OEM_ARM_STL.copy()


def _make_central_plate(geom: ArmGeometry, z_below_arm: bool = True
                         ) -> pv.PolyData:
    """A representative central body plate that the arm bolts into.
    Positioned slightly below the arm root (or above for the top plate).
    """
    L = F450_PLATE_SIZE
    half = L / 2
    # The plate centre is just inboard of the arm root (the screw holes
    # sit at (10, 9.5..28.5) mm; the body extends to negative x).
    # Place plate so its outboard edge meets the arm at x = 0,
    # spans y to cover both screw rows.
    z_off = -F450_PLATE_THICKNESS - 0.5e-3 if z_below_arm \
        else geom.L_z + 0.5e-3
    plate = pv.Box(bounds=(
        -L,  0.0,                       # x: spans body inboard
        geom.L_y / 2 - half, geom.L_y / 2 + half,   # y
        z_off, z_off + F450_PLATE_THICKNESS,
    ))
    return plate


# -----------------------------------------------------------------------------
# Topology — multi-view (4 panels)
# -----------------------------------------------------------------------------
def plot_topology_multiview(geom: ArmGeometry, fem: FEM3D,
                             rho_active: np.ndarray, path: str,
                             title: str = "",
                             threshold: float = 0.5,
                             window_size: tuple = (1800, 1200),
                             with_assembly: bool = False):
    surface = _extract_smooth_surface(geom, fem, rho_active, iso=threshold)
    views = [("iso", "Isometric"),
             ("top", "Top"),
             ("side", "Side"),
             ("front", "Front")]

    pl = pv.Plotter(shape=(2, 2), window_size=window_size, off_screen=True,
                     border=True, border_color="lightgray")
    for k, (view_name, label) in enumerate(views):
        pl.subplot(k // 2, k % 2)
        pl.add_mesh(surface, color="#3a6fa5", show_edges=False,
                     specular=0.4, specular_power=15, ambient=0.30,
                     diffuse=0.85, smooth_shading=True)
        if with_assembly:
            _add_assembly_actors(pl, geom)
        pos, foc, up = _camera_for(view_name, surface.bounds)
        pl.camera_position = [pos, foc, up]
        pl.add_text(label, position="upper_left", font_size=14, color="black")
        pl.show_axes()
    if title:
        pl.subplot(0, 0)
        pl.add_text(title, position="upper_right", font_size=11, color="black")
    pl.screenshot(path, transparent_background=False)
    pl.close()


# -----------------------------------------------------------------------------
# Topology — single isometric, opaque shaded, optional assembly
# -----------------------------------------------------------------------------
def _add_assembly_actors(pl: pv.Plotter, geom: ArmGeometry,
                          show_propeller: bool = False,
                          plate_opacity: float = 0.40,
                          twin_plates: bool = True,
                          plate_gap: float = F450_GAP) -> tuple:
    """Add the F450 frame plates, bolts, motor (and optional propeller).

    The OEM F450 has a TWIN-PLATE central frame (top + bottom) separated by
    a `plate_gap` (≈ 44 mm) so the flight controller, ESCs, and battery can
    be housed between the plates.  By default we render both plates and
    long bolts spanning the gap, matching the OEM assembly.

    If `twin_plates` is False, only a single plate is shown (the simplified
    view used in earlier drafts).

    Returns combined bounds for camera framing.
    """
    actors_bounds = []
    L = F450_PLATE_SIZE
    half = L / 2

    if twin_plates:
        # Top plate sits ABOVE the arm so the arm top face contacts the
        # underside of the top plate; bottom plate sits BELOW with a `gap`
        # of empty space between them for the flight-controller bay.
        top_plate_z0 = geom.L_z         # top plate underside touches arm top
        bot_plate_z1 = -plate_gap       # bottom plate top face is gap below 0
        top_plate = pv.Box(bounds=(
            -L, 0.0,
            geom.L_y / 2 - half, geom.L_y / 2 + half,
            top_plate_z0, top_plate_z0 + F450_PLATE_THICKNESS,
        ))
        bot_plate = pv.Box(bounds=(
            -L, 0.0,
            geom.L_y / 2 - half, geom.L_y / 2 + half,
            bot_plate_z1 - F450_PLATE_THICKNESS, bot_plate_z1,
        ))
        pl.add_mesh(top_plate, color="#2c2c2c", opacity=plate_opacity,
                     smooth_shading=False, specular=0.10, ambient=0.30)
        pl.add_mesh(bot_plate, color="#2c2c2c", opacity=plate_opacity,
                     smooth_shading=False, specular=0.10, ambient=0.30)
        actors_bounds.extend([top_plate.bounds, bot_plate.bounds])
        # Long bolts: head above top plate, shaft long enough to span both
        # plates and the gap
        bolt_full_h = (top_plate_z0 + F450_PLATE_THICKNESS) - (bot_plate_z1 -
                                                                  F450_PLATE_THICKNESS) + 2e-3
        bolt_head_z = top_plate_z0 + F450_PLATE_THICKNESS
        for cx, cy in geom.screw_centres:
            bolt = _make_bolt((cx, cy), z_top_of_arm=bolt_head_z,
                                head_above_arm=True,
                                shaft_height=bolt_full_h)
            pl.add_mesh(bolt, color="#888888", smooth_shading=True,
                         specular=0.50, specular_power=15)
            actors_bounds.append(bolt.bounds)
    else:
        # Single-plate (legacy) view
        plate = _make_central_plate(geom, z_below_arm=True)
        pl.add_mesh(plate, color="#3a3a3a", opacity=plate_opacity,
                     smooth_shading=False, specular=0.10, ambient=0.30)
        actors_bounds.append(plate.bounds)
        for cx, cy in geom.screw_centres:
            bolt = _make_bolt((cx, cy), z_top_of_arm=geom.L_z,
                                head_above_arm=True)
            pl.add_mesh(bolt, color="#888888", smooth_shading=True,
                         specular=0.50, specular_power=15)
            actors_bounds.append(bolt.bounds)

    # Motor on top of the top plate (or top of arm if single-plate)
    motor_z_top = (geom.L_z + F450_PLATE_THICKNESS) if twin_plates else geom.L_z
    motor = _make_motor(geom.motor_centre, z_top_of_arm=motor_z_top)
    pl.add_mesh(motor, color="#883030", smooth_shading=True,
                 specular=0.25, specular_power=15, ambient=0.25)
    actors_bounds.append(motor.bounds)

    if show_propeller:
        cx, cy = geom.motor_centre
        prop_z = motor_z_top + F450_MOTOR_HEIGHT + F450_BELL_HEIGHT + 4e-3
        prop = _make_propeller((cx, cy), prop_z)
        pl.add_mesh(prop, color="#222222", opacity=0.85, smooth_shading=True)
        actors_bounds.append(prop.bounds)

    if not actors_bounds:
        return None
    arr = np.array(actors_bounds)
    return (arr[:, 0].min(), arr[:, 1].max(),
            arr[:, 2].min(), arr[:, 3].max(),
            arr[:, 4].min(), arr[:, 5].max())


def plot_topology_iso(geom: ArmGeometry, fem: FEM3D,
                       rho_active: np.ndarray, path: str,
                       title: str = "",
                       threshold: float = 0.5,
                       window_size: tuple = (1400, 900),
                       with_assembly: bool = False,
                       show_propeller: bool = False):
    surface = _extract_smooth_surface(geom, fem, rho_active, iso=threshold)
    pl = pv.Plotter(window_size=window_size, off_screen=True)
    pl.add_mesh(surface, color="#3a6fa5", show_edges=False,
                 specular=0.4, specular_power=15, ambient=0.30,
                 diffuse=0.85, smooth_shading=True)
    if with_assembly:
        asm_bounds = _add_assembly_actors(pl, geom, show_propeller=show_propeller)
        # Use union of arm + assembly bounds, but exclude very-tall prop column
        s = surface.bounds
        bounds = (min(s[0], asm_bounds[0]), max(s[1], asm_bounds[1]),
                   min(s[2], asm_bounds[2]), max(s[3], asm_bounds[3]),
                   min(s[4], asm_bounds[4]), max(s[5], asm_bounds[5]))
    else:
        bounds = surface.bounds
    pos, foc, up = _camera_for("iso", bounds)
    pl.camera_position = [pos, foc, up]
    if title:
        pl.add_text(title, position="upper_left", font_size=14)
    pl.show_axes()
    pl.screenshot(path, transparent_background=False)
    pl.close()


# -----------------------------------------------------------------------------
# Stress contour on a smooth surface, optional assembly
# -----------------------------------------------------------------------------
def plot_stress_3d(geom: ArmGeometry, fem: FEM3D,
                    rho_active: np.ndarray,
                    stress_vM_Pa: np.ndarray,
                    path: str,
                    title: str = "",
                    density_threshold: float = 0.5,
                    vmax_MPa: float = None,
                    cmap: str = "jet",
                    window_size: tuple = (1600, 900),
                    with_assembly: bool = False,
                    show_propeller: bool = False,
                    multiview: bool = False):
    surface = _extract_smooth_surface(
        geom, fem, rho_active,
        extra_fields={"vm_MPa": stress_vM_Pa / 1e6},
        iso=density_threshold,
    )
    if vmax_MPa is None:
        vmax_MPa = float(np.nanmax(surface["vm_MPa"]))

    scalar_bar_args = dict(
        title="von Mises (MPa)",
        title_font_size=14, label_font_size=12,
        height=0.55, width=0.06, vertical=True,
        position_x=0.90, position_y=0.20,
        n_labels=6, fmt="%.2f",
    )

    if multiview:
        views = [("iso", "Isometric"), ("top", "Top"),
                  ("side", "Side"), ("front", "Front")]
        pl = pv.Plotter(shape=(2, 2), window_size=window_size, off_screen=True,
                         border=True, border_color="lightgray")
        for k, (view_name, label) in enumerate(views):
            pl.subplot(k // 2, k % 2)
            pl.add_mesh(surface, scalars="vm_MPa", cmap=cmap,
                         clim=(0, vmax_MPa),
                         show_edges=False, specular=0.20, ambient=0.30,
                         smooth_shading=True,
                         scalar_bar_args=scalar_bar_args if k == 0 else None,
                         show_scalar_bar=(k == 0))
            if with_assembly:
                _add_assembly_actors(pl, geom, show_propeller=show_propeller)
            pos, foc, up = _camera_for(view_name, surface.bounds)
            pl.camera_position = [pos, foc, up]
            pl.add_text(label, position="upper_left", font_size=12)
            pl.show_axes()
    else:
        pl = pv.Plotter(window_size=window_size, off_screen=True)
        pl.add_mesh(surface, scalars="vm_MPa", cmap=cmap,
                     clim=(0, vmax_MPa),
                     show_edges=False, specular=0.20, ambient=0.30,
                     smooth_shading=True,
                     scalar_bar_args=scalar_bar_args)
        if with_assembly:
            asm_bounds = _add_assembly_actors(pl, geom,
                                                show_propeller=show_propeller,
                                                plate_opacity=0.30)
            s = surface.bounds
            bounds = (min(s[0], asm_bounds[0]), max(s[1], asm_bounds[1]),
                       min(s[2], asm_bounds[2]), max(s[3], asm_bounds[3]),
                       min(s[4], asm_bounds[4]), max(s[5], asm_bounds[5]))
        else:
            bounds = surface.bounds
        pos, foc, up = _camera_for("iso", bounds)
        pl.camera_position = [pos, foc, up]
        if title:
            pl.add_text(title, position="upper_left", font_size=14)
        pl.show_axes()

    pl.screenshot(path, transparent_background=False)
    pl.close()


# -----------------------------------------------------------------------------
# Mode-shape rendering — undeformed iso-surface coloured by |u|, or warped
# -----------------------------------------------------------------------------
def plot_mode_shape_3d(geom: ArmGeometry, fem: FEM3D,
                        rho_active: np.ndarray,
                        mode_shape: np.ndarray,
                        path: str,
                        title: str = "",
                        density_threshold: float = 0.5,
                        warp_factor: float = 0.0,
                        cmap: str = "viridis",
                        window_size: tuple = (1500, 900),
                        with_assembly: bool = False):
    """Render the optimised topology coloured by mode-shape displacement
    magnitude.

    mode_shape : (n_dofs,) one column from FEM3D.modal_analysis.
    warp_factor: if >0, deform the surface by mode_shape × warp_factor for
                 a "deformed mode" view.  If 0, the undeformed iso-surface
                 is shown with the displacement field as a colour overlay.
    """
    # Compute per-node displacement magnitude (n_nodes,) — sum of squared
    # translational DoFs at each node, sqrt.
    u_node = mode_shape.reshape(-1, 3)
    u_mag_node = np.linalg.norm(u_node, axis=1)
    # Map to per-element magnitudes (average over the 8 corner nodes).
    u_mag_elem = u_mag_node[fem.elem_node_ids].mean(axis=1)
    # Normalise so colourbar is in [0, 1]
    u_max = float(u_mag_elem.max()) if u_mag_elem.max() > 0 else 1.0
    u_mag_norm = u_mag_elem / u_max
    surface = _extract_smooth_surface(
        geom, fem, rho_active,
        extra_fields={"u_mag": u_mag_norm},
        iso=density_threshold,
    )

    if warp_factor > 0.0:
        # Warp the surface by interpolating mode_shape's vector field to the
        # surface point coordinates.  Build a point ImageData of the vector
        # field, then sample.
        ux_n = u_node[:, 0].reshape(geom.nx + 1, geom.ny + 1, geom.nz + 1)
        uy_n = u_node[:, 1].reshape(geom.nx + 1, geom.ny + 1, geom.nz + 1)
        uz_n = u_node[:, 2].reshape(geom.nx + 1, geom.ny + 1, geom.nz + 1)
        img = pv.ImageData(
            dimensions=(geom.nx + 1, geom.ny + 1, geom.nz + 1),
            spacing=(geom.dx, geom.dy, geom.dz),
            origin=(0.0, 0.0, 0.0),
        )
        vec = np.stack([ux_n.transpose(2, 1, 0).ravel(order="C"),
                         uy_n.transpose(2, 1, 0).ravel(order="C"),
                         uz_n.transpose(2, 1, 0).ravel(order="C")], axis=1)
        img.point_data["mode"] = vec
        sampled = surface.sample(img)
        surface.points = surface.points + warp_factor * sampled["mode"]

    scalar_bar_args = dict(
        title="|u| (normalised)",
        title_font_size=14, label_font_size=12,
        height=0.55, width=0.06, vertical=True,
        position_x=0.90, position_y=0.20,
        n_labels=5, fmt="%.2f",
    )
    pl = pv.Plotter(window_size=window_size, off_screen=True)
    pl.add_mesh(surface, scalars="u_mag", cmap=cmap, clim=(0, 1),
                 show_edges=False, specular=0.20, ambient=0.30,
                 smooth_shading=True,
                 scalar_bar_args=scalar_bar_args)
    if with_assembly:
        _add_assembly_actors(pl, geom)
    pos, foc, up = _camera_for("iso", surface.bounds)
    pl.camera_position = [pos, foc, up]
    if title:
        pl.add_text(title, position="upper_left", font_size=14)
    pl.show_axes()
    pl.screenshot(path, transparent_background=False)
    pl.close()


# -----------------------------------------------------------------------------
# Goodman-corrected alternating stress on a smooth surface
# -----------------------------------------------------------------------------
def plot_alt_stress_3d(geom: ArmGeometry, fem: FEM3D,
                        rho_active: np.ndarray,
                        sigma_alt_eq_Pa: np.ndarray,
                        path: str,
                        title: str = "",
                        density_threshold: float = 0.5,
                        vmax_MPa: float = None,
                        endurance_MPa: float = 13.0,
                        cmap: str = "jet",
                        window_size: tuple = (1600, 900),
                        with_assembly: bool = False):
    surface = _extract_smooth_surface(
        geom, fem, rho_active,
        extra_fields={"alt_MPa": sigma_alt_eq_Pa / 1e6},
        iso=density_threshold,
    )
    if vmax_MPa is None:
        vmax_MPa = float(max(endurance_MPa, np.nanmax(surface["alt_MPa"])))

    scalar_bar_args = dict(
        title=f"σ_alt,eq (MPa)\n(endurance = {endurance_MPa:.0f} MPa)",
        title_font_size=12, label_font_size=12,
        height=0.55, width=0.06, vertical=True,
        position_x=0.90, position_y=0.20,
        n_labels=6, fmt="%.2f",
    )

    pl = pv.Plotter(window_size=window_size, off_screen=True)
    pl.add_mesh(surface, scalars="alt_MPa", cmap=cmap,
                 clim=(0, vmax_MPa),
                 show_edges=False, specular=0.20, ambient=0.30,
                 smooth_shading=True,
                 scalar_bar_args=scalar_bar_args)
    if with_assembly:
        _add_assembly_actors(pl, geom)
    bounds = pl.bounds if with_assembly else surface.bounds
    pos, foc, up = _camera_for("iso", bounds)
    pl.camera_position = [pos, foc, up]
    if title:
        pl.add_text(title, position="upper_left", font_size=14)
    pl.show_axes()
    pl.screenshot(path, transparent_background=False)
    pl.close()


# -----------------------------------------------------------------------------
# Full quadcopter overview: 4 arms + central body + 4 motors + 4 props
# -----------------------------------------------------------------------------
def plot_oem_assembly(path: str,
                       title: str = "OEM F450 arm in the central frame",
                       stl_path: str = "data/F450_arm.stl",
                       window_size: tuple = (1500, 1000)):
    """Render the OEM F450 arm in the F450 twin-plate frame so the reader
    sees the actual commercial product geometry (a 3D bridge between two
    carbon-fibre plates).  No simulated topology is shown here — this is
    a reference figure for the OEM arm only.
    """
    oem = _load_oem_arm(stl_path)
    # OEM bounds (after mm→m scaling, the arm spans ~218 × 55 × 44 mm).
    # Place the arm so its inboard end sits near x=0 and the bottom plate
    # touches the OEM arm's bottom flange.
    b = oem.bounds
    # Centre the arm in y so the screw-hole rows straddle y=0
    cy = 0.5 * (b[2] + b[3])
    oem.points[:, 1] -= cy   # centre y on 0
    # Shift so the inboard end is at x ≈ 0 (the inboard arm flanges sit just
    # outboard of the central plate edge)
    oem.points[:, 0] -= b[0]
    # Lift so the bottom of the arm is at z = -F450_GAP/2 (centred vertically
    # between the two plates).
    b2 = oem.bounds
    z_centre = 0.5 * (b2[4] + b2[5])
    oem.points[:, 2] -= z_centre

    pl = pv.Plotter(window_size=window_size, off_screen=True)
    pl.add_mesh(oem, color="#3a6fa5", smooth_shading=True,
                 specular=0.4, ambient=0.30, diffuse=0.85)

    # Twin plates straddling the OEM arm
    arm_zmin, arm_zmax = oem.bounds[4], oem.bounds[5]
    L = 0.080
    top_plate = pv.Box(bounds=(-L, 0.0, -L/2, L/2,
                                  arm_zmax, arm_zmax + F450_PLATE_THICKNESS))
    bot_plate = pv.Box(bounds=(-L, 0.0, -L/2, L/2,
                                  arm_zmin - F450_PLATE_THICKNESS, arm_zmin))
    pl.add_mesh(top_plate, color="#2c2c2c", opacity=0.45,
                 smooth_shading=False, ambient=0.30)
    pl.add_mesh(bot_plate, color="#2c2c2c", opacity=0.45,
                 smooth_shading=False, ambient=0.30)

    # Motor centred where the OEM motor-mount ring is — approximately at
    # arm.x_max - 10 mm and arm.y_mean = 0.
    motor_x = oem.bounds[1] - 10e-3
    motor = _make_motor((motor_x, 0.0), z_top_of_arm=arm_zmax + F450_PLATE_THICKNESS)
    pl.add_mesh(motor, color="#883030", smooth_shading=True,
                 specular=0.25, ambient=0.25)

    # Camera: angled iso view that frames arm + assembly
    bx = oem.bounds
    cx, cy_, cz = 0.5*(bx[0]+bx[1]), 0.5*(bx[2]+bx[3]), 0.5*(bx[4]+bx[5])
    Lmax = max(bx[1]-bx[0], bx[3]-bx[2], bx[5]-bx[4])
    pl.camera_position = [
        (cx + Lmax*0.7, cy_ - Lmax*0.8, cz + Lmax*0.6),
        (cx, cy_, cz), (0, 0, 1),
    ]
    if title:
        pl.add_text(title, position="upper_left", font_size=14)
    pl.show_axes()
    pl.screenshot(path, transparent_background=False)
    pl.close()


def plot_quadcopter_overview(geom: ArmGeometry, fem: FEM3D,
                              rho_active: np.ndarray, path: str,
                              title: str = "F450 with optimized arms",
                              threshold: float = 0.5,
                              window_size: tuple = (1500, 1500),
                              show_propellers: bool = False,
                              use_oem_arm: bool = False):
    """Hero figure: render the F450 quadcopter as a complete assembly with
    four of our optimized arms arranged at 45/135/225/315° around a central
    plate.  Propellers are off by default because at 10″ they dwarf the
    structure and obscure the design.
    """
    if use_oem_arm:
        oem = _load_oem_arm()
        # Centre OEM arm: y in middle, inboard end at x=0, z centred at gap/2
        b = oem.bounds
        oem.points[:, 0] -= b[0]
        oem.points[:, 1] -= 0.5 * (b[2] + b[3])
        oem.points[:, 2] -= 0.5 * (b[4] + b[5])
        one_arm = oem
        arm_zmin = oem.bounds[4]; arm_zmax = oem.bounds[5]
        arm_xmax = oem.bounds[1]
    else:
        one_arm = _extract_smooth_surface(geom, fem, rho_active, iso=threshold)
        arm_zmin = -F450_GAP / 2; arm_zmax = F450_GAP / 2
        arm_xmax = geom.L_x

    pl = pv.Plotter(window_size=window_size, off_screen=True)

    # Twin plates (top + bottom) so the F450 architecture is recognizable
    L_plate = 0.080
    top_plate = pv.Box(bounds=(-L_plate/2, L_plate/2,
                                  -L_plate/2, L_plate/2,
                                  arm_zmax, arm_zmax + F450_PLATE_THICKNESS))
    bot_plate = pv.Box(bounds=(-L_plate/2, L_plate/2,
                                  -L_plate/2, L_plate/2,
                                  arm_zmin - F450_PLATE_THICKNESS, arm_zmin))
    pl.add_mesh(top_plate, color="#2c2c2c", opacity=0.45,
                 smooth_shading=False, ambient=0.30)
    pl.add_mesh(bot_plate, color="#2c2c2c", opacity=0.90,
                 smooth_shading=False, ambient=0.30)

    # The arm's body-side screw holes sit at local x ∈ {10, 25} mm; place
    # the arm so both screw holes overlay the central plate.  With the
    # plate edge at x = L_plate/2 = 40 mm and we want the outermost screw
    # (x = 25) at x = 25 mm in world before rotation, the inboard arm
    # translation should be ~ 0 mm (arm starts at the plate centre).  We
    # offset by +10 mm so the inner screw is at x = 20 mm and the outer
    # screw at x = 35 mm — both safely under the plate.
    t_x = (L_plate / 2 - 5e-3) if use_oem_arm else 10e-3
    angles_deg = [45, 135, 225, 315]
    for ang in angles_deg:
        arm = one_arm.copy()
        if use_oem_arm:
            arm = arm.translate((t_x, 0.0, 0.0), inplace=False)
        else:
            arm = arm.translate((t_x, -geom.L_y / 2, 0.0), inplace=False)
        arm = arm.rotate_z(ang, point=(0.0, 0.0, 0.0), inplace=False)
        pl.add_mesh(arm, color="#3a6fa5", smooth_shading=True,
                     specular=0.4, ambient=0.30, diffuse=0.85)

        # Motor world position
        if use_oem_arm:
            r = t_x + (arm_xmax - 10e-3)
        else:
            r = t_x + geom.motor_centre[0]
        mx = r * np.cos(np.radians(ang))
        my = r * np.sin(np.radians(ang))
        motor = _make_motor((mx, my),
                             z_top_of_arm=arm_zmax + F450_PLATE_THICKNESS)
        pl.add_mesh(motor, color="#883030", smooth_shading=True,
                     specular=0.25, ambient=0.25)

        if show_propellers:
            prop_z = geom.L_z + F450_MOTOR_HEIGHT + F450_BELL_HEIGHT + 4e-3
            prop = _make_propeller((mx, my), prop_z,
                                     length=F450_PROP_LENGTH)
            prop = prop.rotate_z(ang, point=(mx, my, prop_z), inplace=False)
            pl.add_mesh(prop, color="#222222", opacity=0.85,
                         smooth_shading=True)

    Rmax = (L_plate / 2 + arm_xmax) * 1.10
    Dcam = Rmax * 2.2
    pl.camera_position = [
        (Dcam * 0.6, -Dcam * 0.6, Dcam * 0.5),
        (0.0, 0.0, 0.0),
        (0, 0, 1),
    ]
    pl.reset_camera_clipping_range()
    if title:
        pl.add_text(title, position="upper_left", font_size=14)
    pl.show_axes()
    pl.screenshot(path, transparent_background=False)
    pl.close()


if __name__ == "__main__":
    import pickle
    from .fem import Material
    with open("results/raw_baseline.pkl", "rb") as f:
        b = pickle.load(f)
    geom = ArmGeometry()
    fem = FEM3D(geom, Material())
    rho = np.ones(fem.n_active)
    plot_stress_3d(geom, fem, rho,
                    b["static"]["LC2_maneuver"]["vm"],
                    "/tmp/test_stress3d.png",
                    title="Baseline vM (LC2)",
                    with_assembly=True)
    print("Wrote /tmp/test_stress3d.png")

