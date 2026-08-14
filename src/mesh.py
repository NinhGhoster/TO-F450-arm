"""
Structured 3D hex mesh of the F450 quadcopter arm.

Geometry:
- Bounding box: L_x = 130 mm (along arm), L_y = 38 mm (width), L_z = 4 mm (thickness)
- Four small screw holes (Ø 2.5 mm) for body-side fixation, arranged in a 2x2
  pattern at the body end of the arm
- One large motor mount hole (Ø 10 mm) at the motor end of the arm

Mesh:
- Structured Cartesian voxel grid
- Default element size: dx=dy=1 mm in plane, dz=1 mm through thickness
- 130 x 38 x 4 = 19,760 voxels at the finest resolution

Element classification:
- 'void'        : voxel outside arm material (inside holes or outside the arm footprint)
- 'fixed'       : voxel adjacent to a screw hole boundary — fully constrained (BC)
- 'load'        : voxel adjacent to motor mount hole boundary on the load-bearing half
- 'always_solid': voxel inside the arm but in a "do-not-optimize" region (around holes,
                  to preserve assembly interfaces)
- 'design'      : voxel inside the arm and free to be optimized
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ArmGeometry:
    """F450 arm geometry parameters (SI units, metres)."""

    # Bounding-box dimensions
    L_x: float = 130e-3       # arm long axis
    L_y: float = 38e-3        # arm width
    L_z: float = 4e-3         # arm thickness

    # Voxel sizes
    dx: float = 1e-3          # in-plane voxel
    dy: float = 1e-3
    dz: float = 1e-3          # through-thickness voxel

    # Body-side screw holes (4x M2.5)
    screw_dia: float = 2.5e-3
    screw_centres: Tuple[Tuple[float, float], ...] = (
        (10e-3, 9.5e-3),
        (10e-3, 28.5e-3),
        (25e-3, 9.5e-3),
        (25e-3, 28.5e-3),
    )

    # Motor mount hole (10 mm)
    motor_dia: float = 10e-3
    motor_centre: Tuple[float, float] = (115e-3, 19e-3)

    # "Keep-solid" ring around each hole — material that must stay (assembly interface)
    keep_solid_ring: float = 2.0e-3   # ring thickness around screw holes
    motor_keep_ring: float = 3.0e-3   # ring thickness around motor hole

    # Derived after __post_init__
    nx: int = field(init=False)
    ny: int = field(init=False)
    nz: int = field(init=False)
    n_elements: int = field(init=False)
    n_nodes: int = field(init=False)
    n_dofs: int = field(init=False)

    def __post_init__(self):
        self.nx = int(round(self.L_x / self.dx))
        self.ny = int(round(self.L_y / self.dy))
        self.nz = int(round(self.L_z / self.dz))
        self.n_elements = self.nx * self.ny * self.nz
        self.n_nodes = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
        self.n_dofs = 3 * self.n_nodes

    # ------------------------------------------------------------------ #
    # Coordinate helpers
    # ------------------------------------------------------------------ #
    def element_centre(self, ex: int, ey: int, ez: int) -> Tuple[float, float, float]:
        return ((ex + 0.5) * self.dx,
                (ey + 0.5) * self.dy,
                (ez + 0.5) * self.dz)

    def node_coord(self, nx: int, ny: int, nz: int) -> Tuple[float, float, float]:
        return (nx * self.dx, ny * self.dy, nz * self.dz)

    def node_index(self, nx: int, ny: int, nz: int) -> int:
        """Flatten 3D node index to a single global index, row-major in (z, y, x)."""
        return nz * (self.nx + 1) * (self.ny + 1) + ny * (self.nx + 1) + nx

    def element_index(self, ex: int, ey: int, ez: int) -> int:
        return ez * self.nx * self.ny + ey * self.nx + ex

    def element_nodes(self, ex: int, ey: int, ez: int) -> np.ndarray:
        """Return 8 node global indices for the hex element (CCW: bottom face then top)."""
        n = []
        for dz in (0, 1):
            for dy_, dx_ in ((0, 0), (0, 1), (1, 1), (1, 0)):
                n.append(self.node_index(ex + dx_, ey + dy_, ez + dz))
        return np.array(n, dtype=np.int64)

    # ------------------------------------------------------------------ #
    # Masks (booleans on the element grid)
    # ------------------------------------------------------------------ #
    def _in_arm_footprint(self, xc: float, yc: float) -> bool:
        """Default arm footprint = full rectangle. Override here for tapered shape."""
        return (0 <= xc <= self.L_x) and (0 <= yc <= self.L_y)

    def _in_any_screw_hole(self, xc: float, yc: float, expand: float = 0.0) -> bool:
        r = self.screw_dia / 2.0 + expand
        for cx, cy in self.screw_centres:
            if (xc - cx) ** 2 + (yc - cy) ** 2 <= r * r:
                return True
        return False

    def _in_motor_hole(self, xc: float, yc: float, expand: float = 0.0) -> bool:
        cx, cy = self.motor_centre
        r = self.motor_dia / 2.0 + expand
        return (xc - cx) ** 2 + (yc - cy) ** 2 <= r * r

    def build_masks(self):
        """Return a dict of boolean arrays over the element grid (nx, ny, nz)."""
        # Element centres
        xs = (np.arange(self.nx) + 0.5) * self.dx
        ys = (np.arange(self.ny) + 0.5) * self.dy
        zs = (np.arange(self.nz) + 0.5) * self.dz
        XC, YC, ZC = np.meshgrid(xs, ys, zs, indexing="ij")

        in_footprint = np.ones_like(XC, dtype=bool)  # full rectangle by default

        # Screw holes (subtract)
        in_screw = np.zeros_like(XC, dtype=bool)
        for cx, cy in self.screw_centres:
            in_screw |= ((XC - cx) ** 2 + (YC - cy) ** 2) <= (self.screw_dia / 2.0) ** 2

        # Motor hole (subtract)
        in_motor = (
            (XC - self.motor_centre[0]) ** 2 + (YC - self.motor_centre[1]) ** 2
        ) <= (self.motor_dia / 2.0) ** 2

        # Material domain
        in_arm = in_footprint & ~in_screw & ~in_motor

        # Keep-solid rings (material that must stay around holes — assembly interface)
        keep_screw = np.zeros_like(XC, dtype=bool)
        for cx, cy in self.screw_centres:
            keep_screw |= ((XC - cx) ** 2 + (YC - cy) ** 2) <= (
                self.screw_dia / 2.0 + self.keep_solid_ring) ** 2
        keep_screw &= in_arm   # only count voxels inside the arm

        keep_motor = (
            (XC - self.motor_centre[0]) ** 2 + (YC - self.motor_centre[1]) ** 2
        ) <= (self.motor_dia / 2.0 + self.motor_keep_ring) ** 2
        keep_motor &= in_arm

        keep_solid = keep_screw | keep_motor  # voxels excluded from TO design domain
        design = in_arm & ~keep_solid

        return dict(
            in_arm=in_arm,
            keep_solid=keep_solid,
            design=design,
            keep_screw=keep_screw,
            keep_motor=keep_motor,
        )

    # ------------------------------------------------------------------ #
    # Boundary conditions and load identification
    # ------------------------------------------------------------------ #
    def fixed_nodes(self) -> np.ndarray:
        """Nodes whose 3 DOFs are fully constrained (inner surface of screw holes)."""
        fixed = []
        for cx, cy in self.screw_centres:
            r_outer = self.screw_dia / 2.0 + self.keep_solid_ring
            r_inner = self.screw_dia / 2.0
            for ix in range(self.nx + 1):
                for iy in range(self.ny + 1):
                    x, y, _ = self.node_coord(ix, iy, 0)
                    d = np.hypot(x - cx, y - cy)
                    # Fix nodes that sit on the keep-ring around the screw hole
                    if r_inner <= d <= r_outer:
                        for iz in range(self.nz + 1):
                            fixed.append(self.node_index(ix, iy, iz))
        return np.unique(np.asarray(fixed, dtype=np.int64))

    def load_nodes(self, load_dir_xy: Tuple[float, float] = (0.0, 1.0)
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify nodes around the motor mount hole where the bearing load is applied,
        and return (node_indices, weights) where the weights distribute a unit total
        force across those nodes according to cos(theta) bearing distribution.

        load_dir_xy = unit vector in the (x, y) plane indicating the in-plane bearing
                       direction. Default (0, 1) = +y (lateral).
        For an in-plane bearing pull, load is applied to nodes on the half of the
        hole's inner surface that is opposite the load direction (where the bolt
        bears against the hole).
        """
        cx, cy = self.motor_centre
        r_inner = self.motor_dia / 2.0
        r_outer = self.motor_dia / 2.0 + self.motor_keep_ring
        ld = np.asarray(load_dir_xy, dtype=float)
        ld /= np.linalg.norm(ld)

        nodes = []
        cos_weights = []
        for ix in range(self.nx + 1):
            for iy in range(self.ny + 1):
                x, y, _ = self.node_coord(ix, iy, 0)
                d = np.hypot(x - cx, y - cy)
                if not (r_inner <= d <= r_outer):
                    continue
                # Unit normal from hole centre to the node (radial direction)
                nx_dir = (x - cx) / d
                ny_dir = (y - cy) / d
                # Cosine of angle between radial and -load_dir
                #   ( bolt pulls bearing surface in -ld direction )
                cos_t = -(nx_dir * ld[0] + ny_dir * ld[1])
                if cos_t <= 1e-6:
                    continue
                # All z layers
                for iz in range(self.nz + 1):
                    nodes.append(self.node_index(ix, iy, iz))
                    cos_weights.append(cos_t)

        nodes = np.asarray(nodes, dtype=np.int64)
        weights = np.asarray(cos_weights, dtype=float)
        weights /= weights.sum()
        return nodes, weights

    def load_dofs_for_force(self, F_xyz: Tuple[float, float, float]
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (global_dofs, magnitudes) so that the assembled load vector can be
        constructed by F[global_dofs] += magnitudes.

        F_xyz = total force vector in Newtons (Fx, Fy, Fz).

        - The in-plane components (Fx, Fy) are applied as a bearing load on the
          half of the motor hole inner surface opposite the load direction
          (cos-weighted distribution).
        - The out-of-plane component (Fz) is applied uniformly to ALL nodes
          on the inner surface of the motor hole, representing the transverse
          thrust transmitted by the bolt/washer to the plate around the
          motor mount.
        """
        Fx, Fy, Fz = F_xyz
        # In-plane: bearing distribution on half the hole
        if abs(Fx) + abs(Fy) > 1e-12:
            ld_xy = (Fx, Fy)
            nodes_xy, w_xy = self.load_nodes(load_dir_xy=ld_xy)
            mag_x = Fx * w_xy
            mag_y = Fy * w_xy
        else:
            nodes_xy = np.array([], dtype=np.int64)
            mag_x = np.array([])
            mag_y = np.array([])

        # Out-of-plane: uniform distribution over all hole-boundary nodes (all z layers)
        if abs(Fz) > 1e-12:
            nodes_z = self._all_motor_boundary_nodes()
            mag_z = (Fz / len(nodes_z)) * np.ones(len(nodes_z))
        else:
            nodes_z = np.array([], dtype=np.int64)
            mag_z = np.array([])

        # Build the DOF index arrays
        dofs = []
        mags = []
        if len(nodes_xy) > 0:
            dofs.append(3 * nodes_xy)
            mags.append(mag_x)
            dofs.append(3 * nodes_xy + 1)
            mags.append(mag_y)
        if len(nodes_z) > 0:
            dofs.append(3 * nodes_z + 2)
            mags.append(mag_z)
        if not dofs:
            return np.array([], dtype=np.int64), np.array([])
        return np.concatenate(dofs), np.concatenate(mags)

    def _all_motor_boundary_nodes(self) -> np.ndarray:
        """All nodes on the inner surface of the motor mount hole (all z layers)."""
        cx, cy = self.motor_centre
        r_inner = self.motor_dia / 2.0
        r_outer = self.motor_dia / 2.0 + self.motor_keep_ring
        nodes = []
        for ix in range(self.nx + 1):
            for iy in range(self.ny + 1):
                x, y, _ = self.node_coord(ix, iy, 0)
                d = np.hypot(x - cx, y - cy)
                if r_inner <= d <= r_outer:
                    for iz in range(self.nz + 1):
                        nodes.append(self.node_index(ix, iy, iz))
        return np.asarray(nodes, dtype=np.int64)

    def load_dofs_for_moment_Mz(self, Mz: float
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a moment Mz about the motor-mount axis (+Z) as a set of
        tangential in-plane forces on the boundary nodes of the motor hole.

        For each node i at angular position theta_i and radius r_i from the
        hole centre, the tangent direction is (-sin theta_i, cos theta_i).
        Applying a tangential force F_t at node i contributes r_i * F_t to
        the moment about the hole centre.  We distribute Mz uniformly:

            F_t at every node = Mz / sum_i(r_i)

        Returns global DOF indices and force magnitudes.
        """
        cx, cy = self.motor_centre
        r_inner = self.motor_dia / 2.0
        r_outer = self.motor_dia / 2.0 + self.motor_keep_ring
        nodes = []
        r_list = []
        cos_list = []
        sin_list = []
        for ix in range(self.nx + 1):
            for iy in range(self.ny + 1):
                x, y, _ = self.node_coord(ix, iy, 0)
                d = np.hypot(x - cx, y - cy)
                if not (r_inner <= d <= r_outer):
                    continue
                cos_t = (x - cx) / d
                sin_t = (y - cy) / d
                for iz in range(self.nz + 1):
                    nodes.append(self.node_index(ix, iy, iz))
                    r_list.append(d)
                    cos_list.append(cos_t)
                    sin_list.append(sin_t)
        nodes = np.asarray(nodes, dtype=np.int64)
        r_arr = np.asarray(r_list)
        cos_arr = np.asarray(cos_list)
        sin_arr = np.asarray(sin_list)
        # F_t per node so that sum_i (r_i * F_t) = Mz
        # Use a constant F_t = Mz / sum(r_i)
        F_t = Mz / r_arr.sum()
        # Tangent direction: (-sin theta, +cos theta)
        Fx = -F_t * sin_arr
        Fy = +F_t * cos_arr
        dofs = np.concatenate([3 * nodes, 3 * nodes + 1])
        mags = np.concatenate([Fx, Fy])
        return dofs, mags

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        masks = self.build_masks()
        n_arm = int(masks["in_arm"].sum())
        n_design = int(masks["design"].sum())
        n_keep = int(masks["keep_solid"].sum())
        return (
            f"ArmGeometry summary:\n"
            f"  Bounding box: {self.L_x*1e3:.1f} x {self.L_y*1e3:.1f} x {self.L_z*1e3:.1f} mm\n"
            f"  Voxel size:   {self.dx*1e3:.2f} x {self.dy*1e3:.2f} x {self.dz*1e3:.2f} mm\n"
            f"  Grid:         {self.nx} x {self.ny} x {self.nz} = {self.n_elements} voxels\n"
            f"  Nodes:        {self.n_nodes}  (DOFs: {self.n_dofs})\n"
            f"  In arm:       {n_arm} elements\n"
            f"  Design dom.:  {n_design} elements\n"
            f"  Keep-solid:   {n_keep} elements\n"
        )


if __name__ == "__main__":
    g = ArmGeometry()
    print(g.summary())
    fixed = g.fixed_nodes()
    print(f"Fixed nodes: {len(fixed)}")
    nodes, w = g.load_nodes()
    print(f"Load nodes (lateral): {len(nodes)} (per layer x {g.nz+1} layers)")
    print(f"Sum of weights = {w.sum():.6f} (should be 1.0)")
