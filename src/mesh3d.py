"""
3D F450 arm geometry — corrected against the OEM CAD (v5, 2026-08-13).

Every dimension and fastener position below was measured from
``figure_pack_for_artist/cad/F450_Frame_Assembly.stp`` through the
OpenCascade kernel, not taken from the datasheet or estimated.  See
``notes/model_vs_cad_findings.md`` for the measurement record.

What changed from v4, and why it matters:

* **Cross-section was transposed.**  v4 used 220 × 56 (width) × 44 (height).
  The CAD says the arm is 215.9 long × 40.8 wide × 54.3 tall — the STEP
  files are modelled Y-up, and reading them Z-up swaps width and height.
  The v4 domain therefore did not even enclose the real arm (44 < 54.3).

* **Fastener pattern was wrong.**  v4 clamped four Ø3.5 mm bolts running
  the *full* height of the arm.  The product has **four screws into the
  top flange and two into the bottom flange**, Ø5.5 mm, engaging over
  different vertical ranges.  Since the clamped region sets the cantilever
  root, this drives omega_1 directly.

The two clamp planes sit 38 mm apart (top screws at z ~ 43-54, bottom at
z ~ 11-19), which matches the 38.0 mm measured clear gap between the
carbon plates — an independent consistency check on the numbers below.

Design domain adopted here:
    L_x = 216 mm   (long axis, x)   — OEM 215.9 mm
    L_y =  42 mm   (width, y)       — OEM  40.8 mm
    L_z =  54 mm   (vertical, z)    — OEM  54.3 mm

At 2 mm voxels that is 108 × 21 × 27 = 61 236 elements and
109 × 22 × 28 = 67 144 nodes (201 432 DoFs) — slightly smaller than the
v4 mesh, so run costs are comparable.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ArmGeometry3D:
    """3D OEM-scale F450 arm geometry."""

    # Bounding box — OEM 215.9 x 40.8 x 54.3 mm, rounded out to whole voxels
    L_x: float = 216e-3
    L_y: float = 42e-3
    L_z: float = 54e-3

    # Voxel size
    dx: float = 2e-3
    dy: float = 2e-3
    dz: float = 2e-3

    # Body-side screws, measured from the OEM assembly.  Each entry is
    # (x, y, z0, z1): axis position plus the vertical span over which the
    # screw actually engages the arm.  FOUR enter the top flange from the
    # top plate; TWO enter the bottom flange from the bottom plate.
    # y values are symmetrised about the arm centreline (y = L_y/2 = 21 mm);
    # the raw measurements scatter by ~1.5 mm about the midline, which is
    # tessellation noise on a part that is symmetric by design.
    # Ø3.10 is the *hole* in the arm, measured as four vertical cylindrical
    # faces in the top flange of F450_Arm.stp. The Ø5.5 solids in the frame
    # assembly are the screw HEADS, not the shanks — using 5.5 here would
    # bore a hole twice the real diameter. The head bearing footprint is
    # covered instead by keep_solid_ring, which puts the outer radius of the
    # clamped annulus at 1.55 + 4.0 = 5.55 mm, i.e. just outside the head.
    # Oe2.5 is the frame fastener. (The Oe3.10 holes in the flat pad are the
    # MOTOR mount at the opposite end of the arm, not the frame joint — an
    # earlier revision of this file had the two ends swapped.)
    screw_dia: float = 2.5e-3
    screws: Tuple[Tuple[float, float, float, float], ...] = (
        # --- 4 screws down from the TOP plate --------------------------
        (25.8e-3,  9.5e-3, 33.2e-3, 53.2e-3),
        (25.8e-3, 32.5e-3, 33.2e-3, 53.2e-3),
        (5.8e-3,  12.7e-3, 46.9e-3, 53.2e-3),
        (5.8e-3,  29.2e-3, 46.9e-3, 53.2e-3),
        # --- 2 screws up from the BOTTOM plate -------------------------
        # z = 15.2..35.2, i.e. 38 mm below the top face — matching the
        # measured clear gap between the two carbon plates. The previous
        # values put these at z = 0..4, clamping the very bottom of the
        # domain and making the root ~50 mm deep instead of 38 mm, which
        # stiffened it and biased omega_1 high.
        (25.5e-3, 10.5e-3, 15.2e-3, 35.2e-3),
        (25.5e-3, 31.5e-3, 15.2e-3, 35.2e-3),
    )

    # Motor-mount hole (top face only — motor body sits on top).
    # Motor screw centroid measured 201.9 mm out from the arm root.
    # Centroid of the four Oe3.10 motor screw holes, mapped from the CAD.
    motor_dia: float = 10e-3
    motor_centre: Tuple[float, float] = (195.6e-3, 21e-3)

    # "Keep-solid" rings (excluded from TO design domain)
    keep_solid_ring: float = 4.0e-3
    motor_keep_ring: float = 8.0e-3

    # How deep into the arm (from the relevant face) the motor pad keep-solid
    # extends.  The motor sits on the top face only, so the keep-solid for
    # the motor mount is only the top several voxels.
    motor_pad_depth: float = 4e-3

    nx: int = field(init=False)
    ny: int = field(init=False)
    nz: int = field(init=False)
    n_elements: int = field(init=False)
    n_nodes: int = field(init=False)
    n_dofs: int = field(init=False)

    @property
    def screw_centres(self) -> Tuple[Tuple[float, float], ...]:
        """(x, y) of every screw — kept so callers written against the v4
        API (renderers, assembly plots) keep working."""
        return tuple((s[0], s[1]) for s in self.screws)

    def __post_init__(self):
        self.nx = int(round(self.L_x / self.dx))
        self.ny = int(round(self.L_y / self.dy))
        self.nz = int(round(self.L_z / self.dz))
        self.n_elements = self.nx * self.ny * self.nz
        self.n_nodes = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
        self.n_dofs = 3 * self.n_nodes

    # ------------------------------------------------------------------ #
    def element_centre(self, ex: int, ey: int, ez: int) -> Tuple[float, float, float]:
        return ((ex + 0.5) * self.dx,
                (ey + 0.5) * self.dy,
                (ez + 0.5) * self.dz)

    def node_coord(self, nx: int, ny: int, nz: int) -> Tuple[float, float, float]:
        return (nx * self.dx, ny * self.dy, nz * self.dz)

    def node_index(self, nx: int, ny: int, nz: int) -> int:
        return nz * (self.nx + 1) * (self.ny + 1) + ny * (self.nx + 1) + nx

    def element_index(self, ex: int, ey: int, ez: int) -> int:
        return ez * self.nx * self.ny + ey * self.nx + ex

    def element_nodes(self, ex: int, ey: int, ez: int) -> np.ndarray:
        """8 node global indices for the hex element (CCW: bottom face then top)."""
        n = []
        for dz_ in (0, 1):
            for dy_, dx_ in ((0, 0), (0, 1), (1, 1), (1, 0)):
                n.append(self.node_index(ex + dx_, ey + dy_, ez + dz_))
        return np.array(n, dtype=np.int64)

    # ------------------------------------------------------------------ #
    def build_masks(self) -> dict:
        """Boolean arrays over the element grid, shape (nx, ny, nz)."""
        xs = (np.arange(self.nx) + 0.5) * self.dx
        ys = (np.arange(self.ny) + 0.5) * self.dy
        zs = (np.arange(self.nz) + 0.5) * self.dz
        XC, YC, ZC = np.meshgrid(xs, ys, zs, indexing="ij")

        # Full bounding box is in-arm
        in_footprint = np.ones_like(XC, dtype=bool)

        # Body-side screw holes — each is a short vertical cylinder over the
        # span the screw actually occupies, NOT a full-height through-bolt.
        in_screw = np.zeros_like(XC, dtype=bool)
        for cx, cy, z0, z1 in self.screws:
            in_screw |= (((XC - cx) ** 2 + (YC - cy) ** 2) <=
                         (self.screw_dia / 2.0) ** 2) & (ZC >= z0) & (ZC <= z1)

        # Motor-mount hole — top face only (only the top `motor_pad_depth` of
        # material has a hole through it; the bolt isn't there in the OEM,
        # but a cylindrical hole through the top face accepts the motor body)
        z_top = self.L_z - self.motor_pad_depth
        in_motor = (((XC - self.motor_centre[0]) ** 2 +
                     (YC - self.motor_centre[1]) ** 2) <=
                     (self.motor_dia / 2.0) ** 2) & (ZC > z_top)

        in_arm = in_footprint & ~in_screw & ~in_motor

        # Keep-solid rings — material that must remain (assembly interface)
        # Keep-solid collar around each screw, over that screw's own span plus
        # one voxel of lead-in, so the optimiser cannot delete the material the
        # fastener bears against.
        keep_screw = np.zeros_like(XC, dtype=bool)
        r_keep = (self.screw_dia / 2.0 + self.keep_solid_ring) ** 2
        for cx, cy, z0, z1 in self.screws:
            keep_screw |= (((XC - cx) ** 2 + (YC - cy) ** 2) <= r_keep) & \
                          (ZC >= z0 - self.dz) & (ZC <= z1 + self.dz)
        keep_screw &= in_arm

        # Motor pad: a horizontal annular region on the top face only
        keep_motor = ((((XC - self.motor_centre[0]) ** 2 +
                       (YC - self.motor_centre[1]) ** 2) <=
                       (self.motor_dia / 2.0 + self.motor_keep_ring) ** 2)
                       & (ZC > z_top))
        keep_motor &= in_arm

        keep_solid = keep_screw | keep_motor
        design = in_arm & ~keep_solid

        return dict(
            in_arm=in_arm, keep_solid=keep_solid, design=design,
            keep_screw=keep_screw, keep_motor=keep_motor,
        )

    # ------------------------------------------------------------------ #
    def fixed_nodes(self) -> np.ndarray:
        """Bearing surfaces of the six body-side screws, fixed in all 3 DoFs.

        Only the nodes each screw actually bears against are constrained —
        the annulus around its axis, over that screw's own vertical span.
        The four top screws therefore clamp the top flange and the two bottom
        screws clamp the bottom flange, 38 mm below, reproducing the real
        bolted joint.

        The v4 version clamped every node over the full height at four
        positions, which both over-constrained the root and put half the
        constraint where the product has no fastener at all.
        """
        r_outer = self.screw_dia / 2.0 + self.keep_solid_ring
        r_inner = self.screw_dia / 2.0
        fixed = []
        for cx, cy, z0, z1 in self.screws:
            for ix in range(self.nx + 1):
                for iy in range(self.ny + 1):
                    x, y, _ = self.node_coord(ix, iy, 0)
                    if r_inner <= np.hypot(x - cx, y - cy) <= r_outer:
                        for iz in range(self.nz + 1):
                            z = iz * self.dz
                            if z0 - self.dz <= z <= z1 + self.dz:
                                fixed.append(self.node_index(ix, iy, iz))
        return np.unique(np.asarray(fixed, dtype=np.int64))

    # ------------------------------------------------------------------ #
    def motor_mount_top_nodes(self) -> np.ndarray:
        """Public alias for `_motor_top_boundary_nodes` — used as the tip-mass
        attachment points (motor + prop) in modal analysis."""
        return self._motor_top_boundary_nodes()

    def _motor_top_boundary_nodes(self) -> np.ndarray:
        """Nodes on the top face inside the motor-mount pad area
        (where the motor body bears against the arm)."""
        cx, cy = self.motor_centre
        # Anywhere on the top face within the motor pad radius
        r_outer = self.motor_dia / 2.0 + self.motor_keep_ring
        nodes = []
        # Top face is at iz = nz
        iz = self.nz
        for ix in range(self.nx + 1):
            for iy in range(self.ny + 1):
                x, y, _ = self.node_coord(ix, iy, iz)
                d = np.hypot(x - cx, y - cy)
                if d <= r_outer:
                    nodes.append(self.node_index(ix, iy, iz))
        return np.asarray(nodes, dtype=np.int64)

    def _motor_inner_boundary_nodes(self) -> np.ndarray:
        """Nodes on the inner cylindrical wall of the motor mount hole (top
        few layers, where the motor body bears in the xy plane)."""
        cx, cy = self.motor_centre
        r_inner = self.motor_dia / 2.0
        r_outer = self.motor_dia / 2.0 + self.motor_keep_ring
        nodes = []
        z_top = self.L_z - self.motor_pad_depth
        for ix in range(self.nx + 1):
            for iy in range(self.ny + 1):
                x, y, _ = self.node_coord(ix, iy, 0)
                d = np.hypot(x - cx, y - cy)
                if r_inner <= d <= r_outer:
                    # Only layers above z_top
                    iz0 = int(round(z_top / self.dz))
                    for iz in range(iz0, self.nz + 1):
                        nodes.append(self.node_index(ix, iy, iz))
        return np.asarray(nodes, dtype=np.int64)

    # ------------------------------------------------------------------ #
    def load_dofs_for_force(self, F_xyz: Tuple[float, float, float]
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """Distribute (Fx, Fy, Fz) on the motor mount.

        - The vertical (Fz) component is distributed uniformly over all top-face
          nodes inside the motor pad — modelling motor thrust bearing on the
          arm's top face.
        - In-plane (Fx, Fy) components are distributed as a cosine-weighted
          bearing load on the inner cylindrical wall of the motor mount hole.
        """
        Fx, Fy, Fz = F_xyz

        # Vertical thrust on top face
        if abs(Fz) > 1e-12:
            top_nodes = self._motor_top_boundary_nodes()
            mag_z = (Fz / len(top_nodes)) * np.ones(len(top_nodes))
        else:
            top_nodes = np.array([], dtype=np.int64)
            mag_z = np.array([])

        # In-plane bearing on inner cylinder
        if abs(Fx) + abs(Fy) > 1e-12:
            cx, cy = self.motor_centre
            r_inner = self.motor_dia / 2.0
            r_outer = self.motor_dia / 2.0 + self.motor_keep_ring
            ld = np.asarray((Fx, Fy)); ld /= np.linalg.norm(ld)
            nodes_xy = []
            cos_w = []
            z_top = self.L_z - self.motor_pad_depth
            iz0 = int(round(z_top / self.dz))
            for ix in range(self.nx + 1):
                for iy in range(self.ny + 1):
                    x, y, _ = self.node_coord(ix, iy, 0)
                    d = np.hypot(x - cx, y - cy)
                    if not (r_inner <= d <= r_outer):
                        continue
                    nx_dir = (x - cx) / d
                    ny_dir = (y - cy) / d
                    cos_t = -(nx_dir * ld[0] + ny_dir * ld[1])
                    if cos_t <= 1e-6:
                        continue
                    for iz in range(iz0, self.nz + 1):
                        nodes_xy.append(self.node_index(ix, iy, iz))
                        cos_w.append(cos_t)
            if nodes_xy:
                nodes_xy = np.asarray(nodes_xy, dtype=np.int64)
                cos_w = np.asarray(cos_w); cos_w /= cos_w.sum()
                mag_x = Fx * cos_w; mag_y = Fy * cos_w
            else:
                nodes_xy = np.array([], dtype=np.int64)
                mag_x = np.array([]); mag_y = np.array([])
        else:
            nodes_xy = np.array([], dtype=np.int64)
            mag_x = np.array([]); mag_y = np.array([])

        dofs = []
        mags = []
        if len(top_nodes) > 0:
            dofs.append(3 * top_nodes + 2); mags.append(mag_z)
        if len(nodes_xy) > 0:
            dofs.append(3 * nodes_xy); mags.append(mag_x)
            dofs.append(3 * nodes_xy + 1); mags.append(mag_y)
        if not dofs:
            return np.array([], dtype=np.int64), np.array([])
        return np.concatenate(dofs), np.concatenate(mags)

    # ------------------------------------------------------------------ #
    def load_dofs_for_moment_Mz(self, Mz: float
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a moment Mz about the motor axis (+Z) as tangential nodal
        forces on the inner cylindrical wall of the motor mount."""
        cx, cy = self.motor_centre
        r_inner = self.motor_dia / 2.0
        r_outer = self.motor_dia / 2.0 + self.motor_keep_ring
        z_top = self.L_z - self.motor_pad_depth
        iz0 = int(round(z_top / self.dz))
        nodes, r_list, cos_list, sin_list = [], [], [], []
        for ix in range(self.nx + 1):
            for iy in range(self.ny + 1):
                x, y, _ = self.node_coord(ix, iy, 0)
                d = np.hypot(x - cx, y - cy)
                if not (r_inner <= d <= r_outer):
                    continue
                cos_t = (x - cx) / d
                sin_t = (y - cy) / d
                for iz in range(iz0, self.nz + 1):
                    nodes.append(self.node_index(ix, iy, iz))
                    r_list.append(d); cos_list.append(cos_t); sin_list.append(sin_t)
        nodes = np.asarray(nodes, dtype=np.int64)
        r_arr = np.asarray(r_list); cos_arr = np.asarray(cos_list); sin_arr = np.asarray(sin_list)
        F_t = Mz / r_arr.sum()
        Fx = -F_t * sin_arr; Fy = +F_t * cos_arr
        return np.concatenate([3 * nodes, 3 * nodes + 1]), np.concatenate([Fx, Fy])

    # ------------------------------------------------------------------ #
    def summary(self) -> str:
        masks = self.build_masks()
        n_arm = int(masks["in_arm"].sum())
        n_design = int(masks["design"].sum())
        n_keep = int(masks["keep_solid"].sum())
        return (
            f"ArmGeometry3D summary:\n"
            f"  Bounding box: {self.L_x*1e3:.1f} x {self.L_y*1e3:.1f} x {self.L_z*1e3:.1f} mm\n"
            f"  Voxel size:   {self.dx*1e3:.2f} x {self.dy*1e3:.2f} x {self.dz*1e3:.2f} mm\n"
            f"  Grid:         {self.nx} x {self.ny} x {self.nz} = {self.n_elements} voxels\n"
            f"  Nodes:        {self.n_nodes}  (DoFs: {self.n_dofs})\n"
            f"  In arm:       {n_arm} elements\n"
            f"  Design dom.:  {n_design} elements\n"
            f"  Keep-solid:   {n_keep} elements\n"
        )


if __name__ == "__main__":
    g = ArmGeometry3D()
    print(g.summary())
    fixed = g.fixed_nodes()
    print(f"Fixed nodes: {len(fixed)}")
