"""
Custom 3D linear-elastic finite element solver using 8-node trilinear hex
elements on a structured Cartesian voxel mesh.

Designed to be efficient for SIMP topology optimization: the element
stiffness matrix K_e for a unit-density solid voxel is computed once and then
scaled per element by rho_e^p during each TO iteration.

References
----------
- Bendsoe & Sigmund, "Topology Optimization", Springer 2003
- Andreassen et al., "Efficient topology optimization in MATLAB using 88 lines
  of code", Struct. Multidisc. Optim. (2011) 43:1
- Sigmund "A 99 line topology optimization code written in Matlab"
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass

from .mesh import ArmGeometry


# -----------------------------------------------------------------------------
# Hex element kernel
# -----------------------------------------------------------------------------
GAUSS_PT = 1.0 / np.sqrt(3.0)
GAUSS = [(s1, s2, s3, 1.0)
         for s1 in (-GAUSS_PT, GAUSS_PT)
         for s2 in (-GAUSS_PT, GAUSS_PT)
         for s3 in (-GAUSS_PT, GAUSS_PT)]

# Local node coordinates in (xi, eta, zeta), matching mesh.element_nodes ordering
# (bottom z=-1 face CCW from (-1,-1), then top z=+1 face CCW from (-1,-1))
LOCAL_NODES = np.array([
    (-1, -1, -1),
    (+1, -1, -1),
    (+1, +1, -1),
    (-1, +1, -1),
    (-1, -1, +1),
    (+1, -1, +1),
    (+1, +1, +1),
    (-1, +1, +1),
], dtype=float)


def shape_funcs_and_grads(xi: float, eta: float, zeta: float):
    """Return (N, dN_dxi) for the 8-node hex.
    N has shape (8,), dN_dxi has shape (8, 3) with columns d/dxi, d/deta, d/dzeta.
    """
    xs, ys, zs = LOCAL_NODES[:, 0], LOCAL_NODES[:, 1], LOCAL_NODES[:, 2]
    N = 0.125 * (1 + xs * xi) * (1 + ys * eta) * (1 + zs * zeta)
    dN = np.empty((8, 3))
    dN[:, 0] = 0.125 * xs * (1 + ys * eta) * (1 + zs * zeta)
    dN[:, 1] = 0.125 * (1 + xs * xi) * ys * (1 + zs * zeta)
    dN[:, 2] = 0.125 * (1 + xs * xi) * (1 + ys * eta) * zs
    return N, dN


def constitutive_D(E: float, nu: float) -> np.ndarray:
    """Isotropic 6x6 elasticity matrix (Voigt notation: eps = [exx, eyy, ezz, gxy, gyz, gxz])."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    D = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            D[i, j] = lam + (2.0 * mu if i == j else 0.0)
    for i in range(3, 6):
        D[i, i] = mu
    return D


def constitutive_D_orthotropic(Ex: float, Ey: float, Ez: float,
                                nu_xy: float, nu_yz: float, nu_xz: float,
                                Gxy: float, Gyz: float, Gxz: float) -> np.ndarray:
    """Orthotropic 6x6 elasticity matrix (for anisotropy study).

    Uses engineering constants. Symmetric (nu_yx = nu_xy * Ey/Ex, etc.).
    """
    # Compliance matrix (Voigt: [exx, eyy, ezz, gxy, gyz, gxz])
    S = np.zeros((6, 6))
    S[0, 0] = 1.0 / Ex
    S[1, 1] = 1.0 / Ey
    S[2, 2] = 1.0 / Ez
    S[3, 3] = 1.0 / Gxy
    S[4, 4] = 1.0 / Gyz
    S[5, 5] = 1.0 / Gxz
    S[0, 1] = -nu_xy / Ex
    S[1, 0] = S[0, 1]
    S[1, 2] = -nu_yz / Ey
    S[2, 1] = S[1, 2]
    S[0, 2] = -nu_xz / Ex
    S[2, 0] = S[0, 2]
    return np.linalg.inv(S)


def unit_element_K(E: float, nu: float, dx: float, dy: float, dz: float,
                   D: np.ndarray = None) -> np.ndarray:
    """Build the 24x24 element stiffness for a rectangular hex with the given
    dimensions and material.  Returns Ke such that K_global += sum_e Ke."""
    if D is None:
        D = constitutive_D(E, nu)

    # Coordinate scaling: dx_local = (dx/2) d_xi etc.
    # Jacobian determinant: |J| = (dx/2)(dy/2)(dz/2) = dx*dy*dz/8
    half = np.array([dx / 2.0, dy / 2.0, dz / 2.0])
    detJ = half[0] * half[1] * half[2]
    inv_half = 1.0 / half  # multiplier from d/dxi to d/dx etc.

    Ke = np.zeros((24, 24))
    for xi, eta, zeta, w in GAUSS:
        _, dN_dxi = shape_funcs_and_grads(xi, eta, zeta)
        # Map to physical gradient: dN/dx = (2/dx) dN/dxi
        dN_dx = dN_dxi * inv_half[np.newaxis, :]
        # Strain-displacement matrix B (6 x 24)
        B = np.zeros((6, 24))
        for n in range(8):
            i = 3 * n
            Nx, Ny, Nz = dN_dx[n, 0], dN_dx[n, 1], dN_dx[n, 2]
            B[0, i + 0] = Nx
            B[1, i + 1] = Ny
            B[2, i + 2] = Nz
            B[3, i + 0] = Ny
            B[3, i + 1] = Nx
            B[4, i + 1] = Nz
            B[4, i + 2] = Ny
            B[5, i + 0] = Nz
            B[5, i + 2] = Nx
        Ke += (B.T @ D @ B) * detJ * w
    return Ke


def unit_element_M(rho_material: float, dx: float, dy: float, dz: float
                    ) -> np.ndarray:
    """24x24 consistent mass matrix for an 8-node trilinear hex element of
    unit-density material.  Scaling by SIMP density factor is applied at
    assembly time, identically to the K pattern.

    M_e[3i+d, 3j+d] = rho_material * integral( N_i N_j ) dV   for d in {0,1,2}
    """
    half = np.array([dx / 2.0, dy / 2.0, dz / 2.0])
    detJ = half[0] * half[1] * half[2]
    Me = np.zeros((24, 24))
    I3 = np.eye(3)
    for xi, eta, zeta, w in GAUSS:
        N, _ = shape_funcs_and_grads(xi, eta, zeta)        # (8,)
        NN = np.outer(N, N)                                # (8, 8)
        # Kronecker: 24x24 block where block (i,j) is NN[i,j] * I_3
        Me += np.kron(NN, I3) * (rho_material * detJ * w)
    return Me


def unit_element_B_at_centre(dx: float, dy: float, dz: float) -> np.ndarray:
    """Strain-displacement matrix B evaluated at the element centre (xi=eta=zeta=0).
    Used for stress recovery: eps = B @ u_e, then sigma = D @ eps.
    """
    half = np.array([dx / 2.0, dy / 2.0, dz / 2.0])
    inv_half = 1.0 / half
    _, dN_dxi = shape_funcs_and_grads(0.0, 0.0, 0.0)
    dN_dx = dN_dxi * inv_half[np.newaxis, :]
    B = np.zeros((6, 24))
    for n in range(8):
        i = 3 * n
        Nx, Ny, Nz = dN_dx[n, 0], dN_dx[n, 1], dN_dx[n, 2]
        B[0, i + 0] = Nx
        B[1, i + 1] = Ny
        B[2, i + 2] = Nz
        B[3, i + 0] = Ny
        B[3, i + 1] = Nx
        B[4, i + 1] = Nz
        B[4, i + 2] = Ny
        B[5, i + 0] = Nz
        B[5, i + 2] = Nx
    return B


# -----------------------------------------------------------------------------
# FEM Problem
# -----------------------------------------------------------------------------
@dataclass
class Material:
    E: float = 1700e6
    nu: float = 0.39
    sigma_y: float = 38e6
    sigma_u: float = 48e6
    rho: float = 930.0  # kg / m^3 — for mass calculations


class FEM3D:
    """
    Linear-elastic 3D FEM on a structured voxel mesh.

    SIMP penalisation: each element's effective Young modulus is
        E_eff(rho_e) = E_min + rho_e^p * (E_solid - E_min)
    where E_min = 1e-9 * E_solid for numerical stability.

    The element stiffness for a unit-density (E_solid) voxel is computed once
    in `__init__`. Per iteration, we scale element contributions by the
    SIMP factor and re-assemble the global K via sparse COO.
    """

    SIMP_PENALTY = 3.0
    E_MIN_FRAC = 1e-9

    def __init__(self, geom: ArmGeometry, mat: Material):
        self.geom = geom
        self.mat = mat

        # 1) Compute the per-element K (24x24) for full solid material
        D_solid = constitutive_D(mat.E, mat.nu)
        self.Ke_solid = unit_element_K(mat.E, mat.nu,
                                       geom.dx, geom.dy, geom.dz,
                                       D=D_solid)
        self.D_solid = D_solid
        self.B_centre = unit_element_B_at_centre(geom.dx, geom.dy, geom.dz)
        # Consistent mass matrix for a unit-density-factor element (uses the
        # material density `mat.rho` directly; SIMP scaling is applied at
        # assembly time).  Reused by `assemble_M` and the modal analysis.
        self.Me_solid = unit_element_M(mat.rho, geom.dx, geom.dy, geom.dz)

        # 2) Pre-compute the sparse assembly index pattern.
        #    For each element, expand 24x24 -> (576,) entries into the global rows/cols
        masks = geom.build_masks()
        in_arm_flat = masks["in_arm"].astype(bool)
        # element indices of active elements (those that participate)
        active_elements = []
        ne = geom.n_elements
        # Pre-allocate arrays sized for active elements only
        n_active_max = int(in_arm_flat.sum())

        rows = np.empty(n_active_max * 576, dtype=np.int64)
        cols = np.empty(n_active_max * 576, dtype=np.int64)
        # We also store for each active element its position in the flat array (size 576)
        # so we can scale values per element rapidly.

        elem_node_ids = np.empty((n_active_max, 8), dtype=np.int64)
        elem_dof_ids = np.empty((n_active_max, 24), dtype=np.int64)
        elem_grid_idx = np.empty((n_active_max, 3), dtype=np.int64)

        pos = 0
        k = 0
        for ez in range(geom.nz):
            for ey in range(geom.ny):
                for ex in range(geom.nx):
                    if not in_arm_flat[ex, ey, ez]:
                        continue
                    nodes = geom.element_nodes(ex, ey, ez)
                    elem_node_ids[k] = nodes
                    dofs = np.empty(24, dtype=np.int64)
                    for n in range(8):
                        dofs[3 * n:3 * n + 3] = (
                            3 * nodes[n], 3 * nodes[n] + 1, 3 * nodes[n] + 2
                        )
                    elem_dof_ids[k] = dofs
                    elem_grid_idx[k] = (ex, ey, ez)
                    # Outer product of dofs to build row/col index pattern
                    rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
                    rows[pos:pos + 576] = rr.ravel()
                    cols[pos:pos + 576] = cc.ravel()
                    pos += 576
                    k += 1

        rows = rows[:pos]
        cols = cols[:pos]
        elem_node_ids = elem_node_ids[:k]
        elem_dof_ids = elem_dof_ids[:k]
        elem_grid_idx = elem_grid_idx[:k]
        active_elements = k

        self.rows = rows
        self.cols = cols
        self.elem_node_ids = elem_node_ids
        self.elem_dof_ids = elem_dof_ids
        self.elem_grid_idx = elem_grid_idx     # (n_active, 3) of (ex, ey, ez)
        self.n_active = active_elements
        # Pre-flatten the Ke / Me templates once
        self.Ke_flat = self.Ke_solid.ravel()       # (576,)
        self.Me_flat = self.Me_solid.ravel()       # (576,)
        # 3) Identify free DOFs
        fixed_nodes = geom.fixed_nodes()
        fixed_dofs = np.concatenate(
            [3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2]
        )
        all_dofs = np.arange(geom.n_dofs)
        self.fixed_dofs = np.unique(fixed_dofs)
        free_mask = np.ones(geom.n_dofs, dtype=bool)
        free_mask[self.fixed_dofs] = False
        self.free_dofs = all_dofs[free_mask]

        # 4) Identify "ghost" DOFs — DOFs of nodes that belong to NO active element.
        #    These have an all-zero row/column in K and would make the matrix
        #    singular. We add them to fixed_dofs (just zero them).
        node_used = np.zeros(geom.n_nodes, dtype=bool)
        node_used[self.elem_node_ids.ravel()] = True
        ghost_nodes = np.where(~node_used)[0]
        ghost_dofs = np.concatenate(
            [3 * ghost_nodes, 3 * ghost_nodes + 1, 3 * ghost_nodes + 2]
        )
        if len(ghost_dofs) > 0:
            self.fixed_dofs = np.unique(np.concatenate([self.fixed_dofs, ghost_dofs]))
            free_mask = np.ones(geom.n_dofs, dtype=bool)
            free_mask[self.fixed_dofs] = False
            self.free_dofs = all_dofs[free_mask]

        # 5) Density array — one entry per active element, default 1.0 (full solid)
        self.rho_active = np.ones(self.n_active, dtype=float)

    # ------------------------------------------------------------------ #
    def assemble_K(self, rho_active: np.ndarray) -> sp.csr_matrix:
        """Assemble the global sparse stiffness using current densities.

        rho_active : (n_active,) of densities (0 < rho <= 1)
        """
        e_min = self.E_MIN_FRAC
        p = self.SIMP_PENALTY
        # SIMP modulus factor per element: e_min + (1 - e_min) * rho^p, applied to Ke_solid
        # i.e. K_e(rho) = factor * K_e_solid
        factor = e_min + (1.0 - e_min) * (rho_active ** p)  # (n_active,)
        # Broadcast factor across the 576 entries of each element
        vals = (factor[:, None] * self.Ke_flat[None, :]).ravel()   # (n_active * 576,)
        n = self.geom.n_dofs
        K = sp.coo_matrix((vals, (self.rows, self.cols)), shape=(n, n))
        return K.tocsr()

    # ------------------------------------------------------------------ #
    # SIMP mass penalty: linear above the threshold (physical mass), polynomial
    # below it (Pedersen 2000, suppresses spurious low-density localised modes
    # by making M decay faster than K).
    MASS_THRESHOLD = 0.1
    MASS_PENALTY_Q = 6.0

    def _mass_simp_factor(self, rho_active: np.ndarray) -> np.ndarray:
        """Mass-penalty factor m(rho) used in modal assembly.
        - rho >= MASS_THRESHOLD: m(rho) = rho (linear, physical mass).
        - rho <  MASS_THRESHOLD: m(rho) = (rho/MASS_THRESHOLD)^Q · MASS_THRESHOLD.
        The piecewise form pushes localised eigenmodes in low-density regions
        out of the frequency band of interest while keeping the bulk mass
        physically correct.
        """
        rho_t = self.MASS_THRESHOLD
        q = self.MASS_PENALTY_Q
        m = np.where(
            rho_active >= rho_t,
            rho_active,
            rho_t * (np.maximum(rho_active, 0.0) / rho_t) ** q,
        )
        return m

    def _mass_simp_factor_grad(self, rho_active: np.ndarray) -> np.ndarray:
        """Derivative dm/drho corresponding to `_mass_simp_factor`."""
        rho_t = self.MASS_THRESHOLD
        q = self.MASS_PENALTY_Q
        rho_safe = np.maximum(rho_active, 0.0)
        return np.where(
            rho_active >= rho_t,
            np.ones_like(rho_active),
            q * (rho_safe / rho_t) ** (q - 1.0),
        )

    def assemble_M(self, rho_active: np.ndarray,
                    tip_mass_kg: float = 0.0,
                    tip_mass_nodes: np.ndarray = None
                    ) -> sp.csr_matrix:
        """Assemble the global sparse consistent mass matrix.

        rho_active : (n_active,) of densities (0 < rho <= 1)
        tip_mass_kg : optional lumped mass (motor + prop) distributed over
                      the supplied tip nodes.  If None and tip_mass_kg > 0,
                      the caller must add the lump separately.
        tip_mass_nodes : (n_tip,) node indices to receive the lump.
        """
        factor_m = self._mass_simp_factor(rho_active)              # (n_active,)
        vals = (factor_m[:, None] * self.Me_flat[None, :]).ravel()
        n = self.geom.n_dofs
        M = sp.coo_matrix((vals, (self.rows, self.cols)), shape=(n, n))
        M = M.tocsr()
        if tip_mass_kg > 0.0 and tip_mass_nodes is not None and len(tip_mass_nodes) > 0:
            M = self.add_tip_mass(M, tip_mass_kg, tip_mass_nodes)
        return M

    def add_tip_mass(self, M: sp.csr_matrix, mass_kg: float,
                      node_ids: np.ndarray) -> sp.csr_matrix:
        """Lump `mass_kg` evenly over the translational DoFs of `node_ids`.
        Adds mass_kg / n_nodes to each of the three diagonal entries per node."""
        n_nodes = len(node_ids)
        if n_nodes == 0 or mass_kg <= 0.0:
            return M
        mass_per_node = mass_kg / n_nodes
        node_ids = np.asarray(node_ids, dtype=np.int64)
        dofs = np.concatenate([3 * node_ids,
                                3 * node_ids + 1,
                                3 * node_ids + 2])
        vals = np.full(len(dofs), mass_per_node)
        n = self.geom.n_dofs
        M_lump = sp.coo_matrix((vals, (dofs, dofs)), shape=(n, n)).tocsr()
        return (M + M_lump).tocsr()

    def modal_analysis(self, rho_active: np.ndarray,
                        n_modes: int = 6,
                        tip_mass_kg: float = 0.060,
                        tip_mass_nodes: np.ndarray = None,
                        X0_free: np.ndarray = None,
                        tol: float = 1e-4,
                        maxiter: int = 500,
                        ) -> tuple:
        """Solve (K - omega^2 M) v = 0 for the lowest `n_modes` natural
        frequencies using LOBPCG with a diagonal preconditioner.  LOBPCG is
        ~10-100× faster than scipy eigsh shift-invert on 200k-DoF 3D problems
        because it avoids the dense LU fill-in of a sparse direct factorisation.

        Parameters
        ----------
        X0_free : (n_free, n_modes) optional warm-start eigenvector guess on
                  free DoFs only.  Pass the previous iter's mode_shapes[free]
                  to dramatically accelerate convergence inside the TO loop.

        Returns
        -------
        frequencies_Hz : (n_modes,) — sorted ascending
        mode_shapes    : (n_dofs, n_modes) — full-length, zeroed at fixed DoFs
        omega_sq       : (n_modes,) — eigenvalues (rad/s)^2
        """
        K = self.assemble_K(rho_active)
        if tip_mass_nodes is None:
            if hasattr(self.geom, "motor_mount_top_nodes"):
                tip_mass_nodes = self.geom.motor_mount_top_nodes()
            elif hasattr(self.geom, "_motor_top_boundary_nodes"):
                tip_mass_nodes = self.geom._motor_top_boundary_nodes()
            else:
                tip_mass_nodes = np.array([], dtype=np.int64)
        M = self.assemble_M(rho_active,
                             tip_mass_kg=tip_mass_kg,
                             tip_mass_nodes=tip_mass_nodes)
        free = self.free_dofs
        K_ff = K[free][:, free].tocsc()
        M_ff = M[free][:, free].tocsc()
        n_free = K_ff.shape[0]
        # Diagonal Jacobi preconditioner — applies (diag K)^-1 to a vector
        diag_K = K_ff.diagonal()
        diag_inv = 1.0 / np.where(diag_K > 0, diag_K, 1.0)
        M_inv_op = sp.diags(diag_inv)
        # Initial guess: warm-start if provided, otherwise random.
        if X0_free is not None and X0_free.shape == (n_free, n_modes):
            X0 = X0_free.copy()
        else:
            rng = np.random.default_rng(0)
            X0 = rng.standard_normal((n_free, n_modes))
        try:
            eigvals, eigvecs_f = spla.lobpcg(
                K_ff, X0, B=M_ff, M=M_inv_op,
                tol=tol, largest=False, maxiter=maxiter,
            )
        except Exception:
            # Fallback to scipy eigsh shift-invert (slower but robust)
            eigvals, eigvecs_f = spla.eigsh(
                K_ff, k=n_modes, M=M_ff,
                sigma=1.0, which="LM", mode="normal",
                tol=1e-5, maxiter=2000,
            )
        # Sort ascending, clamp small negatives from numerical noise
        order = np.argsort(eigvals)
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs_f = eigvecs_f[:, order]
        frequencies_Hz = np.sqrt(eigvals) / (2.0 * np.pi)
        # Pad eigenvectors back to full-DoF length
        mode_shapes = np.zeros((self.geom.n_dofs, n_modes))
        mode_shapes[free, :] = eigvecs_f
        return frequencies_Hz, mode_shapes, eigvals

    def eigenvalue_sensitivity(self, rho_active: np.ndarray,
                                omega_sq_k: float,
                                v_k: np.ndarray,
                                tip_mass_kg: float = 0.060,
                                tip_mass_nodes: np.ndarray = None,
                                ) -> np.ndarray:
        """Sensitivity of the k-th eigenvalue (omega_k^2) to design densities.

        d(omega^2)/d(rho_e) = ( v^T (dK/drho_e) v
                                - omega^2 * v^T (dM/drho_e) v ) / (v^T M v)

        For SIMP:
            dK/drho_e = (1-e_min) p rho_e^(p-1) * Ke_solid    (only element e)
            dM/drho_e = m'(rho_e) * Me_solid                  (only element e)
        Tip mass is independent of rho and so drops out.

        Returns
        -------
        dlam_drho : (n_active,) — gradient at each active element
        """
        p = self.SIMP_PENALTY
        e_min = self.E_MIN_FRAC
        rho_safe = np.maximum(rho_active, 1e-9)
        factor_K_prime = (1.0 - e_min) * p * (rho_safe ** (p - 1.0))
        factor_M_prime = self._mass_simp_factor_grad(rho_active)

        # Per-element quadratic forms
        v_all = v_k[self.elem_dof_ids]                     # (n_active, 24)
        veKev = np.einsum("ij,jk,ik->i", v_all, self.Ke_solid, v_all)
        veMev = np.einsum("ij,jk,ik->i", v_all, self.Me_solid, v_all)

        # v^T M v including the tip-mass lump (rho-independent so still divides)
        if tip_mass_nodes is None:
            if hasattr(self.geom, "motor_mount_top_nodes"):
                tip_mass_nodes = self.geom.motor_mount_top_nodes()
            elif hasattr(self.geom, "_motor_top_boundary_nodes"):
                tip_mass_nodes = self.geom._motor_top_boundary_nodes()
            else:
                tip_mass_nodes = np.array([], dtype=np.int64)
        vMv_bulk = float(np.sum(self._mass_simp_factor(rho_active) * veMev))
        if tip_mass_kg > 0.0 and len(tip_mass_nodes) > 0:
            mass_per_node = tip_mass_kg / len(tip_mass_nodes)
            tip_dofs = np.concatenate([3 * tip_mass_nodes,
                                        3 * tip_mass_nodes + 1,
                                        3 * tip_mass_nodes + 2])
            vMv_tip = float(mass_per_node * np.sum(v_k[tip_dofs] ** 2))
        else:
            vMv_tip = 0.0
        vMv = vMv_bulk + vMv_tip
        if vMv <= 0.0:
            vMv = 1.0  # eigsh-normalised modes should give ~1; this is a guard

        dlam_drho = (factor_K_prime * veKev
                      - omega_sq_k * factor_M_prime * veMev) / vMv
        return dlam_drho

    # ------------------------------------------------------------------ #
    def apply_BCs_and_solve(self, K: sp.csr_matrix, F: np.ndarray,
                            method: str = "cg",
                            x0: np.ndarray = None,
                            rtol: float = 1e-5) -> np.ndarray:
        """Solve K_ff u_f = F_f then assemble full u with u_fixed = 0.

        method : "cg" (preconditioned conjugate gradient) or "lu" (direct).
        x0     : initial guess for CG (the full-length displacement vector;
                 the function extracts the free part). Used for warm-starting.
        """
        free = self.free_dofs
        K_ff = K[free][:, free].tocsc()
        F_f = F[free]
        if method == "cg":
            diag = K_ff.diagonal()
            M_inv = sp.diags(1.0 / np.where(diag != 0, diag, 1.0))
            x0_f = x0[free] if x0 is not None else None
            u_f, info = spla.cg(K_ff, F_f, M=M_inv, rtol=rtol, maxiter=5000,
                                 x0=x0_f)
            if info != 0:
                # Fall back to direct
                u_f = spla.spsolve(K_ff, F_f)
        else:
            u_f = spla.spsolve(K_ff, F_f)
        u = np.zeros(self.geom.n_dofs)
        u[free] = u_f
        return u

    # ------------------------------------------------------------------ #
    def solve(self, rho_active: np.ndarray, F: np.ndarray,
              method: str = "cg", x0: np.ndarray = None) -> np.ndarray:
        K = self.assemble_K(rho_active)
        return self.apply_BCs_and_solve(K, F, method=method, x0=x0)

    # ------------------------------------------------------------------ #
    def compute_stress(self, u: np.ndarray, rho_active: np.ndarray
                       ) -> np.ndarray:
        """Compute Voigt stress vector (6,) at each active element centre.
        Returns array shape (n_active, 6).

        Stress = E_factor * D_solid * (B @ u_e)
        We use linear rho scaling for stress evaluation (the standard SIMP
        practice for post-processing rather than objective).
        """
        # Vectorised: u_all shape (n_active, 24)
        u_all = u[self.elem_dof_ids]
        # eps_all = (B @ u^T)^T  =>  shape (n_active, 6)
        eps_all = u_all @ self.B_centre.T
        # sigma_all = D @ eps_all^T  =>  shape (n_active, 6)
        sigma_all = eps_all @ self.D_solid.T
        return rho_active[:, None] * sigma_all

    # ------------------------------------------------------------------ #
    def von_mises(self, stress: np.ndarray) -> np.ndarray:
        """Compute von Mises equivalent stress from Voigt stress array (..., 6)."""
        sxx, syy, szz = stress[..., 0], stress[..., 1], stress[..., 2]
        sxy, syz, sxz = stress[..., 3], stress[..., 4], stress[..., 5]
        vm2 = 0.5 * (
            (sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
        ) + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)
        return np.sqrt(np.maximum(vm2, 0.0))

    # ------------------------------------------------------------------ #
    def principal_stress_max(self, stress: np.ndarray) -> np.ndarray:
        """Maximum (most tensile) principal stress."""
        out = np.zeros(stress.shape[0])
        for k in range(stress.shape[0]):
            s = stress[k]
            T = np.array([[s[0], s[3], s[5]],
                          [s[3], s[1], s[4]],
                          [s[5], s[4], s[2]]])
            w = np.linalg.eigvalsh(T)
            out[k] = w.max()
        return out

    # ------------------------------------------------------------------ #
    def elem_compliance(self, u: np.ndarray, rho_active: np.ndarray = None
                        ) -> np.ndarray:
        """Return per-element solid-material strain energy u_e^T K_e_solid u_e
        (used for SIMP sensitivity).  rho_active is unused but kept for the
        interface (the multiplication by SIMP factor is done by the caller).
        """
        # Vectorised: extract u_e for all active elements, then quadratic form.
        u_all = u[self.elem_dof_ids]                # (n_active, 24)
        Ke = self.Ke_solid                          # (24, 24)
        # (u_all @ Ke) has shape (n_active, 24); element-wise product with u_all and sum over axis 1
        return np.einsum("ij,jk,ik->i", u_all, Ke, u_all)

    # ------------------------------------------------------------------ #
    def total_mass(self, rho_active: np.ndarray) -> float:
        """Total mass = sum(rho_e * V_e * rho_material), where V_e = dx*dy*dz."""
        V_e = self.geom.dx * self.geom.dy * self.geom.dz
        return float(np.sum(rho_active) * V_e * self.mat.rho)

    def total_solid_mass(self) -> float:
        """Mass if all design-domain elements are solid (rho=1)."""
        return self.total_mass(np.ones(self.n_active))


# -----------------------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------------------
def _cantilever_freq_with_tip_mass(L: float, b: float, h: float,
                                     E: float, rho: float,
                                     m_tip: float = 0.0) -> float:
    """Closed-form first natural frequency of a uniform clamped-free cantilever
    beam with optional concentrated tip mass.  Euler-Bernoulli assumption
    (valid only for slender beams; thick blocks deviate significantly).

    For pure beam (m_tip = 0):
        omega_1 = (1.8751/L)^2 · sqrt(E*I / (rho*A))
    With tip mass:
        omega_1 ≈ sqrt(k_tip / (m_tip + 0.235·m_beam))
        where k_tip = 3 E I / L^3, m_beam = rho·A·L.
    """
    I = b * h**3 / 12.0
    A = b * h
    if m_tip <= 0.0:
        return ((1.8751 / L) ** 2) * np.sqrt(E * I / (rho * A))
    k_tip = 3.0 * E * I / L**3
    m_beam = rho * A * L
    return np.sqrt(k_tip / (m_tip + 0.235 * m_beam))


if __name__ == "__main__":
    import time
    g = ArmGeometry()
    mat = Material()
    print(g.summary())

    t0 = time.time()
    fem = FEM3D(g, mat)
    t1 = time.time()
    print(f"FEM3D init: {t1 - t0:.2f} s")
    print(f"  active elements: {fem.n_active}")
    print(f"  free DOFs:       {len(fem.free_dofs)}")

    # Build a unit load in -Y on the motor mount and solve baseline (rho=1)
    F = np.zeros(g.n_dofs)
    dofs, mags = g.load_dofs_for_force((0.0, -5.88, 0.0))   # in-plane, -Y direction
    F[dofs] += mags
    print(f"Total applied force: ({F[0::3].sum():.3f}, {F[1::3].sum():.3f}, {F[2::3].sum():.3f}) N")

    t0 = time.time()
    rho_solid = np.ones(fem.n_active)
    K = fem.assemble_K(rho_solid)
    print(f"K assembled: nnz = {K.nnz}, shape = {K.shape}")
    t1 = time.time()
    print(f"  assembly: {t1 - t0:.2f} s")

    t0 = time.time()
    u = fem.apply_BCs_and_solve(K, F)
    t1 = time.time()
    print(f"  solve:    {t1 - t0:.2f} s")
    print(f"  max disp: {np.abs(u).max() * 1e6:.3f} um")

    t0 = time.time()
    stress = fem.compute_stress(u, rho_solid)
    vm = fem.von_mises(stress)
    t1 = time.time()
    print(f"  stress:   {t1 - t0:.2f} s, max von Mises = {vm.max()/1e6:.3f} MPa")
    print(f"  mean vM:  {vm.mean()/1e6:.4f} MPa")
    print(f"  mass (solid arm): {fem.total_solid_mass()*1e3:.3f} g")

    # ----------------------------------------------------------------- #
    # Modal smoke-test on the 3D bridge geometry — verifies M assembly,
    # eigsh shift-invert, and an Euler-Bernoulli cross-check.
    # ----------------------------------------------------------------- #
    print()
    print("=" * 60)
    print("Modal analysis smoke-test on ArmGeometry3D (the bridge)")
    print("=" * 60)
    from .mesh3d import ArmGeometry3D
    g3d = ArmGeometry3D()
    print(g3d.summary())
    fem3d = FEM3D(g3d, mat)
    rho_solid_3d = np.ones(fem3d.n_active)

    t0 = time.time()
    freqs, modes, lam = fem3d.modal_analysis(rho_solid_3d, n_modes=6,
                                              tip_mass_kg=0.060)
    t1 = time.time()
    print(f"  modal_analysis (n=6 modes): {t1 - t0:.2f} s")
    print(f"  baseline natural freqs (Hz): "
          f"{', '.join(f'{f:.1f}' for f in freqs)}")

    # Closed-form Euler-Bernoulli check (cantilever, bending in -z).
    # Effective cantilever length: bolt centroid at x ≈ 19 mm; motor at 200 mm
    L_eff = 0.200 - 0.019
    omega_eb = _cantilever_freq_with_tip_mass(
        L=L_eff, b=g3d.L_y, h=g3d.L_z,
        E=mat.E, rho=mat.rho, m_tip=0.060,
    )
    f_eb = omega_eb / (2.0 * np.pi)
    print(f"  EB cantilever closed-form (with tip mass): {f_eb:.1f} Hz")
    print(f"  Ratio FEM/EB for first mode: {freqs[0] / f_eb:.2f}")
    print(f"  (Note: ArmGeometry3D is stubby — h/L = {g3d.L_z/L_eff:.2f};")
    print(f"   EB underestimates because it ignores shear deformation.)")
