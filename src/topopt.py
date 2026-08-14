"""
SIMP-based topology optimization on the structured voxel mesh.

Features
--------
- Multi-load case compliance minimization with arbitrary load weights
- Density filter (cone weights) — enforces mesh-independence and a min member size
- Optimality criteria (OC) update with adaptive bisection on the Lagrange multiplier
- Keep-solid regions (around screw holes, motor mount) excluded from optimization
- Volume / mass-fraction constraint on the design domain only

References
----------
- Sigmund (2001) "A 99 line topology optimization code written in Matlab"
- Andreassen et al. (2011) "Efficient topology optimization in MATLAB using
  88 lines of code"
- Bendsoe & Sigmund "Topology Optimization" (2003)
"""
from __future__ import annotations

import time
import numpy as np
import scipy.ndimage as ndi

from .mesh import ArmGeometry
from .fem import FEM3D


# -----------------------------------------------------------------------------
# Cone filter kernel for the density / sensitivity filter
# -----------------------------------------------------------------------------
def cone_kernel(r_min_vox: float) -> np.ndarray:
    """3D cone-shaped weighting kernel:  w(d) = max(0, r_min - d)
    where d is the Euclidean distance from kernel centre, in voxel units.
    """
    r = int(np.ceil(r_min_vox))
    coords = np.arange(-r, r + 1)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    D = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    K = np.maximum(0.0, r_min_vox - D)
    return K


def filter_field(field_grid: np.ndarray, mask_grid: np.ndarray,
                  kernel: np.ndarray) -> np.ndarray:
    """Apply a density / sensitivity filter on a 3D grid, masked by `mask_grid`.

    field_grid : (nx, ny, nz) values on the active grid (zero outside)
    mask_grid  : (nx, ny, nz) boolean — only these cells contribute / are updated
    kernel     : (kx, ky, kz) filter weights (cone kernel)

    Returns the filtered field, same shape, with values outside the mask = 0.
    """
    num = ndi.convolve(field_grid * mask_grid, kernel, mode="constant", cval=0.0)
    den = ndi.convolve(mask_grid.astype(float), kernel, mode="constant", cval=0.0)
    out = np.zeros_like(field_grid)
    out[mask_grid] = num[mask_grid] / np.maximum(den[mask_grid], 1e-12)
    return out


# -----------------------------------------------------------------------------
# SIMP topology optimization solver
# -----------------------------------------------------------------------------
class SIMPOptimizer:
    """Run SIMP topology optimization on a multi-load problem."""

    def __init__(self, geom: ArmGeometry, fem: FEM3D,
                 load_cases: list,
                 vol_frac: float = 0.30,
                 r_min_mm: float = 2.0,
                 move: float = 0.2,
                 max_iter: int = 100,
                 tol_change: float = 0.01,
                 weights: list = None,
                 min_freq_Hz: float = None,
                 tip_mass_kg: float = 0.060,
                 n_modes_track: int = 4,
                 mu_freq_init: float = 0.10,
                 mu_freq_max: float = 50.0,
                 mu_freq_ramp_every: int = 5,
                 mu_freq_ramp_factor: float = 1.5):
        """
        load_cases : list of dicts, each having either:
                      - 'F'  = (Fx, Fy, Fz) Newtons at the motor mount, or
                      - 'Mz' = Mz in N·m at the motor mount.
                      A dict may include both 'F' and 'Mz' (forces and moment).
        vol_frac   : target mass / volume fraction on the design domain
        r_min_mm   : filter radius in mm (also enforces min member ≥ this size)
        move       : OC move limit per step
        max_iter   : maximum iterations
        tol_change : convergence threshold on max change of physical density per iter
        weights    : per-load-case weights (default uniform)
        min_freq_Hz: optional lower bound on first natural frequency [Hz].
                     If set, the optimizer enforces the bound via an
                     augmented-Lagrangian penalty whose multiplier ramps each
                     `mu_freq_ramp_every` iterations until satisfied.
                     None disables modal handling (v3 behaviour).
        tip_mass_kg: lumped motor+prop mass at the outboard mount (default 60 g
                     for a Sunnysky 2212 + 10×4.5 prop).
        n_modes_track : number of lowest modes to compute every iteration
                        (modal eigsolve cost is roughly linear in this).
        mu_freq_init, mu_freq_max, mu_freq_ramp_every, mu_freq_ramp_factor:
                     augmented-Lagrangian schedule for the frequency penalty.
        """
        self.geom = geom
        self.fem = fem
        self.load_cases = load_cases
        self.vol_frac = vol_frac
        self.r_min_vox = max(1.0, r_min_mm / (geom.dx * 1e3))  # convert mm to voxels
        self.move = move
        self.max_iter = max_iter
        self.tol_change = tol_change
        self.min_freq_Hz = min_freq_Hz
        self.tip_mass_kg = tip_mass_kg
        self.n_modes_track = max(1, int(n_modes_track))
        self.mu_freq = mu_freq_init
        self.mu_freq_max = mu_freq_max
        self.mu_freq_ramp_every = max(1, int(mu_freq_ramp_every))
        self.mu_freq_ramp_factor = mu_freq_ramp_factor

        if weights is None:
            weights = [1.0 / len(load_cases)] * len(load_cases)
        self.weights = np.asarray(weights, dtype=float)
        self.weights /= self.weights.sum()

        # Build masks & assignment between active-element index and 3D grid
        masks = geom.build_masks()
        self.in_arm = masks["in_arm"]
        self.design = masks["design"]
        self.keep_solid = masks["keep_solid"]

        # Vectors of grid indices for active elements
        self.active_idx = fem.elem_grid_idx     # (n_active, 3) array of (ex, ey, ez)

        # Which active elements are design vs. keep-solid
        ex_a = self.active_idx[:, 0]
        ey_a = self.active_idx[:, 1]
        ez_a = self.active_idx[:, 2]
        self.is_design_active = self.design[ex_a, ey_a, ez_a]
        self.n_design = int(self.is_design_active.sum())

        # Pre-build the filter kernel
        self.kernel = cone_kernel(self.r_min_vox)

        # Pre-compute the in-design mask on the full grid (used by filter)
        self.design_mask_grid = self.design.astype(bool)

        # Pre-build per-load-case force vectors (forces + optional moment)
        self.F_list = []
        for lc in load_cases:
            F = np.zeros(geom.n_dofs)
            if "F" in lc and lc["F"] is not None:
                dofs, mags = geom.load_dofs_for_force(lc["F"])
                F[dofs] += mags
            if "Mz" in lc and lc["Mz"] is not None:
                dofs_m, mags_m = geom.load_dofs_for_moment_Mz(lc["Mz"])
                F[dofs_m] += mags_m
            self.F_list.append(F)

        # Volume per element = dx*dy*dz
        self.V_e = geom.dx * geom.dy * geom.dz
        self.design_volume = self.n_design * self.V_e   # full solid design domain
        self.target_volume = vol_frac * self.design_volume

    # ------------------------------------------------------------------ #
    def _density_to_grid(self, rho_active: np.ndarray) -> np.ndarray:
        """Map active-element densities to a 3D grid for filtering."""
        g = np.zeros((self.geom.nx, self.geom.ny, self.geom.nz))
        ex, ey, ez = self.active_idx.T
        g[ex, ey, ez] = rho_active
        return g

    def _grid_to_active(self, g: np.ndarray) -> np.ndarray:
        ex, ey, ez = self.active_idx.T
        return g[ex, ey, ez]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_mode_by_mac(modes_new: np.ndarray, v_prev: np.ndarray) -> int:
        """Pick the column of `modes_new` most similar to `v_prev` by Modal
        Assurance Criterion MAC = (aᵀb)² / (‖a‖² ‖b‖²). Guards against
        bending↔torsion mode swaps as the topology evolves."""
        num = modes_new.T @ v_prev                               # (n_modes,)
        a2 = (modes_new ** 2).sum(axis=0)
        b2 = float((v_prev ** 2).sum())
        denom = np.maximum(a2 * b2, 1e-24)
        mac = (num ** 2) / denom
        return int(np.argmax(mac))

    def _oc_update(self, x_design: np.ndarray, dc_design: np.ndarray
                   ) -> np.ndarray:
        """Optimality criteria update with bisection on the Lagrange multiplier."""
        l1, l2 = 1e-9, 1e9
        x_new = np.empty_like(x_design)
        # Avoid sqrt of negative sensitivities by ensuring dc is negative
        # (compliance objective is always decreasing in density, so dc/dx < 0)
        # We use abs() to be safe, since the gradient should be negative.
        sens = np.abs(dc_design)
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lmid = 0.5 * (l1 + l2)
            B = sens / (lmid * self.V_e)
            # Damped OC update with move-limit
            tmp = x_design * np.sqrt(B)
            x_new = np.maximum(0.001,
                                np.maximum(x_design - self.move,
                                            np.minimum(1.0,
                                                        np.minimum(x_design + self.move,
                                                                    tmp))))
            v = x_new.sum() * self.V_e
            if v > self.target_volume:
                l1 = lmid
            else:
                l2 = lmid
        return x_new

    # ------------------------------------------------------------------ #
    def run(self, verbose: bool = True):
        """Run the topology optimization.

        Returns a dict with:
            'rho_active'        : final density per active element (n_active,)
            'compliance_hist'   : list of weighted compliance per iter
            'mass_hist'         : list of total mass (active solid + design) per iter
            'change_hist'       : max-change-of-density per iter
            'u_per_lc_final'    : list of full-DOF displacement vectors (one per LC)
        """
        geom, fem = self.geom, self.fem
        n_active = fem.n_active

        # Initialize density:  design cells start at vol_frac; keep_solid cells = 1.
        rho_active = np.where(self.is_design_active, self.vol_frac, 1.0)
        # rho on the 3D grid for filtering
        rho_grid = self._density_to_grid(rho_active)
        # Track previous physical density for change measurement
        rho_phys_grid = rho_grid.copy()

        u_prev = [None] * len(self.load_cases)
        history = {"compliance": [], "mass": [], "change": [], "vol_frac": [],
                    "omega_1_Hz": [], "mu_freq": [], "freq_violation": []}
        modal_active = self.min_freq_Hz is not None
        if modal_active:
            omega_target_sq = (2.0 * np.pi * self.min_freq_Hz) ** 2
            # Cached previous-iter mode shape for MAC-similarity mode tracking
            self._v_prev_mode1 = None
            # Cached previous-iter eigenvectors on free DoFs for LOBPCG warm-start
            self._X0_free_prev = None
            # Tip-mass attachment node list (geometry-resolved once)
            if hasattr(geom, "motor_mount_top_nodes"):
                self._tip_nodes = geom.motor_mount_top_nodes()
            elif hasattr(geom, "_motor_top_boundary_nodes"):
                self._tip_nodes = geom._motor_top_boundary_nodes()
            else:
                self._tip_nodes = np.array([], dtype=np.int64)
        else:
            omega_target_sq = None
            self._tip_nodes = np.array([], dtype=np.int64)
            self._X0_free_prev = None

        if verbose:
            extra = ""
            if modal_active:
                extra = (f", min ω₁ = {self.min_freq_Hz:.1f} Hz "
                         f"(tip mass {self.tip_mass_kg*1e3:.0f} g)")
            print(f"SIMP TO start — target Vf = {self.vol_frac:.3f}, "
                  f"r_min = {self.r_min_vox:.2f} vox ({self.r_min_vox * geom.dx * 1e3:.2f} mm){extra}")

        for it in range(1, self.max_iter + 1):
            t0 = time.time()
            # --- Apply density filter: design vars -> physical densities ---
            rho_phys_grid_new = filter_field(rho_grid, self.design_mask_grid, self.kernel)
            # Outside the design mask, keep_solid stays at 1, and so on; for
            # active-but-not-design cells we forcibly set rho = 1
            rho_phys_grid_new[~self.design_mask_grid] = 0.0
            # Composite physical density on full grid:
            rho_phys_grid_full = np.where(
                self.design, rho_phys_grid_new,
                np.where(self.in_arm, 1.0, 0.0)
            )

            # Density per active element
            rho_phys_active = self._grid_to_active(rho_phys_grid_full)

            # --- FEM solve for each load case ---
            sens_grid_total = np.zeros((geom.nx, geom.ny, geom.nz))
            c_total = 0.0
            u_list = []
            for lc_i, F in enumerate(self.F_list):
                u = fem.solve(rho_phys_active, F, method="cg", x0=u_prev[lc_i])
                u_list.append(u)
                u_prev[lc_i] = u
                # Element-wise strain energy for SIMP sensitivity
                ce = fem.elem_compliance(u, rho_phys_active)  # u_e^T K_e_solid u_e
                # Weighted compliance: w * sum_e (E_factor * ce) = w * u^T K u
                # But we want the actual compliance c = u^T K(rho) u for the obj
                # which is sum_e factor_e * ce, where factor_e = E_min + (1-E_min) * rho^p
                p = fem.SIMP_PENALTY
                emin = fem.E_MIN_FRAC
                factor = emin + (1.0 - emin) * (rho_phys_active ** p)
                c_lc = float((factor * ce).sum())
                c_total += self.weights[lc_i] * c_lc
                # Sensitivity wrt physical density: dc/drho = -p * (1-emin) * rho^(p-1) * ce
                # (negative -> filling material reduces compliance)
                dc_lc = -p * (1.0 - emin) * (rho_phys_active ** (p - 1)) * ce
                # Project sensitivity to grid for filtering
                sens_grid = self._density_to_grid(dc_lc)
                sens_grid_total += self.weights[lc_i] * sens_grid

            # --- Apply filter to the sensitivity (sensitivity filter via density chain rule)
            # We use the standard density-filter approach: filter the sensitivity by
            # convolving with the same cone weights.
            sens_filt_grid = filter_field(sens_grid_total, self.design_mask_grid, self.kernel)

            # --- Modal constraint: augmented-Lagrangian on first natural frequency
            if modal_active:
                freqs, modes, lam = fem.modal_analysis(
                    rho_phys_active,
                    n_modes=self.n_modes_track,
                    tip_mass_kg=self.tip_mass_kg,
                    tip_mass_nodes=self._tip_nodes,
                    X0_free=self._X0_free_prev,
                )
                # Cache eigenvectors on free DoFs for next iteration's warm-start
                self._X0_free_prev = modes[fem.free_dofs, :].copy()
                # Mode-tracking: identify the mode in the new set that's most
                # similar to last iter's first mode (guards against bending↔
                # torsion swaps as the topology evolves).
                if self._v_prev_mode1 is None:
                    track_idx = 0
                else:
                    track_idx = self._select_mode_by_mac(modes, self._v_prev_mode1)
                lam_1 = float(lam[track_idx])
                omega_1_Hz = float(freqs[track_idx])
                v_1 = modes[:, track_idx]
                self._v_prev_mode1 = v_1.copy()

                # Dimensionless violation in eigenvalue space.  Using
                # (1 − λ₁/λ_target) keeps it in [0, 1] and avoids the
                # 10⁶-scale issue of the bare squared-frequency residual.
                violation_rel = max(0.0, 1.0 - lam_1 / omega_target_sq)
                if violation_rel > 0.0:
                    dlam_drho = fem.eigenvalue_sensitivity(
                        rho_phys_active, lam_1, v_1,
                        tip_mass_kg=self.tip_mass_kg,
                        tip_mass_nodes=self._tip_nodes,
                    )
                    dlam_grid = self._density_to_grid(dlam_drho)
                    dlam_filt = filter_field(dlam_grid, self.design_mask_grid, self.kernel)
                    # Auto-scale: bring modal sens onto the same magnitude
                    # scale as the compliance sens, so mu_freq is a pure
                    # relative weight (mu_freq=1 means "equal weight").
                    s_c = float(np.mean(np.abs(
                        sens_filt_grid[self.design_mask_grid])))
                    s_m = float(np.mean(np.abs(
                        dlam_filt[self.design_mask_grid])))
                    scale = s_c / max(s_m, 1e-30)
                    # Penalty contribution: -mu * violation * (scaled dlam)
                    # Sign: dlam_drho is typically POSITIVE (more density →
                    # higher ω²), so subtracting drives sens MORE negative,
                    # which the OC update reads as "add material here".
                    penalty_grid = -self.mu_freq * violation_rel * scale * dlam_filt
                    sens_filt_grid = sens_filt_grid + penalty_grid

                history["omega_1_Hz"].append(omega_1_Hz)
                history["mu_freq"].append(self.mu_freq)
                history["freq_violation"].append(violation_rel)

                # Ramp mu every few iters if still violating
                if (it % self.mu_freq_ramp_every == 0 and violation_rel > 0.0
                        and self.mu_freq < self.mu_freq_max):
                    self.mu_freq = min(self.mu_freq_max,
                                        self.mu_freq * self.mu_freq_ramp_factor)
            else:
                history["omega_1_Hz"].append(np.nan)
                history["mu_freq"].append(np.nan)
                history["freq_violation"].append(np.nan)

            # --- OC update on design cells only ---
            x_design = rho_grid[self.design]
            dc_design = sens_filt_grid[self.design]
            x_design_new = self._oc_update(x_design, dc_design)
            # Write back to grid
            rho_grid_new = rho_grid.copy()
            rho_grid_new[self.design] = x_design_new
            # Compute change vs previous PHYSICAL density (post-filter)
            # Re-filter to get the new physical density for the change measure
            rho_phys_grid_check = filter_field(rho_grid_new, self.design_mask_grid, self.kernel)
            change = float(np.max(np.abs(rho_phys_grid_check - rho_phys_grid)))

            rho_grid = rho_grid_new
            rho_phys_grid = rho_phys_grid_check

            # Bookkeeping
            current_mass_design = x_design_new.sum() * self.V_e
            history["compliance"].append(c_total)
            history["mass"].append(current_mass_design)
            history["change"].append(change)
            history["vol_frac"].append(current_mass_design / self.design_volume)

            dt = time.time() - t0
            if verbose:
                extra = ""
                if modal_active and not np.isnan(history["omega_1_Hz"][-1]):
                    extra = (f"  ω₁={history['omega_1_Hz'][-1]:6.1f}Hz"
                             f"  μ={self.mu_freq:.2e}")
                print(f"  iter {it:3d}  c={c_total:.4e}  "
                      f"Vf={current_mass_design/self.design_volume:.3f}  "
                      f"change={change:.4f}  [{dt:.1f}s]{extra}")

            if change < self.tol_change and it > 10:
                if verbose:
                    print(f"  Converged at iter {it} (change < {self.tol_change})")
                break

        # Final assembly: rho_active is the physical density (post-filter)
        rho_phys_final_grid = filter_field(rho_grid, self.design_mask_grid, self.kernel)
        rho_phys_full = np.where(
            self.design, rho_phys_final_grid,
            np.where(self.in_arm, 1.0, 0.0)
        )
        rho_active_final = self._grid_to_active(rho_phys_full)

        return {
            "rho_active": rho_active_final,
            "rho_grid": rho_phys_full,
            "history": history,
            "u_per_lc": u_list,
        }


if __name__ == "__main__":
    from .fem import Material
    g = ArmGeometry()
    fem = FEM3D(g, Material())
    opt = SIMPOptimizer(
        g, fem,
        load_cases=[
            dict(F=(0.0, -2.94, 0.0)),     # hover (lateral component as in-plane proxy)
            dict(F=(0.0, -5.88, 0.0)),     # maneuver
            dict(F=(0.0, 14.7, 0.0)),      # hard landing (opposite direction)
        ],
        weights=[0.4, 0.4, 0.2],
        vol_frac=0.30,
        r_min_mm=2.0,
        max_iter=10,
    )
    res = opt.run()
    print("\nFinal compliance:", res["history"]["compliance"][-1])
    print("Final mass fraction:", res["history"]["vol_frac"][-1])
