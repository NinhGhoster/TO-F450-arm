"""
Master run script: builds the F450 arm finite element model, runs baseline
and topology-optimized analyses across a mass-fraction sweep, predicts fatigue
life under the hover-maneuver cycle, and produces all figures and result
tables required for the manuscript.

Execution: from the project root,
    ./venv/bin/python -m sims.run_all
"""
from __future__ import annotations

import json
import os
import time
import pickle
import numpy as np

from src.mesh import ArmGeometry
from src.fem import FEM3D, Material
from src.topopt import SIMPOptimizer
from src.fatigue import fatigue_life_field, sn_alt_stress_at
from src import plotting as P


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# Material — PA12 SLS
MATERIAL = Material(
    E=1700e6, nu=0.39, sigma_y=38e6, sigma_u=48e6, rho=930.0,
)

# Load cases — F = (Fx, Fy, Fz) in Newtons applied at the motor mount;
# Mz = N·m moment about +Z (propeller torque reaction).  Each LC may
# include forces, a moment, or both.
LOAD_CASES = [
    dict(name="LC1_hover",     F=(0.0, 0.0,  2.94),  Mz=None),
    dict(name="LC2_maneuver",  F=(0.0, 0.0,  5.88),  Mz=None),
    dict(name="LC3_landing",   F=(0.0, 0.0, -14.7),  Mz=None),
    # NEW — 30° banked maneuver at T/W=2:  T sin30° lateral + T cos30° vertical
    dict(name="LC4_banked",    F=(0.0, 2.94, 5.0922), Mz=None),
    # NEW — propeller torque reaction at maneuver throttle: Mz = 0.10 N·m
    dict(name="LC5_proptorque", F=None, Mz=0.10),
]
# Weights for multi-load TO: hover dominates flight duty cycle; banked
# manoeuvre is a sustained state during cornering; hard landing is rare
# but impulsive; prop torque is continuous but small in magnitude.
LOAD_WEIGHTS = [0.30, 0.20, 0.10, 0.25, 0.15]

# Mass-fraction sweep
VOL_FRACS = [0.20, 0.30, 0.40, 0.50]
NOMINAL_VF = 0.30

# TO settings
TO_MAX_ITER = 60
TO_R_MIN_MM = 2.0


# ---------------------------------------------------------------------------
def build_force_vectors(geom: ArmGeometry):
    F_list = []
    for lc in LOAD_CASES:
        F = np.zeros(geom.n_dofs)
        if lc.get("F") is not None:
            dofs, mags = geom.load_dofs_for_force(lc["F"])
            F[dofs] += mags
        if lc.get("Mz") is not None:
            dofs_m, mags_m = geom.load_dofs_for_moment_Mz(lc["Mz"])
            F[dofs_m] += mags_m
        F_list.append(F)
    return F_list


def stats_field(field: np.ndarray) -> dict:
    """Return summary stats of a per-element field (e.g. vM stress)."""
    return dict(
        max=float(np.nanmax(field)),
        min=float(np.nanmin(field)),
        mean=float(np.nanmean(field)),
        p95=float(np.nanpercentile(field, 95)),
        p99=float(np.nanpercentile(field, 99)),
    )


def static_analysis(fem: FEM3D, F_list: list, rho_active: np.ndarray
                    ) -> dict:
    """Solve all load cases and return per-LC displacement and stress fields."""
    out = {}
    u_prev = None
    for lc, F in zip(LOAD_CASES, F_list):
        t0 = time.time()
        u = fem.solve(rho_active, F, method="cg", x0=u_prev)
        u_prev = u
        sigma = fem.compute_stress(u, rho_active)
        vm = fem.von_mises(sigma)
        smax = fem.principal_stress_max(sigma)
        # Total deformation magnitude per node
        u_node = u.reshape(-1, 3)
        d_node = np.linalg.norm(u_node, axis=1)
        # SF against yield
        sf_yield = MATERIAL.sigma_y / np.maximum(vm, 1e-3)
        dt = time.time() - t0
        out[lc["name"]] = dict(
            u=u, sigma=sigma, vm=vm, sigma_max=smax,
            disp_node=d_node, sf_yield=sf_yield, solve_time=dt,
        )
        print(f"  {lc['name']:15s}: max vM = {vm.max()/1e6:8.4f} MPa  "
              f"max disp = {d_node.max()*1e6:8.2f} um  "
              f"min SF = {sf_yield.min():6.2f}  [{dt:.1f}s]")
    return out


def fatigue_analysis(static_results: dict) -> dict:
    """Compute fatigue life under LC1 <-> LC2 cycle."""
    vm1 = static_results["LC1_hover"]["vm"]
    vm2 = static_results["LC2_maneuver"]["vm"]
    fat = fatigue_life_field(vm1, vm2, MATERIAL.sigma_u)
    return fat


def save_records(records: list, path: str):
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=float)


def field_to_csv(geom: ArmGeometry, fem: FEM3D, field: np.ndarray, name: str,
                 path: str):
    """Save per-active-element field with its grid coords."""
    arr = np.column_stack([
        fem.elem_grid_idx[:, 0], fem.elem_grid_idx[:, 1], fem.elem_grid_idx[:, 2],
        field
    ])
    np.savetxt(path, arr, fmt=("%d", "%d", "%d", "%.6e"),
                header=f"ex,ey,ez,{name}", delimiter=",", comments="")


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("F450 quadcopter arm — full TO + fatigue analysis pipeline")
    print("=" * 78)
    overall_start = time.time()

    # 1. Geometry & FEM
    geom = ArmGeometry()
    print("\n" + geom.summary())
    fem = FEM3D(geom, MATERIAL)
    F_list = build_force_vectors(geom)
    for lc, F in zip(LOAD_CASES, F_list):
        Ftot = (F[0::3].sum(), F[1::3].sum(), F[2::3].sum())
        Mz_label = f", Mz={lc['Mz']:.3f} N.m" if lc.get('Mz') else ""
        print(f"  {lc['name']}: F=({Ftot[0]:.3f}, {Ftot[1]:.3f}, {Ftot[2]:.3f}) N{Mz_label}")

    # Annotation figure of the geometry
    P.plot_arm_footprint(geom, os.path.join(FIGURES_DIR, "fig04_geometry_BC.png"))
    P.plot_sn_curve(os.path.join(FIGURES_DIR, "fig03_sn_curve.png"))

    # 2. Mesh convergence proxy (not full study, just report values at this mesh)
    baseline_mesh_info = dict(
        nx=geom.nx, ny=geom.ny, nz=geom.nz,
        n_elements=int(geom.n_elements),
        n_active_elements=int(fem.n_active),
        n_nodes=int(geom.n_nodes), n_dofs=int(geom.n_dofs),
    )

    # 3. Baseline static + fatigue
    print("\n--- Baseline static analysis ---")
    rho_baseline = np.ones(fem.n_active)
    static_base = static_analysis(fem, F_list, rho_baseline)
    base_mass_g = fem.total_mass(rho_baseline) * 1e3
    print(f"  Baseline arm mass: {base_mass_g:.3f} g")

    print("\n--- Baseline fatigue analysis ---")
    fat_base = fatigue_analysis(static_base)
    print(f"  Baseline: min life = {fat_base['life'].min():.3e} cycles, "
          f"min FS = {fat_base['factor_of_safety'].min():.3f}, "
          f"max alt_eq = {fat_base['sigma_alt_eq'].max()/1e6:.3f} MPa")

    # Save baseline figures
    P.plot_stress_topview(geom, fem, static_base["LC2_maneuver"]["vm"],
                            os.path.join(FIGURES_DIR, "fig05_baseline_vm_LC2.png"),
                            title="Baseline arm — von Mises (LC2 Maneuver, 5.88 N)")
    P.plot_life_topview(geom, fem, fat_base["life"],
                         os.path.join(FIGURES_DIR, "fig06_baseline_life.png"),
                         title="Baseline arm — predicted life (cycles, LC1↔LC2)")

    # 4. Topology optimisation sweep
    pareto_records = []
    optimized_data = {}
    for vf in VOL_FRACS:
        print(f"\n--- Topology optimization at Vf = {vf:.2f} ---")
        opt = SIMPOptimizer(
            geom, fem,
            load_cases=LOAD_CASES,
            weights=LOAD_WEIGHTS,
            vol_frac=vf,
            r_min_mm=TO_R_MIN_MM,
            max_iter=TO_MAX_ITER,
            tol_change=0.005,
        )
        t0 = time.time()
        res = opt.run(verbose=True)
        print(f"  TO done in {(time.time() - t0)/60:.1f} min "
              f"({len(res['history']['compliance'])} iters)")

        rho_opt = res["rho_active"]
        # Validate: re-run static at full LCs and fatigue
        print(f"  -- Validating optimized arm --")
        static_opt = static_analysis(fem, F_list, rho_opt)
        fat_opt = fatigue_analysis(static_opt)
        mass_g = fem.total_mass(rho_opt) * 1e3
        sig_LC2 = static_opt["LC2_maneuver"]["vm"].max() / 1e6
        sig_LC1 = static_opt["LC1_hover"]["vm"].max() / 1e6
        sig_LC3 = static_opt["LC3_landing"]["vm"].max() / 1e6
        life_min = float(fat_opt["life"].min())
        fs_fat = float(fat_opt["factor_of_safety"].min())

        sig_LC4 = static_opt["LC4_banked"]["vm"].max() / 1e6
        sig_LC5 = static_opt["LC5_proptorque"]["vm"].max() / 1e6
        record = dict(
            Vf=float(vf), mass_g=float(mass_g),
            sigma_max_LC1_MPa=float(sig_LC1),
            sigma_max_MPa=float(sig_LC2),  # LC2 still the primary single metric
            sigma_max_LC3_MPa=float(sig_LC3),
            sigma_max_LC4_MPa=float(sig_LC4),
            sigma_max_LC5_MPa=float(sig_LC5),
            life_min=float(life_min),
            fs_fatigue=float(fs_fat),
            n_iter=int(len(res["history"]["compliance"])),
        )
        pareto_records.append(record)
        optimized_data[vf] = dict(rho=rho_opt, static=static_opt, fatigue=fat_opt,
                                  history=res["history"])

        # Per-Vf figures
        P.plot_density_topview(
            geom, fem, rho_opt,
            os.path.join(FIGURES_DIR, f"fig08_topo_Vf{int(vf*100):02d}.png"),
            title=f"Optimized topology, $V_f$ = {vf:.2f}",
            midplane=True,
        )
        if vf == NOMINAL_VF:
            P.plot_density_layers(
                geom, fem, rho_opt,
                os.path.join(FIGURES_DIR, "fig08b_topo_Vf30_layers.png"),
                title=f"Optimized topology (Vf = {vf:.2f}) — per-layer view",
            )
            P.plot_to_convergence(
                res["history"],
                os.path.join(FIGURES_DIR, "fig07_to_convergence.png"),
                title=f"TO convergence ($V_f$ = {vf:.2f})",
            )
            P.plot_stress_topview(
                geom, fem, static_opt["LC2_maneuver"]["vm"],
                os.path.join(FIGURES_DIR, "fig09_optimized_vm_LC2.png"),
                title=f"Optimized arm — von Mises (LC2 Maneuver), Vf = {vf:.2f}",
                vmax=static_base["LC2_maneuver"]["vm"].max() / 1e6,
            )
            P.plot_life_topview(
                geom, fem, fat_opt["life"],
                os.path.join(FIGURES_DIR, "fig10_optimized_life.png"),
                title=f"Optimized arm — predicted life (cycles), Vf = {vf:.2f}",
            )

    # 5. Pareto figure & table
    P.plot_pareto(pareto_records, os.path.join(FIGURES_DIR, "fig11_pareto.png"))

    # 6. Orientation / anisotropy study at nominal Vf
    print(f"\n--- Build-orientation sensitivity at Vf = {NOMINAL_VF:.2f} ---")
    from src.fem import constitutive_D_orthotropic, unit_element_K
    rho_opt_nom = optimized_data[NOMINAL_VF]["rho"]
    orient_records = []

    # Iso baseline (reuse existing)
    rec0 = pareto_records[VOL_FRACS.index(NOMINAL_VF)]
    orient_records.append(dict(
        label="Isotropic\n(E=1700)",
        sigma_max_MPa=rec0["sigma_max_MPa"],
        life_min=rec0["life_min"],
        case="iso",
    ))

    # Reduced E in the build (z) direction
    E_reduced = 1500e6
    D_orth_z = constitutive_D_orthotropic(
        Ex=1700e6, Ey=1700e6, Ez=E_reduced,
        nu_xy=0.39, nu_yz=0.39, nu_xz=0.39,
        Gxy=1700e6/(2*(1+0.39)),
        Gyz=(1700e6 + E_reduced)/2/(2*(1+0.39)),
        Gxz=(1700e6 + E_reduced)/2/(2*(1+0.39)),
    )
    fem_z = FEM3D(geom, MATERIAL)
    fem_z.D_solid = D_orth_z
    fem_z.Ke_solid = unit_element_K(MATERIAL.E, MATERIAL.nu,
                                     geom.dx, geom.dy, geom.dz, D=D_orth_z)
    fem_z.B_centre = fem.B_centre
    fem_z.Ke_flat = fem_z.Ke_solid.ravel()
    static_z = static_analysis(fem_z, build_force_vectors(geom), rho_opt_nom)
    fat_z = fatigue_analysis(static_z)
    orient_records.append(dict(
        label="Vertical build\n(E_z=1500)",
        sigma_max_MPa=static_z["LC2_maneuver"]["vm"].max() / 1e6,
        life_min=float(fat_z["life"].min()),
        case="z_reduced",
    ))

    # Reduced E in the in-plane (xy) direction
    D_orth_xy = constitutive_D_orthotropic(
        Ex=E_reduced, Ey=E_reduced, Ez=1700e6,
        nu_xy=0.39, nu_yz=0.39, nu_xz=0.39,
        Gxy=E_reduced/(2*(1+0.39)),
        Gyz=(1700e6 + E_reduced)/2/(2*(1+0.39)),
        Gxz=(1700e6 + E_reduced)/2/(2*(1+0.39)),
    )
    fem_xy = FEM3D(geom, MATERIAL)
    fem_xy.D_solid = D_orth_xy
    fem_xy.Ke_solid = unit_element_K(MATERIAL.E, MATERIAL.nu,
                                       geom.dx, geom.dy, geom.dz, D=D_orth_xy)
    fem_xy.B_centre = fem.B_centre
    fem_xy.Ke_flat = fem_xy.Ke_solid.ravel()
    static_xy = static_analysis(fem_xy, build_force_vectors(geom), rho_opt_nom)
    fat_xy = fatigue_analysis(static_xy)
    orient_records.append(dict(
        label="In-plane build\n(E_xy=1500)",
        sigma_max_MPa=static_xy["LC2_maneuver"]["vm"].max() / 1e6,
        life_min=float(fat_xy["life"].min()),
        case="xy_reduced",
    ))

    P.plot_orientation_bar(orient_records, os.path.join(FIGURES_DIR, "fig12_orientation.png"))

    # 7. Persist all numerical results
    print("\n--- Saving outputs ---")
    summary = dict(
        mesh=baseline_mesh_info,
        material=dict(E=MATERIAL.E, nu=MATERIAL.nu,
                       sigma_y=MATERIAL.sigma_y,
                       sigma_u=MATERIAL.sigma_u,
                       rho=MATERIAL.rho),
        load_cases=[dict(name=lc["name"], F=lc["F"]) for lc in LOAD_CASES],
        load_weights=LOAD_WEIGHTS,
        baseline=dict(
            mass_g=base_mass_g,
            static={lc: dict(
                max_vM_MPa=float(static_base[lc]["vm"].max() / 1e6),
                mean_vM_MPa=float(static_base[lc]["vm"].mean() / 1e6),
                max_disp_um=float(static_base[lc]["disp_node"].max() * 1e6),
                max_sigma_max_MPa=float(static_base[lc]["sigma_max"].max() / 1e6),
                min_sf_yield=float(static_base[lc]["sf_yield"].min()),
            ) for lc in [c["name"] for c in LOAD_CASES]},
            fatigue=dict(
                min_life=float(fat_base["life"].min()),
                min_fs=float(fat_base["factor_of_safety"].min()),
                max_alt_eq_MPa=float(fat_base["sigma_alt_eq"].max() / 1e6),
            ),
        ),
        pareto=pareto_records,
        orientation=orient_records,
        nominal_Vf=NOMINAL_VF,
        runtime_min=(time.time() - overall_start) / 60.0,
    )
    save_records(summary, os.path.join(RESULTS_DIR, "summary.json"))
    # Save dense data as pickle for follow-up plotting
    with open(os.path.join(RESULTS_DIR, "raw_data.pkl"), "wb") as f:
        pickle.dump(dict(
            baseline=dict(static=static_base, fatigue=fat_base),
            optimized=optimized_data,
            orientation=dict(z=dict(static=static_z, fatigue=fat_z),
                              xy=dict(static=static_xy, fatigue=fat_xy)),
            mesh_info=baseline_mesh_info,
            material=MATERIAL,
        ), f)

    print(f"\n*** Total runtime: {(time.time() - overall_start)/60:.1f} min ***")
    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"Figures saved to {FIGURES_DIR}")
    print("\n=== SUMMARY ===")
    print(f"Baseline mass:      {base_mass_g:.3f} g")
    print(f"Baseline max vM (LC2): {static_base['LC2_maneuver']['vm'].max()/1e6:.3f} MPa")
    print(f"Baseline min life: {fat_base['life'].min():.2e} cycles")
    for r in pareto_records:
        print(f"  Vf = {r['Vf']:.2f}:  mass = {r['mass_g']:6.2f} g  "
              f"max vM (LC2) = {r['sigma_max_MPa']:7.3f} MPa  "
              f"min life = {r['life_min']:.2e}  "
              f"fs_fatigue = {r['fs_fatigue']:5.2f}")


if __name__ == "__main__":
    main()
