"""
the HPC system-friendly parallel version of run_all.py.

Stages:
  1. Build geometry and FEM once, serially.
  2. Run baseline static + fatigue analysis (serial, ~40 s).
  3. Fork 4 worker processes, each running TO + validation for one Vf
     in {0.20, 0.30, 0.40, 0.50}.  Each worker reuses the shared geometry
     code and builds its own FEM/SIMP objects.
  4. After workers finish, run orientation study on the V_f = 0.30 result
     (serial, ~70 s).
  5. Consolidate results into summary.json and figures.

Threading:
  Each Python worker uses a single OpenBLAS / MKL thread (set via env var
  before importing numpy).  This prevents the 4 workers from oversubscribing
  the same physical cores during sparse linear-algebra operations.

Usage:
  ./venv/bin/python -u -m sims.run_parallel
"""
from __future__ import annotations

import os
# IMPORTANT: set these before importing numpy/scipy.  Each worker uses 1 thread.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import pickle
import time
import multiprocessing as mp
from multiprocessing import get_context
import numpy as np

from src.mesh import ArmGeometry
from src.fem import FEM3D, Material
from src.topopt import SIMPOptimizer
from src.fatigue import fatigue_life_field
from src import plotting as P


# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


MATERIAL = Material(E=1700e6, nu=0.39, sigma_y=38e6, sigma_u=48e6, rho=930.0)

LOAD_CASES = [
    dict(name="LC1_hover",     F=(0.0, 0.0,  2.94),  Mz=None),
    dict(name="LC2_maneuver",  F=(0.0, 0.0,  5.88),  Mz=None),
    dict(name="LC3_landing",   F=(0.0, 0.0, -14.7),  Mz=None),
    dict(name="LC4_banked",    F=(0.0, 2.94, 5.0922), Mz=None),
    dict(name="LC5_proptorque", F=None, Mz=0.10),
]
LOAD_WEIGHTS = [0.30, 0.20, 0.10, 0.25, 0.15]
VOL_FRACS = [0.20, 0.30, 0.40, 0.50]
NOMINAL_VF = 0.30
TO_MAX_ITER = 60


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


def static_analysis(fem: FEM3D, F_list, rho_active):
    out = {}
    u_prev = None
    for lc, F in zip(LOAD_CASES, F_list):
        t0 = time.time()
        u = fem.solve(rho_active, F, method="cg", x0=u_prev)
        u_prev = u
        sigma = fem.compute_stress(u, rho_active)
        vm = fem.von_mises(sigma)
        smax = fem.principal_stress_max(sigma)
        u_node = u.reshape(-1, 3)
        d_node = np.linalg.norm(u_node, axis=1)
        sf_yield = MATERIAL.sigma_y / np.maximum(vm, 1e-3)
        dt = time.time() - t0
        out[lc["name"]] = dict(u=u, sigma=sigma, vm=vm, sigma_max=smax,
                                disp_node=d_node, sf_yield=sf_yield,
                                solve_time=dt)
    return out


def fatigue_analysis(static_results):
    vm1 = static_results["LC1_hover"]["vm"]
    vm2 = static_results["LC2_maneuver"]["vm"]
    return fatigue_life_field(vm1, vm2, MATERIAL.sigma_u)


# ---------------------------------------------------------------------------
def worker_run_vf(vf_args):
    """Worker entry point. Each process re-builds its own geometry + FEM
    + SIMP optimizer to avoid pickling large objects across processes."""
    vf, max_iter = vf_args
    print(f"[worker pid={os.getpid()}] Starting Vf = {vf:.2f}", flush=True)
    t0 = time.time()

    geom = ArmGeometry()
    fem = FEM3D(geom, MATERIAL)
    F_list = build_force_vectors(geom)

    opt = SIMPOptimizer(
        geom, fem,
        load_cases=LOAD_CASES,
        weights=LOAD_WEIGHTS,
        vol_frac=vf,
        r_min_mm=2.0,
        max_iter=max_iter,
        tol_change=0.005,
    )
    res = opt.run(verbose=False)
    print(f"[worker pid={os.getpid()}] Vf={vf:.2f} TO done in {(time.time()-t0)/60:.1f} min "
          f"({len(res['history']['compliance'])} iters)", flush=True)

    rho_opt = res["rho_active"]
    static_opt = static_analysis(fem, F_list, rho_opt)
    fat_opt = fatigue_analysis(static_opt)

    mass_g = fem.total_mass(rho_opt) * 1e3
    sig_per_lc = {lc["name"]: float(static_opt[lc["name"]]["vm"].max() / 1e6)
                  for lc in LOAD_CASES}
    disp_per_lc = {lc["name"]: float(static_opt[lc["name"]]["disp_node"].max() * 1e6)
                   for lc in LOAD_CASES}
    sf_y_per_lc = {lc["name"]: float(static_opt[lc["name"]]["sf_yield"].min())
                   for lc in LOAD_CASES}
    life_min = float(fat_opt["life"].min())
    fs_fat = float(fat_opt["factor_of_safety"].min())

    record = dict(
        Vf=float(vf), mass_g=float(mass_g),
        sigma_max_LC1_MPa=sig_per_lc["LC1_hover"],
        sigma_max_MPa=sig_per_lc["LC2_maneuver"],
        sigma_max_LC3_MPa=sig_per_lc["LC3_landing"],
        sigma_max_LC4_MPa=sig_per_lc["LC4_banked"],
        sigma_max_LC5_MPa=sig_per_lc["LC5_proptorque"],
        disp_LC2_um=disp_per_lc["LC2_maneuver"],
        disp_LC3_um=disp_per_lc["LC3_landing"],
        sf_yield_LC2=sf_y_per_lc["LC2_maneuver"],
        sf_yield_LC3=sf_y_per_lc["LC3_landing"],
        life_min=life_min,
        fs_fatigue=fs_fat,
        n_iter=int(len(res["history"]["compliance"])),
        wall_time_min=(time.time() - t0) / 60,
    )
    print(f"[worker pid={os.getpid()}] Vf={vf:.2f} record: mass={mass_g:.2f} g, "
          f"σ_LC2={sig_per_lc['LC2_maneuver']:.3f} MPa, σ_LC3={sig_per_lc['LC3_landing']:.3f} MPa, "
          f"fs_fat={fs_fat:.3f}", flush=True)

    # Save the heavy data per Vf
    payload_path = os.path.join(RESULTS_DIR, f"vf_{int(vf*100):02d}.pkl")
    with open(payload_path, "wb") as f:
        pickle.dump(dict(rho=rho_opt, static=static_opt, fatigue=fat_opt,
                          history=res["history"], record=record), f)
    return record


# ---------------------------------------------------------------------------
def main():
    print("=" * 78, flush=True)
    print("F450 quadcopter arm — parallel TO + fatigue (the HPC system-ready)", flush=True)
    print("=" * 78, flush=True)
    overall_start = time.time()

    # 1. Build geometry & FEM (serial)
    geom = ArmGeometry()
    print(geom.summary(), flush=True)
    fem = FEM3D(geom, MATERIAL)
    F_list = build_force_vectors(geom)
    for lc, F in zip(LOAD_CASES, F_list):
        Ftot = (F[0::3].sum(), F[1::3].sum(), F[2::3].sum())
        Mz_label = f", Mz={lc['Mz']:.3f} N.m" if lc.get('Mz') else ""
        print(f"  {lc['name']}: F=({Ftot[0]:.3f}, {Ftot[1]:.3f}, {Ftot[2]:.3f}) N{Mz_label}", flush=True)

    P.plot_arm_footprint(geom, os.path.join(FIGURES_DIR, "fig04_geometry_BC.png"))
    P.plot_sn_curve(os.path.join(FIGURES_DIR, "fig03_sn_curve.png"))

    baseline_mesh_info = dict(
        nx=geom.nx, ny=geom.ny, nz=geom.nz,
        n_elements=int(geom.n_elements),
        n_active_elements=int(fem.n_active),
        n_nodes=int(geom.n_nodes), n_dofs=int(geom.n_dofs),
    )

    # 2. Baseline static + fatigue (serial)
    print("\n--- Baseline static analysis ---", flush=True)
    rho_baseline = np.ones(fem.n_active)
    static_base = static_analysis(fem, F_list, rho_baseline)
    for lc in LOAD_CASES:
        s = static_base[lc["name"]]
        print(f"  {lc['name']:15s}: max vM = {s['vm'].max()/1e6:8.4f} MPa "
              f" disp = {s['disp_node'].max()*1e6:8.2f} um "
              f" min SF = {s['sf_yield'].min():6.2f}",
              flush=True)
    base_mass_g = fem.total_mass(rho_baseline) * 1e3
    print(f"  Baseline mass: {base_mass_g:.3f} g", flush=True)

    print("\n--- Baseline fatigue analysis ---", flush=True)
    fat_base = fatigue_analysis(static_base)
    print(f"  Baseline: min life = {fat_base['life'].min():.3e} cycles, "
          f"min FS = {fat_base['factor_of_safety'].min():.3f}, "
          f"max alt_eq = {fat_base['sigma_alt_eq'].max()/1e6:.3f} MPa", flush=True)

    # Save baseline figures
    P.plot_stress_topview(geom, fem, static_base["LC2_maneuver"]["vm"],
                            os.path.join(FIGURES_DIR, "fig05_baseline_vm_LC2.png"),
                            title="Baseline arm — von Mises (LC2 Maneuver, 5.88 N)")
    P.plot_life_topview(geom, fem, fat_base["life"],
                         os.path.join(FIGURES_DIR, "fig06_baseline_life.png"),
                         title="Baseline arm — predicted life (cycles, LC1↔LC2)")

    # 3. Parallel TO across all Vf
    print(f"\n--- Parallel TO across {len(VOL_FRACS)} Vf values ---", flush=True)
    print(f"  Using {min(len(VOL_FRACS), mp.cpu_count())} workers, "
          f"each pinned to a single BLAS thread", flush=True)
    t_parallel0 = time.time()
    # Use 'fork' so children inherit imported modules (faster startup);
    # falls back to 'spawn' on macOS/Windows where fork is unsafe.
    ctx = get_context("fork") if hasattr(os, "fork") and os.name == "posix" else get_context("spawn")
    args = [(vf, TO_MAX_ITER) for vf in VOL_FRACS]
    with ctx.Pool(processes=len(VOL_FRACS)) as pool:
        pareto_records = pool.map(worker_run_vf, args)
    print(f"  Parallel TO done in {(time.time() - t_parallel0)/60:.1f} min", flush=True)

    # Reload Vf=0.30 result for downstream plotting and orientation
    nominal_path = os.path.join(RESULTS_DIR, f"vf_{int(NOMINAL_VF*100):02d}.pkl")
    with open(nominal_path, "rb") as f:
        nominal_data = pickle.load(f)
    rho_opt_nom = nominal_data["rho"]

    # Per-Vf density figures
    for vf in VOL_FRACS:
        path = os.path.join(RESULTS_DIR, f"vf_{int(vf*100):02d}.pkl")
        with open(path, "rb") as fp:
            d = pickle.load(fp)
        P.plot_density_topview(
            geom, fem, d["rho"],
            os.path.join(FIGURES_DIR, f"fig08_topo_Vf{int(vf*100):02d}.png"),
            title=f"Optimized topology, $V_f$ = {vf:.2f}",
            midplane=True,
        )
        if vf == NOMINAL_VF:
            P.plot_density_layers(
                geom, fem, d["rho"],
                os.path.join(FIGURES_DIR, "fig08b_topo_Vf30_layers.png"),
                title=f"Optimized topology (Vf = {vf:.2f}) — per-layer view",
            )
            P.plot_to_convergence(
                d["history"],
                os.path.join(FIGURES_DIR, "fig07_to_convergence.png"),
                title=f"TO convergence ($V_f$ = {vf:.2f})",
            )
            P.plot_stress_topview(
                geom, fem, d["static"]["LC2_maneuver"]["vm"],
                os.path.join(FIGURES_DIR, "fig09_optimized_vm_LC2.png"),
                title=f"Optimized arm — von Mises (LC2 Maneuver), Vf = {vf:.2f}",
                vmax=static_base["LC2_maneuver"]["vm"].max() / 1e6,
            )
            P.plot_life_topview(
                geom, fem, d["fatigue"]["life"],
                os.path.join(FIGURES_DIR, "fig10_optimized_life.png"),
                title=f"Optimized arm — predicted life (cycles), Vf = {vf:.2f}",
            )

    # 4. Orientation / anisotropy study at nominal Vf
    print(f"\n--- Build-orientation sensitivity at Vf = {NOMINAL_VF:.2f} ---", flush=True)
    from src.fem import constitutive_D_orthotropic, unit_element_K
    orient_records = []
    rec0 = [r for r in pareto_records if r["Vf"] == NOMINAL_VF][0]
    orient_records.append(dict(
        label="Isotropic\n(E=1700)",
        sigma_max_MPa=rec0["sigma_max_MPa"],
        life_min=rec0["life_min"],
        case="iso",
    ))
    E_reduced = 1500e6
    nu = MATERIAL.nu
    # Vertical-build worst case (reduced E_z)
    D_orth_z = constitutive_D_orthotropic(
        Ex=1700e6, Ey=1700e6, Ez=E_reduced,
        nu_xy=nu, nu_yz=nu, nu_xz=nu,
        Gxy=1700e6/(2*(1+nu)),
        Gyz=(1700e6 + E_reduced)/2/(2*(1+nu)),
        Gxz=(1700e6 + E_reduced)/2/(2*(1+nu)),
    )
    fem_z = FEM3D(geom, MATERIAL)
    fem_z.D_solid = D_orth_z
    fem_z.Ke_solid = unit_element_K(MATERIAL.E, MATERIAL.nu,
                                     geom.dx, geom.dy, geom.dz, D=D_orth_z)
    fem_z.Ke_flat = fem_z.Ke_solid.ravel()
    static_z = static_analysis(fem_z, F_list, rho_opt_nom)
    fat_z = fatigue_analysis(static_z)
    orient_records.append(dict(
        label="Vertical build\n(E_z=1500)",
        sigma_max_MPa=static_z["LC2_maneuver"]["vm"].max() / 1e6,
        life_min=float(fat_z["life"].min()),
        case="z_reduced",
    ))
    # In-plane reduced
    D_orth_xy = constitutive_D_orthotropic(
        Ex=E_reduced, Ey=E_reduced, Ez=1700e6,
        nu_xy=nu, nu_yz=nu, nu_xz=nu,
        Gxy=E_reduced/(2*(1+nu)),
        Gyz=(1700e6 + E_reduced)/2/(2*(1+nu)),
        Gxz=(1700e6 + E_reduced)/2/(2*(1+nu)),
    )
    fem_xy = FEM3D(geom, MATERIAL)
    fem_xy.D_solid = D_orth_xy
    fem_xy.Ke_solid = unit_element_K(MATERIAL.E, MATERIAL.nu,
                                       geom.dx, geom.dy, geom.dz, D=D_orth_xy)
    fem_xy.Ke_flat = fem_xy.Ke_solid.ravel()
    static_xy = static_analysis(fem_xy, F_list, rho_opt_nom)
    fat_xy = fatigue_analysis(static_xy)
    orient_records.append(dict(
        label="In-plane build\n(E_xy=1500)",
        sigma_max_MPa=static_xy["LC2_maneuver"]["vm"].max() / 1e6,
        life_min=float(fat_xy["life"].min()),
        case="xy_reduced",
    ))
    P.plot_orientation_bar(orient_records, os.path.join(FIGURES_DIR, "fig12_orientation.png"))

    # 5. Pareto figure (informative version)
    sigma_y_MPa = MATERIAL.sigma_y / 1e6
    Vf_arr = np.array([r["Vf"] for r in pareto_records])
    mass_arr = np.array([r["mass_g"] for r in pareto_records])
    sig_LC2 = np.array([r["sigma_max_MPa"] for r in pareto_records])
    sig_LC3 = np.array([r["sigma_max_LC3_MPa"] for r in pareto_records])
    fs_fat = np.array([r["fs_fatigue"] for r in pareto_records])
    base_LC2 = static_base["LC2_maneuver"]["vm"].max() / 1e6
    base_LC3 = static_base["LC3_landing"]["vm"].max() / 1e6
    base_fs_fat = float(fat_base["factor_of_safety"].min())

    all_mass = np.concatenate([[base_mass_g], mass_arr])
    all_LC2 = np.concatenate([[base_LC2], sig_LC2])
    all_LC3 = np.concatenate([[base_LC3], sig_LC3])
    all_fs_fat = np.concatenate([[base_fs_fat], fs_fat])
    all_labels = ["Baseline"] + [f"Vf={v:.2f}" for v in Vf_arr]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    ax = axes[0]
    ax.plot(all_mass, all_LC2, "o-", color="C0", label="LC2 (Maneuver)")
    ax.plot(all_mass, all_LC3, "s-", color="C3", label="LC3 (Hard landing)")
    ax.axhline(sigma_y_MPa, color="grey", ls="--", lw=1,
                label=fr"$\sigma_y$ = {sigma_y_MPa:.0f} MPa")
    for i, lab in enumerate(all_labels):
        ax.annotate(lab, (all_mass[i], all_LC3[i]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Arm mass (g)"); ax.set_ylabel("Peak von Mises stress (MPa)")
    ax.set_title("Mass vs. peak stress")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=7, loc="upper right")
    ax = axes[1]
    ax.plot(all_mass, sigma_y_MPa/all_LC2, "o-", color="C0", label="SF yield (LC2)")
    ax.plot(all_mass, sigma_y_MPa/all_LC3, "s-", color="C3", label="SF yield (LC3)")
    ax.plot(all_mass, all_fs_fat, "^-", color="C2", label="SF fatigue (LC1↔LC2)")
    ax.axhline(1.0, color="black", ls="--", lw=1, label="SF = 1")
    ax.axhline(1.5, color="grey", ls=":", lw=1, label="SF = 1.5 (target)")
    for i, lab in enumerate(all_labels):
        ax.annotate(lab, (all_mass[i], all_fs_fat[i]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Arm mass (g)"); ax.set_ylabel("Factor of safety")
    ax.set_title("Mass vs. factor of safety")
    ax.set_yscale("log"); ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=7, loc="upper left", ncol=1)
    ax = axes[2]
    x = np.arange(len(pareto_records))
    width = 0.30
    ax.bar(x - width, mass_arr, width, color="C0", label="Mass (g)")
    ax.set_xticks(x); ax.set_xticklabels([f"Vf={v:.2f}" for v in Vf_arr])
    ax.set_ylabel("Mass (g)", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.axhline(base_mass_g, color="C0", ls=":", lw=1,
                label=f"Baseline {base_mass_g:.1f} g")
    ax.legend(loc="upper left", fontsize=7)
    ax2 = ax.twinx()
    ax2.bar(x, fs_fat, width, color="C2", alpha=0.7, label="SF fatigue")
    ax2.bar(x + width, sigma_y_MPa/sig_LC3, width, color="C3", alpha=0.7,
             label="SF yield (LC3)")
    ax2.set_ylabel("Safety factor")
    ax2.axhline(1.5, color="grey", ls=":", lw=1)
    ax2.legend(loc="upper right", fontsize=7)
    ax.set_title("Optimized configurations")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig11_pareto.png"))
    plt.close(fig)

    # 6. Save all numerical results
    print("\n--- Saving outputs ---", flush=True)
    summary = dict(
        mesh=baseline_mesh_info,
        material=dict(E=MATERIAL.E, nu=MATERIAL.nu,
                       sigma_y=MATERIAL.sigma_y,
                       sigma_u=MATERIAL.sigma_u,
                       rho=MATERIAL.rho),
        load_cases=[dict(name=lc["name"], F=lc.get("F"), Mz=lc.get("Mz"))
                     for lc in LOAD_CASES],
        load_weights=LOAD_WEIGHTS,
        baseline=dict(
            mass_g=base_mass_g,
            static={lc["name"]: dict(
                max_vM_MPa=float(static_base[lc["name"]]["vm"].max() / 1e6),
                mean_vM_MPa=float(static_base[lc["name"]]["vm"].mean() / 1e6),
                max_disp_um=float(static_base[lc["name"]]["disp_node"].max() * 1e6),
                max_sigma_max_MPa=float(static_base[lc["name"]]["sigma_max"].max() / 1e6),
                min_sf_yield=float(static_base[lc["name"]]["sf_yield"].min()),
            ) for lc in LOAD_CASES},
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
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    with open(os.path.join(RESULTS_DIR, "raw_baseline.pkl"), "wb") as f:
        pickle.dump(dict(static=static_base, fatigue=fat_base, mesh_info=baseline_mesh_info,
                          material=MATERIAL), f)
    with open(os.path.join(RESULTS_DIR, "raw_orientation.pkl"), "wb") as f:
        pickle.dump(dict(z=dict(static=static_z, fatigue=fat_z),
                          xy=dict(static=static_xy, fatigue=fat_xy)), f)

    print(f"\n*** Total runtime: {(time.time() - overall_start)/60:.1f} min ***", flush=True)
    print(f"Results saved to {RESULTS_DIR}", flush=True)
    print(f"Figures saved to {FIGURES_DIR}", flush=True)
    print("\n=== SUMMARY ===", flush=True)
    print(f"Baseline mass:      {base_mass_g:.3f} g", flush=True)
    print(f"Baseline max vM (LC2): {static_base['LC2_maneuver']['vm'].max()/1e6:.3f} MPa", flush=True)
    print(f"Baseline min life: {fat_base['life'].min():.2e} cycles", flush=True)
    for r in sorted(pareto_records, key=lambda x: x["Vf"]):
        print(f"  Vf={r['Vf']:.2f}: m={r['mass_g']:6.2f} g, "
              f"σ_LC2={r['sigma_max_MPa']:7.3f}, σ_LC3={r['sigma_max_LC3_MPa']:7.3f}, "
              f"σ_LC4={r['sigma_max_LC4_MPa']:7.3f}, σ_LC5={r['sigma_max_LC5_MPa']:7.3f}, "
              f"fs_fat={r['fs_fatigue']:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
