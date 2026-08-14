"""
Parallel pipeline runner — 3D bridge version (218×55×44 mm OEM bounding box).

Differs from `sims/run_parallel.py` only in: (a) the imported geometry class
(`ArmGeometry3D` instead of `ArmGeometry`), and (b) the V_f sweep, which is
shifted to lower values because the 3D bounding box is ~ 28× larger by
volume than the planar idealization so the baseline solid block is now
~ 500 g.
"""
from __future__ import annotations

import os
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

from src.mesh3d import ArmGeometry3D
from src.fem import FEM3D, Material
from src.topopt import SIMPOptimizer
from src.fatigue import fatigue_life_field
from src import plotting as P


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_3d")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures_3d")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


MATERIAL = Material(E=1700e6, nu=0.39, sigma_y=38e6, sigma_u=48e6, rho=930.0)

LOAD_CASES = [
    dict(name="LC1_hover",     F=(0.0, 0.0,  2.94),   Mz=None),
    dict(name="LC2_maneuver",  F=(0.0, 0.0,  5.88),   Mz=None),
    dict(name="LC3_landing",   F=(0.0, 0.0, -14.7),   Mz=None),
    dict(name="LC4_banked",    F=(0.0, 2.94, 5.0922), Mz=None),
    dict(name="LC5_proptorque", F=None, Mz=0.10),
]
LOAD_WEIGHTS = [0.30, 0.20, 0.10, 0.25, 0.15]

# Vf sweep shifted lower because the OEM bounding box is ~ 28× larger by
# volume than the planar idealization.  At V_f = 0.05 the optimised arm
# weighs ~ 25 g (close to OEM mass), at V_f = 0.20 ~ 100 g.
VOL_FRACS = [0.05, 0.10, 0.15, 0.20]
NOMINAL_VF = 0.10
TO_MAX_ITER = 60


def build_force_vectors(geom):
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


def static_analysis(fem, F_list, rho_active):
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


def worker_run_vf(vf_args):
    vf, max_iter = vf_args
    print(f"[worker pid={os.getpid()}] Starting Vf = {vf:.2f}", flush=True)
    t0 = time.time()

    geom = ArmGeometry3D()
    fem = FEM3D(geom, MATERIAL)
    F_list = build_force_vectors(geom)

    opt = SIMPOptimizer(
        geom, fem,
        load_cases=LOAD_CASES,
        weights=LOAD_WEIGHTS,
        vol_frac=vf,
        r_min_mm=3.0,                  # min member size 3 mm
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

    payload_path = os.path.join(RESULTS_DIR, f"vf_{int(vf*100):02d}.pkl")
    with open(payload_path, "wb") as f:
        pickle.dump(dict(rho=rho_opt, static=static_opt, fatigue=fat_opt,
                          history=res["history"], record=record), f)
    return record


def main():
    print("=" * 78, flush=True)
    print("F450 3D-bridge arm — parallel TO + fatigue (OEM bounding box)", flush=True)
    print("=" * 78, flush=True)
    overall_start = time.time()

    geom = ArmGeometry3D()
    print(geom.summary(), flush=True)
    fem = FEM3D(geom, MATERIAL)
    F_list = build_force_vectors(geom)
    for lc, F in zip(LOAD_CASES, F_list):
        Ftot = (F[0::3].sum(), F[1::3].sum(), F[2::3].sum())
        Mz_label = f", Mz={lc['Mz']:.3f} N.m" if lc.get('Mz') else ""
        print(f"  {lc['name']}: F=({Ftot[0]:.3f}, {Ftot[1]:.3f}, {Ftot[2]:.3f}) N{Mz_label}", flush=True)

    baseline_mesh_info = dict(
        nx=geom.nx, ny=geom.ny, nz=geom.nz,
        n_elements=int(geom.n_elements),
        n_active_elements=int(fem.n_active),
        n_nodes=int(geom.n_nodes), n_dofs=int(geom.n_dofs),
        bounding_box_mm=(geom.L_x * 1e3, geom.L_y * 1e3, geom.L_z * 1e3),
        voxel_size_mm=(geom.dx * 1e3, geom.dy * 1e3, geom.dz * 1e3),
    )

    print("\n--- Baseline static analysis ---", flush=True)
    rho_baseline = np.ones(fem.n_active)
    static_base = static_analysis(fem, F_list, rho_baseline)
    for lc in LOAD_CASES:
        s = static_base[lc["name"]]
        print(f"  {lc['name']:15s}: max vM = {s['vm'].max()/1e6:8.4f} MPa "
              f" disp = {s['disp_node'].max()*1e6:8.2f} um "
              f" min SF = {s['sf_yield'].min():6.2f}", flush=True)
    base_mass_g = fem.total_mass(rho_baseline) * 1e3
    print(f"  Baseline solid-block mass: {base_mass_g:.2f} g", flush=True)

    print("\n--- Baseline fatigue analysis ---", flush=True)
    fat_base = fatigue_analysis(static_base)
    print(f"  Baseline: min life = {fat_base['life'].min():.3e} cycles, "
          f"min FS = {fat_base['factor_of_safety'].min():.3f}, "
          f"max alt_eq = {fat_base['sigma_alt_eq'].max()/1e6:.3f} MPa", flush=True)

    print(f"\n--- Parallel TO across {len(VOL_FRACS)} V_f values ---", flush=True)
    t_parallel0 = time.time()
    ctx = get_context("fork") if hasattr(os, "fork") and os.name == "posix" else get_context("spawn")
    args = [(vf, TO_MAX_ITER) for vf in VOL_FRACS]
    with ctx.Pool(processes=len(VOL_FRACS)) as pool:
        pareto_records = pool.map(worker_run_vf, args)
    print(f"  Parallel TO done in {(time.time() - t_parallel0)/60:.1f} min", flush=True)

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
        nominal_Vf=NOMINAL_VF,
        runtime_min=(time.time() - overall_start) / 60.0,
    )
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    with open(os.path.join(RESULTS_DIR, "raw_baseline.pkl"), "wb") as f:
        pickle.dump(dict(static=static_base, fatigue=fat_base,
                          mesh_info=baseline_mesh_info, material=MATERIAL), f)

    print(f"\n*** Total runtime: {(time.time() - overall_start)/60:.1f} min ***", flush=True)
    print("\n=== SUMMARY ===", flush=True)
    print(f"Baseline solid-block mass:      {base_mass_g:.2f} g", flush=True)
    print(f"Baseline max vM (LC2 maneuver): {static_base['LC2_maneuver']['vm'].max()/1e6:.4f} MPa", flush=True)
    print(f"Baseline min life:             {fat_base['life'].min():.2e} cycles", flush=True)
    for r in sorted(pareto_records, key=lambda x: x["Vf"]):
        print(f"  Vf={r['Vf']:.2f}: m={r['mass_g']:7.2f} g, "
              f"σ_LC2={r['sigma_max_MPa']:7.3f}, σ_LC3={r['sigma_max_LC3_MPa']:7.3f}, "
              f"σ_LC4={r['sigma_max_LC4_MPa']:7.3f}, σ_LC5={r['sigma_max_LC5_MPa']:7.3f}, "
              f"fs_fat={r['fs_fatigue']:.2f}", flush=True)


if __name__ == "__main__":
    main()
