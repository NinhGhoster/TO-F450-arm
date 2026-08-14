"""
Single-V_f runner for the 3D-bridge mesh, GPU-accelerated on the HPC system.

Designed to be invoked once per PBS array task — one V_f per H100 GPU.
The V_f value is taken from the `PBS_ARRAY_INDEX` environment variable
indexing into the global `VOL_FRACS` list at the top of this file, or
from `--vf` on the command line for local testing.

Each task runs baseline + TO + validation + fatigue for ONE mass
fraction.  Per-V_f outputs are written to `results_3d/vf_NN.pkl`.  A
final consolidation step (`sims/consolidate_3d.py`) collects all tasks
and writes the summary.json + figures.
"""
from __future__ import annotations

import os
# Allow BLAS to multi-thread freely — we are alone on the node
os.environ.pop("OMP_NUM_THREADS", None)
os.environ.pop("OPENBLAS_NUM_THREADS", None)
os.environ.pop("MKL_NUM_THREADS", None)
os.environ.pop("NUMEXPR_NUM_THREADS", None)

import argparse
import json
import pickle
import time
import sys
import numpy as np

from src.mesh3d import ArmGeometry3D
from src.fem import FEM3D, Material
from src.fem_gpu import enable_gpu_solver, gpu_available
from src.topopt import SIMPOptimizer
from src.fatigue import fatigue_life_field


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_3d")
os.makedirs(RESULTS_DIR, exist_ok=True)


VOL_FRACS = [0.05, 0.10, 0.15, 0.20]
NOMINAL_VF = 0.10
TO_MAX_ITER = 60

MATERIAL = Material(E=1700e6, nu=0.39, sigma_y=38e6, sigma_u=48e6, rho=930.0)

LOAD_CASES = [
    dict(name="LC1_hover",     F=(0.0, 0.0,  2.94),   Mz=None),
    dict(name="LC2_maneuver",  F=(0.0, 0.0,  5.88),   Mz=None),
    dict(name="LC3_landing",   F=(0.0, 0.0, -14.7),   Mz=None),
    dict(name="LC4_banked",    F=(0.0, 2.94, 5.0922), Mz=None),
    dict(name="LC5_proptorque", F=None, Mz=0.10),
]
LOAD_WEIGHTS = [0.30, 0.20, 0.10, 0.25, 0.15]


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


def run_one_vf(vf: float):
    overall_t0 = time.time()
    print("=" * 60, flush=True)
    print(f"V_f = {vf:.2f}  on host {os.uname().nodename}", flush=True)
    gpu = enable_gpu_solver()
    print(f"GPU solver: {'ENABLED' if gpu else 'disabled (CPU fallback)'}", flush=True)
    if gpu:
        import cupy as cp
        ndev = cp.cuda.runtime.getDeviceCount()
        for i in range(ndev):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
            print(f"  Visible GPU {i}: {name}", flush=True)
    print("=" * 60, flush=True)

    geom = ArmGeometry3D()
    print(geom.summary(), flush=True)
    fem = FEM3D(geom, MATERIAL)
    F_list = build_force_vectors(geom)

    # Run baseline once per V_f task (each task has its own GPU)
    print("--- Baseline static + fatigue ---", flush=True)
    t0 = time.time()
    rho_baseline = np.ones(fem.n_active)
    static_base = static_analysis(fem, F_list, rho_baseline)
    fat_base = fatigue_analysis(static_base)
    print(f"  baseline elapsed: {time.time()-t0:.1f}s", flush=True)
    for lc in LOAD_CASES:
        s = static_base[lc["name"]]
        print(f"  {lc['name']:15s}: max vM = {s['vm'].max()/1e6:.4f} MPa  "
              f"disp = {s['disp_node'].max()*1e6:7.2f} um  "
              f"SF = {s['sf_yield'].min():6.2f}", flush=True)

    # TO
    print(f"--- TO at V_f = {vf:.2f} ---", flush=True)
    t0 = time.time()
    opt = SIMPOptimizer(
        geom, fem,
        load_cases=LOAD_CASES,
        weights=LOAD_WEIGHTS,
        vol_frac=vf,
        r_min_mm=3.0,
        max_iter=TO_MAX_ITER,
        tol_change=0.005,
    )
    res = opt.run(verbose=True)
    print(f"  TO done in {(time.time()-t0)/60:.1f} min  "
          f"({len(res['history']['compliance'])} iters)", flush=True)

    rho_opt = res["rho_active"]
    print(f"--- Validation static + fatigue ---", flush=True)
    static_opt = static_analysis(fem, F_list, rho_opt)
    fat_opt = fatigue_analysis(static_opt)
    mass_g = fem.total_mass(rho_opt) * 1e3
    sig_per_lc = {lc["name"]: float(static_opt[lc["name"]]["vm"].max() / 1e6)
                  for lc in LOAD_CASES}
    disp_per_lc = {lc["name"]: float(static_opt[lc["name"]]["disp_node"].max() * 1e6)
                   for lc in LOAD_CASES}
    sf_y_per_lc = {lc["name"]: float(static_opt[lc["name"]]["sf_yield"].min())
                   for lc in LOAD_CASES}

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
        life_min=float(fat_opt["life"].min()),
        fs_fatigue=float(fat_opt["factor_of_safety"].min()),
        n_iter=int(len(res["history"]["compliance"])),
        wall_time_min=(time.time() - overall_t0) / 60,
        gpu_used=gpu,
    )
    print("=== RESULT ===", flush=True)
    for k, v in record.items():
        print(f"  {k}: {v}", flush=True)

    payload_path = os.path.join(RESULTS_DIR, f"vf_{int(vf*100):02d}.pkl")
    with open(payload_path, "wb") as f:
        pickle.dump(dict(rho=rho_opt, static=static_opt, fatigue=fat_opt,
                          history=res["history"], record=record), f)

    # Also save baseline (it's the same for every task, last writer wins)
    baseline_path = os.path.join(RESULTS_DIR, "raw_baseline.pkl")
    with open(baseline_path, "wb") as f:
        pickle.dump(dict(static=static_base, fatigue=fat_base,
                          material=MATERIAL,
                          mesh_info=dict(
                              nx=geom.nx, ny=geom.ny, nz=geom.nz,
                              n_elements=int(geom.n_elements),
                              n_active_elements=int(fem.n_active),
                              n_nodes=int(geom.n_nodes),
                              n_dofs=int(geom.n_dofs),
                          )), f)

    print(f"Saved: {payload_path}", flush=True)
    print(f"Total wall time: {(time.time()-overall_t0)/60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vf", type=float, default=None,
                    help="V_f value (e.g. 0.10). If not set, taken from PBS_ARRAY_INDEX.")
    args = p.parse_args()
    if args.vf is not None:
        vf = args.vf
    else:
        idx = int(os.environ.get("PBS_ARRAY_INDEX", "0"))
        vf = VOL_FRACS[idx]
    run_one_vf(vf)


if __name__ == "__main__":
    main()
