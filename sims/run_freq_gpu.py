"""
Single-(V_f, omega_target) runner for the 3D-bridge mesh, GPU-accelerated.

This is the v4 production runner.  Unlike `run_vf_gpu.py` (v3), the
optimisation includes an active first-natural-frequency constraint
enforced via an augmented-Lagrangian penalty.  The headline contribution
is therefore "what is the lowest-mass design that retains ω₁ above the
rotor blade-passage band?", not "how light can we go without yielding?".

One PBS array task processes one ω_target value at the nominal V_f
(default 0.10).  Per-task outputs are written to
`results_3d/freq_omega_NNN.pkl`.

The optimisation problem at each task:
    minimise  Σ_i w_i u_i^T K(ρ) u_i
    subject to ω₁(ρ) ≥ ω_target
              Σ_e ρ_e V_e ≤ V_f · V_design

(implemented as compliance objective + augmented-Lagrangian frequency
penalty + strict OC bisection on the volume).
"""
from __future__ import annotations

import os
# Allow BLAS to multi-thread freely — we are alone on the node
os.environ.pop("OMP_NUM_THREADS", None)
os.environ.pop("OPENBLAS_NUM_THREADS", None)
os.environ.pop("MKL_NUM_THREADS", None)
os.environ.pop("NUMEXPR_NUM_THREADS", None)

import argparse
import pickle
import time
import numpy as np

from src.mesh3d import ArmGeometry3D
from src.fem import FEM3D, Material
from src.fem_gpu import enable_gpu_solver
from src.topopt import SIMPOptimizer
from src.fatigue import fatigue_life_field


# Overridable so a corrected-geometry sweep can be written alongside the v4
# results instead of overwriting them (QARM_RESULTS_DIR=results_3d_v5).
RESULTS_DIR = os.environ.get("QARM_RESULTS_DIR") or os.path.join(
    os.path.dirname(__file__), "..", "results_3d")
os.makedirs(RESULTS_DIR, exist_ok=True)


# Sweep design: fixed ω_target (rotor BPF rule of thumb), sweep V_f.
# Per PBS array index, run one V_f.  Story: "how much arm mass is needed
# to keep ω₁ above the rotor BPF?"  The smallest V_f that achieves
# ω₁ ≥ ω_target is the v4 headline minimum mass.
# 0.065 and 0.08 bracket the OEM arm, which occupies 7.6 % of the design
# box (34.5 g against 455 g solid) — that is what makes an EQUAL-MASS
# comparison possible rather than only a same-V_f one.
# 0.05 is retained but is expected to be ill-conditioned: at that density
# the stiffness range spans ~1e9 and LOBPCG struggles (see
# notes/voxel_vs_cad_validation.md).
VOL_FRACS = [0.05, 0.065, 0.08, 0.10, 0.20, 0.30, 0.50]
NOMINAL_OMEGA_Hz = 500.0
TIP_MASS_KG = 0.060  # motor (~50 g Sunnysky 2212) + 10×4.5 prop (~10 g)
TO_MAX_ITER = 80

MATERIAL = Material(E=1700e6, nu=0.39, sigma_y=38e6, sigma_u=48e6, rho=930.0)

LOAD_CASES = [
    dict(name="LC1_hover",      F=(0.0, 0.0,  2.94),   Mz=None),
    dict(name="LC2_maneuver",   F=(0.0, 0.0,  5.88),   Mz=None),
    dict(name="LC3_landing",    F=(0.0, 0.0, -14.7),   Mz=None),
    dict(name="LC4_banked",     F=(0.0, 2.94, 5.0922), Mz=None),
    dict(name="LC5_proptorque", F=None,                Mz=0.10),
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


def run_one(omega_target_Hz: float, vf: float):
    overall_t0 = time.time()
    print("=" * 60, flush=True)
    print(f"ω_target = {omega_target_Hz:.0f} Hz   V_f = {vf:.2f}   "
          f"host = {os.uname().nodename}", flush=True)
    gpu = enable_gpu_solver()
    print(f"GPU solver: {'ENABLED' if gpu else 'disabled (CPU fallback)'}",
          flush=True)
    if gpu:
        import cupy as cp
        for i in range(cp.cuda.runtime.getDeviceCount()):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = (props["name"].decode() if isinstance(props["name"], bytes)
                    else props["name"])
            print(f"  Visible GPU {i}: {name}", flush=True)
    print("=" * 60, flush=True)

    geom = ArmGeometry3D()
    print(geom.summary(), flush=True)
    fem = FEM3D(geom, MATERIAL)
    F_list = build_force_vectors(geom)

    # --- Baseline static + fatigue + modal ---
    print("--- Baseline static + fatigue + modal ---", flush=True)
    t0 = time.time()
    rho_baseline = np.ones(fem.n_active)
    static_base = static_analysis(fem, F_list, rho_baseline)
    fat_base = fatigue_analysis(static_base)
    freqs_base, modes_base, lam_base = fem.modal_analysis(
        rho_baseline, n_modes=6, tip_mass_kg=TIP_MASS_KG,
    )
    print(f"  baseline elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"  baseline first 6 ω (Hz): "
          f"{', '.join(f'{f:.1f}' for f in freqs_base)}", flush=True)
    for lc in LOAD_CASES:
        s = static_base[lc["name"]]
        print(f"  {lc['name']:15s}: max vM = {s['vm'].max()/1e6:.4f} MPa  "
              f"SF_y = {s['sf_yield'].min():6.2f}", flush=True)

    # --- Frequency-constrained TO ---
    print(f"--- TO at V_f = {vf:.2f}, min ω₁ ≥ {omega_target_Hz:.0f} Hz ---",
          flush=True)
    t0 = time.time()
    opt = SIMPOptimizer(
        geom, fem,
        load_cases=LOAD_CASES,
        weights=LOAD_WEIGHTS,
        vol_frac=vf,
        r_min_mm=3.0,
        max_iter=TO_MAX_ITER,
        tol_change=0.005,
        # New v4 modal arguments:
        min_freq_Hz=omega_target_Hz,
        tip_mass_kg=TIP_MASS_KG,
        n_modes_track=4,
        # mu_freq is auto-scaled to compliance sens magnitude in topopt;
        # mu=1 means "modal sens contributes the same scale as compliance".
        mu_freq_init=0.20,
        mu_freq_max=50.0,
        mu_freq_ramp_every=5,
        mu_freq_ramp_factor=1.5,
    )
    res = opt.run(verbose=True)
    print(f"  TO done in {(time.time()-t0)/60:.1f} min  "
          f"({len(res['history']['compliance'])} iters)", flush=True)

    rho_opt = res["rho_active"]
    print("--- Validation static + fatigue + modal ---", flush=True)
    t0 = time.time()
    static_opt = static_analysis(fem, F_list, rho_opt)
    fat_opt = fatigue_analysis(static_opt)
    freqs_opt, modes_opt, lam_opt = fem.modal_analysis(
        rho_opt, n_modes=6, tip_mass_kg=TIP_MASS_KG,
    )
    print(f"  validation elapsed: {time.time()-t0:.1f}s", flush=True)
    print(f"  optimised first 6 ω (Hz): "
          f"{', '.join(f'{f:.1f}' for f in freqs_opt)}", flush=True)

    mass_g = fem.total_mass(rho_opt) * 1e3
    sig_per_lc = {lc["name"]: float(static_opt[lc["name"]]["vm"].max() / 1e6)
                  for lc in LOAD_CASES}
    sf_y_per_lc = {lc["name"]: float(static_opt[lc["name"]]["sf_yield"].min())
                   for lc in LOAD_CASES}

    record = dict(
        omega_target_Hz=float(omega_target_Hz),
        Vf=float(vf),
        mass_g=float(mass_g),
        omega_1_Hz=float(freqs_opt[0]),
        omega_2_Hz=float(freqs_opt[1]),
        omega_3_Hz=float(freqs_opt[2]),
        constraint_satisfied=bool(float(freqs_opt[0]) >= omega_target_Hz),
        sigma_max_LC1_MPa=sig_per_lc["LC1_hover"],
        sigma_max_LC2_MPa=sig_per_lc["LC2_maneuver"],
        sigma_max_LC3_MPa=sig_per_lc["LC3_landing"],
        sigma_max_LC4_MPa=sig_per_lc["LC4_banked"],
        sigma_max_LC5_MPa=sig_per_lc["LC5_proptorque"],
        sf_yield_LC2=sf_y_per_lc["LC2_maneuver"],
        sf_yield_LC3=sf_y_per_lc["LC3_landing"],
        fs_fatigue=float(fat_opt["factor_of_safety"].min()),
        n_iter=int(len(res["history"]["compliance"])),
        wall_time_min=(time.time() - overall_t0) / 60,
        gpu_used=gpu,
    )
    print("=== RESULT ===", flush=True)
    for k, v in record.items():
        print(f"  {k}: {v}", flush=True)

    payload_path = os.path.join(
        RESULTS_DIR,
        f"freq_vf{int(vf*100):02d}_omega{int(omega_target_Hz):04d}.pkl",
    )
    with open(payload_path, "wb") as f:
        pickle.dump(dict(
            rho=rho_opt,
            static=static_opt,
            fatigue=fat_opt,
            modal=dict(frequencies_Hz=freqs_opt,
                        mode_shapes=modes_opt,
                        omega_sq=lam_opt,
                        tip_mass_kg=TIP_MASS_KG),
            history=res["history"],
            record=record,
        ), f)

    # Persist baseline (modal + static) once — last writer wins
    baseline_path = os.path.join(RESULTS_DIR, "raw_baseline_modal.pkl")
    with open(baseline_path, "wb") as f:
        pickle.dump(dict(
            static=static_base,
            fatigue=fat_base,
            modal=dict(frequencies_Hz=freqs_base,
                        mode_shapes=modes_base,
                        omega_sq=lam_base,
                        tip_mass_kg=TIP_MASS_KG),
            material=MATERIAL,
            mesh_info=dict(
                nx=geom.nx, ny=geom.ny, nz=geom.nz,
                n_elements=int(geom.n_elements),
                n_active_elements=int(fem.n_active),
                n_nodes=int(geom.n_nodes),
                n_dofs=int(geom.n_dofs),
            ),
        ), f)

    print(f"Saved: {payload_path}", flush=True)
    print(f"Total wall time: {(time.time()-overall_t0)/60:.1f} min", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--omega-target", type=float, default=NOMINAL_OMEGA_Hz,
                    help=f"First-natural-frequency target [Hz] "
                         f"(default {NOMINAL_OMEGA_Hz:.0f}).")
    p.add_argument("--vf", type=float, default=None,
                    help="Volume fraction. If unset, taken from "
                         "PBS_ARRAY_INDEX into VOL_FRACS.")
    args = p.parse_args()
    if args.vf is not None:
        vf = args.vf
    else:
        idx = int(os.environ.get("PBS_ARRAY_INDEX", "0"))
        vf = VOL_FRACS[idx]
    run_one(args.omega_target, vf=vf)


if __name__ == "__main__":
    main()
