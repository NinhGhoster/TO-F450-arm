"""
Complete v4 figure regeneration — modal-pivot version.

Renders every figure referenced in the v4 manuscript from the saved pkls
in `results_3d/` without re-running the FEM.  Use whenever rendering
style, labels, or axis choices change.

Includes:
  * fig02b      — quadcopter overview with the modal-optimised arms
  * fig04       — geometry + BC top view (unchanged from v3)
  * fig05/06    — baseline stress + fatigue (unchanged; framed as
                  "non-binding evidence" in v4)
  * fig07       — modal-constrained TO convergence at the nominal
                  ω_target (3-panel: ω₁, V_f, μ)
  * fig08       — topology per ω_target (4 panels)
  * fig09       — first 4 mode shapes for the nominal design
  * fig10       — FRF at motor mount, baseline + 4 optimised designs
  * fig11       — mass-frequency Pareto plus structural-safety
                  verification panel
  * fig12       — vM stress check on the modal-optimised nominal design
"""
from __future__ import annotations

import os
import pickle
import numpy as np

from src.mesh3d import ArmGeometry3D
from src.fem import FEM3D, Material
from src.fatigue import equivalent_alt_stress_goodman
from src.plotting_3d import (
    plot_topology_iso,
    plot_mode_shape_3d,
    plot_stress_3d,
    plot_alt_stress_3d,
    plot_quadcopter_overview,
)
from src.plotting import (
    plot_arm_footprint,
    plot_freq_convergence,
    plot_frf,
    plot_mass_freq_pareto,
    modal_frf_at_dof,
)


FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
# Overridable so a corrected-geometry sweep renders from its own directory
# instead of the superseded one.
RESULTS_DIR = os.environ.get("QARM_RESULTS_DIR") or os.path.join(
    os.path.dirname(__file__), "..", "results_3d")
os.makedirs(FIGURES_DIR, exist_ok=True)

MAT = Material()
geom = ArmGeometry3D()
fem = FEM3D(geom, MAT)

# Load baseline (modal + static)
with open(os.path.join(RESULTS_DIR, "raw_baseline_modal.pkl"), "rb") as f:
    baseline = pickle.load(f)
baseline_modal = baseline["modal"]
baseline_static = baseline["static"]

# Load all (V_f, ω_target) sweep results.  Headline V_f for figures
# (mode shapes, FRF reference) is the smallest one that satisfies the
# constraint; default to V_f = 0.20.
import glob
# Whatever the sweep actually produced, rather than a hard-coded list that
# silently drops points or looks for ones that were never run.
VOL_FRACS = sorted(
    int(os.path.basename(f).split("_")[1][2:]) / 100.0
    for f in glob.glob(os.path.join(RESULTS_DIR, "freq_vf*_omega*.pkl")))
NOMINAL_OMEGA_Hz = 500
NOMINAL_VF = 0.50  # the best-modal design within the sweep (highest ω₁)
opt_data = {}
for vf in VOL_FRACS:
    path = os.path.join(
        RESULTS_DIR, f"freq_vf{int(vf*100):02d}_omega{NOMINAL_OMEGA_Hz:04d}.pkl"
    )
    if not os.path.exists(path):
        print(f"  [skip] {path} missing — run not finished?")
        continue
    with open(path, "rb") as f:
        opt_data[vf] = pickle.load(f)
if NOMINAL_VF not in opt_data:
    raise SystemExit(f"Nominal V_f = {NOMINAL_VF} result is missing.")
d_nom = opt_data[NOMINAL_VF]


def _alt_eq(static_results):
    vm1 = static_results["LC1_hover"]["vm"]
    vm2 = static_results["LC2_maneuver"]["vm"]
    s_peak = np.maximum(vm1, vm2)
    s_valley = np.minimum(vm1, vm2)
    sigma_mean = 0.5 * (s_peak + s_valley)
    sigma_alt = 0.5 * (s_peak - s_valley)
    return equivalent_alt_stress_goodman(sigma_alt, sigma_mean, MAT.sigma_u)


print("=" * 60)
print("Re-generating ALL v4 figures (modal-pivot)")
print("=" * 60)

# ---- fig04: geometry + BC top view (carried over) -------------------------
plot_arm_footprint(geom, os.path.join(FIGURES_DIR, "fig04_geometry_BC.png"))
print("  fig04 geometry + BC done")

# ---- fig02b: quadcopter overview with the modal-optimised nominal arm -----
plot_quadcopter_overview(
    geom, fem, d_nom["rho"],
    os.path.join(FIGURES_DIR, "fig02b_quadcopter_overview.png"),
    title=(f"F450 with modal-optimised arms "
           fr"($\omega_\mathrm{{target}}$ = {NOMINAL_OMEGA_Hz} Hz)"),
)
print("  fig02b quadcopter overview done")

# ---- fig05: baseline vM under LC2 -----------------------------------------
rho_baseline_active = np.ones(fem.n_active)
vm_base_LC2 = baseline_static["LC2_maneuver"]["vm"]
plot_stress_3d(geom, fem, rho_baseline_active, vm_base_LC2,
                os.path.join(FIGURES_DIR, "fig05_baseline_vm_LC2.png"),
                title="Baseline arm — von Mises stress under LC2 (Maneuver)",
                vmax_MPa=float(vm_base_LC2.max() / 1e6))
print("  fig05 baseline vM done")

# ---- fig06: baseline σ_alt,eq under LC1↔LC2 -------------------------------
alt_base = _alt_eq(baseline_static)
plot_alt_stress_3d(geom, fem, rho_baseline_active, alt_base,
                    os.path.join(FIGURES_DIR, "fig06_baseline_life.png"),
                    title=(r"Baseline arm — Goodman-corrected $\sigma_\mathrm{alt,eq}$ "
                           "under LC1↔LC2"),
                    vmax_MPa=13.0)
print("  fig06 baseline σ_alt,eq done")

# ---- fig07: modal-TO convergence at the nominal (V_f, ω_target) -----------
plot_freq_convergence(
    d_nom["history"],
    os.path.join(FIGURES_DIR, "fig07_modal_convergence.png"),
    title=(fr"TO convergence — $V_f$ = {d_nom['record']['Vf']:.2f}, "
           fr"$\omega_\mathrm{{target}}$ = {NOMINAL_OMEGA_Hz} Hz"),
)
print("  fig07 modal convergence done")

# ---- fig08 topology per V_f at fixed ω_target ----------------------------
for vf in VOL_FRACS:
    if vf not in opt_data:
        continue
    rec = opt_data[vf]["record"]
    plot_topology_iso(
        geom, fem, opt_data[vf]["rho"],
        os.path.join(FIGURES_DIR, f"fig08_topo_vf{int(vf*100):02d}.png"),
        title=(fr"Optimised topology, $V_f$ = {vf:.2f}, "
               fr"$\omega_\mathrm{{target}}$ = {NOMINAL_OMEGA_Hz} Hz "
               fr"(achieved $\omega_1$ = {rec['omega_1_Hz']:.1f} Hz, "
               fr"mass = {rec['mass_g']:.1f} g)"),
    )
    print(f"  fig08 V_f={vf:.2f} done")

# ---- fig09: first 4 mode shapes of the nominal design --------------------
freqs_nom = d_nom["modal"]["frequencies_Hz"]
modes_nom = d_nom["modal"]["mode_shapes"]
for k in range(min(4, modes_nom.shape[1])):
    plot_mode_shape_3d(
        geom, fem, d_nom["rho"], modes_nom[:, k],
        os.path.join(FIGURES_DIR, f"fig09_mode_{k+1}.png"),
        title=(fr"Mode {k+1} — $\omega_{{{k+1}}}$ = {freqs_nom[k]:.1f} Hz "
               f"(nominal design)"),
    )
    print(f"  fig09 mode {k+1} (ω = {freqs_nom[k]:.1f} Hz) done")

# ---- fig10: FRF at motor mount -------------------------------------------
# The z-translation DoF at a motor-mount node.  Taking the geometrically
# central node of the pad does not work: after optimisation many pad nodes
# carry no material, their mode-shape entries are identically zero, and the
# resulting FRF is zero at every frequency — which plots as nothing at all on
# log axes.  Pick instead the pad node that actually moves the most, summed
# over the tracked modes, which is the one an accelerometer would be put on.
motor_nodes = np.asarray(geom.motor_mount_top_nodes())
_z = np.abs(baseline_modal["mode_shapes"][3 * motor_nodes + 2, :]).sum(axis=1)
for _vf, _d in opt_data.items():
    _z = _z + np.abs(_d["modal"]["mode_shapes"][3 * motor_nodes + 2, :]).sum(axis=1)
ref_node = int(motor_nodes[int(np.argmax(_z))])
ref_dof_z = 3 * ref_node + 2
curves = {}
# Baseline
fb, mb = modal_frf_at_dof(
    baseline_modal["frequencies_Hz"], baseline_modal["mode_shapes"],
    dof_idx=ref_dof_z,
)
curves["Solid baseline"] = (fb, mb)
# Each optimised V_f at the nominal ω_target
for vf in VOL_FRACS:
    if vf not in opt_data:
        continue
    fk, mk = modal_frf_at_dof(
        opt_data[vf]["modal"]["frequencies_Hz"],
        opt_data[vf]["modal"]["mode_shapes"],
        dof_idx=ref_dof_z,
    )
    curves[fr"$V_f$ = {vf:.2f}"] = (fk, mk)
plot_frf(curves, os.path.join(FIGURES_DIR, "fig10_frf_motor_mount.png"),
          bpf_band_Hz=(83.0, 267.0),
          omega_target_Hz=None,
          title="Vertical response at the motor mount, modal superposition")
print("  fig10 FRF at motor mount done")

# ---- fig11: mass-frequency Pareto + structural-safety verification --------
baseline_record = dict(
    mass_g=float(fem.total_mass(rho_baseline_active) * 1e3),
    omega_1_Hz=float(baseline_modal["frequencies_Hz"][0]),
    sf_yield_LC3=float(baseline_static["LC3_landing"]["sf_yield"].min()),
    fs_fatigue=float(baseline.get("fatigue", {}).get("factor_of_safety",
                                                       np.array([1e9])).min())
        if "fatigue" in baseline else float("nan"),
)
records = [opt_data[vf]["record"] for vf in VOL_FRACS if vf in opt_data]
plot_mass_freq_pareto(
    records, baseline_record,
    os.path.join(FIGURES_DIR, "fig11_pareto_mass_freq.png"),
    omega_target_default_Hz=NOMINAL_OMEGA_Hz,
    bpf_band_Hz=(200.0, 400.0),
)
print("  fig11 mass-frequency Pareto done")

# ---- fig12: vM verification on the nominal modal-optimised design --------
opt_vm_LC2 = d_nom["static"]["LC2_maneuver"]["vm"]
plot_stress_3d(geom, fem, d_nom["rho"], opt_vm_LC2,
                os.path.join(FIGURES_DIR, "fig12_optimised_vm_LC2.png"),
                title=(fr"Modal-optimised arm vM under LC2 "
                       fr"($\omega_\mathrm{{target}}$ = {NOMINAL_OMEGA_Hz} Hz, "
                       fr"$V_f$ = {d_nom['record']['Vf']:.2f})"),
                vmax_MPa=float(opt_vm_LC2.max() / 1e6))
print("  fig12 stress verification done")

# ---- Cleanup: remove v3-only figures that are obsolete in v4 -------------
# NOTE: macOS HFS+/APFS is case-insensitive, so legacy `fig08_topo_VfNN.png`
# (uppercase V) is the SAME file as the new v4 `fig08_topo_vfNN.png` (lowercase v).
# Do not list these in the stale set, otherwise the cleanup would delete our
# fresh outputs.  v4 file names with V_f are all lowercase-v throughout.
for stale in (
    "fig07_to_convergence.png",        # v3 compliance-only convergence
    "fig08_mv_Vf05.png", "fig08_mv_Vf10.png",
    "fig08_mv_Vf15.png", "fig08_mv_Vf20.png",
    "fig09_vm_Vf05.png", "fig09_vm_Vf10.png",
    "fig09_vm_Vf15.png", "fig09_vm_Vf20.png",
    "fig09_optimized_vm_LC2.png",
    "fig10_alt_Vf05.png", "fig10_alt_Vf10.png",
    "fig10_alt_Vf15.png", "fig10_alt_Vf20.png",
    "fig10_optimized_life.png",
):
    p = os.path.join(FIGURES_DIR, stale)
    if os.path.exists(p):
        os.remove(p)
        print(f"  removed legacy {stale}")

print()
print("All v4 figures regenerated.")
