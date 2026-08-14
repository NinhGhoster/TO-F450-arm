"""
Regenerate all the manuscript figures using the 3D PyVista renderers,
keeping the same filenames so the manuscript markdown does not need to
change its figure references.

Mapping:
    fig05_baseline_vm_LC2.png    — 3D iso, baseline arm, LC2 vM
    fig06_baseline_life.png       — 3D iso, baseline arm, σ_alt,eq (Goodman-corrected)
    fig08_topo_Vf{20,30,40,50}.png — 3D iso, optimized topology (single panel for sweep)
    fig08b_topo_Vf30_layers.png   — 2D per-layer view (kept as supplementary, unchanged)
    fig09_optimized_vm_LC2.png    — 3D iso, optimized arm Vf=0.30, LC2 vM
    fig10_optimized_life.png      — 3D iso, optimized arm Vf=0.30, σ_alt,eq
    fig08_multiview_Vf30.png      — NEW: 4-panel multiview of Vf=0.30 topology

The 2D Pareto figure (fig11) and orientation bar chart (fig12) are kept
as 2D — these are not 3D objects.
"""
from __future__ import annotations

import os
import pickle
import numpy as np

from src.mesh import ArmGeometry
from src.fem import FEM3D, Material
from src.fatigue import equivalent_alt_stress_goodman
from src.plotting_3d import (
    plot_topology_multiview,
    plot_topology_iso,
    plot_stress_3d,
    plot_alt_stress_3d,
    plot_quadcopter_overview,
    plot_oem_assembly,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)

MAT = Material()
geom = ArmGeometry()
fem = FEM3D(geom, MAT)

# Load saved data
with open(os.path.join(RESULTS_DIR, "raw_baseline.pkl"), "rb") as f:
    baseline = pickle.load(f)
opt_data = {}
for vf in (0.20, 0.30, 0.40, 0.50):
    path = os.path.join(RESULTS_DIR, f"vf_{int(vf*100):02d}.pkl")
    with open(path, "rb") as f:
        opt_data[vf] = pickle.load(f)
NOMINAL_VF = 0.30
d30 = opt_data[NOMINAL_VF]


def _alt_eq(static_results, sigma_uts_Pa):
    """Compute Goodman-corrected alternating stress field per element."""
    vm1 = static_results["LC1_hover"]["vm"]
    vm2 = static_results["LC2_maneuver"]["vm"]
    s_peak = np.maximum(vm1, vm2)
    s_valley = np.minimum(vm1, vm2)
    sigma_mean = 0.5 * (s_peak + s_valley)
    sigma_alt = 0.5 * (s_peak - s_valley)
    return equivalent_alt_stress_goodman(sigma_alt, sigma_mean, sigma_uts_Pa)


print("Rendering 3D figures...")

# ---------------------- Baseline ----------------------
rho_baseline = np.ones(fem.n_active)
vm_base_LC2 = baseline["static"]["LC2_maneuver"]["vm"]
base_LC2_max_MPa = float(vm_base_LC2.max() / 1e6)
plot_stress_3d(geom, fem, rho_baseline, vm_base_LC2,
                os.path.join(FIGURES_DIR, "fig05_baseline_vm_LC2.png"),
                title="Baseline arm — von Mises stress under LC2 (Maneuver, 5.88 N)",
                vmax_MPa=base_LC2_max_MPa)
print("  fig05 baseline vM (3D iso) done")

alt_base = _alt_eq(baseline["static"], MAT.sigma_u)
plot_alt_stress_3d(geom, fem, rho_baseline, alt_base,
                    os.path.join(FIGURES_DIR, "fig06_baseline_life.png"),
                    title="Baseline arm — Goodman-corrected σ_alt,eq under LC1↔LC2 cycle",
                    vmax_MPa=13.0)   # endurance limit as scale top
print("  fig06 baseline alt-stress (3D iso) done")

# ---------------------- Optimized: per-Vf single-iso topology ----------------------
for vf in (0.20, 0.30, 0.40, 0.50):
    plot_topology_iso(geom, fem, opt_data[vf]["rho"],
                       os.path.join(FIGURES_DIR, f"fig08_topo_Vf{int(vf*100):02d}.png"),
                       title=f"Optimized topology, $V_f$ = {vf:.2f}")
    print(f"  fig08 Vf={vf:.2f} (3D iso) done")

# Multi-view (iso, top, side, front) for the nominal Vf=0.30
plot_topology_multiview(geom, fem, opt_data[NOMINAL_VF]["rho"],
                         os.path.join(FIGURES_DIR, "fig08_multiview_Vf30.png"),
                         title=f"Optimized topology, $V_f$ = {NOMINAL_VF:.2f}")
print("  fig08 multi-view (4 panels) done")

# Hero figures —
# (a) Reference: the OEM F450 arm in the twin-plate frame, no simulation
plot_oem_assembly(os.path.join(FIGURES_DIR, "fig02a_OEM_assembly.png"),
                   title="OEM F450 arm in the twin-plate central frame "
                          "(top plate semi-transparent)")
print("  fig02a OEM arm reference done")

# (b) F450 quadcopter overview with our SIMULATED simplified-domain arms
plot_quadcopter_overview(geom, fem, opt_data[NOMINAL_VF]["rho"],
                          os.path.join(FIGURES_DIR, "fig02b_quadcopter_overview.png"),
                          title=f"F450 quadcopter with simulated optimized arms ($V_f$ = {NOMINAL_VF:.2f})")
print("  fig02b simulated quadcopter overview done")

# Single arm with assembly context (motor + bolts + plate) for topology
plot_topology_iso(geom, fem, opt_data[NOMINAL_VF]["rho"],
                   os.path.join(FIGURES_DIR, "fig08_arm_in_assembly_Vf30.png"),
                   title=f"Optimized arm ($V_f$ = {NOMINAL_VF:.2f}) with F450 motor, bolts, and central frame plate",
                   with_assembly=True, show_propeller=False)
print("  fig08 single arm + assembly done")

# ---------------------- Optimized stress + fatigue at Vf=0.30 ----------------------
opt_vm_LC2 = d30["static"]["LC2_maneuver"]["vm"]
opt_LC2_max_MPa = float(opt_vm_LC2.max() / 1e6)
plot_stress_3d(geom, fem, d30["rho"], opt_vm_LC2,
                os.path.join(FIGURES_DIR, "fig09_optimized_vm_LC2.png"),
                title=f"Optimized arm — von Mises stress under LC2, $V_f$ = {NOMINAL_VF:.2f}",
                vmax_MPa=opt_LC2_max_MPa)
print("  fig09 optimized vM (3D iso) done")

alt_opt = _alt_eq(d30["static"], MAT.sigma_u)
plot_alt_stress_3d(geom, fem, d30["rho"], alt_opt,
                    os.path.join(FIGURES_DIR, "fig10_optimized_life.png"),
                    title=f"Optimized arm — Goodman-corrected σ_alt,eq, $V_f$ = {NOMINAL_VF:.2f}",
                    vmax_MPa=13.0)
print("  fig10 optimized alt-stress (3D iso) done")

print()
print("All 3D figures regenerated in figures/.")
print("Note: fig07 (TO convergence), fig08b (per-layer 2D), fig11 (Pareto),")
print("      fig12 (orientation bar) are 2D and left unchanged.")
