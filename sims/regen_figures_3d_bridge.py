"""
Regenerate all manuscript figures using the new 3D-bridge mesh
(ArmGeometry3D, 220×56×44 mm).
"""
from __future__ import annotations

import os
import pickle
import numpy as np

from src.mesh3d import ArmGeometry3D
from src.fem import FEM3D, Material
from src.fatigue import equivalent_alt_stress_goodman
from src.plotting_3d import (
    plot_topology_multiview,
    plot_topology_iso,
    plot_stress_3d,
    plot_alt_stress_3d,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_3d")
os.makedirs(FIGURES_DIR, exist_ok=True)

MAT = Material()
geom = ArmGeometry3D()
fem = FEM3D(geom, MAT)

with open(os.path.join(RESULTS_DIR, "raw_baseline.pkl"), "rb") as f:
    baseline = pickle.load(f)
opt_data = {}
for vf in (0.05, 0.10, 0.15, 0.20):
    path = os.path.join(RESULTS_DIR, f"vf_{int(vf*100):02d}.pkl")
    with open(path, "rb") as f:
        opt_data[vf] = pickle.load(f)
NOMINAL_VF = 0.10
d_nom = opt_data[NOMINAL_VF]


def _alt_eq(static_results, sigma_uts_Pa):
    vm1 = static_results["LC1_hover"]["vm"]
    vm2 = static_results["LC2_maneuver"]["vm"]
    s_peak = np.maximum(vm1, vm2)
    s_valley = np.minimum(vm1, vm2)
    sigma_mean = 0.5 * (s_peak + s_valley)
    sigma_alt = 0.5 * (s_peak - s_valley)
    return equivalent_alt_stress_goodman(sigma_alt, sigma_mean, sigma_uts_Pa)


print("Rendering 3D-bridge figures...")

# Baseline static
rho_baseline = np.ones(fem.n_active)
vm_base_LC2 = baseline["static"]["LC2_maneuver"]["vm"]
base_LC2_max_MPa = float(vm_base_LC2.max() / 1e6)
plot_stress_3d(geom, fem, rho_baseline, vm_base_LC2,
                os.path.join(FIGURES_DIR, "fig05_baseline_vm_LC2.png"),
                title="Baseline arm — von Mises stress under LC2 (Maneuver, 5.88 N)",
                vmax_MPa=base_LC2_max_MPa)
print(f"  fig05 baseline vM done (max {base_LC2_max_MPa:.3f} MPa)")

alt_base = _alt_eq(baseline["static"], MAT.sigma_u)
plot_alt_stress_3d(geom, fem, rho_baseline, alt_base,
                    os.path.join(FIGURES_DIR, "fig06_baseline_life.png"),
                    title="Baseline arm — Goodman-corrected σ_alt,eq under LC1↔LC2 cycle",
                    vmax_MPa=13.0)
print("  fig06 baseline alt-stress done")

# Per-Vf topology
for vf in (0.05, 0.10, 0.15, 0.20):
    plot_topology_iso(geom, fem, opt_data[vf]["rho"],
                       os.path.join(FIGURES_DIR, f"fig08_topo_Vf{int(vf*100):02d}.png"),
                       title=f"Optimized topology, $V_f$ = {vf:.2f}")
    print(f"  fig08 Vf={vf:.2f} done")

# Multi-view of nominal Vf
plot_topology_multiview(geom, fem, d_nom["rho"],
                         os.path.join(FIGURES_DIR, "fig08_multiview_Vf30.png"),
                         title=f"Optimized topology, $V_f$ = {NOMINAL_VF:.2f}")
print("  fig08 multi-view done")

# Single arm + assembly
plot_topology_iso(geom, fem, d_nom["rho"],
                   os.path.join(FIGURES_DIR, "fig08_arm_in_assembly_Vf30.png"),
                   title=f"Optimized 3D-bridge arm ($V_f$ = {NOMINAL_VF:.2f}) in F450 frame",
                   with_assembly=True, show_propeller=False)
print("  fig08 arm + assembly done")

# Optimized static + fatigue at nominal Vf
opt_vm_LC2 = d_nom["static"]["LC2_maneuver"]["vm"]
opt_LC2_max_MPa = float(opt_vm_LC2.max() / 1e6)
plot_stress_3d(geom, fem, d_nom["rho"], opt_vm_LC2,
                os.path.join(FIGURES_DIR, "fig09_optimized_vm_LC2.png"),
                title=f"Optimized arm — von Mises under LC2, $V_f$ = {NOMINAL_VF:.2f}",
                vmax_MPa=opt_LC2_max_MPa)
print(f"  fig09 optimized vM done (max {opt_LC2_max_MPa:.3f} MPa)")

alt_opt = _alt_eq(d_nom["static"], MAT.sigma_u)
plot_alt_stress_3d(geom, fem, d_nom["rho"], alt_opt,
                    os.path.join(FIGURES_DIR, "fig10_optimized_life.png"),
                    title=f"Optimized arm — Goodman-corrected σ_alt,eq, $V_f$ = {NOMINAL_VF:.2f}",
                    vmax_MPa=13.0)
print("  fig10 optimized alt-stress done")

print()
print("All 3D-bridge figures regenerated.")
