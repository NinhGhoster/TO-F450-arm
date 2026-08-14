"""
COMPLETE figure regeneration for the 3D-bridge geometry.

Renders, in one pass, every figure referenced in the manuscript using the
new ArmGeometry3D mesh and the GPU-computed results in `results_3d/`.

Includes:
  * fig02b      — quadcopter overview with the 3D bridge arm
  * fig04       — geometry + BC top view (annotated)
  * fig05       — baseline vM stress (LC2)
  * fig06       — baseline σ_alt,eq
  * fig07       — TO convergence at the nominal V_f (from saved history)
  * fig08_Vfxx  — isometric topology per V_f  (x 4)
  * fig08_mv_Vfxx — multi-view (iso/top/side/front) per V_f  (x 4)
  * fig08_arm_in_assembly — arm in F450 frame at nominal V_f
  * fig09_Vfxx  — vM stress per V_f under LC2  (x 4)
  * fig10_Vfxx  — σ_alt,eq per V_f  (x 4)
  * fig11       — Pareto plot (mass vs σ vs SF)
  * fig12       — (deferred — orientation not yet re-run for 3D bridge)
"""
from __future__ import annotations

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from src.mesh3d import ArmGeometry3D
from src.fem import FEM3D, Material
from src.fatigue import equivalent_alt_stress_goodman
from src.plotting_3d import (
    plot_topology_multiview,
    plot_topology_iso,
    plot_stress_3d,
    plot_alt_stress_3d,
    plot_quadcopter_overview,
)
from src.plotting import plot_arm_footprint, plot_to_convergence


FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_3d")
os.makedirs(FIGURES_DIR, exist_ok=True)

MAT = Material()
geom = ArmGeometry3D()
fem = FEM3D(geom, MAT)

with open(os.path.join(RESULTS_DIR, "raw_baseline.pkl"), "rb") as f:
    baseline = pickle.load(f)

VOL_FRACS = [0.05, 0.10, 0.15, 0.20]
NOMINAL_VF = 0.10

opt_data = {}
for vf in VOL_FRACS:
    path = os.path.join(RESULTS_DIR, f"vf_{int(vf*100):02d}.pkl")
    with open(path, "rb") as f:
        opt_data[vf] = pickle.load(f)
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
print("Re-generating ALL figures for 3D bridge")
print("=" * 60)

# ---- fig04: geometry + BC top view ----------------------------------------
plot_arm_footprint(geom, os.path.join(FIGURES_DIR, "fig04_geometry_BC.png"))
print("  fig04 geometry + BC (3D bridge) done")

# ---- fig02b: quadcopter overview with our 3D-bridge arm at nominal V_f -----
plot_quadcopter_overview(
    geom, fem, d_nom["rho"],
    os.path.join(FIGURES_DIR, "fig02b_quadcopter_overview.png"),
    title=f"F450 with simulated optimized 3D-bridge arms ($V_f$ = {NOMINAL_VF:.2f})",
)
print("  fig02b quadcopter overview (3D bridge) done")

# ---- fig05: baseline vM under LC2 -----------------------------------------
rho_baseline = np.ones(fem.n_active)
vm_base_LC2 = baseline["static"]["LC2_maneuver"]["vm"]
plot_stress_3d(geom, fem, rho_baseline, vm_base_LC2,
                os.path.join(FIGURES_DIR, "fig05_baseline_vm_LC2.png"),
                title="Baseline arm — von Mises stress under LC2 (Maneuver, 5.88 N)",
                vmax_MPa=float(vm_base_LC2.max() / 1e6))
print("  fig05 baseline vM done")

# ---- fig06: baseline σ_alt,eq under LC1↔LC2 cycle -------------------------
alt_base = _alt_eq(baseline["static"])
plot_alt_stress_3d(geom, fem, rho_baseline, alt_base,
                    os.path.join(FIGURES_DIR, "fig06_baseline_life.png"),
                    title="Baseline arm — Goodman-corrected σ_alt,eq under LC1↔LC2 cycle",
                    vmax_MPa=13.0)
print("  fig06 baseline σ_alt,eq done")

# ---- fig07: TO convergence at the nominal V_f -----------------------------
hist_nom = d_nom["history"]
plot_to_convergence(hist_nom,
                     os.path.join(FIGURES_DIR, "fig07_to_convergence.png"),
                     title=f"TO convergence ($V_f$ = {NOMINAL_VF:.2f}, 3D bridge)")
print("  fig07 TO convergence (3D bridge) done")

# ---- fig08 isometric + multi-view per V_f --------------------------------
for vf in VOL_FRACS:
    rho = opt_data[vf]["rho"]
    plot_topology_iso(
        geom, fem, rho,
        os.path.join(FIGURES_DIR, f"fig08_topo_Vf{int(vf*100):02d}.png"),
        title=f"Optimized topology, $V_f$ = {vf:.2f}",
    )
    plot_topology_multiview(
        geom, fem, rho,
        os.path.join(FIGURES_DIR, f"fig08_mv_Vf{int(vf*100):02d}.png"),
        title=f"Optimized topology, $V_f$ = {vf:.2f}",
    )
    print(f"  fig08 Vf={vf:.2f} (iso + multiview) done")

# ---- fig08_arm_in_assembly at nominal V_f ---------------------------------
plot_topology_iso(geom, fem, d_nom["rho"],
                   os.path.join(FIGURES_DIR, "fig08_arm_in_assembly.png"),
                   title=f"Optimized 3D-bridge arm ($V_f$ = {NOMINAL_VF:.2f}) in F450 frame",
                   with_assembly=True, show_propeller=False)
print("  fig08 arm in assembly done")

# ---- fig09 vM stress under LC2 per V_f -----------------------------------
# Use a common colour scale across V_f for fair comparison
max_LC2_all = max(opt_data[vf]["static"]["LC2_maneuver"]["vm"].max() / 1e6
                  for vf in VOL_FRACS)
for vf in VOL_FRACS:
    plot_stress_3d(
        geom, fem, opt_data[vf]["rho"],
        opt_data[vf]["static"]["LC2_maneuver"]["vm"],
        os.path.join(FIGURES_DIR, f"fig09_vm_Vf{int(vf*100):02d}.png"),
        title=f"Optimized arm vM under LC2, $V_f$ = {vf:.2f}",
        vmax_MPa=max_LC2_all,
    )
    print(f"  fig09 vM Vf={vf:.2f} done")

# Also keep a single nominal-V_f vM figure as the manuscript's headline
plot_stress_3d(
    geom, fem, d_nom["rho"],
    d_nom["static"]["LC2_maneuver"]["vm"],
    os.path.join(FIGURES_DIR, "fig09_optimized_vm_LC2.png"),
    title=f"Optimized arm — von Mises under LC2, $V_f$ = {NOMINAL_VF:.2f}",
    vmax_MPa=float(d_nom["static"]["LC2_maneuver"]["vm"].max() / 1e6),
)

# ---- fig10 σ_alt,eq per V_f -----------------------------------------------
for vf in VOL_FRACS:
    alt = _alt_eq(opt_data[vf]["static"])
    plot_alt_stress_3d(
        geom, fem, opt_data[vf]["rho"], alt,
        os.path.join(FIGURES_DIR, f"fig10_alt_Vf{int(vf*100):02d}.png"),
        title=f"σ_alt,eq, $V_f$ = {vf:.2f}",
        vmax_MPa=13.0,
    )
    print(f"  fig10 σ_alt,eq Vf={vf:.2f} done")

plot_alt_stress_3d(
    geom, fem, d_nom["rho"], _alt_eq(d_nom["static"]),
    os.path.join(FIGURES_DIR, "fig10_optimized_life.png"),
    title=f"Optimized arm — Goodman-corrected σ_alt,eq, $V_f$ = {NOMINAL_VF:.2f}",
    vmax_MPa=13.0,
)

print()
# Clean up obsolete legacy figures
for stale in (
    "fig08b_topo_Vf30_layers.png",   # 2D per-layer view doesn't apply to 3D
    "fig08_multiview_Vf30.png",      # renamed; nominal V_f is 0.10 now
    "fig08_topo_Vf30.png",
    "fig08_topo_Vf40.png",
    "fig08_topo_Vf50.png",
    "fig08_arm_in_assembly_Vf30.png",
    "fig12_orientation.png",         # orientation study not re-run for bridge
):
    p = os.path.join(FIGURES_DIR, stale)
    if os.path.exists(p):
        os.remove(p)
        print(f"  removed legacy {stale}")

print()
print("All 3D-bridge figures regenerated.")
