"""Regenerate Pareto figure with stress (LC2 & LC3) and fatigue SF axes."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "savefig.dpi": 220, "savefig.bbox": "tight",
})

root = os.path.join(os.path.dirname(__file__), "..")
with open(os.path.join(root, "results", "summary.json")) as f:
    d = json.load(f)

base_mass = d["baseline"]["mass_g"]
base_LC2 = d["baseline"]["static"]["LC2_maneuver"]["max_vM_MPa"]
base_LC3 = d["baseline"]["static"]["LC3_landing"]["max_vM_MPa"]
base_fs_fat = d["baseline"]["fatigue"]["min_fs"]
sigma_y_MPa = d["material"]["sigma_y"] / 1e6

records = d["pareto"]
Vf = np.array([r["Vf"] for r in records])
mass = np.array([r["mass_g"] for r in records])
sig_LC2 = np.array([r["sigma_max_MPa"] for r in records])
sig_LC3 = np.array([r["sigma_max_LC3_MPa"] for r in records])
fs_fat = np.array([r["fs_fatigue"] for r in records])
fs_yield_LC2 = sigma_y_MPa / sig_LC2
fs_yield_LC3 = sigma_y_MPa / sig_LC3

# Include baseline as a point
all_mass = np.concatenate([[base_mass], mass])
all_LC2 = np.concatenate([[base_LC2], sig_LC2])
all_LC3 = np.concatenate([[base_LC3], sig_LC3])
all_fs_fat = np.concatenate([[base_fs_fat], fs_fat])
all_labels = ["Baseline"] + [f"Vf={v:.2f}" for v in Vf]
all_mass_red = (1 - all_mass / base_mass) * 100   # % reduction

fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))

# Panel 1: peak stress under LC2 and LC3 vs mass
ax = axes[0]
ax.plot(all_mass, all_LC2, "o-", color="C0", label="LC2 (Maneuver, 5.88 N)")
ax.plot(all_mass, all_LC3, "s-", color="C3", label="LC3 (Hard landing, 14.7 N)")
ax.axhline(sigma_y_MPa, color="grey", ls="--", lw=1, label=fr"$\sigma_y$ = {sigma_y_MPa:.0f} MPa")
for i, lab in enumerate(all_labels):
    ax.annotate(lab, (all_mass[i], all_LC3[i]),
                xytext=(4, 4), textcoords="offset points", fontsize=7)
ax.set_xlabel("Arm mass (g)")
ax.set_ylabel("Peak von Mises stress (MPa)")
ax.set_title("Mass vs. peak stress")
ax.grid(True, ls=":", alpha=0.5)
ax.legend(fontsize=7, loc="upper right")

# Panel 2: factor of safety vs mass
ax = axes[1]
ax.plot(all_mass, sigma_y_MPa/all_LC2, "o-", color="C0", label="SF yield (LC2)")
ax.plot(all_mass, sigma_y_MPa/all_LC3, "s-", color="C3", label="SF yield (LC3)")
ax.plot(all_mass, all_fs_fat, "^-", color="C2", label="SF fatigue (LC1↔LC2)")
ax.axhline(1.0, color="black", ls="--", lw=1, label="SF = 1")
ax.axhline(1.5, color="grey", ls=":", lw=1, label="SF = 1.5 (target)")
for i, lab in enumerate(all_labels):
    ax.annotate(lab, (all_mass[i], all_fs_fat[i]),
                xytext=(4, 4), textcoords="offset points", fontsize=7)
ax.set_xlabel("Arm mass (g)")
ax.set_ylabel("Factor of safety")
ax.set_title("Mass vs. factor of safety")
ax.set_yscale("log")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(fontsize=7, loc="upper left", ncol=1)

# Panel 3: mass reduction summary
ax = axes[2]
x = np.arange(len(records))
width = 0.30
bars1 = ax.bar(x - width, mass, width, color="C0", label="Mass (g)")
ax.set_xticks(x)
ax.set_xticklabels([f"Vf={v:.2f}" for v in Vf])
ax.set_ylabel("Mass (g)", color="C0")
ax.tick_params(axis="y", labelcolor="C0")
ax.axhline(base_mass, color="C0", ls=":", lw=1, label=f"Baseline {base_mass:.1f} g")
ax.legend(loc="upper left", fontsize=7)

ax2 = ax.twinx()
bars2 = ax2.bar(x, fs_fat, width, color="C2", alpha=0.7, label="SF fatigue")
bars3 = ax2.bar(x + width, sigma_y_MPa/sig_LC3, width, color="C3", alpha=0.7,
                 label="SF yield (LC3)")
ax2.set_ylabel("Safety factor", color="black")
ax2.axhline(1.5, color="grey", ls=":", lw=1)
ax2.legend(loc="upper right", fontsize=7)
ax.set_title("Optimized configurations")

plt.tight_layout()
out_path = os.path.join(root, "figures", "fig11_pareto.png")
fig.savefig(out_path)
plt.close(fig)
print(f"Saved {out_path}")
print()
print("Numerical comparison (relative to baseline):")
print(f"{'Config':<10}{'Mass(g)':<10}{'ΔMass%':<10}{'σ_LC2':<10}{'σ_LC3':<10}{'SF_yLC3':<10}{'SF_fat':<10}")
print(f"{'Baseline':<10}{base_mass:<10.2f}{0:<10.1f}{base_LC2:<10.3f}{base_LC3:<10.3f}{sigma_y_MPa/base_LC3:<10.2f}{base_fs_fat:<10.2f}")
for i, r in enumerate(records):
    dm = (1 - r['mass_g'] / base_mass) * 100
    sf_y_lc3 = sigma_y_MPa / r['sigma_max_LC3_MPa']
    vf_label = f"Vf={r['Vf']:.2f}"
    print(f"{vf_label:<10}{r['mass_g']:<10.2f}{dm:<10.1f}{r['sigma_max_MPa']:<10.3f}{r['sigma_max_LC3_MPa']:<10.3f}{sf_y_lc3:<10.2f}{r['fs_fatigue']:<10.2f}")
