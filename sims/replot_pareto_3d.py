"""Regenerate Pareto figure with the 3D-bridge results."""
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "savefig.dpi": 220, "savefig.bbox": "tight",
})

root = os.path.join(os.path.dirname(__file__), "..")
res_dir = os.path.join(root, "results_3d")
fig_dir = os.path.join(root, "figures")

# Load baseline
with open(os.path.join(res_dir, "raw_baseline.pkl"), "rb") as f:
    base = pickle.load(f)
base_LC2 = float(base["static"]["LC2_maneuver"]["vm"].max() / 1e6)
base_LC3 = float(base["static"]["LC3_landing"]["vm"].max() / 1e6)
base_fs_fat = float(base["fatigue"]["factor_of_safety"].min())
# Baseline mass = solid block
base_mass = 67552 * (2e-3 ** 3) * 930.0 * 1e3  # active elements * voxel vol * rho * 1000

records = []
for vf in (0.05, 0.10, 0.15, 0.20):
    with open(os.path.join(res_dir, f"vf_{int(vf*100):02d}.pkl"), "rb") as f:
        d = pickle.load(f)
    records.append(d["record"])

records = sorted(records, key=lambda r: r["Vf"])
sigma_y_MPa = 38.0

Vf = np.array([r["Vf"] for r in records])
mass = np.array([r["mass_g"] for r in records])
sig_LC2 = np.array([r["sigma_max_MPa"] for r in records])
sig_LC3 = np.array([r["sigma_max_LC3_MPa"] for r in records])
fs_fat = np.array([r["fs_fatigue"] for r in records])

all_mass = np.concatenate([[base_mass], mass])
all_LC2 = np.concatenate([[base_LC2], sig_LC2])
all_LC3 = np.concatenate([[base_LC3], sig_LC3])
all_fs_fat = np.concatenate([[base_fs_fat], fs_fat])
all_labels = ["Baseline"] + [f"Vf={v:.2f}" for v in Vf]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

ax = axes[0]
ax.plot(all_mass, all_LC2, "o-", color="C0", label="LC2 (Maneuver)")
ax.plot(all_mass, all_LC3, "s-", color="C3", label="LC3 (Hard landing)")
ax.axhline(sigma_y_MPa, color="grey", ls="--", lw=1,
            label=fr"$\sigma_y$ = {sigma_y_MPa:.0f} MPa")
for i, lab in enumerate(all_labels):
    ax.annotate(lab, (all_mass[i], all_LC3[i]),
                xytext=(4, 4), textcoords="offset points", fontsize=7)
ax.set_xlabel("Arm mass (g)")
ax.set_ylabel("Peak von Mises stress (MPa)")
ax.set_title("Mass vs. peak stress (3D bridge)")
ax.set_yscale("log")
ax.grid(True, which="both", ls=":", alpha=0.5)
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
ax.set_xlabel("Arm mass (g)")
ax.set_ylabel("Factor of safety")
ax.set_title("Mass vs. factor of safety")
ax.set_yscale("log")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(fontsize=7, loc="upper right", ncol=1)

ax = axes[2]
x = np.arange(len(records))
width = 0.30
ax.bar(x - width, mass, width, color="C0", label="Mass (g)")
ax.set_xticks(x); ax.set_xticklabels([f"Vf={v:.2f}" for v in Vf])
ax.set_ylabel("Mass (g)", color="C0")
ax.tick_params(axis="y", labelcolor="C0")
ax.axhline(base_mass, color="C0", ls=":", lw=1,
            label=f"Baseline {base_mass:.0f} g")
ax.axhline(30, color="C4", ls="--", lw=1, label="OEM ≈ 30 g")
ax.legend(loc="upper left", fontsize=7)
ax2 = ax.twinx()
ax2.bar(x, fs_fat, width, color="C2", alpha=0.7, label="SF fatigue")
ax2.bar(x + width, sigma_y_MPa/sig_LC3, width, color="C3", alpha=0.7,
         label="SF yield (LC3)")
ax2.set_ylabel("Safety factor"); ax2.set_yscale("log")
ax2.axhline(1.5, color="grey", ls=":", lw=1)
ax2.legend(loc="upper right", fontsize=7)
ax.set_title("Configurations")

plt.tight_layout()
fig.savefig(os.path.join(fig_dir, "fig11_pareto.png"))
plt.close(fig)
print("Saved fig11_pareto.png")

print()
print(f"Baseline: mass {base_mass:.1f} g, σ_LC2 {base_LC2:.3f}, σ_LC3 {base_LC3:.3f}, "
      f"SF_yLC3 {sigma_y_MPa/base_LC3:.1f}, SF_fat {base_fs_fat:.1f}")
for r in records:
    print(f"Vf={r['Vf']:.2f}: mass {r['mass_g']:6.2f} g, "
          f"σ_LC2 {r['sigma_max_MPa']:.3f}, σ_LC3 {r['sigma_max_LC3_MPa']:.3f}, "
          f"SF_yLC3 {sigma_y_MPa/r['sigma_max_LC3_MPa']:.2f}, "
          f"SF_fat {r['fs_fatigue']:.2f}")
