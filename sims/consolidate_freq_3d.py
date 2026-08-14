"""
After all ω_target PBS tasks finish, this collects their per-task pickles
into a single `results_3d/summary_freq.json` for use in the manuscript
and to drive Pareto / table figures.
"""
from __future__ import annotations

import os
import json
import pickle
import numpy as np


RESULTS_DIR = os.environ.get("QARM_RESULTS_DIR") or os.path.join(
    os.path.dirname(__file__), "..", "results_3d")


def main():
    rows = []
    baseline_path = os.path.join(RESULTS_DIR, "raw_baseline_modal.pkl")
    if os.path.exists(baseline_path):
        with open(baseline_path, "rb") as f:
            base = pickle.load(f)
        freqs_b = base["modal"]["frequencies_Hz"]
        # Compute baseline mass from mesh info (full solid)
        mi = base["mesh_info"]
        # 2 mm voxels: V_e = (2e-3)^3 = 8e-9 m^3; rho_PA12 = 930 kg/m^3
        V_e = 8e-9
        mass_g = mi["n_active_elements"] * V_e * 930.0 * 1e3
        sf_y_LC3 = float(base["static"]["LC3_landing"]["sf_yield"].min())
        sf_fat = float(base["fatigue"]["factor_of_safety"].min())
        rows.append(dict(
            label="Solid baseline", Vf=1.00, omega_target_Hz=None,
            mass_g=mass_g,
            omega_1_Hz=float(freqs_b[0]),
            omega_2_Hz=float(freqs_b[1]),
            omega_3_Hz=float(freqs_b[2]),
            sf_yield_LC3=sf_y_LC3,
            fs_fatigue=sf_fat,
        ))

    # Sweep is over V_f at the nominal ω_target (500 Hz by default).  If
    # multiple ω_target sweeps are present they will all be aggregated.
    import glob
    pattern = os.path.join(RESULTS_DIR, "freq_vf*_omega*.pkl")
    for p in sorted(glob.glob(pattern)):
        with open(p, "rb") as f:
            d = pickle.load(f)
        r = d["record"]
        rows.append(dict(
            label=(f"V_f = {r['Vf']:.2f}, "
                    f"ω_target = {int(r['omega_target_Hz'])} Hz"),
            Vf=r["Vf"], omega_target_Hz=r["omega_target_Hz"],
            mass_g=r["mass_g"],
            omega_1_Hz=r["omega_1_Hz"],
            omega_2_Hz=r["omega_2_Hz"],
            omega_3_Hz=r["omega_3_Hz"],
            constraint_satisfied=r["constraint_satisfied"],
            sigma_max_LC2_MPa=r["sigma_max_LC2_MPa"],
            sigma_max_LC3_MPa=r["sigma_max_LC3_MPa"],
            sf_yield_LC2=r["sf_yield_LC2"],
            sf_yield_LC3=r["sf_yield_LC3"],
            fs_fatigue=r["fs_fatigue"],
            n_iter=r["n_iter"],
            wall_time_min=r["wall_time_min"],
        ))

    out_path = os.path.join(RESULTS_DIR, "summary_freq.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, default=lambda x: float(x))

    print(f"Wrote {out_path}")
    print()
    fmt = "{:25s} {:>7s} {:>9s} {:>9s} {:>9s} {:>9s}"
    print(fmt.format("design", "Vf", "mass(g)", "ω₁(Hz)", "SF_y(LC3)", "SF_fat"))
    print("-" * 75)
    for r in rows:
        print(fmt.format(
            r["label"],
            f"{r['Vf']:.2f}",
            f"{r['mass_g']:.1f}",
            f"{r['omega_1_Hz']:.1f}",
            f"{r['sf_yield_LC3']:.2f}",
            f"{r['fs_fatigue']:.2f}",
        ))


if __name__ == "__main__":
    main()
