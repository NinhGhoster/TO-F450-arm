"""Emit the manuscript's numeric tables straight from the saved results.

Every number the manuscript quotes for the baseline and the V_f sweep is
derived here, so the tables in the paper can be regenerated rather than
retyped.  Retyping is how the v4 draft ended up claiming a 5.70 MPa peak
stress that its own Table 3 contradicted.

Usage::

    QARM_RESULTS_DIR=results_3d_v5 ./venv/bin/python -m sims.report_numbers
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np

RESULTS = os.environ.get("QARM_RESULTS_DIR", "results_3d")
LC_ORDER = ["LC1_hover", "LC2_maneuver", "LC3_landing", "LC4_banked",
            "LC5_proptorque"]
LC_LABEL = {"LC1_hover": "LC1 Hover", "LC2_maneuver": "LC2 Maneuver",
            "LC3_landing": "LC3 Hard landing", "LC4_banked": "LC4 Banked",
            "LC5_proptorque": "LC5 Prop torque"}
SIGMA_Y_MPA = 38.0


def load(name):
    with open(os.path.join(RESULTS, name), "rb") as f:
        return pickle.load(f)


def baseline_tables():
    b = load("raw_baseline_modal.pkl")
    mi = b.get("mesh_info", {})
    print(f"### Mesh\n")
    print(f"{mi.get('nx')} x {mi.get('ny')} x {mi.get('nz')} = "
          f"{mi.get('n_elements')} voxels, {mi.get('n_active_elements')} active, "
          f"{mi.get('n_nodes')} nodes, {mi.get('n_dofs')} DoF\n")

    print("### Table 3 — baseline static response\n")
    hdr = "| Quantity | " + " | ".join(LC_LABEL[k] for k in LC_ORDER) + " |"
    print(hdr)
    print("|" + "---|" * (len(LC_ORDER) + 1))
    rows = {
        "Peak von Mises (MPa)":
            lambda s: f"{np.nanmax(s['vm'])/1e6:.4g}",
        "Max total deformation (μm)":
            lambda s: f"{np.nanmax(np.abs(s['disp_node']))*1e6:.1f}",
        "Max principal stress (MPa)":
            lambda s: f"{np.nanmax(s['sigma_max'])/1e6:.4g}",
        "Yield safety factor":
            lambda s: f"{SIGMA_Y_MPA/(np.nanmax(s['vm'])/1e6):.1f}",
    }
    for label, fn in rows.items():
        print(f"| {label} | " +
              " | ".join(fn(b["static"][k]) for k in LC_ORDER) + " |")

    f = b["fatigue"]
    print(f"\nWorst Goodman sigma_alt_eq = "
          f"{np.nanmax(f['sigma_alt_eq'])/1e6:.4g} MPa; "
          f"min fatigue SF = {np.nanmin(f['factor_of_safety']):.1f}")

    print("\n### Table 4 — baseline modal\n")
    print("| Mode | Frequency (Hz) |")
    print("|---|---|")
    for i, fr in enumerate(b["modal"]["frequencies_Hz"], start=1):
        print(f"| ω{i} | {fr:.1f} |")
    return b


def pareto_table():
    path = os.path.join(RESULTS, "summary_freq.json")
    if not os.path.exists(path):
        print("\n(no summary_freq.json yet — run sims.consolidate_freq_3d)")
        return None
    rows = json.load(open(path))
    print("\n### Table 5 — mass-frequency Pareto\n")
    print("| Config | Mass (g) | ω₁ (Hz) | ω₂ (Hz) | ω₃ (Hz) | "
          "SF_yield (LC3) | SF_fat | ω₁ ≥ target |")
    print("|" + "---|" * 8)
    for r in rows:
        tgt = r.get("omega_target_Hz")
        met = ("—" if not tgt else
               ("yes" if r["omega_1_Hz"] >= tgt
                else f"no ({tgt - r['omega_1_Hz']:.0f} Hz short)"))
        print(f"| {r['label']} | {r['mass_g']:.1f} | {r['omega_1_Hz']:.1f} | "
              f"{r.get('omega_2_Hz', float('nan')):.1f} | "
              f"{r.get('omega_3_Hz', float('nan')):.1f} | "
              f"{r.get('sf_yield_LC3', float('nan')):.1f} | "
              f"{r.get('fs_fatigue', float('nan')):.1f} | {met} |")

    swept = [r for r in rows if r.get("omega_target_Hz")]
    if swept:
        best = max(swept, key=lambda r: r["omega_1_Hz"])
        base = next((r for r in rows if not r.get("omega_target_Hz")), None)
        print(f"\n**Derived statements** (recompute, do not assume):")
        print(f"- best omega_1 in the sweep: {best['label']} at "
              f"{best['omega_1_Hz']:.1f} Hz, mass {best['mass_g']:.1f} g")
        if base:
            better = [r for r in swept
                      if r["omega_1_Hz"] > base["omega_1_Hz"]]
            print(f"- baseline omega_1 = {base['omega_1_Hz']:.1f} Hz "
                  f"at {base['mass_g']:.1f} g")
            print(f"- designs beating the baseline omega_1 at lower mass: "
                  f"{[r['label'] for r in better] or 'none'}")
        tgt = swept[0]["omega_target_Hz"]
        print(f"- any design reaching omega_target = {tgt:.0f} Hz? "
              f"{'YES' if any(r['omega_1_Hz'] >= tgt for r in swept) else 'NO'}")
        print(f"- monotonic in V_f? "
              f"{'yes' if all(a['omega_1_Hz'] <= b['omega_1_Hz'] for a, b in zip(swept, swept[1:])) else 'NO — non-monotonic'}")
    return rows


if __name__ == "__main__":
    print(f"# Numbers from {RESULTS}\n")
    baseline_tables()
    pareto_table()
