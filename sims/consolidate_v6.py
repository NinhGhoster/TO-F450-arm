"""Collect the v6 sweep into the numbers the manuscript quotes.

Reads every ``freq_vfNN_omega0500.pkl`` in ``QARM_RESULTS_DIR`` (default
``results_3d_v6``) plus the OEM reference from ``notes/oem_baseline.json``,
and prints the mass-frequency table together with the two things that decide
whether a design is acceptable: where its first mode sits relative to the
rotor excitation band, and whether it still passes the structural checks.

Every value is read from the saved run.  Nothing here is recomputed from
memory or carried over from an earlier revision.

Usage::

    ./venv/bin/python -m sims.consolidate_v6
"""
from __future__ import annotations

import glob
import json
import os
import pickle

import numpy as np

from sims.modal_participation import (EXCITABLE_FRAC, effective_mass_voxel,
                                      first_excitable)

RESULTS = os.environ.get("QARM_RESULTS_DIR", "results_3d_v6")
OUT = "notes/v6_numbers.md"

# Rotor excitation band and admissible windows, from manuscript section 1.5.
F_1P = (83.0, 133.0)
F_BPF = (167.0, 267.0)
MARGIN = 0.20
SUB_MAX = F_1P[0] * (1 - MARGIN)      # 66.4 Hz
SUPER_MIN = F_BPF[1] * (1 + MARGIN)   # 320.4 Hz


def verdict(w):
    """Where this first mode sits relative to the excitation band."""
    if w <= SUB_MAX:
        return "sub-critical, clear"
    if w >= SUPER_MIN:
        return "super-critical, clear"
    return "IN BAND"


def rows():
    from src.fem import FEM3D, Material
    from src.mesh3d import ArmGeometry3D
    geom = ArmGeometry3D()
    fem = FEM3D(geom, Material())
    out = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "freq_vf*_omega*.pkl"))):
        d = pickle.load(open(f, "rb"))
        r = d["record"]
        h = d["history"]
        w = np.array(h["omega_1_Hz"], dtype=float)
        freqs = np.asarray(d["modal"]["frequencies_Hz"], dtype=float)
        # The lowest mode is not necessarily the one that matters: low-density
        # SIMP regions produce localised modes below the structural ones.
        # Rank by what rotor thrust can actually excite.
        eff = effective_mass_voxel(fem, geom, d["rho"], d["modal"]["mode_shapes"],
                                   tip_mass_kg=d["modal"].get("tip_mass_kg", 0.060))
        w_exc, e_exc, k_exc = first_excitable(freqs, eff)
        out.append(dict(
            w_exc=w_exc, e_exc=e_exc, k_exc=k_exc,
            eff=[float(x) for x in eff[:6]],
            vf=float(r["Vf"]), mass_g=float(r["mass_g"]),
            # The validation eigensolve, not the tracked value from the last
            # optimiser iteration — the two differ slightly and only the
            # validation run is an unambiguous first mode.
            w1=float(freqs[0]), w2=float(freqs[1]), w3=float(freqs[2]),
            w1_tracked_final=float(w[-1]), w1_tracked_best=float(np.nanmax(w)),
            best_iter=int(np.nanargmax(w)) + 1, n_iter=int(r["n_iter"]),
            sf_y=float(r["sf_yield_LC3"]), sf_f=float(r["fs_fatigue"]),
            wall_min=float(r["wall_time_min"]),
            vm3=float(r["sigma_max_LC3_MPa"]),
        ))
    return sorted(out, key=lambda x: x["vf"])


def main():
    rs = rows()
    oem = json.load(open("notes/oem_baseline.json"))
    oem_lc3 = next(c for c in oem["cases"] if c["tag"] == "LC3_landing")

    print("=" * 96)
    print(f"v6 modal-constrained sweep  ({RESULTS})    target omega_1 >= 500 Hz")
    print(f"excitation band {F_1P[0]:.0f}-{F_BPF[1]:.0f} Hz; admissible: "
          f"<= {SUB_MAX:.0f} Hz or >= {SUPER_MIN:.0f} Hz")
    print("=" * 96)
    print(f"{'design':<22}{'mass':>8}{'excitable':>9}{'eff.m':>8}"
          f"{'nominal':>8}{'SF_y':>8}{'SF_fat':>8}   {'verdict':<22}")
    print(f"{'':<22}{'(g)':>8}{'w (Hz)':>9}{'(%)':>8}"
          f"{'w1 (Hz)':>8}{'(LC3)':>8}{'':>8}")
    print("-" * 96)
    print(f"{'OEM arm (CAD tets)':<22}{oem['mass_g']:>8.1f}"
          f"{oem['first_excitable_Hz']:>9.1f}{oem['first_excitable_frac']*100:>8.1f}"
          f"{oem['frequencies_Hz'][0]:>8.1f}{oem_lc3['sf_yield']:>8.2f}"
          f"{oem['sf_fatigue']:>8.2f}   {verdict(oem['first_excitable_Hz']):<22}")
    print("-" * 96)
    for r in rs:
        print(f"{'V_f = ' + format(r['vf'], '.3f'):<22}{r['mass_g']:>8.1f}"
              f"{r['w_exc']:>9.1f}{r['e_exc']*100:>8.1f}{r['w1']:>8.1f}"
              f"{r['sf_y']:>8.1f}{r['sf_f']:>8.1f}   {verdict(r['w_exc']):<22}")
    print("-" * 96)

    best = max(rs, key=lambda r: r["w_exc"])
    print(f"\nhighest excitable mode in the sweep: {best['w_exc']:.1f} Hz at V_f = "
          f"{best['vf']:.3f} ({best['mass_g']:.1f} g)")
    print(f"super-critical window needs {SUPER_MIN:.0f} Hz — short by "
          f"{SUPER_MIN - best['w_exc']:.0f} Hz")
    print(f"designs reaching the 500 Hz target: "
          f"{sum(1 for r in rs if r['w_exc'] >= 500)} of {len(rs)}")
    print(f"designs inside the excitation band: "
          f"{sum(1 for r in rs if verdict(r['w_exc']) == 'IN BAND')} of {len(rs)}")
    print(f"\n(nominal first modes at V_f = "
          + ", ".join(f"{r['vf']:.3f}" for r in rs if r['k_exc'] and r['k_exc'] > 0)
          + " carry <"
          + f"{EXCITABLE_FRAC:.0%} effective mass and are not thrust-excitable)")

    # The augmented Lagrangian is only useful while it is still improving the
    # constrained quantity; flag any run where it stopped doing so.
    print("\noptimiser trajectory (tracked mode 1 during the run):")
    for r in rs:
        drop = r["w1_tracked_best"] - r["w1_tracked_final"]
        note = (f"  <-- peaked at iter {r['best_iter']}, then lost {drop:.1f} Hz"
                if drop > 1.0 else "")
        print(f"  V_f = {r['vf']:.3f}: final {r['w1_tracked_final']:6.1f} Hz, "
              f"best {r['w1_tracked_best']:6.1f} Hz{note}")

    os.makedirs("notes", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("# v6 sweep — numbers as run\n\n")
        fh.write(f"Results dir `{RESULTS}`, target omega_1 >= 500 Hz, "
                 f"60 g tip mass.\nExcitation band {F_1P[0]:.0f}-{F_BPF[1]:.0f} Hz; "
                 f"admissible <= {SUB_MAX:.0f} Hz or >= {SUPER_MIN:.0f} Hz.\n\n")
        fh.write("| Design | Mass (g) | First thrust-excitable mode (Hz) | Its effective "
                 "mass (%) | Nominal omega_1 (Hz) | SF_yield (LC3) | SF_fatigue | Verdict |"
                 "\n|---|---|---|---|---|---|---|---|\n")
        fh.write(f"| OEM arm (CAD tets) | {oem['mass_g']:.1f} | "
                 f"{oem['first_excitable_Hz']:.1f} | {oem['first_excitable_frac']*100:.1f} | "
                 f"{oem['frequencies_Hz'][0]:.1f} | {oem_lc3['sf_yield']:.2f} | "
                 f"{oem['sf_fatigue']:.2f} | {verdict(oem['first_excitable_Hz'])} |\n")
        for r in rs:
            fh.write(f"| V_f = {r['vf']:.3f} | {r['mass_g']:.1f} | {r['w_exc']:.1f} | "
                     f"{r['e_exc']*100:.1f} | {r['w1']:.1f} | {r['sf_y']:.1f} | "
                     f"{r['sf_f']:.1f} | {verdict(r['w_exc'])} |\n")
        fh.write(f"\nHighest thrust-excitable mode: {best['w_exc']:.1f} Hz at V_f = "
                 f"{best['vf']:.3f} ({best['mass_g']:.1f} g); super-critical window "
                 f"needs {SUPER_MIN:.0f} Hz.\n")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
