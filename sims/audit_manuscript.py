"""Check the manuscript's headline numbers against the saved results.

Every quantitative claim in a paper is a chance to mistype a digit, and this
manuscript was rewritten around a new set of results, so each number was
re-entered by hand at least once.  This script re-derives the important ones
from ``notes/oem_baseline.json`` and the sweep pickles and confirms the text
says the same thing.

It is a spot check on the claims that carry the argument, not a full parse of
the prose.  A FAIL means the manuscript and the data disagree and one of them
is wrong.

Usage::

    QARM_RESULTS_DIR=results_3d_v6 ./venv/bin/python -m sims.audit_manuscript
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

from sims.consolidate_v6 import SUB_MAX, SUPER_MIN, rows, verdict

PAPER = "paper_v4_modal.md"


def main():
    text = open(PAPER).read()
    rs = {round(r["vf"], 3): r for r in rows()}
    oem = json.load(open("notes/oem_baseline.json"))
    oem_lc3 = next(c for c in oem["cases"] if c["tag"] == "LC3_landing")

    checks = []

    def want(label, value, fmt="{:.1f}"):
        """Assert the formatted value appears somewhere in the manuscript."""
        s = fmt.format(value)
        checks.append((label, s, s in text))

    # --- the OEM reference -------------------------------------------------
    want("OEM mass", oem["mass_g"])
    want("OEM first excitable mode", oem["first_excitable_Hz"])
    want("OEM effective mass %", oem["first_excitable_frac"] * 100)
    want("OEM yield SF (LC3)", oem_lc3["sf_yield"], "{:.2f}")
    want("OEM fatigue SF", oem["sf_fatigue"], "{:.1f}")
    for c in oem["cases"]:
        want(f"OEM peak vM {c['tag']}", c["max_vm_MPa"], "{:.3f}")
    for f in oem["frequencies_Hz"][:3]:
        want(f"OEM mode {f:.1f} Hz", f)

    # --- the sweep ---------------------------------------------------------
    for vf, r in rs.items():
        want(f"V_f={vf} mass", r["mass_g"])
        want(f"V_f={vf} excitable mode", r["w_exc"])
        want(f"V_f={vf} yield SF", r["sf_y"])
        want(f"V_f={vf} fatigue SF", r["sf_f"])

    # --- band arithmetic ---------------------------------------------------
    checks.append(("sub-critical limit 66 Hz", "66", f"{SUB_MAX:.0f}" == "66"))
    checks.append(("super-critical limit 320 Hz", "320",
                   f"{SUPER_MIN:.0f}" == "320"))
    best = max(rs.values(), key=lambda r: r["w_exc"])
    checks.append(("best excitable mode is the one quoted",
                   f"{best['w_exc']:.1f}", f"{best['w_exc']:.1f}" in text))
    checks.append(("shortfall to super-critical window",
                   f"{SUPER_MIN - best['w_exc']:.0f}",
                   f"{SUPER_MIN - best['w_exc']:.0f}" in text))

    # --- claims that must hold for the argument ----------------------------
    all_in_band = all(verdict(r["w_exc"]) == "IN BAND" for r in rs.values())
    checks.append(("every optimised design is in band", "True", all_in_band))
    oem_clear = verdict(oem["first_excitable_Hz"]) == "sub-critical, clear"
    checks.append(("OEM arm is sub-critical and clear", "True", oem_clear))
    none_at_target = all(r["w_exc"] < 500 for r in rs.values())
    checks.append(("no design reaches the 500 Hz target", "True", none_at_target))
    # The two modes the paper says thrust cannot excite.
    z2, z3 = oem["eff_mass_z"][1], oem["eff_mass_z"][2]
    checks.append(("OEM in-band modes carry ~0 % effective mass",
                   f"{z2:.4f}/{z3:.4f}", z2 < 0.005 and z3 < 0.005))

    # --- figure captions ---------------------------------------------------
    # Captions carry numbers too, and they are edited by hand more often than
    # tables are, so they get checked the same way.
    import glob, os, pickle
    from scipy import ndimage
    from src.fem import FEM3D, Material
    from src.mesh3d import ArmGeometry3D
    geom = ArmGeometry3D()
    fem = FEM3D(geom, Material())
    ex, ey, ez = fem.elem_grid_idx.T
    nb, det = [], []
    for f in sorted(glob.glob(os.path.join(os.environ.get(
            "QARM_RESULTS_DIR", "results_3d_v6"), "freq_vf*_omega*.pkl"))):
        rho = pickle.load(open(f, "rb"))["rho"]
        keep = rho > 0.5
        g = np.zeros((geom.nx, geom.ny, geom.nz), bool)
        g[ex[keep], ey[keep], ez[keep]] = True
        lab, n = ndimage.label(g)
        sizes = np.bincount(lab.ravel())[1:]
        nb.append(n)
        det.append(100.0 * (sizes.sum() - sizes.max()) / sizes.sum())
    checks.append(("Fig 9: body-count range quoted",
                   f"{min(nb)}–{max(nb)}",
                   f"between {min(nb)} and {max(nb)} separate solid bodies" in text))
    checks.append(("Fig 9: detached-volume range quoted",
                   f"{min(det):.1f}–{max(det):.1f} %",
                   f"{min(det):.1f}–{max(det):.1f} %" in text))
    lo, hi = min(r["w_exc"] for r in rs.values()), max(r["w_exc"] for r in rs.values())
    checks.append(("Fig 10: FRF peak range quoted",
                   f"{lo:.0f}/{hi:.0f} Hz",
                   f"{lo:.0f} Hz" in text and f"{hi:.0f} Hz" in text))

    width = max(len(c[0]) for c in checks) + 2
    bad = 0
    print("=" * (width + 26))
    print("Manuscript numbers against the saved results")
    print("=" * (width + 26))
    for label, val, ok in checks:
        if not ok:
            bad += 1
        print(f"{'ok  ' if ok else 'FAIL'}  {label:<{width}} {val}")
    print("-" * (width + 26))
    print(f"{len(checks) - bad}/{len(checks)} consistent")
    if bad:
        print(f"\n{bad} value(s) in the manuscript do not match the data.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
