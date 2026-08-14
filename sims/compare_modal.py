"""Compare CalculiX eigenfrequencies against the in-house solver.

Reads the ``.dat`` file CalculiX writes for a ``*FREQUENCY`` step and puts its
frequencies next to the ones stored in ``results_3d/freq_vfNN_omega0500.pkl``.
Nothing here recomputes anything — it only reports what two independent codes
produced for the same problem.
"""
from __future__ import annotations

import argparse
import pickle
import re

import numpy as np


def read_ccx_frequencies(dat_path: str) -> np.ndarray:
    """Cycles/time (Hz) from a CalculiX .dat eigenvalue block."""
    txt = open(dat_path, errors="ignore").read()
    m = re.search(r"E I G E N V A L U E   O U T P U T(.*?)(?:\n\s*\n\s*\n|\Z)",
                  txt, re.S)
    if not m:
        raise RuntimeError(f"no eigenvalue block in {dat_path}")
    freqs = []
    for line in m.group(1).splitlines():
        parts = line.split()
        # mode, eigenvalue, omega(rad/time), f(cycles/time), imag
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                freqs.append(float(parts[3]))
            except ValueError:
                pass
    if not freqs:
        raise RuntimeError(f"eigenvalue block in {dat_path} had no rows")
    return np.array(freqs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vf", type=float, default=0.50)
    ap.add_argument("--dat", nargs="*", default=None,
                    help="CalculiX .dat files (default: both decks)")
    a = ap.parse_args()

    tag = f"vf{int(a.vf*100):02d}"
    with open(f"results_3d/freq_{tag}_omega0500.pkl", "rb") as fh:
        inhouse = pickle.load(fh)["modal"]["frequencies_Hz"]

    dats = a.dat or [f"verification/{tag}_solid.dat",
                     f"verification/{tag}_simp.dat"]

    print(f"Modal verification — V_f = {a.vf:.2f}, tip mass 60 g\n")
    cols, results = ["in-house"], {}
    for d in dats:
        try:
            results[d.split("/")[-1].replace(".dat", "")] = \
                read_ccx_frequencies(d)
            cols.append(d.split("_")[-1].replace(".dat", "") + " (CalculiX)")
        except Exception as e:
            print(f"  [skip {d}: {e}]")

    head = f"{'mode':>5}" + "".join(f"{c:>22}" for c in cols)
    print(head)
    print("-" * len(head))
    n = max([len(inhouse)] + [len(v) for v in results.values()] or [0])
    for k in range(min(n, 6)):
        row = f"{k+1:>5}" + f"{inhouse[k]:>22.1f}"
        for key in results:
            v = results[key]
            if k < len(v):
                d = 100.0 * (v[k] - inhouse[k]) / inhouse[k]
                row += f"{v[k]:>14.1f} ({d:+5.1f}%)"
            else:
                row += f"{'-':>22}"
        print(row)

    print("\nin-house omega_1 = %.1f Hz (the manuscript's headline)" % inhouse[0])
    for key, v in results.items():
        print(f"{key:>18} omega_1 = {v[0]:.1f} Hz  "
              f"({100*(v[0]-inhouse[0])/inhouse[0]:+.1f} %)")


if __name__ == "__main__":
    main()
