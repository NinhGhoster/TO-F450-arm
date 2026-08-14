"""Read a CalculiX .frd result file: displacements and von Mises stress.

Standalone on purpose — it does not need FreeCAD, so the same reader works
for decks written by FreeCAD's FEM workbench and for the ones
``sims/export_abaqus_inp.py`` writes from the voxel model. That is what lets
the OEM arm and the topology-optimised designs be compared on equal terms.

.frd stores raw stress components (SXX, SYY, SZZ, SXY, SYZ, SZX); von Mises
is computed here rather than read, since CalculiX does not store it.

Usage::

    ./venv/bin/python -m sims.read_frd fea_oem/LC2_maneuver/ArmMesh.frd
"""
from __future__ import annotations

import re
import sys

import numpy as np


def _values(line: str, n: int):
    """Parse the numeric fields of a '-1' data record.

    .frd is nominally fixed-width (node in 10 chars, then 12-char fields) but
    writers vary, so fall back to whitespace splitting when the widths do not
    produce n clean floats.
    """
    body = line[3:]
    try:
        node = int(body[:10])
        vals = [float(body[10 + 12 * i: 22 + 12 * i]) for i in range(n)]
        return node, vals
    except ValueError:
        p = line.split()
        return int(p[1]), [float(x) for x in p[2:2 + n]]


def read_frd(path: str):
    disp: dict[int, list] = {}
    stress: dict[int, list] = {}
    block = None
    ncomp = 0
    for line in open(path, errors="ignore"):
        if line.startswith(" -4"):
            name = line.split()[1].upper()
            if name.startswith("DISP"):
                block, ncomp = "disp", 3
            elif name.startswith("STRESS"):
                block, ncomp = "stress", 6
            else:
                block = None
            continue
        if line.startswith(" -3"):
            block = None
            continue
        if block and line.startswith(" -1"):
            node, vals = _values(line, ncomp)
            (disp if block == "disp" else stress)[node] = vals
    return disp, stress


def von_mises(s):
    sxx, syy, szz, sxy, syz, szx = np.asarray(s).T
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                   + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))


def summarise(path: str, sigma_y_mpa: float = 38.0):
    disp, stress = read_frd(path)
    out = {"file": path, "nodes": len(disp)}
    if disp:
        d = np.array([disp[k] for k in sorted(disp)])
        mag = np.linalg.norm(d, axis=1)
        out["u_max_mm"] = float(mag.max())
    if stress:
        vm = von_mises(np.array([stress[k] for k in sorted(stress)]))
        out["vm_max_MPa"] = float(vm.max())
        out["vm_mean_MPa"] = float(vm.mean())
        out["sf_yield"] = float(sigma_y_mpa / vm.max()) if vm.max() > 0 else float("inf")
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = summarise(p)
        print(f"{p}")
        print(f"   nodes {r['nodes']}   |u|max {r.get('u_max_mm', 0):.4f} mm   "
              f"vM max {r.get('vm_max_MPa', 0):.4f} MPa   "
              f"SF_yield {r.get('sf_yield', 0):.1f}")
