"""Build the LC4 (banked) and LC5 (propeller torque) decks for the OEM arm.

FreeCAD's ConstraintForce takes a single direction from a referenced face and
has no moment constraint, so neither of these two cases can be expressed in
the GUI. Both are written here by rewriting the ``*CLOAD`` block of the LC2
deck, leaving mesh, material, boundary conditions and step definition byte-for
-byte identical — so any difference in the results is the load and nothing
else.

LC4  F = (0, 2.94, 5.0922) N — banked maneuver, reusing LC2's nodal weights
     so the load is spread over the motor screw holes the same way.

LC5  Mz = 0.10 N·m about the motor axis. Applied as tangential nodal forces
     scaled by radius, f_i = k · r_i, which is the traction distribution of a
     rigid rotation. With k = Mz / Σ r_j² the moment sums exactly to Mz and
     the net force cancels.

Usage::

    ./venv/bin/python -m sims.make_lc45
"""
from __future__ import annotations

import os
import shutil

import numpy as np

SRC = "fea_oem/LC2_maneuver/ArmMesh.inp"
OUT = "fea_oem"


def parse(path):
    lines = open(path, errors="ignore").read().splitlines()
    coords, in_node = {}, False
    for l in lines:
        if l.upper().startswith("*NODE"):
            in_node = True
            continue
        if l.startswith("*"):
            in_node = False
        if in_node and l.strip() and not l.startswith("**"):
            p = [x.strip() for x in l.split(",")]
            if len(p) >= 4:
                coords[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
    i = next(k for k, l in enumerate(lines) if l.startswith("*CLOAD"))
    j = i + 1
    weights = {}
    while j < len(lines):
        l = lines[j]
        if l.startswith("**"):
            j += 1
            continue
        if l.startswith("*"):
            break
        if l.strip():
            p = [x.strip() for x in l.split(",")]
            if len(p) >= 3 and p[1] == "3":
                weights[int(p[0])] = float(p[2])
        j += 1
    return lines, i, j, coords, weights


def write(name, block):
    lines, i, j, _, _ = parse(SRC)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    out = lines[:i] + block + lines[j:]
    p = os.path.join(d, "ArmMesh.inp")
    open(p, "w").write("\n".join(out) + "\n")
    return p


def main():
    lines, i, j, coords, w = parse(SRC)
    nodes = sorted(w)
    tot = sum(w[n] for n in nodes)
    frac = {n: w[n] / tot for n in nodes}          # LC2's own distribution
    print(f"motor-mount nodes carrying load: {len(nodes)}  (LC2 total {tot:.4f} N)")

    # ---- LC4: banked maneuver -------------------------------------------
    Fy, Fz = 2.94, 5.0922
    blk = ["*CLOAD", "** LC4 banked maneuver: F = (0, 2.94, 5.0922) N"]
    for n in nodes:
        blk.append(f"{n},2,{Fy * frac[n]:.10e}")
        blk.append(f"{n},3,{Fz * frac[n]:.10e}")
    p4 = write("LC4_banked", blk)
    print(f"LC4 -> {p4}   sum Fy={Fy:.4f}  sum Fz={Fz:.4f}")

    # ---- LC5: propeller torque ------------------------------------------
    Mz = 100.0                                     # 0.10 N.m in N.mm
    xy = np.array([coords[n][:2] for n in nodes])
    c = xy.mean(axis=0)                            # motor axis
    d = xy - c
    r2 = (d ** 2).sum()
    k = Mz / r2
    blk = ["*CLOAD",
           "** LC5 propeller torque: Mz = 0.10 N.m about the motor axis at "
           f"({c[0]:.2f}, {c[1]:.2f}), applied as tangential nodal forces"]
    mz_chk = fx_chk = fy_chk = 0.0
    for n, (dx, dy) in zip(nodes, d):
        fx, fy = -k * dy, k * dx
        blk.append(f"{n},1,{fx:.10e}")
        blk.append(f"{n},2,{fy:.10e}")
        mz_chk += dx * fy - dy * fx
        fx_chk += fx
        fy_chk += fy
    p5 = write("LC5_proptorque", blk)
    print(f"LC5 -> {p5}")
    print(f"     applied Mz = {mz_chk:.4f} N.mm (target {Mz:.1f}); "
          f"net force ({fx_chk:.2e}, {fy_chk:.2e}) N — cancels as it should")


if __name__ == "__main__":
    main()
