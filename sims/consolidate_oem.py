"""Collect the OEM-arm CalculiX results into one table.

Five static load cases and one modal run were solved on a tet mesh of the real
F450 arm CAD, with the same six bolted cylinders restrained and the same motor
thrust path as the topology model.  This script reads the ``.frd`` result files
back, applies the same yield and Goodman-corrected fatigue post-processing the
topology designs get, and writes ``notes/oem_baseline.json`` plus a console
table.

The point is that the OEM arm and every optimised design are then judged by
identical criteria, so the comparison in the manuscript is a like-for-like one.

Usage::

    ./venv/bin/python -m sims.consolidate_oem
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from src.fatigue import fatigue_life_field
from sims.read_frd import read_frd, von_mises
from sims.modal_participation import effective_mass_tet, first_excitable

FEA = "fea_oem"
OUT = "notes/oem_baseline.json"

SIGMA_Y = 38.0        # MPa, PA12 SLS
SIGMA_U = 48.0        # MPa

CASES = [
    ("LC1_hover",      "Hover",              "F_z = 2.94 N"),
    ("LC2_maneuver",   "Maneuver (2 g)",     "F_z = 5.88 N"),
    ("LC3_landing",    "Hard landing",       "F_z = -14.7 N"),
    ("LC4_banked",     "Banked turn",        "F = (0, 2.94, 5.09) N"),
    ("LC5_proptorque", "Propeller torque",   "M_z = 0.10 N.m"),
]


def static_case(tag):
    """Peak von Mises, yield safety factor and peak displacement for one case."""
    disp, stress = read_frd(f"{FEA}/{tag}/ArmMesh.frd")
    nodes = np.array(sorted(stress))
    s = np.array([stress[n] for n in nodes])
    vm = von_mises(s)                       # MPa (deck is in N/mm^2)
    d = np.array([disp[n] for n in nodes])
    dmag = np.linalg.norm(d, axis=1)        # mm
    i = int(np.argmax(vm))
    return dict(
        max_vm_MPa=float(vm.max()),
        sf_yield=float(SIGMA_Y / max(vm.max(), 1e-9)),
        max_disp_mm=float(dmag.max()),
        peak_node=int(nodes[i]),
        n_nodes=len(nodes),
        vm_field=vm,
    )


def modal():
    """First N natural frequencies, read from the CalculiX .dat file."""
    txt = open(f"{FEA}/MODAL/ArmMesh.dat", errors="ignore").read()
    # CalculiX prints "E I G E N V A L U E   O U T P U T" then a table whose
    # columns are mode / eigenvalue / rad-per-time / CYCLES-per-time / imag.
    # Column 3 is the one we want; column 4 is the imaginary part and is zero
    # for an undamped run, so indexing it silently yields a table of noughts.
    freqs = []
    grab = False
    for line in txt.splitlines():
        if "E I G E N V A L U E   O U T P U T" in line:
            grab = True
            continue
        if grab:
            p = line.split()
            if len(p) == 5 and p[0].isdigit():
                freqs.append(float(p[3]))
            elif freqs:
                break
    return freqs


def arm_mass_g():
    """Mass of the meshed OEM solid, summed from tet element volumes.

    CalculiX does not print a total mass, so it is integrated here from the
    deck itself.  That keeps the figure tied to the mesh that was actually
    solved rather than to a CAD property read separately.
    """
    p = f"{FEA}/LC2_maneuver/ArmMesh.inp"
    lines = open(p, errors="ignore").read().splitlines()
    rho = 9.3e-10                                # t/mm^3, PA12 at 930 kg/m^3
    for i, l in enumerate(lines):
        if l.upper().startswith("*DENSITY"):
            rho = float(lines[i + 1].split(",")[0])
            break

    coords, conn, block = {}, [], None
    for l in lines:
        if l.startswith("**"):
            continue
        if l.startswith("*"):
            u = l.upper()
            block = ("node" if u.startswith("*NODE") else
                     "elem" if u.startswith("*ELEMENT") else None)
            continue
        if not l.strip():
            continue
        p_ = [x.strip() for x in l.split(",") if x.strip()]
        if block == "node" and len(p_) >= 4:
            coords[int(p_[0])] = (float(p_[1]), float(p_[2]), float(p_[3]))
        elif block == "elem" and len(p_) >= 5:
            conn.append([int(x) for x in p_[1:5]])   # first 4 = tet corners

    if not conn:
        return None
    c = np.array([[coords[n] for n in e] for e in conn])
    # Signed volume of a tetrahedron from its four corners.
    v = np.abs(np.einsum("ij,ij->i",
                         c[:, 1] - c[:, 0],
                         np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]))) / 6.0
    return float(v.sum() * rho * 1e6)            # tonnes -> grams


def main():
    rows, fields = [], {}
    for tag, label, load in CASES:
        r = static_case(tag)
        fields[tag] = r.pop("vm_field")
        r.update(tag=tag, label=label, load=load)
        rows.append(r)

    # Fatigue: the hover<->maneuver cycle is the one the airframe actually
    # sees every flight, so that pair sets the alternating and mean stress.
    fat = fatigue_life_field(fields["LC1_hover"] * 1e6,
                             fields["LC2_maneuver"] * 1e6, SIGMA_U * 1e6)
    sf_f = float(np.min(fat["factor_of_safety"]))
    n_cyc = float(np.min(fat["life"]))

    freqs = modal()
    mass = arm_mass_g()

    # Which of those modes rotor thrust can actually drive.  Reported for the
    # OEM arm and for every optimised design by the same criterion, so the
    # resonance comparison is like-for-like.
    from sims.render_oem_figures import read_mesh, read_frd_blocks
    grid, ids = read_mesh()
    lut = {n: i for i, n in enumerate(ids)}
    tips = []
    txt = open(f"{FEA}/MODAL/ArmMesh.inp", errors="ignore").read().splitlines()
    i0 = next(k for k, l in enumerate(txt) if l.startswith("*ELEMENT, TYPE=MASS"))
    for l in txt[i0 + 1:]:
        if l.startswith("*"):
            break
        if l.strip() and not l.startswith("**"):
            pp = [x.strip() for x in l.split(",") if x.strip()]
            if len(pp) >= 2:
                tips.append(int(pp[1]))
    tip_idx = [lut[n] for n in set(tips) if n in lut]
    eff, struct_g, total_g = effective_mass_tet(
        grid.points, grid.cells_dict[10],
        read_frd_blocks(f"{FEA}/MODAL/ArmMesh.frd", "DISP", 3), ids,
        tip_nodes=tip_idx)
    w_exc, e_exc, k_exc = first_excitable(freqs, eff)

    print("=" * 74)
    print("OEM F450 arm — CalculiX on the CAD solid (tet mesh)")
    print("=" * 74)
    print(f"{'case':<18}{'load':<24}{'max vM':>9}{'SF_y':>8}{'max u':>9}")
    print(f"{'':<18}{'':<24}{'(MPa)':>9}{'':>8}{'(mm)':>9}")
    print("-" * 74)
    for r in rows:
        print(f"{r['label']:<18}{r['load']:<24}"
              f"{r['max_vm_MPa']:>9.3f}{r['sf_yield']:>8.2f}"
              f"{r['max_disp_mm']:>9.4f}")
    print("-" * 74)
    if sf_f is not None:
        print(f"fatigue (hover<->maneuver, Goodman):  SF = {sf_f:.2f}"
              + (f"   N = {n_cyc:.3g} cycles" if n_cyc else ""))
    if freqs:
        print("first natural frequencies (Hz): "
              + ", ".join(f"{f:.1f}" for f in freqs[:6]))
        print("modal effective mass in z (%): "
              + ", ".join(f"{e*100:.2f}" for e in eff[:6]))
        print(f"first THRUST-EXCITABLE mode: {w_exc:.1f} Hz "
              f"(mode {k_exc+1}, {e_exc*100:.1f} % of total mass)")
    if mass:
        print(f"meshed mass: {mass:.1f} g")
    print(f"mesh: {rows[0]['n_nodes']} nodes")

    os.makedirs("notes", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(dict(cases=rows, frequencies_Hz=freqs[:6],
                       eff_mass_z=[float(x) for x in eff[:6]],
                       first_excitable_Hz=w_exc, first_excitable_frac=e_exc,
                       first_excitable_mode=(k_exc + 1) if k_exc is not None else None,
                       sf_fatigue=sf_f, N_cycles=n_cyc,
                       mass_g=mass, sigma_y_MPa=SIGMA_Y,
                       sigma_u_MPa=SIGMA_U), fh, indent=2)
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
