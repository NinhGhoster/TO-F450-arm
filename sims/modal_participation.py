"""Modal effective mass in the thrust direction.

Which natural frequency matters is not simply "the lowest one".  A mode that
rotor thrust cannot excite is irrelevant to resonance no matter where it sits,
and low-density SIMP regions readily produce such modes — localised wobbles of
near-void material that appear below the real structural modes and would
otherwise be mistaken for the fundamental.

The standard discriminator is modal effective mass.  For mode k with shape
phi_k, mass matrix M and a unit rigid-body translation r in the excitation
direction,

    M_eff,k = (phi_k^T M r)^2 / (phi_k^T M phi_k)

and dividing by r^T M r expresses it as a fraction of the total mass.  A mode
the thrust can drive carries a large fraction; an artifact carries almost none.

This module provides that calculation for both model families used in the
paper — the voxel-hex designs and the tet-meshed OEM arm — so the two are
judged by the same criterion.

Usage::

    ./venv/bin/python -m sims.modal_participation
"""
from __future__ import annotations

import numpy as np

# A mode carrying at least this fraction of the total mass is treated as
# excitable.  The gap between the two populations is wide — in the v6 sweep
# the excitable modes carry 53-79 % and the rest 0.0-4 % — so nothing in this
# study is sensitive to the exact value.
EXCITABLE_FRAC = 0.05


def effective_mass_voxel(fem, geom, rho_active, mode_shapes, tip_mass_kg,
                         direction=2):
    """Effective-mass fraction per mode for a voxel design.

    Uses the same consistent mass matrix the eigensolve used, so the numbers
    are exactly those of the model that produced the modes.
    """
    M = fem.assemble_M(rho_active, tip_mass_kg=tip_mass_kg,
                       tip_mass_nodes=geom.motor_mount_top_nodes())
    r = np.zeros(geom.n_dofs)
    r[direction::3] = 1.0
    fd = fem.free_dofs
    Mf, rf = M[fd][:, fd], r[fd]
    total = float(rf @ (Mf @ rf))
    out = []
    for k in range(mode_shapes.shape[1]):
        p = mode_shapes[fd, k]
        gm = float(p @ (Mf @ p))
        L = float(p @ (Mf @ rf))
        out.append(L * L / max(gm, 1e-30) / total)
    return np.array(out)


def effective_mass_tet(points, tets, mode_list, node_ids, rho_t=9.3e-10,
                       tip_nodes=None, tip_mass_kg=0.060, direction=2):
    """Effective-mass fraction per mode for a tet mesh.

    CalculiX does not export its mass matrix, so a lumped approximation is
    used: a quarter of each tetrahedron's mass to each of its four nodes, plus
    the tip mass spread over the same node set the deck loads it onto.  For
    separating excitable modes from artifacts the lumped and consistent
    formulations agree closely, because the quantity is an integral over the
    whole structure rather than a local one.
    """
    c = points[tets]
    vol = np.abs(np.einsum("ij,ij->i", c[:, 1] - c[:, 0],
                           np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]))) / 6.0
    m = np.zeros(len(points))
    np.add.at(m, tets.ravel(), np.repeat(vol * rho_t / 4.0, 4))
    struct_g = m.sum() * 1e6
    if tip_nodes is not None and len(tip_nodes):
        m[tip_nodes] += (tip_mass_kg * 1e-3) / len(tip_nodes)   # kg -> tonnes
    total = m.sum()
    out = []
    for md in mode_list:
        u = np.array([md[n] for n in node_ids])
        gm = float((m[:, None] * u * u).sum())
        L = float((m * u[:, direction]).sum())
        out.append(L * L / max(gm, 1e-30) / total)
    return np.array(out), struct_g, total * 1e6


def first_excitable(freqs, eff, thresh=EXCITABLE_FRAC):
    """(frequency, effective-mass fraction, index) of the lowest excitable mode."""
    for k, (f, e) in enumerate(zip(freqs, eff)):
        if e >= thresh:
            return float(f), float(e), k
    return None, None, None


if __name__ == "__main__":
    print(__doc__)
    print(f"excitable threshold: {EXCITABLE_FRAC:.0%} of total mass")
