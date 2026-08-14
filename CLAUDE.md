# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this is

The finite-element, modal and topology-optimisation pipeline behind a paper on
the F450 quadcopter arm in SLS polyamide-12. It is a research codebase, not a
library: the point is that every number in the paper can be re-derived from it.

The paper's finding is counter-intuitive and the code exists to support it:
**raising the first natural frequency — the usual modal objective — moves this
arm out of a safe sub-critical regime and into the rotor excitation band.**
The OEM arm sits at 16.8 Hz, five times below the 83–267 Hz band; every
optimised design lands inside it, from 137.6 Hz at 44.7 g to 253.9 Hz at
232.2 g.

## Layout

- `src/` — the solver. `mesh3d.py` (voxel geometry), `fem.py` (hex FEM, K and M
  assembly, CG, LOBPCG modal, eigenvalue sensitivity), `fem_gpu.py` (CuPy
  drop-in), `topopt.py` (SIMP + OC + augmented-Lagrangian frequency penalty),
  `fatigue.py` (Goodman + S-N), `plotting*.py`.
- `sims/` — orchestration, the CalculiX cross-validation, figure generation,
  and `audit_manuscript.py`, which re-derives the paper's headline numbers from
  the saved results and checks the text against them.
- `figures/`, `notes/` — published figures and the consolidated result tables.

`src/mesh.py` is the superseded planar geometry, kept only for comparison. Do
not present it as the current model.

## Things that are easy to get wrong

- **Rank modes by modal effective mass, never by eigenvalue index.** Low-density
  SIMP regions support localised modes *below* the structural ones. At four of
  five volume fractions the optimiser's tracked "mode 1" carried under 2.3 % of
  the effective mass in the thrust direction — the constraint was acting on a
  mode the rotor cannot excite. Use `sims/modal_participation.py`.
- **The 2 mm voxel model is ~18 % softer than a tet mesh of the same CAD**
  (13.7 Hz against 16.8 Hz on the OEM arm, whose walls are themselves ~2 mm).
  Compare voxel-against-voxel. Frequency is an integral quantity and survives
  the coarse grid; peak stress is local and does not — do not claim the
  optimised designs are stronger than the OEM arm on the voxel stress numbers.
- **The in-house LOBPCG fails below roughly 10 % solid fraction**, returning
  spurious near-zero modes, because the stiffness range spans ~1e9. CalculiX on
  the solid elements is fine. The same effect stalls the sweep at V_f = 0.05
  and 0.065.
- **The augmented Lagrangian degrades the design once its target is
  unreachable.** At V_f = 0.50 the tracked frequency peaks at iteration 25 and
  then loses 26.7 Hz as mu ramps to its cap. Stop ramping when consecutive
  ramps stop helping, and return the best iterate.
- **Modal sensitivity must be auto-scaled to the compliance sensitivity** before
  the two are combined. Combining them dimensionally (J against rad^2/s^2) puts
  the penalty six orders of magnitude out and breaks the volume constraint
  inside the OC bisection.
- **Do not render 3D surfaces through `regen_figures_v4.py`.** It marches cubes
  over the zero-padded density field, which puts domain-face nodes exactly on
  the 0.5 iso-value and leaves four hairlines tracing the box. Use
  `render_topologies.py` / `render_figures_v5.py`, which threshold the cells and
  take the external surface.
- **SIMP has no connectivity constraint.** In every converged design the
  keep-solid regions of bolts 3 and 4 end up as floating islands carrying no
  load. That is a reported finding, not a bug to hide — do not silently clean
  the detached bodies out of figures.

## Not in this repository

The OEM CAD (vendor property) and the multi-hundred-megabyte `.pkl`/`.frd`
result archives. Every dimension measured from the CAD is in `src/mesh3d.py`.

## Commands

```bash
./venv/bin/python -m src.mesh3d                    # geometry summary
./venv/bin/python -m src.fem                       # FEM + modal self-test
./venv/bin/python -u -m sims.run_freq_gpu --vf 0.08 --omega-target 500

QARM_RESULTS_DIR=results_3d_v6 ./venv/bin/python -m sims.consolidate_v6
QARM_RESULTS_DIR=results_3d_v6 ./venv/bin/python -m sims.render_topologies
QARM_RESULTS_DIR=results_3d_v6 ./venv/bin/python -m sims.audit_manuscript
```

Every consolidation and rendering entry point reads `QARM_RESULTS_DIR`, so a
new sweep can be rendered without overwriting an old one.
