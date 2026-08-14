# TO-F450-arm

Topology optimisation of an F450 quadcopter arm in SLS polyamide-12, with the
finite-element, modal and fatigue pipeline used to produce the results in

> *Frequency-Maximising Topology Optimization Can Drive a Polymer Quadcopter
> Arm Into Resonance: The Case for a Two-Sided Modal Design Rule, Demonstrated
> on an Additively Manufactured F450 Arm in PA12.*

The paper's finding is a cautionary one: raising the first natural frequency,
the usual modal objective, moves this component **out of** a safe sub-critical
regime and **into** the rotor excitation band. The code here is what
establishes that.

## What is in here

| Path | Contents |
|---|---|
| `src/` | The solver: voxel mesh, hex FEM with consistent mass matrix, SIMP optimiser with an augmented-Lagrangian frequency penalty, Goodman/S-N fatigue, plotting |
| `sims/` | Orchestration, the CalculiX cross-validation, figure generation, and the manuscript number audit |
| `figures/` | Every figure in the paper, as published |
| `notes/` | The consolidated result tables the manuscript quotes |

Roughly 3 600 lines of NumPy/SciPy, with an optional CuPy path for GPU. No
commercial solver is in the optimisation loop.

## Not included

- **The OEM CAD.** The F450 arm STEP/STL geometry belongs to its vendor and is
  not redistributed here. `src/mesh3d.py` carries every dimension measured from
  it (Table A1 of the paper), so the design domain is reproducible without it.
- **Raw result archives.** The per-run `.pkl` files and the CalculiX `.frd`
  outputs are several hundred megabytes. `notes/` carries the consolidated
  numbers; the runs regenerate them.

## Reproducing

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt

./venv/bin/python -m src.mesh3d                    # geometry summary
./venv/bin/python -m src.fem                       # FEM + modal self-test

# One volume fraction locally
./venv/bin/python -u -m sims.run_freq_gpu --vf 0.08 --omega-target 500

# Or the full sweep as a PBS array (edit the placeholders in the .pbs first)
qsub sims/submit_freq_array_volta.pbs

# Consolidate, render, and check the manuscript's numbers against the results
QARM_RESULTS_DIR=results_3d_v6 ./venv/bin/python -m sims.consolidate_v6
QARM_RESULTS_DIR=results_3d_v6 ./venv/bin/python -m sims.render_topologies
QARM_RESULTS_DIR=results_3d_v6 ./venv/bin/python -m sims.audit_manuscript
```

`sims/submit_freq_array_volta.pbs` and `sims/watch_sweep.sh` contain
`<project-code>`, `<username>` and `<hpc-host>` placeholders; set them for your
own site.

## Two things worth knowing before you trust a number

1. **Rank modes by modal effective mass, not by eigenvalue index.**
   Low-density SIMP regions support localised modes *below* the structural
   ones. At four of the five volume fractions in this study the optimiser's
   tracked "mode 1" carried under 2.3 % of the effective mass in the thrust
   direction — the frequency constraint was acting on a mode the rotor cannot
   excite. `sims/modal_participation.py` does this properly.
2. **The 2 mm voxel model is ~18 % softer than a tet mesh of the same CAD**
   (13.7 Hz against 16.8 Hz on the OEM arm, whose walls are themselves ~2 mm).
   Comparisons in the paper are therefore made voxel-against-voxel. See
   `notes/voxel_vs_cad_validation.md`.

## Licence

MIT, see `LICENSE`.
