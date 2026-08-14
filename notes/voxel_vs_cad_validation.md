# Voxel pipeline vs CAD tets — how far apart are they?

The topology designs come out of the in-house voxel FEM; the OEM baseline was
solved with CalculiX on a tet mesh of the real CAD solid. Comparing them is
only meaningful if we know the discretisation error, so the OEM arm was
voxelised onto the production 216 x 42 x 54 mm / 2 mm grid and solved again.

Same solver (CalculiX), same boundary conditions, same 60 g tip mass — only
the discretisation differs.

| | omega_1 | omega_2 | omega_3 | mass |
|---|---|---|---|---|
| CAD tets (206 155 linear tets) | 16.8 Hz | 51.2 Hz | 109.9 Hz | 34.5 g |
| voxel, 2 mm (4 384 hexes) | 13.7 Hz | 32.1 Hz | 62.4 Hz | 33.0 g |
| difference | **-18 %** | -37 % | -43 % | -4 % |

**The voxel model is softer.** The OEM arm is thin-walled — spar walls are
about 2 mm, the same as the voxel edge — so voxelisation loses material at
thin sections and thins the load paths. Higher modes diverge more because they
are increasingly local, and local features are exactly what a 2 mm grid cannot
resolve.

## What follows from this

1. **Compare like with like.** The OEM arm and the topology designs must both
   be judged on the voxel grid, i.e. against the 13.7 Hz figure, not 16.8 Hz.
   The tet result stands as the better estimate of the *real* arm.
2. **The bias should be smaller for the optimised designs.** They fill much
   more of the envelope (V_f 0.10-0.50 against the OEM arm's 0.076) with
   chunkier members, so 2 mm voxels resolve them far better than they resolve
   a 2 mm wall.
3. **The in-house LOBPCG could not solve this case at all.** With only 7.3 %
   of the domain solid and the rest at the SIMP floor, the stiffness range
   spans 10^9 and the eigensolve returned six spurious ~0.2 Hz modes without
   converging. CalculiX, solving the solid elements only, had no such trouble.
   This is the same ill-conditioning CLAUDE.md records for V_f = 0.05, and it
   is a real limit on how low the volume-fraction sweep can usefully go.

Reproduce with `sims/validate_voxel_vs_cad.py` and
`ccx -i verification/oem_voxel`.
