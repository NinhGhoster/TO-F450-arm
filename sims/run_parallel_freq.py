"""
Local-CPU parallel runner of the frequency sweep — same problem as
`run_freq_gpu.py` but using a multiprocessing.Pool of 4 workers, each
processing one ω_target with the CPU LOBPCG.

Use when no GPU is available; production runs go through `run_freq_gpu.py`
on the HPC system.  The 4 targets take ~ 30–90 min each on a single Mac CPU core,
so the whole sweep is ~ 2–4 h wall (well under the v3 V_f sweep).
"""
from __future__ import annotations

import multiprocessing as mp
import os

# Ensure each worker has BLAS threads available
os.environ.pop("OMP_NUM_THREADS", None)
os.environ.pop("OPENBLAS_NUM_THREADS", None)
os.environ.pop("MKL_NUM_THREADS", None)

from sims.run_freq_gpu import run_one, VOL_FRACS, NOMINAL_OMEGA_Hz


def _worker(vf: float):
    run_one(NOMINAL_OMEGA_Hz, vf=vf)


def main():
    # Use "fork" so children inherit the imported modules cheaply
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(4, len(VOL_FRACS))) as pool:
        pool.map(_worker, VOL_FRACS)


if __name__ == "__main__":
    main()
