"""
GPU-accelerated drop-in for `FEM3D.apply_BCs_and_solve`.

When `cupy` is importable and a CUDA device is visible, we transfer the
stiffness matrix and load vector to the device, run a diagonally-
preconditioned conjugate gradient on the GPU, and copy the displacement
back to the host.  Falls back to the CPU implementation otherwise.

Why this matters
----------------
For the 67k-element / 222k-DoF 3D-bridge mesh, the CPU CG solver takes
~ 7–15 s per state on a single core.  On an NVIDIA H100 the same solve
typically completes in well under 1 s thanks to the GPU's order-of-
magnitude higher memory bandwidth — sparse CG is bandwidth-bound, not
compute-bound, so the speed-up tracks the HBM3-vs-DDR4 ratio.

Usage
-----
The simplest integration point is to monkey-patch `FEM3D` so all callers
benefit without code changes:

    from src.fem_gpu import enable_gpu_solver
    enable_gpu_solver()        # no-op on CPU-only nodes
    # ... then use FEM3D as normal
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Try to import cupy.  We only patch the solver if the import works AND
# a device is actually visible.
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    import cupyx.scipy.sparse.linalg as cpspla
    _HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUPY = False


def gpu_available() -> bool:
    return _HAS_CUPY


def _solve_cg_gpu(K_csr: sp.csr_matrix, F: np.ndarray, free,
                   x0: np.ndarray = None,
                   rtol: float = 1e-5,
                   maxiter: int = 5000,
                   solver_name: str = "cg") -> np.ndarray:
    """Solve K[free,free] u_f = F[free] on the GPU and return the full u."""
    K_ff = K_csr[free][:, free].tocsr()
    F_f = F[free].astype(np.float64)
    # Move to GPU
    K_gpu = cpsp.csr_matrix(
        (cp.asarray(K_ff.data), cp.asarray(K_ff.indices), cp.asarray(K_ff.indptr)),
        shape=K_ff.shape,
    )
    F_gpu = cp.asarray(F_f)
    # Diagonal Jacobi preconditioner
    diag = K_gpu.diagonal()
    M_inv = cpsp.diags(1.0 / cp.where(diag != 0, diag, 1.0))

    if x0 is not None:
        x0_gpu = cp.asarray(x0[free].astype(np.float64))
    else:
        x0_gpu = None

    if solver_name == "cg":
        solver = cpspla.cg
    elif solver_name == "bicgstab":
        solver = cpspla.bicgstab
    else:
        raise ValueError(f"Unknown solver_name: {solver_name}")

    # CuPy's CG signature varies by version: older versions take `tol=`,
    # newer (≥ 13) take `rtol=`.  Try the new spelling first, fall back
    # to the old one if needed.
    common_kw = dict(M=M_inv, maxiter=maxiter, x0=x0_gpu)
    try:
        u_f_gpu, info = solver(K_gpu, F_gpu, rtol=rtol, atol=0.0, **common_kw)
    except TypeError:
        u_f_gpu, info = solver(K_gpu, F_gpu, tol=rtol, **common_kw)
    if info != 0:
        # Fall back to CPU CG (NOT spsolve — that direct solve is brutally
        # slow for this size and would dominate the run time).
        import scipy.sparse.linalg as spla
        diag_cpu = K_ff.diagonal()
        Minv_cpu = sp.diags(1.0 / np.where(diag_cpu != 0, diag_cpu, 1.0))
        u_f, _ = spla.cg(K_ff, F_f, M=Minv_cpu, rtol=rtol, maxiter=maxiter,
                          x0=(x0[free] if x0 is not None else None))
    else:
        u_f = cp.asnumpy(u_f_gpu)

    n = F.shape[0]
    u = np.zeros(n)
    u[free] = u_f
    return u


def _modal_gpu(self, rho_active: np.ndarray,
                n_modes: int = 6,
                tip_mass_kg: float = 0.060,
                tip_mass_nodes: np.ndarray = None,
                X0_free: np.ndarray = None,
                tol: float = 1e-4,
                maxiter: int = 500,
                ) -> tuple:
    """GPU drop-in for `FEM3D.modal_analysis`: assembles K, M on host then
    runs LOBPCG on the device.  Falls back to CPU LOBPCG on failure.

    Signature matches the CPU implementation exactly.
    """
    # Assemble on the host first (cheap; assembly is just data manipulation).
    K = self.assemble_K(rho_active)
    if tip_mass_nodes is None:
        if hasattr(self.geom, "motor_mount_top_nodes"):
            tip_mass_nodes = self.geom.motor_mount_top_nodes()
        elif hasattr(self.geom, "_motor_top_boundary_nodes"):
            tip_mass_nodes = self.geom._motor_top_boundary_nodes()
        else:
            tip_mass_nodes = np.array([], dtype=np.int64)
    M = self.assemble_M(rho_active,
                         tip_mass_kg=tip_mass_kg,
                         tip_mass_nodes=tip_mass_nodes)
    free = self.free_dofs
    K_ff = K[free][:, free].tocsr()
    M_ff = M[free][:, free].tocsr()
    n_free = K_ff.shape[0]

    # Move sparse matrices and initial guess to the GPU
    K_gpu = cpsp.csr_matrix(
        (cp.asarray(K_ff.data), cp.asarray(K_ff.indices), cp.asarray(K_ff.indptr)),
        shape=K_ff.shape,
    )
    M_gpu = cpsp.csr_matrix(
        (cp.asarray(M_ff.data), cp.asarray(M_ff.indices), cp.asarray(M_ff.indptr)),
        shape=M_ff.shape,
    )
    diag_K = K_gpu.diagonal()
    M_inv_op = cpsp.diags(1.0 / cp.where(diag_K > 0, diag_K, 1.0))
    if X0_free is not None and X0_free.shape == (n_free, n_modes):
        X0_gpu = cp.asarray(X0_free.astype(np.float64))
    else:
        rng = cp.random.default_rng(0)
        X0_gpu = rng.standard_normal((n_free, n_modes), dtype=cp.float64)

    try:
        eigvals_gpu, eigvecs_gpu = cpspla.lobpcg(
            K_gpu, X0_gpu, B=M_gpu, M=M_inv_op,
            tol=tol, largest=False, maxiter=maxiter,
        )
        eigvals = cp.asnumpy(eigvals_gpu)
        eigvecs_f = cp.asnumpy(eigvecs_gpu)
    except Exception as e:
        print(f"[GPU LOBPCG failed: {e}; falling back to CPU LOBPCG]",
              flush=True)
        import scipy.sparse.linalg as spla_cpu
        diag_K_cpu = K_ff.diagonal()
        Minv_cpu = sp.diags(1.0 / np.where(diag_K_cpu > 0, diag_K_cpu, 1.0))
        X0_cpu = cp.asnumpy(X0_gpu) if isinstance(X0_gpu, cp.ndarray) else X0_gpu
        eigvals, eigvecs_f = spla_cpu.lobpcg(
            K_ff, X0_cpu, B=M_ff, M=Minv_cpu,
            tol=tol, largest=False, maxiter=maxiter,
        )

    order = np.argsort(eigvals)
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs_f = eigvecs_f[:, order]
    frequencies_Hz = np.sqrt(eigvals) / (2.0 * np.pi)
    mode_shapes = np.zeros((self.geom.n_dofs, n_modes))
    mode_shapes[free, :] = eigvecs_f
    return frequencies_Hz, mode_shapes, eigvals


def enable_gpu_solver(force: bool = False) -> bool:
    """Monkey-patch `FEM3D.apply_BCs_and_solve` and `FEM3D.modal_analysis` to
    route through the GPU.

    Returns True if the patch was applied, False otherwise (no GPU).
    Pass `force=True` to apply even without a visible GPU (raises at
    runtime — useful for debugging on the login node).
    """
    if not (_HAS_CUPY or force):
        return False

    from .fem import FEM3D

    original_solve = FEM3D.apply_BCs_and_solve
    original_modal = FEM3D.modal_analysis

    def _patched_solve(self, K, F, method="cg", x0=None, rtol=1e-5):
        if method == "cg" and (_HAS_CUPY or force):
            try:
                return _solve_cg_gpu(K, F, self.free_dofs,
                                       x0=x0, rtol=rtol)
            except Exception as e:
                # Soft-fail: fall back to CPU
                print(f"[GPU solver failed: {e}; falling back to CPU]",
                      flush=True)
                return original_solve(self, K, F, method=method,
                                       x0=x0, rtol=rtol)
        return original_solve(self, K, F, method=method, x0=x0, rtol=rtol)

    def _patched_modal(self, rho_active, n_modes=6, tip_mass_kg=0.060,
                        tip_mass_nodes=None, X0_free=None,
                        tol=1e-4, maxiter=500):
        if _HAS_CUPY or force:
            try:
                return _modal_gpu(self, rho_active,
                                    n_modes=n_modes,
                                    tip_mass_kg=tip_mass_kg,
                                    tip_mass_nodes=tip_mass_nodes,
                                    X0_free=X0_free,
                                    tol=tol, maxiter=maxiter)
            except Exception as e:
                print(f"[GPU modal failed: {e}; falling back to CPU]",
                      flush=True)
        return original_modal(self, rho_active,
                                n_modes=n_modes,
                                tip_mass_kg=tip_mass_kg,
                                tip_mass_nodes=tip_mass_nodes,
                                X0_free=X0_free,
                                tol=tol, maxiter=maxiter)

    FEM3D.apply_BCs_and_solve = _patched_solve
    FEM3D.modal_analysis = _patched_modal
    return True


if __name__ == "__main__":
    print(f"CuPy available: {_HAS_CUPY}")
    if _HAS_CUPY:
        print(f"CuPy version: {cp.__version__}")
        for i in range(cp.cuda.runtime.getDeviceCount()):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
            mem_gb = props["totalGlobalMem"] / (1024 ** 3)
            print(f"  GPU {i}: {name} ({mem_gb:.1f} GB)")
