"""
Plotting helpers for paper figures.

All figures are generated as PNG with consistent styling so that they can
be referenced directly from the manuscript.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, BoundaryNorm

from .mesh import ArmGeometry
from .fem import FEM3D

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.0,
})


def _active_to_grid(geom: ArmGeometry, fem: FEM3D, values: np.ndarray,
                     fill_outside: float = np.nan) -> np.ndarray:
    """Inflate a per-active-element array to a full (nx, ny, nz) grid."""
    g = np.full((geom.nx, geom.ny, geom.nz), fill_outside, dtype=float)
    ex, ey, ez = fem.elem_grid_idx.T
    g[ex, ey, ez] = values
    return g


def plot_density_topview(geom: ArmGeometry, fem: FEM3D, rho_active: np.ndarray,
                          path: str, title: str = "Density distribution",
                          midplane: bool = True):
    """Top view (xy) of the density field, averaged through the thickness
    (or just at the mid-plane)."""
    grid = _active_to_grid(geom, fem, rho_active, fill_outside=0.0)
    if midplane and geom.nz > 1:
        z_idx = geom.nz // 2
        img = grid[:, :, z_idx]
    else:
        img = grid.mean(axis=2)
    fig, ax = plt.subplots(figsize=(8, 3))
    extent = [0, geom.L_x * 1e3, 0, geom.L_y * 1e3]
    im = ax.imshow(img.T, origin="lower", extent=extent, cmap="gray_r",
                    vmin=0, vmax=1, aspect="equal", interpolation="nearest")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"density $\rho$")
    fig.savefig(path)
    plt.close(fig)


def plot_density_layers(geom: ArmGeometry, fem: FEM3D, rho_active: np.ndarray,
                         path: str, title: str = "Density distribution"):
    """One panel per through-thickness layer (xy slices)."""
    grid = _active_to_grid(geom, fem, rho_active, fill_outside=0.0)
    nz = geom.nz
    fig, axes = plt.subplots(nz, 1, figsize=(8, 1.6 * nz), sharex=True)
    if nz == 1:
        axes = [axes]
    extent = [0, geom.L_x * 1e3, 0, geom.L_y * 1e3]
    for k in range(nz):
        ax = axes[k]
        im = ax.imshow(grid[:, :, k].T, origin="lower", extent=extent,
                        cmap="gray_r", vmin=0, vmax=1,
                        aspect="equal", interpolation="nearest")
        ax.set_ylabel(f"layer {k+1}\n(z={(k+0.5)*geom.dz*1e3:.1f} mm)")
    axes[-1].set_xlabel("x (mm)")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_stress_topview(geom: ArmGeometry, fem: FEM3D, stress_vM: np.ndarray,
                         path: str, title: str, vmax: float = None,
                         midplane: bool = True):
    """Top view of von Mises stress (MPa), at mid-plane or thickness-max."""
    grid = _active_to_grid(geom, fem, stress_vM / 1e6, fill_outside=np.nan)
    if midplane:
        img = grid[:, :, geom.nz // 2]
    else:
        # The largest through-thickness value at each (x,y)
        img = np.nanmax(grid, axis=2)
    fig, ax = plt.subplots(figsize=(8, 3))
    extent = [0, geom.L_x * 1e3, 0, geom.L_y * 1e3]
    if vmax is None:
        vmax = np.nanmax(img)
    norm = Normalize(vmin=0, vmax=vmax)
    im = ax.imshow(img.T, origin="lower", extent=extent, cmap="jet",
                    norm=norm, aspect="equal", interpolation="nearest")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                  label="von Mises stress (MPa)")
    fig.savefig(path)
    plt.close(fig)


def plot_life_topview(geom: ArmGeometry, fem: FEM3D, life: np.ndarray,
                       path: str, title: str, midplane: bool = True):
    """Top view of fatigue life (log scale, cycles)."""
    grid = _active_to_grid(geom, fem, life, fill_outside=np.nan)
    if midplane:
        img = grid[:, :, geom.nz // 2]
    else:
        img = np.nanmin(grid, axis=2)
    img = np.clip(img, 1.0, 1e9)
    fig, ax = plt.subplots(figsize=(8, 3))
    extent = [0, geom.L_x * 1e3, 0, geom.L_y * 1e3]
    norm = LogNorm(vmin=1e3, vmax=1e9)
    im = ax.imshow(img.T, origin="lower", extent=extent, cmap="viridis_r",
                    norm=norm, aspect="equal", interpolation="nearest")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="life (cycles)")
    fig.savefig(path)
    plt.close(fig)


def plot_sn_curve(path: str):
    """The PA12 SLS S-N curve used in the paper."""
    from .fatigue import SN_CYCLES, SN_ALT_MPA
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.loglog(SN_CYCLES, SN_ALT_MPA, "o-", color="C0", markerfacecolor="white",
                markersize=7, label="PA12 SLS, R=-1")
    ax.set_xlabel("Cycles to failure, N")
    ax.set_ylabel(r"Alternating stress $\sigma_a$ (MPa)")
    ax.set_title("S-N curve for PA12 (SLS), R=-1")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    ax.set_xlim(50, 2e7)
    ax.set_ylim(8, 60)
    fig.savefig(path)
    plt.close(fig)


def plot_to_convergence(history: dict, path: str, title: str = "TO convergence"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    it = np.arange(1, len(history["compliance"]) + 1)
    axes[0].semilogy(it, history["compliance"], "o-", markersize=3)
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Compliance (J)")
    axes[0].set_title("Compliance")
    axes[0].grid(True, ls=":", alpha=0.5)
    axes[1].plot(it, history["change"], "o-", markersize=3, color="C1")
    axes[1].set_xlabel("Iteration"); axes[1].set_ylabel(r"$\max|\Delta \rho|$")
    axes[1].set_title("Density change per iter")
    axes[1].grid(True, ls=":", alpha=0.5)
    axes[1].axhline(0.01, ls="--", color="grey", label="convergence threshold")
    axes[1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pareto(records: list, path: str):
    """records is a list of dicts: {Vf, mass_g, sigma_max_MPa, life_min}"""
    Vf = np.array([r["Vf"] for r in records])
    mass = np.array([r["mass_g"] for r in records])
    sig = np.array([r["sigma_max_MPa"] for r in records])
    life = np.array([r["life_min"] for r in records])

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    ax = axes[0]
    ax.plot(mass, sig, "o-", color="C3")
    for r in records:
        ax.annotate(f"Vf={r['Vf']:.2f}", (r["mass_g"], r["sigma_max_MPa"]),
                     xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Optimized arm mass (g)")
    ax.set_ylabel("Peak von Mises under LC2 (MPa)")
    ax.set_title("Mass vs. peak stress")
    ax.grid(True, ls=":", alpha=0.5)

    ax2 = axes[1]
    ax2.semilogy(mass, life, "o-", color="C2")
    for r in records:
        ax2.annotate(f"Vf={r['Vf']:.2f}", (r["mass_g"], r["life_min"]),
                      xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax2.axhline(1e6, ls="--", color="grey", label="design life 1e6")
    ax2.set_xlabel("Optimized arm mass (g)")
    ax2.set_ylabel("Min predicted fatigue life (cycles)")
    ax2.set_title("Mass vs. fatigue life")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_orientation_bar(records: list, path: str):
    """records is a list of {label, sigma_max_MPa, life_min}"""
    labels = [r["label"] for r in records]
    sigs = [r["sigma_max_MPa"] for r in records]
    lifes = [r["life_min"] for r in records]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(x, sigs, color=["C0", "C3", "C1"], width=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylabel("Peak von Mises (MPa)")
    axes[0].set_title("Stress vs. build orientation")
    axes[0].grid(True, axis="y", ls=":", alpha=0.5)
    axes[1].bar(x, lifes, color=["C0", "C3", "C1"], width=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylabel("Min predicted life (cycles)")
    axes[1].set_yscale("log")
    axes[1].axhline(1e6, ls="--", color="grey", label="design life")
    axes[1].set_title("Life vs. build orientation")
    axes[1].grid(True, which="both", axis="y", ls=":", alpha=0.5)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_arm_footprint(geom, path: str):
    """Two views of the design domain with the boundary conditions marked.

    A top view alone cannot show what matters about this joint: the arm is
    bolted at six points, four from the top plate and two from the bottom,
    each restrained only over its own z-span. In plan the top and bottom
    screws very nearly coincide, so a single view makes six bolts look like
    four; the side view carries that information, and draws the two frame
    plates so it is clear the bolts land on them rather than floating.

    All explanation lives in the manuscript caption; the figure itself carries
    only what it must to be read.
    """
    screws = getattr(geom, "screws", None)
    two_panel = screws is not None and hasattr(geom, "L_z")
    Lx, Ly = geom.L_x * 1e3, geom.L_y * 1e3

    if two_panel:
        fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.0),
                                 gridspec_kw=dict(height_ratios=[1.0, 1.2]))
        ax, sx = axes
    else:
        fig, ax = plt.subplots(figsize=(9, 3))
        sx = None

    tops = list(screws[:4]) if screws is not None else []
    bots = list(screws[4:]) if screws is not None else []

    # ---- top view -------------------------------------------------------
    ax.add_patch(plt.Rectangle((0, 0), Lx, Ly, fill=False, ec="black", lw=1.5))
    # The top- and bottom-plate holes are within 0.3 mm of each other in x, so
    # drawing both as circles of the same size stacks one exactly on the other
    # and reads as a smudge. The bottom pair is drawn larger and open so both
    # remain visible.
    for hx, hy, *_ in tops:
        ax.add_patch(plt.Circle((hx * 1e3, hy * 1e3), geom.screw_dia / 2 * 1e3,
                                fc="#a9cce3", ec="C0", lw=1.2, zorder=4))
    for hx, hy, *_ in bots:
        ax.add_patch(plt.Circle((hx * 1e3, hy * 1e3),
                                geom.screw_dia / 2 * 1e3 + 1.6,
                                fc="none", ec="C0", lw=1.3, ls=(0, (3, 2)),
                                zorder=3))
    ax.add_patch(plt.Circle((geom.motor_centre[0] * 1e3,
                             geom.motor_centre[1] * 1e3),
                            geom.motor_dia / 2 * 1e3,
                            fc="mistyrose", ec="C3", lw=1.3))
    ax.text(geom.screw_centres[0][0] * 1e3 + 2, Ly + 3.5, "frame bolts",
            ha="center", va="bottom", fontsize=8.5, color="C0")
    ax.text(geom.motor_centre[0] * 1e3, Ly + 3.5, "motor mount",
            ha="center", va="bottom", fontsize=8.5, color="C3")
    ax.set_xlim(-10, Lx + 10)
    ax.set_ylim(-6, Ly + 16)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    ax.set_title("(a) Top view", fontsize=10, loc="left")

    # ---- side view ------------------------------------------------------
    if two_panel:
        Lz = geom.L_z * 1e3
        z_top = max(z1 for *_, z1 in tops) * 1e3
        z_bot = min(z0 for *_, z0, _ in bots) * 1e3
        sx.add_patch(plt.Rectangle((0, 0), Lx, Lz, fill=False, ec="black",
                                   lw=1.5))
        # The frame plates. Without them the bottom bolts appear to stop in
        # mid-air; they in fact land on the lower plate.
        for z, lab in ((z_top, "top plate"), (z_bot, "bottom plate")):
            sx.plot([-6, 46], [z, z], color="#555555", lw=2.4,
                    solid_capstyle="butt", zorder=2)
            sx.text(-8, z, lab, ha="right", va="center", fontsize=8,
                    color="#555555")
        sx.annotate("", xy=(40, z_bot), xytext=(40, z_top),
                    arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.9))
        sx.text(41.5, (z_top + z_bot) / 2, f"{z_top - z_bot:.1f} mm",
                fontsize=8, color="#555555", va="center")

        d = geom.screw_dia * 1e3
        SEP = 6.0
        for k, (cx, _cy, z0, z1) in enumerate(screws):
            off = -SEP / 2 if k < 4 else SEP / 2
            style = (dict(fc="#a9cce3", ec="C0", lw=1.1) if k < 4 else
                     dict(fc="none", ec="C0", lw=1.2, ls=(0, (3, 2))))
            sx.add_patch(plt.Rectangle((cx * 1e3 + off - d / 2, z0 * 1e3), d,
                                       (z1 - z0) * 1e3, zorder=5, **style))
        sx.plot([geom.motor_centre[0] * 1e3], [Lz], marker="v", ms=9,
                color="C3", clip_on=False)
        sx.set_xlim(-10, Lx + 10)
        sx.set_ylim(-6, Lz + 10)
        sx.set_xlabel("x (mm)")
        sx.set_ylabel("z (mm)")
        sx.set_aspect("equal")
        sx.set_title("(b) Side view", fontsize=10, loc="left")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_freq_convergence(history: dict, path: str, title: str = ""):
    """Three-panel: ω₁(iter), V_f(iter), μ(iter).  Used as fig07 in v4."""
    iters = np.arange(1, len(history["omega_1_Hz"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

    ax = axes[0]
    omega = np.array(history["omega_1_Hz"], dtype=float)
    ax.plot(iters, omega, "o-", color="C0", ms=3)
    # The peak matters: past it the multiplier saturates and the constrained
    # quantity gets worse, which is the point section 3.4 makes.
    k = int(np.nanargmax(omega))
    if omega[k] - omega[-1] > 1.0:
        ax.plot([iters[k]], [omega[k]], "o", ms=8, mfc="none", mec="C3", mew=1.6)
        ax.annotate(f"peak {omega[k]:.1f} Hz\n(iter {iters[k]})",
                    (iters[k], omega[k]), textcoords="offset points",
                    xytext=(12, -26), fontsize=8, color="C3")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"tracked $\omega_1$ (Hz)")
    ax.set_title("Tracked first mode")
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[1]
    vf = np.array(history["vol_frac"], dtype=float)
    ax.plot(iters, vf, "s-", color="C2", ms=3)
    # Plotted on an absolute 0-1 axis. Autoscaling this series produces an
    # offset axis label like "1e-5+5e-1", which reads as though the volume
    # were wandering when in fact the OC bisection holds it to five decimals.
    ax.set_ylim(0.0, 1.0)
    ax.axhline(vf[-1], color="grey", lw=0.8, ls="--")
    ax.annotate(f"held at {vf[-1]:.3f}\n(max drift {np.abs(vf - vf[-1]).max():.1e})",
                (iters[len(iters) // 2], vf[-1]), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=8, color="dimgrey")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$V_f$ (design domain)")
    ax.set_title("Volume fraction")
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[2]
    mu = np.array(history["mu_freq"], dtype=float)
    ax.plot(iters, mu, "^-", color="C3", ms=3)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\mu_\mathrm{freq}$ (rel. weight)")
    ax.set_yscale("log")
    ax.set_title("Augmented-Lagrangian multiplier")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    if title:
        fig.suptitle(title, y=1.05, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def modal_frf_at_dof(frequencies_Hz: np.ndarray,
                      mode_shapes: np.ndarray,
                      dof_idx: int,
                      freq_range_Hz: tuple = (10.0, 1500.0),
                      n_pts: int = 600,
                      modal_damping: float = 0.02) -> tuple:
    """Modal-superposition FRF |H_{dof,dof}(f)| = | Σ_k v_k[dof]^2 /
    ((ω_k^2 − ω^2) + 2i ζ ω_k ω) |.

    Returns (freqs_Hz, |H|).  Uses simple proportional modal damping ζ
    (default 2 %) so peaks at ω_k stay finite.
    """
    f = np.geomspace(freq_range_Hz[0], freq_range_Hz[1], n_pts)
    omega = 2.0 * np.pi * f
    omega_k = 2.0 * np.pi * np.asarray(frequencies_Hz)
    H = np.zeros(n_pts, dtype=complex)
    v_dof = mode_shapes[dof_idx, :]
    for k in range(len(frequencies_Hz)):
        denom = (omega_k[k] ** 2 - omega ** 2
                  + 2j * modal_damping * omega_k[k] * omega)
        H += v_dof[k] ** 2 / denom
    return f, np.abs(H)


def plot_frf(curves: dict, path: str,
              bpf_band_Hz: tuple = (200.0, 400.0),
              omega_target_Hz: float = None,
              title: str = ""):
    """Plot one or more FRF curves on shared axes.

    `curves` is a dict {label: (freqs_Hz, |H|)} so multiple designs can be
    overlaid.  `bpf_band_Hz` adds a shaded rotor-blade-passage band.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axvspan(*bpf_band_Hz, color="lightcoral", alpha=0.25,
                label=f"rotor BPF band ({bpf_band_Hz[0]:.0f}-{bpf_band_Hz[1]:.0f} Hz)")
    if omega_target_Hz is not None:
        ax.axvline(omega_target_Hz, color="orange", ls="--", lw=1,
                    label=fr"$\omega_\mathrm{{target}}$ = {omega_target_Hz:.0f} Hz")
    for label, (f, mag) in curves.items():
        ax.plot(f, mag, "-", lw=1.2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"|H| (mode-superposition, arb. units)")
    ax.set_title(title or "Frequency Response at motor mount (z-direction)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    # An FRF is flat at low frequency and decays with peaks at high frequency,
    # so the lower-left corner is reliably empty while the upper right is not.
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_mass_freq_pareto(records: list, baseline: dict, path: str,
                           omega_target_default_Hz: float = 500.0,
                           bpf_band_Hz: tuple = (200.0, 400.0),
                           title: str = ""):
    """Mass vs ω₁ Pareto.

    `records` is a list of dicts with at least: mass_g, omega_1_Hz,
    omega_target_Hz, sf_yield_LC3, sf_fatigue.
    `baseline` is a single dict with mass_g, omega_1_Hz, sf_yield_LC3,
    sf_fatigue (the solid arm).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))

    ax = axes[0]
    ax.axhspan(*bpf_band_Hz, color="lightcoral", alpha=0.20,
                label=f"rotor BPF band ({bpf_band_Hz[0]:.0f}-{bpf_band_Hz[1]:.0f} Hz)")
    ax.axhline(omega_target_default_Hz, color="orange", ls="--", lw=1,
                label=fr"$\omega_\mathrm{{target}}$ = {omega_target_default_Hz:.0f} Hz")
    if baseline is not None:
        ax.scatter([baseline["mass_g"]], [baseline["omega_1_Hz"]],
                    color="black", marker="*", s=80, zorder=5, label="Solid baseline")
        ax.annotate("baseline",
                    (baseline["mass_g"], baseline["omega_1_Hz"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=8)
    mass = np.array([r["mass_g"] for r in records])
    omg = np.array([r["omega_1_Hz"] for r in records])
    target = np.array([r["omega_target_Hz"] for r in records])
    sc = ax.scatter(mass, omg, c=target, cmap="viridis", s=60, zorder=4,
                     edgecolors="black")
    for r in records:
        ax.annotate(f"{r['omega_target_Hz']:.0f}",
                    (r["mass_g"], r["omega_1_Hz"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$\omega_\mathrm{target}$ (Hz)")
    ax.set_xlabel("Arm mass (g)")
    ax.set_ylabel(r"Achieved $\omega_1$ (Hz)")
    ax.set_title(r"Mass-frequency Pareto")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    ax.scatter(mass, [r["sf_yield_LC3"] for r in records],
                color="C0", marker="o", s=50, label="SF yield (LC3 landing)")
    ax.scatter(mass, [r["fs_fatigue"] for r in records],
                color="C2", marker="^", s=50, label="SF fatigue (LC1↔LC2)")
    ax.axhline(1.5, color="grey", ls=":", lw=1, label="SF = 1.5 (target)")
    ax.axhline(1.0, color="black", ls="--", lw=1, label="SF = 1")
    ax.set_xlabel("Arm mass (g)")
    ax.set_ylabel("Safety factor (verification)")
    ax.set_yscale("log")
    ax.set_title("Structural safety of modal-optimised designs")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")

    if title:
        fig.suptitle(title, y=1.04, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
