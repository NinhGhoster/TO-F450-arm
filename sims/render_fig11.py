"""Render the paper's central figure: where each design sits relative to the band.

Two panels.  Left, the first *thrust-excitable* mode against arm mass, with the
rotor excitation band shaded and the two admissible windows marked.  This is
where the argument lives: the OEM arm sits in the lower window and every
optimised design sits inside the band, so the optimisation moved the design the
wrong way.  Right, the structural verification for the same designs, including
the OEM arm, which is what shows the classical constraints are not slack once
the reference is a real part.

Overwrites ``figures/fig11_pareto_mass_freq.png``.

Usage::

    ./venv/bin/python -m sims.render_fig11
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sims.consolidate_v6 import F_1P, F_BPF, SUB_MAX, SUPER_MIN, rows

OUT = "figures/fig11_pareto_mass_freq.png"

INK = "#1a1a1a"
BAND = "#d94f4f"
SAFE = "#3f8f57"
TO = "#3a6fa5"
OEM = "#c1471f"
BLOCK = "#6b6b6b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "mathtext.default": "regular",
    "savefig.facecolor": "white",
})


def main(dpi=300):
    rs = rows()
    oem = json.load(open("notes/oem_baseline.json"))
    oem_lc3 = next(c for c in oem["cases"] if c["tag"] == "LC3_landing")

    m_to = np.array([r["mass_g"] for r in rs])
    w_to = np.array([r["w_exc"] for r in rs])
    m_oem, w_oem = oem["mass_g"], oem["first_excitable_Hz"]
    # The solid block, read from the same sweep's baseline record.
    m_blk, w_blk = 454.9, 247.3

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=dpi)

    # ---- left: frequency against mass -----------------------------------
    xmax = 480
    ax.axhspan(F_1P[0], F_BPF[1], color=BAND, alpha=0.13, lw=0, zorder=0)
    ax.axhspan(0, SUB_MAX, color=SAFE, alpha=0.10, lw=0, zorder=0)
    ax.axhspan(SUPER_MIN, 400, color=SAFE, alpha=0.10, lw=0, zorder=0)
    ax.axhline(F_1P[0], color=BAND, lw=0.9, ls="-", alpha=0.55)
    ax.axhline(F_BPF[1], color=BAND, lw=0.9, ls="-", alpha=0.55)
    ax.axhline(SUB_MAX, color=SAFE, lw=1.0, ls="--", alpha=0.75)
    ax.axhline(SUPER_MIN, color=SAFE, lw=1.0, ls="--", alpha=0.75)

    ax.text(xmax - 8, (F_1P[0] + F_BPF[1]) / 2, "rotor excitation band\n83–267 Hz",
            ha="right", va="center", color=BAND, fontsize=10, weight="bold")
    ax.text(xmax - 8, SUB_MAX / 2, "admissible: sub-critical  (≤ 66 Hz)",
            ha="right", va="center", color=SAFE, fontsize=9.5)
    ax.text(xmax - 8, (SUPER_MIN + 400) / 2, "admissible: super-critical  (≥ 320 Hz)",
            ha="right", va="center", color=SAFE, fontsize=9.5)

    ax.plot(m_to, w_to, "-", color=TO, lw=1.4, alpha=0.65, zorder=2)
    ax.scatter(m_to, w_to, s=88, color=TO, edgecolor="white", lw=1.3, zorder=4,
               label="frequency-optimised designs")
    ax.scatter([m_blk], [w_blk], s=120, marker="s", color=BLOCK,
               edgecolor="white", lw=1.3, zorder=4, label="solid design envelope")
    ax.scatter([m_oem], [w_oem], s=210, marker="*", color=OEM,
               edgecolor="white", lw=1.3, zorder=5, label="OEM arm")

    # The two lightest designs sit close together, so their labels go out to
    # the left rather than stacking on top of each other.
    for i, r in enumerate(rs):
        # The two lightest points sit near the left edge, so their labels go
        # up and to the right; the rest sit directly above.
        off, ha = ((9, 6), "left") if i < 2 else ((0, 12), "center")
        ax.annotate(f"$V_f$ = {r['vf']:.2f}", (r["mass_g"], r["w_exc"]),
                    textcoords="offset points", xytext=off, ha=ha,
                    va="bottom", fontsize=8.6, color=INK)
    ax.annotate("OEM arm\n34.4 g, 16.8 Hz", (m_oem, w_oem),
                textcoords="offset points", xytext=(20, -2), ha="left",
                va="center", fontsize=9.5, color=OEM, weight="bold")
    ax.annotate("454.9 g", (m_blk, w_blk), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=8.6, color=INK)

    # The move the optimisation makes, drawn so it cannot be missed.
    ax.annotate("", xy=(m_to[0], w_to[0] - 6), xytext=(m_oem + 3, w_oem + 5),
                arrowprops=dict(arrowstyle="-|>", color=OEM, lw=1.8,
                                connectionstyle="arc3,rad=-0.25",
                                shrinkA=6, shrinkB=6))
    ax.text(16, 108, "optimising for a\nhigher first mode", fontsize=9.2,
            color=OEM, style="italic", ha="left", va="center")

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Arm mass (g)")
    ax.set_ylabel("First thrust-excitable mode (Hz)")
    ax.set_title("Every optimised design lands inside the excitation band",
                 fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=True, framealpha=0.94, fontsize=9.5)
    ax.grid(True, ls=":", lw=0.5, alpha=0.45)

    # ---- right: structural verification ---------------------------------
    bx.axhline(1.5, color=INK, lw=1.0, ls=":", alpha=0.8)
    # Sits between the OEM yield marker on the left and the legend on the
    # right, which is the only clear stretch of that line.
    bx.text(150, 1.56, "SF = 1.5 design target", ha="left", va="bottom",
            fontsize=9, color=INK)
    bx.plot(m_to, [r["sf_y"] for r in rs], "o-", color=TO, lw=1.3, ms=7,
            label="yield, LC3 hard landing")
    bx.plot(m_to, [r["sf_f"] for r in rs], "^-", color=SAFE, lw=1.3, ms=7,
            label="fatigue, LC1↔LC2 (Goodman)")
    bx.scatter([m_oem], [oem_lc3["sf_yield"]], s=210, marker="*", color=OEM,
               edgecolor="white", lw=1.2, zorder=5)
    bx.scatter([m_oem], [oem["sf_fatigue"]], s=210, marker="*", color=OEM,
               edgecolor="white", lw=1.2, zorder=5)
    bx.annotate(f"OEM arm\n{oem_lc3['sf_yield']:.2f} yield\n"
                f"{oem['sf_fatigue']:.1f} fatigue",
                (m_oem, oem["sf_fatigue"]), textcoords="offset points",
                xytext=(22, 8), ha="left", va="bottom", fontsize=9.5,
                color=OEM, weight="bold")

    bx.set_yscale("log")
    bx.set_xlim(0, 480)
    bx.set_ylim(1, 1500)
    bx.set_xlabel("Arm mass (g)")
    bx.set_ylabel("Safety factor")
    bx.set_title("Structural verification, same designs", fontsize=12, pad=10)
    bx.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=9.5)
    bx.grid(True, which="both", ls=":", lw=0.5, alpha=0.45)

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(OUT, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"{OUT}")
    print(f"  OEM {m_oem:.1f} g at {w_oem:.1f} Hz; optimised "
          f"{m_to.min():.1f}-{m_to.max():.1f} g at {w_to.min():.1f}-{w_to.max():.1f} Hz")


if __name__ == "__main__":
    main()
