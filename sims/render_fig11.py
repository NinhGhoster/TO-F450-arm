"""Render the paper's central figure: where each design sits against the band.

Two panels. Left, the first *thrust-excitable* mode against arm mass, with the
rotor excitation band shaded and the two admissible windows marked. Right, the
structural verification for the same designs, including the OEM arm.

The visual language follows the earlier version of this figure — markers keyed
to a frequency colour ramp, with a colourbar — because that reads well. What
changed is what the colours and the axes mean. The old figure coloured every
point by *omega_target*, which was 500 Hz for all of them, so the colourbar
carried no information and each point was labelled "500". Here the ramp encodes
the frequency each design actually achieves, which is the quantity the paper is
about and which varies across the sweep.

Overlap is handled explicitly rather than left to chance: the colourbar sits
outside the axes, each legend is placed in a corner checked to be empty, and
the per-point labels are offset away from the trend line and from each other.

Overwrites ``figures/fig11_pareto_mass_freq.png``.

Usage::

    QARM_RESULTS_DIR=results_3d_v6 ./venv/bin/python -m sims.render_fig11
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from sims.consolidate_v6 import F_1P, F_BPF, SUB_MAX, SUPER_MIN, rows

OUT = "figures/fig11_pareto_mass_freq.png"

INK = "#1a1a1a"
BAND = "#d94f4f"
SAFE = "#3f8f57"
OEMC = "#c1471f"
CMAP = "viridis"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.linewidth": 0.9, "mathtext.default": "regular",
    "savefig.facecolor": "white",
})


def main(dpi=300):
    rs = rows()
    oem = json.load(open("notes/oem_baseline.json"))
    oem_lc3 = next(c for c in oem["cases"] if c["tag"] == "LC3_landing")

    m_to = np.array([r["mass_g"] for r in rs])
    w_to = np.array([r["w_exc"] for r in rs])
    m_oem, w_oem = oem["mass_g"], oem["first_excitable_Hz"]
    m_blk, w_blk = 454.9, 247.3

    # The ramp encodes the frequency achieved by the *optimised sweep*, so it
    # is normalised over that series alone. The OEM arm and the solid envelope
    # are not members of it — they are reference points, and each keeps one
    # fixed colour in both panels so a marker means the same thing wherever it
    # appears. Colour therefore does one job here (magnitude within the sweep)
    # and shape does the other (which object this is).
    norm = plt.Normalize(vmin=np.floor(w_to.min() / 10) * 10,
                         vmax=np.ceil(w_to.max() / 10) * 10)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.8, 5.6), dpi=dpi,
                                 gridspec_kw=dict(wspace=0.30))

    # ---- (a) frequency against mass -------------------------------------
    xmax = 520
    ax.axhspan(F_1P[0], F_BPF[1], color=BAND, alpha=0.12, lw=0, zorder=0)
    ax.axhspan(0, SUB_MAX, color=SAFE, alpha=0.10, lw=0, zorder=0)
    ax.axhspan(SUPER_MIN, 400, color=SAFE, alpha=0.10, lw=0, zorder=0)
    for y in (F_1P[0], F_BPF[1]):
        ax.axhline(y, color=BAND, lw=0.9, alpha=0.55, zorder=1)
    for y in (SUB_MAX, SUPER_MIN):
        ax.axhline(y, color=SAFE, lw=1.0, ls="--", alpha=0.75, zorder=1)

    # Band captions hug the right edge; the rightmost marker is the 454.9 g
    # envelope, so there is clear space beyond it.
    ax.text(xmax - 8, (F_1P[0] + F_BPF[1]) / 2, "rotor excitation band\n83–267 Hz",
            ha="right", va="center", color=BAND, fontsize=9.5, weight="bold")
    ax.text(xmax - 8, SUB_MAX / 2, "admissible: sub-critical  (≤ 66 Hz)",
            ha="right", va="center", color=SAFE, fontsize=9)
    ax.text(xmax - 8, SUPER_MIN + 3, "admissible: super-critical  (≥ 320 Hz)",
            ha="right", va="bottom", color=SAFE, fontsize=9)

    ax.plot(m_to, w_to, "-", color="#8899aa", lw=1.3, zorder=2)
    sc = ax.scatter(m_to, w_to, c=w_to, cmap=CMAP, norm=norm, s=115,
                    edgecolor=INK, lw=0.9, zorder=4)
    # Deliberately NOT on the colour ramp: it is not a design, it is the
    # envelope filled solid, and giving it a design's colouring invited it to
    # be read as the heaviest optimised result.
    ax.scatter([m_blk], [w_blk], s=170, marker="s", facecolor="none",
               edgecolor="#6b6b6b", lw=1.8, zorder=4)
    # Red, not ramp-filled: the OEM arm is the reference the paper argues
    # against, not the lightest member of the sweep, and it carries the same
    # red star in panel (b). Its frequency is already legible from its height
    # and from its label, so nothing is lost by taking it off the ramp.
    ax.scatter([m_oem], [w_oem], s=320, marker="*", c=OEMC,
               edgecolor=INK, lw=0.9, zorder=5)

    # Labels go above the trend for the three light designs and below for the
    # two heavy ones, so no caption crosses the line or another caption.
    # V_f = 0.08 and 0.10 sit 9 g and 20 Hz apart, too close for stacked
    # captions. They go down and to the right — below the rising trend line,
    # and inside the axes, which a left-hand offset would not be at x = 45.
    place = {0: ((14, -11), "left"), 1: ((14, -11), "left"),
             2: ((0, 15), "center"), 3: ((-6, 15), "center"),
             4: ((6, 15), "center")}
    for i, r in enumerate(rs):
        off, ha = place.get(i, ((0, 15), "center"))
        ax.annotate(f"$V_f$ = {r['vf']:.2f}", (r["mass_g"], r["w_exc"]),
                    textcoords="offset points", xytext=off, ha=ha,
                    va="center" if i < 2 else "bottom",
                    fontsize=8.8, color=INK)
    ax.annotate("OEM arm\n34.4 g, 16.8 Hz", (m_oem, w_oem),
                textcoords="offset points", xytext=(22, 4), ha="left",
                fontsize=9.5, color=OEMC, weight="bold")
    ax.annotate("454.9 g, 247.3 Hz",
                (m_blk, w_blk), textcoords="offset points", xytext=(0, 20),
                ha="center", va="bottom", fontsize=8.8, color="#6b6b6b")

    ax.annotate("", xy=(m_to[0], w_to[0] - 9), xytext=(m_oem + 5, w_oem + 7),
                arrowprops=dict(arrowstyle="-|>", color=OEMC, lw=1.8,
                                connectionstyle="arc3,rad=-0.3",
                                shrinkA=8, shrinkB=8), zorder=3)
    ax.text(58, 96, "optimising for a\nhigher first mode", fontsize=9,
            color=OEMC, style="italic", ha="left", va="center", zorder=5)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Arm mass (g)")
    ax.set_ylabel("First thrust-excitable mode (Hz)")
    ax.set_title("(a) Every optimised design lands inside the band",
                 fontsize=11, loc="left", pad=8)
    ax.grid(True, ls=":", lw=0.5, alpha=0.4, zorder=0)
    # A key is needed after all: without one the square was read as the
    # heaviest optimised design rather than as the solid envelope. Anchored in
    # the empty strip between the band captions, which no marker reaches.
    ax.legend(handles=[
        Line2D([], [], marker="*", ls="none", mfc=OEMC, mec=INK, ms=14,
               label="OEM arm"),
        Line2D([], [], marker="o", ls="none", mfc="#35b779", mec=INK, ms=9,
               label="optimised designs (fill = frequency)"),
        Line2D([], [], marker="s", ls="none", mfc="none", mec="#6b6b6b",
               mew=1.8, ms=9, label="solid envelope (not a design)"),
    ], loc="upper left", bbox_to_anchor=(0.012, 0.995), frameon=True,
        framealpha=0.96, fontsize=8.6, handletextpad=0.6, borderpad=0.5)

    cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.046)
    cb.set_label("First thrust-excitable mode (Hz)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ---- (b) structural verification ------------------------------------
    bx.axhline(1.5, color=INK, lw=1.0, ls=":", alpha=0.8)
    # Short form: the gap between the OEM star at 34 g and the legend leaves
    # room for about 95 units of text, not 145.
    bx.text(58, 1.62, "safety factor = 1.5 target", ha="left", va="bottom",
            fontsize=9, color=INK)
    bx.plot(m_to, [r["sf_y"] for r in rs], "o-", color="#2f6fae", lw=1.4, ms=8,
            mec=INK, mew=0.7, label="yield, LC3 hard landing")
    bx.plot(m_to, [r["sf_f"] for r in rs], "^-", color=SAFE, lw=1.4, ms=8,
            mec=INK, mew=0.7, label="fatigue, LC1↔LC2 (Goodman)")
    # Same red star as panel (a), for the same object. Two stars appear here
    # because the OEM arm has both a yield and a fatigue safety factor, and
    # this panel plots one axis for both quantities.
    bx.scatter([m_oem, m_oem], [oem_lc3["sf_yield"], oem["sf_fatigue"]],
               marker="*", s=320, c=OEMC, edgecolor=INK, lw=0.9, zorder=5)
    bx.annotate(f"OEM arm\n{oem_lc3['sf_yield']:.2f} yield, "
                f"{oem['sf_fatigue']:.1f} fatigue",
                (m_oem, oem["sf_fatigue"]), textcoords="offset points",
                xytext=(24, 2), ha="left", fontsize=9.5, color=OEMC,
                weight="bold")
    bx.set_yscale("log")
    bx.set_xlim(0, 300)
    bx.set_ylim(1, 3000)
    bx.set_xlabel("Arm mass (g)")
    bx.set_ylabel("Safety factor")
    bx.set_title("(b) Structural verification, same designs", fontsize=11,
                 loc="left", pad=8)
    bx.grid(True, which="both", ls=":", lw=0.5, alpha=0.4)
    # The star was drawn but never keyed, so the only thing identifying it was
    # the annotation beside it — identity resting on an adjacent caption rather
    # than on the legend. Handles are built explicitly so the star is listed
    # with the two series it is compared against.
    bx.legend(handles=[
        Line2D([], [], marker="o", ls="-", color="#2f6fae", mec=INK, mew=0.7,
               ms=8, lw=1.4, label="yield, LC3 hard landing"),
        Line2D([], [], marker="^", ls="-", color=SAFE, mec=INK, mew=0.7,
               ms=8, lw=1.4, label="fatigue, LC1↔LC2 (Goodman)"),
        Line2D([], [], marker="*", ls="none", mfc=OEMC, mec=INK, mew=0.9,
               ms=14, label="OEM arm (both quantities)"),
    # Upper left, not lower right: the third entry made the box wide enough to
    # run over the 1.5-target label on the baseline. Both safety factors rise
    # with mass, so the top-left of a log axis is the one large empty region.
    ], loc="upper left", frameon=True, framealpha=0.95, fontsize=9)

    fig.savefig(OUT, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"{OUT}")
    print(f"  colour ramp {norm.vmin:.0f}–{norm.vmax:.0f} Hz — the frequency "
          f"achieved, not the target")
    print(f"  OEM {m_oem:.1f} g at {w_oem:.1f} Hz; optimised "
          f"{m_to.min():.1f}–{m_to.max():.1f} g at "
          f"{w_to.min():.1f}–{w_to.max():.1f} Hz")


if __name__ == "__main__":
    main()
