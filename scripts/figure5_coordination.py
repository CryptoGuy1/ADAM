#!/usr/bin/env python3
"""Figure 5 for Section 4.2: stage decomposition and end-to-end latency ECDF.

Panel (a): mean per-stage latency over the 446 completed deployment events on a
logarithmic axis, with one-standard-deviation error bars and the share of the
end-to-end budget annotated per stage.

Panel (b): empirical CDF of end-to-end decision latency. Completed events form
the curve; the 13 events that reached the 30-second deadline without committing
an action are shown as censored mass at the deadline. The deadline and the
completed-event P95 are marked.

All values are read from 05_D2_Coordination_Log in the deposited workbook.
Output at 600 dpi PNG plus vector PDF.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
INK = "#1a1a1a"

WB = sys.argv[1] if len(sys.argv) > 1 else "data/ADAM_Dataset_Master.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

co = pd.read_excel(WB, sheet_name="05_D2_Coordination_Log")
done = co[co["Success"].astype(str).str.lower() == "yes"]
fail = co[co["Success"].astype(str).str.lower() != "yes"]

STAGES = [
    ("Crew formation\n$T_{\\mathrm{form}}$", "T_form_ms", ORANGE),
    ("Aggregation\n$T_{\\mathrm{agg}}$", "T_aggregate_ms", GREEN),
    ("On-device reasoning\n$T_{\\mathrm{reason}}$", "T_reason_ms", BLUE),
    ("Governance validation\n$T_{\\mathrm{gov}}$", "T_validate_ms", GREEN),
    ("Semantic memory\n$T_{\\mathrm{weav}}$", "T_weaviate_ms", GREEN),
    ("Blockchain commit\n$T_{\\mathrm{bc}}$", "T_blockchain_ms", GREEN),
]

means = [done[c].mean() for _, c, _ in STAGES]
sds = [done[c].std(ddof=1) for _, c, _ in STAGES]
total = sum(means)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.8, 6.4),
                               gridspec_kw={"width_ratios": [1.15, 1.0]})

# ---- (a) stage decomposition, log scale
y = np.arange(len(STAGES))[::-1]
for yi, (label, col, color), m, s in zip(y, STAGES, means, sds):
    ax1.barh(yi, m, xerr=s, color=color, edgecolor=INK, linewidth=1.0,
             height=0.62, capsize=4, error_kw={"elinewidth": 1.2, "ecolor": INK},
             zorder=3)
    if col == "T_reason_ms":
        # Two-line annotation so the label stays inside the axis limits.
        ax1.text(m * 1.45, yi + 0.10, f"{m:,.0f} ms", va="bottom", ha="left",
                 fontsize=13, color=INK)
        ax1.text(m * 1.45, yi - 0.10, f"({m / total:.1%})", va="top", ha="left",
                 fontsize=13, color=INK)
    else:
        ax1.text(m * 1.45, yi, f"{m:,.0f} ms  ({m / total:.1%})",
                 va="center", ha="left", fontsize=13, color=INK)
ax1.set_yticks(y)
ax1.set_yticklabels([s[0] for s in STAGES], fontsize=13)
ax1.set_xscale("log")
ax1.set_xlim(1.2e2, 8e4)
ax1.set_xlabel("Mean stage latency (ms, log scale)", labelpad=10)
ax1.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.65, which="both")
ax1.set_axisbelow(True)
ax1.text(0.985, 0.04,
         f"Total {total / 1000:.2f} s   ($n={len(done)}$ completed)",
         transform=ax1.transAxes, ha="right", fontsize=13,
         bbox=dict(boxstyle="square,pad=0.35", facecolor="white",
                   edgecolor=INK, linewidth=0.9))
ax1.text(-0.30, 1.03, "(a)", transform=ax1.transAxes, fontsize=19,
         fontweight="bold")

# ---- (b) ECDF against the deadline, censored failures at 30 s
lat = np.sort(done["T_decision_total_ms"].values) / 1000.0
n_all = len(co)
# Completed events climb to 446/459; the 13 censored events account for the rest.
ecdf_y = np.arange(1, len(lat) + 1) / n_all
ax2.step(lat, ecdf_y, where="post", color=BLUE, linewidth=2.4, zorder=3,
         label=f"Completed events ($n={len(done)}$)")
# Censored mass at the deadline
ax2.step([lat[-1], 30.0], [ecdf_y[-1], ecdf_y[-1]], where="post",
         color=BLUE, linewidth=2.4, zorder=3)
ax2.plot([30.0, 30.0], [ecdf_y[-1], 1.0], color=RED, linewidth=3.6,
         zorder=4, solid_capstyle="butt",
         label=f"Deadline-censored ($n={len(fail)}$)")
ax2.plot(30.0, 1.0, marker="o", color=RED, markersize=7,
         markeredgecolor=INK, zorder=5)

p95 = np.percentile(done["T_decision_total_ms"], 95) / 1000.0
med = np.median(done["T_decision_total_ms"]) / 1000.0
ax2.axvline(30.0, color=INK, linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)
ax2.text(29.45, 0.30, "$\\delta_{\\mathrm{deadline}} = 30$ s", rotation=90,
         va="center", ha="right", fontsize=13, color=INK)
ax2.axvline(p95, color=INK, linestyle=":", linewidth=1.2, zorder=2)
ax2.text(p95 + 0.25, 0.18, f"P95 = {p95:.1f} s", fontsize=13, color=INK)
ax2.plot(med, 0.5 * len(done) / n_all, marker="D", color=ORANGE, markersize=8,
         markeredgecolor=INK, zorder=5, linestyle="none",
         label=f"Median {med:.1f} s")

ax2.set_xlim(16.5, 31.5)
ax2.set_ylim(0, 1.04)
ax2.set_xlabel("End-to-end decision latency (s)", labelpad=10)
ax2.set_ylabel("Fraction of the 459 deployment events")
ax2.grid(linestyle=":", linewidth=0.7, alpha=0.65)
ax2.set_axisbelow(True)
ax2.legend(frameon=False, loc="upper left")
ax2.text(-0.16, 1.03, "(b)", transform=ax2.transAxes, fontsize=19,
         fontweight="bold")

fig.tight_layout(w_pad=3.0)
fig.savefig(os.path.join(OUT, "figure5_coordination.pdf"))
fig.savefig(os.path.join(OUT, "figure5_coordination.png"))
plt.close(fig)

print(f"stages sum {total:.1f} ms | median {med:.2f} s | p95 {p95:.2f} s | "
      f"max {done['T_decision_total_ms'].max()/1000:.2f} s | "
      f"completed {len(done)}/{n_all}")
