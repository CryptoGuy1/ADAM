#!/usr/bin/env python3
"""Decision-Agent substitution figure from deposited paired trial records."""
from __future__ import annotations

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WB = sys.argv[1] if len(sys.argv) > 1 else "data/ADAM_Dataset_Master.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14,
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

summary = pd.read_excel(WB, sheet_name="17_Swap_Study_Summary", header=1)
stats = pd.read_excel(WB, sheet_name="18_Revised_Statistical_Tests", header=1)
stats = stats[stats["Family"].astype(str).str.lower().eq("decision-agent substitution")]

labels = ["Static", "Gemma 3 1B", "LogReg", "RF", "GBM"]
# Workbook rows are already in the manuscript order.
if len(summary) != 5:
    raise SystemExit(f"expected five substitution rows, found {len(summary)}")
alone = summary["Standalone_F1"].astype(float).to_numpy()
crew = summary["InCrew_F1"].astype(float).to_numpy()
delta = crew - alone

# Use revised bootstrap CIs in the same comparison order where available.
ci_low = np.empty(5); ci_high = np.empty(5)
for i, row in enumerate(stats.itertuples(index=False)):
    ci_low[i] = float(getattr(row, "Bootstrap_95CI_Low"))
    ci_high[i] = float(getattr(row, "Bootstrap_95CI_High"))
err = np.vstack([delta - ci_low, ci_high - delta])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2))
# Panel a: paired mean operating points, diagonal indicates equality.
ax1.scatter(alone, crew, s=85, zorder=3)
for x, y, label in zip(alone, crew, labels):
    ax1.annotate(label, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=11)
lo = min(alone.min(), crew.min()) - 0.015
hi = max(alone.max(), crew.max()) + 0.015
ax1.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1)
ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
ax1.set_xlabel("Standalone mean $F_1$")
ax1.set_ylabel("In-crew mean $F_1$")
ax1.grid(linestyle=":", linewidth=0.7, alpha=0.65)
ax1.text(-0.11, 1.03, "(a)", transform=ax1.transAxes, fontsize=16, fontweight="bold")

# Panel b: paired mean change and 95% bootstrap CI.
x = np.arange(5)
ax2.bar(x, delta, yerr=err, capsize=4, edgecolor="black", linewidth=0.8)
ax2.axhline(0, linewidth=1.0)
ax2.set_xticks(x, labels, rotation=22, ha="right")
ax2.set_ylabel(r"In-crew $-$ standalone $\Delta F_1$")
ax2.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
for xi, d in zip(x, delta):
    ax2.text(xi, d + 0.003, f"{d:+.3f}", ha="center", va="bottom", fontsize=10)
ax2.text(-0.11, 1.03, "(b)", transform=ax2.transAxes, fontsize=16, fontweight="bold")

fig.tight_layout(w_pad=3.0)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"figure_swap_study.{ext}"))
plt.close(fig)
print("wrote figure_swap_study.{pdf,png}")
