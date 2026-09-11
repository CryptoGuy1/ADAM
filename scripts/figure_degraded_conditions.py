#!/usr/bin/env python3
"""Degraded-condition figure from the deposited robustness-study trial records."""
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
    "font.size": 11.5, "axes.titlesize": 13, "axes.labelsize": 13,
    "legend.fontsize": 9.5, "figure.dpi": 150, "savefig.dpi": 600,
    "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})

df = pd.read_excel(WB, sheet_name="14_Degraded_Conditions", header=1)
conditions = ["Clean", "One-Node Dropout", "Mild Drift", "Noise", "Strong Drift"]
systems = ["Static_Threshold", "RandomForest_Rich", "GBM_Rich", "Single_Agent", "ADAM_GBM", "ADAM_LLM"]
labels = {
    "Static_Threshold": "Static",
    "RandomForest_Rich": "RF (fused)",
    "GBM_Rich": "GBM (fused)",
    "Single_Agent": "Single Agent",
    "ADAM_GBM": "ADAM_GBM",
    "ADAM_LLM": "ADAM_LLM",
}

means = df.groupby(["System", "Condition"])["F1_Score"].mean()
sds = df.groupby(["System", "Condition"])["F1_Score"].std(ddof=1)

fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.3))
# (a) condition means + SD
x = np.arange(len(conditions)); width = 0.125
for k, sysname in enumerate(systems):
    vals = np.array([means.loc[(sysname, c)] for c in conditions])
    errs = np.array([sds.loc[(sysname, c)] for c in conditions])
    axes[0].bar(x + (k-2.5)*width, vals, width, yerr=errs, capsize=2,
                edgecolor="black", linewidth=0.5, label=labels[sysname])
axes[0].set_xticks(x, ["Clean", "Dropout", "Mild", "Noise", "Strong"], rotation=20, ha="right")
axes[0].set_ylabel("Mean $F_1$")
axes[0].set_ylim(0.45, 1.02)
axes[0].grid(axis="y", linestyle=":", alpha=0.6)

# (b) delta from each system's clean arm
for sysname in systems:
    clean = means.loc[(sysname, "Clean")]
    vals = [means.loc[(sysname, c)] - clean for c in conditions[1:]]
    axes[1].plot(np.arange(4), vals, marker="o", label=labels[sysname])
axes[1].axhline(0, linewidth=1.0)
axes[1].set_xticks(np.arange(4), ["Dropout", "Mild", "Noise", "Strong"], rotation=20, ha="right")
axes[1].set_ylabel(r"Change from clean $\Delta F_1$")
axes[1].grid(linestyle=":", alpha=0.6)

# (c) paired crew gain for Gemma and GBM across conditions.
pairs = [("Single_Agent", "ADAM_LLM", "Gemma"), ("GBM_Rich", "ADAM_GBM", "GBM")]
for standalone, crew, label in pairs:
    gains = [means.loc[(crew, c)] - means.loc[(standalone, c)] for c in conditions]
    axes[2].plot(np.arange(5), gains, marker="o", linewidth=2, label=label)
axes[2].axhline(0, linewidth=1.0)
axes[2].set_xticks(np.arange(5), ["Clean", "Dropout", "Mild", "Noise", "Strong"], rotation=20, ha="right")
axes[2].set_ylabel(r"In-crew $-$ standalone $\Delta F_1$")
axes[2].grid(linestyle=":", alpha=0.6)
axes[2].legend(frameon=False)

for i, ax in enumerate(axes):
    ax.text(-0.12, 1.03, f"({chr(97+i)})", transform=ax.transAxes,
            fontsize=15, fontweight="bold")
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.36, -0.01))
fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=2.0)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"figure_degraded_conditions.{ext}"))
plt.close(fig)
print("wrote figure_degraded_conditions.{pdf,png}")
