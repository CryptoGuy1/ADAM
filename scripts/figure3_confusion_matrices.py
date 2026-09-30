#!/usr/bin/env python3
"""Figure 3: mean per-trial confusion matrices from the deposited D1 results."""
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

WB = sys.argv[1] if len(sys.argv) > 1 else "data/ADAM_Dataset_Master_v14_reconciled.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "font.size": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

df = pd.read_excel(WB, sheet_name="03_D1_Trial_Results")
SYSTEMS = [
    ("Static_Threshold", "Static Threshold"),
    ("Random_Forest", "Random Forest (raw)"),
    ("RandomForest_Fused", "Random Forest (fused)"),
    ("ADAM_LLM", "ADAM_LLM"),
]

fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.8))
for ax, (key, title) in zip(axes, SYSTEMS):
    g = df[df["System"] == key]
    if len(g) != 10:
        raise SystemExit(f"expected 10 rows for {key}, found {len(g)}")
    # Rows: true NORMAL/ANOMALY; columns: predicted NORMAL/ANOMALY.
    cm = np.array([[g["TN"].mean(), g["FP"].mean()],
                   [g["FN"].mean(), g["TP"].mean()]])
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=110)
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    color="white" if value > 55 else "black", fontsize=13,
                    fontweight="bold")
    ax.set_xticks([0, 1], ["NORMAL", "ANOMALY"], rotation=20, ha="right")
    ax.set_yticks([0, 1], ["NORMAL", "ANOMALY"])
    ax.set_xlabel("Predicted")
    if ax is axes[0]:
        ax.set_ylabel("Reference")
    ax.set_title(title)
    ax.text(-0.10, 1.04, f"({chr(97 + list(axes).index(ax))})",
            transform=ax.transAxes, fontweight="bold", fontsize=14)

fig.subplots_adjust(left=0.055, right=0.99, bottom=0.20, top=0.86, wspace=0.36)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"figure3_confusion_matrices.{ext}"))
plt.close(fig)
print("wrote figure3_confusion_matrices.{pdf,png}")
