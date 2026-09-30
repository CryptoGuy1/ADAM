#!/usr/bin/env python3
"""Figure 4 operating-point analysis for Section 4.1, computed from the deposited workbook.

Panel (a): recall decomposed by gate region (above / below the 1,000 ppm
screening threshold, and overall) for Static Threshold, ADAM under deployment
mode, and ADAM under benchmark mode. Event-level, pooled over the 2,000 D1
events, with region sizes annotated.

Panel (b): per-trial precision, F1, and false-alarm rate (mean over the 10
trials, error bars one standard deviation) for the same three systems.

The benchmark table and statistical family are verified separately by
``scripts/verify_manuscript_numbers.py``.
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
SKY = "#56B4E9"
INK = "#1a1a1a"

WB = sys.argv[1] if len(sys.argv) > 1 else "data/ADAM_Dataset_Master_v14_reconciled.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

xl = pd.ExcelFile(WB)
pred = pd.read_excel(xl, "06A_Event_Predictions", header=3).dropna(subset=["Event_ID"])
trig = pd.read_excel(xl, "D1_RawTrigger_Log")
m = trig.merge(pred[["Event_ID", "ADAM_LLM", "Static_Threshold"]], on="Event_ID")

above = m["Raw_Instantaneous_PPM"] >= 1000
anom = m["Ground_Truth_Label"] == "anomaly"

def recall_regions(col):
    ra = ((m[col] == "anomaly") & anom & above).sum() / (anom & above).sum()
    rb = ((m[col] == "anomaly") & anom & ~above).sum() / (anom & ~above).sum()
    ro = ((m[col] == "anomaly") & anom).sum() / anom.sum()
    return [ra, rb, ro]

m["ADAM_Derived"] = np.where(above, m["ADAM_LLM"], "normal")

systems = [
    ("Static Threshold", "Static_Threshold", ORANGE),
    ("ADAM (deployment)", "ADAM_Derived", SKY),
    ("ADAM (benchmark)", "ADAM_LLM", BLUE),
]

# Per-trial metrics for panel (b)
tr = pd.read_excel(xl, "03_D1_Trial_Results")

def per_trial(name):
    g = tr[tr["System"] == name]
    out = {}
    for c in ("Precision", "F1", "FAR"):
        out[c] = (g[c].mean(), g[c].std(ddof=1))
    return out

def deployment_per_trial():
    rows = []
    for _trial, g in m.groupby("Trial", sort=True):
        yy = g["Ground_Truth_Label"].str.lower().eq("anomaly")
        pp = g["ADAM_Derived"].str.lower().eq("anomaly")
        tp = int((yy & pp).sum()); fp = int((~yy & pp).sum())
        fn = int((yy & ~pp).sum()); tn = int((~yy & ~pp).sum())
        precision = tp / (tp + fp); recall = tp / (tp + fn)
        rows.append({
            "Precision": precision,
            "F1": 2 * precision * recall / (precision + recall),
            "FAR": fp / (fp + tn),
        })
    g = pd.DataFrame(rows)
    return {c: (g[c].mean(), g[c].std(ddof=1)) for c in ("Precision", "F1", "FAR")}

pt = {
    "Static Threshold": per_trial("Static_Threshold"),
    "ADAM (deployment)": deployment_per_trial(),
    "ADAM (benchmark)": per_trial("ADAM_LLM"),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0))

# ---- (a) recall by gate region
regions = [
    f"Above gate\n($n={int((anom & above).sum())}$)",
    f"Below gate\n($n={int((anom & ~above).sum())}$)",
    f"Overall\n($n={int(anom.sum())}$)",
]
x = np.arange(len(regions))
w = 0.26
for k, (label, col, color) in enumerate(systems):
    vals = recall_regions(col)
    bars = ax1.bar(x + (k - 1) * w, vals, w, color=color, edgecolor=INK,
                   linewidth=1.0, label=label, zorder=3)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=12, color=INK)
ax1.set_xticks(x)
ax1.set_xticklabels(regions)
ax1.set_ylabel("Recall")
ax1.set_ylim(0, 1.30)
ax1.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
ax1.set_axisbelow(True)
ax1.legend(frameon=False, loc="upper right", borderaxespad=0.2)
ax1.text(-0.11, 1.03, "(a)", transform=ax1.transAxes, fontsize=19,
         fontweight="bold")

# ---- (b) per-trial precision / F1 / FAR
metrics = ["Precision", "F1", "FAR"]
x2 = np.arange(len(metrics))
for k, (label, _col, color) in enumerate(systems):
    means = [pt[label][c][0] for c in metrics]
    sds = [pt[label][c][1] for c in metrics]
    bars = ax2.bar(x2 + (k - 1) * w, means, w, yerr=sds, capsize=4,
                   color=color, edgecolor=INK, linewidth=1.0,
                   error_kw={"elinewidth": 1.2, "ecolor": INK}, zorder=3,
                   label=label)
    for b, v in zip(bars, means):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.035, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=12, color=INK)
ax2.set_xticks(x2)
ax2.set_xticklabels(["Precision", "$F_1$", "FAR"])
ax2.set_ylabel("Per-trial mean")
ax2.set_ylim(0, 1.12)
ax2.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
ax2.set_axisbelow(True)
ax2.text(-0.11, 1.03, "(b)", transform=ax2.transAxes, fontsize=19,
         fontweight="bold")

fig.tight_layout(w_pad=3.5)
fig.savefig(os.path.join(OUT, "figure4_operating_point.pdf"))
fig.savefig(os.path.join(OUT, "figure4_operating_point.png"))
plt.close(fig)

print("wrote figure4_operating_point.{pdf,png}")
for label, _c, _k in systems:
    p = pt[label]
    print(f"  {label:20s} P {p['Precision'][0]:.3f}±{p['Precision'][1]:.3f}  "
          f"F1 {p['F1'][0]:.3f}±{p['F1'][1]:.3f}  FAR {p['FAR'][0]:.3f}±{p['FAR'][1]:.3f}")
