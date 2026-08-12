#!/usr/bin/env python3
"""Figure 4 and Table 5 for Section 4.1, computed from the deposited workbook.

Panel (a): recall decomposed by gate region (above / below the 1,000 ppm
screening threshold, and overall) for Static Threshold, ADAM under deployment
mode, and ADAM under benchmark mode. Event-level, pooled over the 2,000 D1
events, with region sizes annotated.

Panel (b): per-trial precision, F1, and false-alarm rate (mean over the 10
trials, error bars one standard deviation) for the same three systems.

Table 5: benchmark-mode comparison of all nine systems with the trial-level
Wilcoxon results, plus a descriptive deployment-mode row for ADAM outside the
comparison family.
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

WB = sys.argv[1] if len(sys.argv) > 1 else "data/ADAM_Dataset_Master.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

xl = pd.ExcelFile(WB)
pred = pd.read_excel(xl, "06A_Event_Predictions", header=3).dropna(subset=["Event_ID"])
trig = pd.read_excel(xl, "D1_RawTrigger_Log")
m = trig.merge(pred[["Event_ID", "ADAM_Full", "Static_Threshold"]], on="Event_ID")

above = m["Raw_Instantaneous_PPM"] >= 1000
anom = m["Ground_Truth_Label"] == "anomaly"

def recall_regions(col):
    ra = ((m[col] == "anomaly") & anom & above).sum() / (anom & above).sum()
    rb = ((m[col] == "anomaly") & anom & ~above).sum() / (anom & ~above).sum()
    ro = ((m[col] == "anomaly") & anom).sum() / anom.sum()
    return [ra, rb, ro]

systems = [
    ("Static Threshold", "Static_Threshold", ORANGE),
    ("ADAM (deployment)", "ADAM_Prediction", SKY),
    ("ADAM (benchmark)", "ADAM_Full", BLUE),
]

# Per-trial metrics for panel (b)
tr = pd.read_excel(xl, "03_D1_Trial_Results")
gated = pd.read_excel(xl, "D1_RawTrigger_Summary")
gated = gated[gated["Trial"].astype(str) != "Overall"]

def per_trial(name):
    if name == "gated":
        g = gated
    else:
        g = tr[tr["System"] == name]
    out = {}
    for c in ("Precision", "F1", "FAR"):
        out[c] = (g[c].mean(), g[c].std(ddof=1))
    return out

pt = {
    "Static Threshold": per_trial("Static_Threshold"),
    "ADAM (deployment)": per_trial("gated"),
    "ADAM (benchmark)": per_trial("ADAM_Full"),
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

# ---------------------------------------------------------------- Table 5
tests = pd.read_excel(xl, "04_D1_Statistical_Tests", header=1)
tests = tests.set_index(tests["Comparison"].astype(str))

ROWS = [
    ("ADAM (Full)", "ADAM_Full", None),
    ("Static Threshold", "Static_Threshold", "ADAM_vs_Static"),
    ("Random Forest", "Random_Forest", "ADAM_vs_RF"),
    ("Cloud-Only", "Cloud_Only", "ADAM_vs_Cloud"),
    ("Single-Agent", "SingleAgent", "ADAM_vs_SingleAgent"),
    ("ADAM-No-Aggregator", "ADAM_NoAgg", "ADAM_vs_NoAgg"),
    ("ADAM-No-LLM", "ADAM_NoLLM", "ADAM_vs_NoLLM"),
    ("ADAM-No-Blockchain", "ADAM_NoBlockchain", "ADAM_vs_NoBlockchain"),
    ("ADAM-No-Weaviate", "ADAM_NoWeaviate", "ADAM_vs_NoWeaviate"),
]

def cell(mean, sd):
    return f"{mean:.3f} $\\pm$ {sd:.3f}"

lines = []
for label, sysname, test in ROWS:
    g = tr[tr["System"] == sysname]
    row = [label]
    for c in ("Precision", "Recall", "F1", "FAR"):
        row.append(cell(g[c].mean(), g[c].std(ddof=1)))
    if test is None:
        row += ["--", "--"]
    else:
        t = tests.loc[test]
        pe = float(t["P_Exact"])
        ph = float(t["P_Holm"])
        mark = "\\tnote{a}" if sysname == "ADAM_NoBlockchain" else ""
        row += [f"{pe:.3f}{mark}", f"{ph:.3f}"]
    lines.append(" & ".join(row) + " \\\\")

# Deployment-mode descriptive row
dep = ["ADAM (deployment mode)\\tnote{b}"]
for c in ("Precision", "Recall", "F1", "FAR"):
    dep.append(cell(gated[c].mean(), gated[c].std(ddof=1)))
dep += ["--", "--"]
dep_line = " & ".join(dep) + " \\\\"

table = r"""\begin{table}[H]
\centering
\caption{Anomaly detection performance across systems on the labeled trials.}
\label{tab:detection}
\renewcommand{\arraystretch}{1.15}
\footnotesize
\begin{threeparttable}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{System} & \textbf{Prec.} & \textbf{Recall} & \textbf{$F_1$} & \textbf{FAR} & $p_{\text{exact}}$ & $p_{\text{Holm}}$ \\
\midrule
""" + "\n".join(lines[:5]) + r"""
\midrule
""" + "\n".join(lines[5:]) + r"""
\midrule
""" + dep_line + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\footnotesize
\item Values are mean $\pm$ standard deviation across the 10 labeled trials under the benchmark mode of Section~\ref{sec:evalmodes}. $p_{\text{exact}}$ is a two-sided exact Wilcoxon signed-rank test on paired per-trial $F_1$ against ADAM~(Full); $p_{\text{Holm}}$ applies the Holm correction across the eight comparisons. The $p_{\text{exact}}=0.002$ floor is the smallest value attainable at $n=10$ and corresponds to all 10 trials favoring ADAM. Random Forest uses 100 estimators with \texttt{max\_depth=5}. Cloud-Only uses GPT-4o-mini through the OpenAI API; all other LLM-based systems use on-device Gemma~3 1B inference.
\item[a] One trial produced identical $F_1$ for both systems and is discarded by the signed-rank test, giving an effective $n=9$.
\item[b] Deployment mode of Section~\ref{sec:evalmodes}: readings below the screening threshold are classified normal without crew formation. Reported descriptively; not part of the benchmark comparison family.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
with open(os.path.join(OUT, "table5_detection.tex"), "w") as fh:
    fh.write(table)

print("wrote figure4_operating_point.{pdf,png} and table5_detection.tex")
for label, _c, _k in systems:
    p = pt[label]
    print(f"  {label:20s} P {p['Precision'][0]:.3f}±{p['Precision'][1]:.3f}  "
          f"F1 {p['F1'][0]:.3f}±{p['F1'][1]:.3f}  FAR {p['FAR'][0]:.3f}±{p['FAR'][1]:.3f}")
