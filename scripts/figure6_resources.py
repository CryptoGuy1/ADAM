#!/usr/bin/env python3
"""Figure 6 and Table 7 for Section 4.3, computed from the deposited workbook.

Panel (a): per-window peak CPU over the deployment, colored by operational
state, against the C2 budget (80%) and the sustained utilization outside
active-inference windows.

Panel (b): per-node memory by state as the stacked reconciliation of
16_Memory_Budget, annotated with the measured state means from
07_D2_Resource_Log.

Table 7: per-system CPU, memory, and bandwidth from 03_D1_Trial_Results.
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
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12.5,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
INK = "#1a1a1a"
GRAY = "#8a8a8a"

WB = sys.argv[1] if len(sys.argv) > 1 else "data/ADAM_Dataset_Master.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

r = pd.read_excel(WB, sheet_name="07_D2_Resource_Log")
t0 = r["Timestamp"].min()
hours = (r["Timestamp"] - t0).dt.total_seconds() / 3600.0
sustained = r.loc[r["State"] != "crew_active", "CPU_Peak_%"].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 6.2),
                               gridspec_kw={"width_ratios": [1.35, 1.0]})

# ---- (a) CPU timeline against the budget
STATE_STYLE = {
    "idle": (GREEN, "o", "Idle"),
    "monitoring": (ORANGE, "s", "Monitoring"),
    "crew_active": (BLUE, "^", "Crew active"),
}
for state, (color, marker, label) in STATE_STYLE.items():
    g = r[r["State"] == state]
    ax1.scatter(hours[g.index], g["CPU_Peak_%"], s=16, c=color, marker=marker,
                edgecolors="none", alpha=0.75, zorder=3,
                label=f"{label} ($n={len(g)}$)")

ax1.axhline(80, color=RED, linestyle="--", linewidth=1.6, zorder=2)
ax1.text(13.0, 82.0, "C2 budget: 80% sustained", fontsize=13, color=RED)
ax1.axhline(sustained, color=INK, linestyle=":", linewidth=1.4, zorder=2)
ax1.text(13.0, sustained + 2.2,
         f"Sustained (outside inference): {sustained:.1f}%",
         fontsize=13, color=INK,
         bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                   edgecolor="none", alpha=0.85))
ax1.set_xlabel("Deployment time (hours from first sample)", labelpad=10)
ax1.set_ylabel("Per-window peak CPU (%)")
ax1.set_ylim(0, 104)
ax1.set_xlim(-1, hours.max() + 1)
ax1.grid(linestyle=":", linewidth=0.7, alpha=0.6)
ax1.set_axisbelow(True)
ax1.legend(frameon=False, loc="center left", handletextpad=0.2,
           borderaxespad=0.3)
ax1.text(-0.115, 1.03, "(a)", transform=ax1.transAxes, fontsize=19,
         fontweight="bold")

# ---- (b) stacked memory reconciliation by state
COMPONENTS = [
    ("Base OS and services", 1050, "#c7c7c7"),
    ("Sensing and logging", 180, "#9e9e9e"),
    ("ADAM core services", 300, "#6f6f6f"),
    ("Weaviate and index cache", 520, ORANGE),
    ("PoA client and buffers", 250, "#f2c14e"),
    ("Working buffers", 50, "#b8860b"),
    ("Model weights (INT4)", 1300, BLUE),
    ("Ollama runtime and KV cache", 240, SKY),
]
STATE_STACK = {"Idle": 3, "Monitoring": 6, "Crew active": 8}
measured = {"Idle": 1530.2, "Monitoring": 2349.3, "Crew active": 3890.0}

x = np.arange(len(STATE_STACK))
for xi, (state, ncomp) in enumerate(STATE_STACK.items()):
    bottom = 0.0
    for name, mb, color in COMPONENTS[:ncomp]:
        ax2.bar(xi, mb, 0.62, bottom=bottom, color=color, edgecolor=INK,
                linewidth=0.8, zorder=3,
                label=name if xi == len(STATE_STACK) - 1 else None)
        bottom += mb
    ax2.text(xi, bottom + 70, f"{measured[state]:,.0f} MB\nmeasured mean",
             ha="center", va="bottom", fontsize=12.5, color=INK)

ax2.set_xticks(x)
ax2.set_xticklabels(list(STATE_STACK.keys()))
ax2.set_ylabel("Per-node memory (MB)")
ax2.set_ylim(0, 4650)
ax2.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
ax2.set_axisbelow(True)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[::-1], labels[::-1], frameon=False, loc="upper left",
           fontsize=11.5, handlelength=1.1, handletextpad=0.4,
           labelspacing=0.35)
ax2.text(-0.155, 1.03, "(b)", transform=ax2.transAxes, fontsize=19,
         fontweight="bold")

fig.tight_layout(w_pad=3.2)
fig.savefig(os.path.join(OUT, "figure6_resources.pdf"))
fig.savefig(os.path.join(OUT, "figure6_resources.png"))
plt.close(fig)

# ---------------------------------------------------------------- Table 7
tr = pd.read_excel(WB, sheet_name="03_D1_Trial_Results")
ROWS = [
    ("ADAM (Full)", "ADAM_Full"),
    ("Static Threshold", "Static_Threshold"),
    ("Random Forest", "Random_Forest"),
    ("Cloud-Only", "Cloud_Only"),
    ("Single-Agent", "SingleAgent"),
    ("ADAM-No-Aggregator", "ADAM_NoAgg"),
    ("ADAM-No-LLM", "ADAM_NoLLM"),
    ("ADAM-No-Blockchain", "ADAM_NoBlockchain"),
    ("ADAM-No-Weaviate", "ADAM_NoWeaviate"),
]
lines = []
for label, sysname in ROWS:
    g = tr[tr["System"] == sysname]
    cpu = g["CPU_WindowPeak_Mean_%"].mean()
    ram = g["RAM_MB"].mean()
    bw = g["Bandwidth_KB_per_60s_4nodes"].mean()
    lines.append(f"{label} & {cpu:.1f} & {ram:,.0f} & {bw:.1f} \\\\")

table = r"""\begin{table}[H]
\centering
\caption{Per-system resource profile during active operation, measured from matched Raspberry~Pi resource windows.}
\label{tab:resources}
\renewcommand{\arraystretch}{1.15}
\footnotesize
\begin{threeparttable}
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{System} & \textbf{CPU (\%)\tnote{a}} & \textbf{Memory (MB)} & \textbf{Bandwidth (KB/min)\tnote{b}} \\
\midrule
""" + "\n".join(lines[:5]) + r"""
\midrule
""" + "\n".join(lines[5:]) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\footnotesize
\item[a] Mean of per-window peak CPU utilization during active windows; sustained utilization outside inference is reported in the text.
\item[b] Aggregate transfer across the four nodes per 60-second measurement window. For Cloud-Only this includes external API traffic; Section~\ref{sec:confidentiality} separates traffic that crosses the deployment boundary.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
with open(os.path.join(OUT, "table7_resources.tex"), "w") as fh:
    fh.write(table)

print(f"sustained outside inference {sustained:.2f}% | samples {len(r)} | "
      f"span {hours.max():.1f} h")
for label, sysname in ROWS:
    g = tr[tr["System"] == sysname]
    print(f"  {label:22s} {g['CPU_WindowPeak_Mean_%'].mean():5.1f}  "
          f"{g['RAM_MB'].mean():7.1f}  {g['Bandwidth_KB_per_60s_4nodes'].mean():6.1f}")
