#!/usr/bin/env python3
"""Figure 7 and Table 8 for Section 4.4, computed from the deposited workbook.

Panel (a): hardware domain N = 1-4. Raw replicates, level means with SD,
power-law fit, and the matched simulation runs as hollow validation markers.

Panel (b): scale-out domain N = 4-16, same treatment, labeled as model-based.

Panel (c): hardware stage decomposition against node count -- the measured
account of where the latency growth originates.

Table 8: fitted models and simulator validation with provenance.
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
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "font.size": 14.5,
    "axes.labelsize": 15.5,
    "xtick.labelsize": 13.5,
    "ytick.labelsize": 13.5,
    "legend.fontsize": 11.8,
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
PURPLE = "#CC79A7"
INK = "#1a1a1a"

WB = sys.argv[1] if len(sys.argv) > 1 else "data/ADAM_Dataset_Master.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

s = pd.read_excel(WB, sheet_name="08_Scalability_Log")
hw = s[s["Run_Mode"] == "HARDWARE"]
sim = s[s["Run_Mode"] == "PYTHON_SIMULATION"]
so = s[(s["Fit_Eligible"].astype(str).str.lower() == "yes") & (s["Node_Count"] >= 4)]

model = lambda N, t0, a, b: t0 + a * np.power(N, b)


def fit(df, p0):
    g = df.groupby("Node_Count")["T_decision_ms"].mean()
    popt, _ = curve_fit(model, g.index.values.astype(float), g.values,
                        p0=p0, maxfev=60000)
    xr = df["Node_Count"].values.astype(float)
    yr = df["T_decision_ms"].values.astype(float)
    r2_raw = 1 - np.sum((yr - model(xr, *popt)) ** 2) / np.sum((yr - yr.mean()) ** 2)
    ym = df.groupby("Node_Count")["T_decision_ms"].mean()
    r2_mean = 1 - np.sum((ym.values - model(ym.index.values.astype(float), *popt)) ** 2) \
        / np.sum((ym.values - ym.values.mean()) ** 2)
    return popt, r2_raw, r2_mean, len(df)


pw_hw, r2r_hw, r2m_hw, n_hw = fit(hw, [17000, 200, 1.4])
pw_so, r2r_so, r2m_so, n_so = fit(so, [14000, 3000, 0.3])

# Validation statistics on matched levels
hwm = hw.groupby("Node_Count")["T_decision_ms"].mean()
simm = sim.groupby("Node_Count")["T_decision_ms"].mean()
rel = [(simm[k] - hwm[k]) / hwm[k] for k in sorted(set(hwm.index) & set(simm.index))]
mape, bias = np.mean(np.abs(rel)) * 100, np.mean(rel) * 100

RNG = np.random.default_rng(42)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19.5, 5.9))


def domain_panel(ax, df, popt, r2raw, color, marker, label, sim_overlay=None):
    levels = sorted(df["Node_Count"].unique())
    for n in levels:
        y = df.loc[df["Node_Count"] == n, "T_decision_ms"] / 1000
        x = n + RNG.uniform(-0.13, 0.13, len(y))
        ax.scatter(x, y, s=13, color=color, alpha=0.30, edgecolors="none",
                   zorder=2)
    g = df.groupby("Node_Count")["T_decision_ms"].agg(["mean", "std"]) / 1000
    fine = np.linspace(min(levels), max(levels), 200)
    ax.plot(fine, model(fine, *popt) / 1000, "-", color=RED, linewidth=2.2,
            zorder=3,
            label=f"$T(N)={popt[0]:,.0f}+{popt[1]:,.0f}N^{{{popt[2]:.2f}}}$")
    ax.errorbar(g.index, g["mean"], yerr=g["std"], fmt=marker, color=color,
                markersize=9.5, markeredgecolor=INK, markeredgewidth=1.1,
                capsize=4.5, linewidth=0, elinewidth=1.5, ecolor=INK, zorder=4,
                label=label)
    if sim_overlay is not None:
        gs = sim_overlay.groupby("Node_Count")["T_decision_ms"].agg(["mean", "std"]) / 1000
        ax.errorbar(gs.index + 0.16, gs["mean"], yerr=gs["std"], fmt="s",
                    color="none", markersize=8.5, markeredgecolor=INK,
                    markeredgewidth=1.3, capsize=4, linewidth=0,
                    elinewidth=1.1, ecolor=INK, zorder=4,
                    label="Scale-out model (validation)")
    ax.set_xticks(levels)
    ax.grid(linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.text(0.965, 0.055, f"$R^2={r2raw:.2f}$ (raw)", transform=ax.transAxes,
            ha="right", fontsize=13, color="#4d4d4d")
    return g


g1 = domain_panel(ax1, hw, pw_hw, r2r_hw, BLUE, "o",
                  "Hardware (Pi 5), $n=18$/level", sim_overlay=sim)
grow_hw = 100 * (g1["mean"].loc[4] / g1["mean"].loc[1] - 1)
ax1.set_xlabel("Nodes $N$ (hardware, measured)", labelpad=9)
ax1.set_ylabel("Decision latency (s)", labelpad=9)
ax1.set_ylim(16.2, 20.4)
ax1.legend(frameon=False, loc="upper left", handletextpad=0.3)
ax1.text(0.965, 0.13, "+%.1f%% from $N{=}1$ to $N{=}4$" % grow_hw,
         transform=ax1.transAxes, ha="right", fontsize=13, color=INK)
ax1.text(-0.155, 1.04, "(a)", transform=ax1.transAxes, fontsize=19,
         fontweight="bold")

g2 = domain_panel(ax2, so, pw_so, r2r_so, SKY, "s", "Scale-out model, $n=18$/level")
grow_so = 100 * (g2["mean"].loc[16] / g2["mean"].loc[4] - 1)
ax2.set_xlabel("Effective nodes $N$ (scale-out model)", labelpad=9)
ax2.set_ylabel("Decision latency (s)", labelpad=9)
ax2.set_ylim(17.4, 22.6)
ax2.legend(frameon=False, loc="upper left", handletextpad=0.3)
ax2.text(0.965, 0.13, "+%.1f%% from $N{=}4$ to $N{=}16$" % grow_so,
         transform=ax2.transAxes, ha="right", fontsize=13, color=INK)
ax2.text(-0.155, 1.04, "(b)", transform=ax2.transAxes, fontsize=19,
         fontweight="bold")

# ---- (c) hardware stage decomposition
STAGES = [
    ("Reasoning $T_{\\mathrm{reason}}$", "T_reason_ms", BLUE, "o"),
    ("Cross-node exchange", "T_cross_node_ms", ORANGE, "s"),
    ("Blockchain commit", "T_blockchain_ms", GREEN, "^"),
    ("Memory retrieval", "T_query_ms", PURPLE, "D"),
    ("Network transfer", "T_network_ms", RED, "v"),
    ("Result merge", "T_merge_ms", "#8a8a8a", "P"),
]
gm = hw.groupby("Node_Count")[[c for _, c, _, _ in STAGES]].mean()
for label, col, color, marker in STAGES:
    ax3.plot(gm.index, gm[col], marker=marker, color=color, linewidth=1.8,
             markersize=7.5, markeredgecolor=INK, markeredgewidth=0.8,
             zorder=3, label=label)
ax3.set_yscale("log")
ax3.set_ylim(25, 8e4)
ax3.set_xticks([1, 2, 3, 4])
ax3.set_xlabel("Nodes $N$ (hardware, measured)", labelpad=9)
ax3.set_ylabel("Mean stage latency (ms, log scale)", labelpad=9)
ax3.grid(linestyle=":", linewidth=0.7, alpha=0.6, which="both")
ax3.set_axisbelow(True)
ax3.legend(frameon=False, loc="center right", fontsize=10.8,
           handletextpad=0.3, labelspacing=0.3)
ax3.text(1.0, 2600, "Cross-node exchange is $0$ at $N{=}1$\n(single node); plotted from $N{=}2$.",
         fontsize=11.0, color=INK, ha="left", va="center")
ax3.text(-0.175, 1.04, "(c)", transform=ax3.transAxes, fontsize=19,
         fontweight="bold")

fig.tight_layout(w_pad=2.6)
fig.savefig(os.path.join(OUT, "figure7_scalability.pdf"))
fig.savefig(os.path.join(OUT, "figure7_scalability.png"))
plt.close(fig)

# ---------------------------------------------------------------- Table 8
table = rf"""\begin{{table}}[H]
\centering
\caption{{Node-scaling models and simulator validation. Load is fixed at the reference configuration throughout, so latency changes attribute to node count.}}
\label{{tab:scalability}}
\renewcommand{{\arraystretch}}{{1.2}}
\footnotesize
\begin{{threeparttable}}
\begin{{tabular}}{{@{{}}llccccc@{{}}}}
\toprule
\textbf{{Domain}} & \textbf{{Basis}} & $N$ & $n$ & $T_0$ (ms) & $\alpha$ & $\beta$ \\
\midrule
Hardware & Measured (Raspberry~Pi~5) & 1--4 & {n_hw} & {pw_hw[0]:,.0f} & {pw_hw[1]:.1f} & {pw_hw[2]:.3f} \\
Scale-out & Validated software model & 4--16 & {n_so} & {pw_so[0]:,.0f} & {pw_so[1]:,.1f} & {pw_so[2]:.3f} \\
\midrule
\multicolumn{{7}}{{@{{}}l}}{{Validation on matched levels ($N=1$--$4$, 18 replicates each): MAPE ${mape:.2f}\%$, mean bias ${bias:+.2f}\%$.}} \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item Fits are $T(N)=T_0+\alpha N^{{\beta}}$ to level means. $R^2$ evaluated over the raw replicate observations is ${r2r_hw:.2f}$ (hardware) and ${r2r_so:.2f}$ (scale-out); evaluated over level means it is ${r2m_hw:.2f}$ and ${r2m_so:.2f}$, as recorded in the deposited fitted-models sheet, the difference being the within-level replicate variance that averaging removes. The scale-out results at $N>4$ are conditional on the validation above.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
with open(os.path.join(OUT, "table8_scalability.tex"), "w") as fh:
    fh.write(table)

# ---------------------------------------------------------------- verify
print(f"HW fit T0={pw_hw[0]:,.0f} a={pw_hw[1]:.2f} b={pw_hw[2]:.3f} "
      f"R2raw={r2r_hw:.3f} R2mean={r2m_hw:.3f}")
print(f"SO fit T0={pw_so[0]:,.0f} a={pw_so[1]:.2f} b={pw_so[2]:.3f} "
      f"R2raw={r2r_so:.3f} R2mean={r2m_so:.3f}")
print(f"validation MAPE {mape:.3f}% bias {bias:+.3f}%")
print(f"growth: HW +{grow_hw:.1f}%  SO +{grow_so:.1f}%  "
      f"overall {100*(g2['mean'].loc[16]/g1['mean'].loc[1]-1):.1f}%")
tot_growth = (gm.loc[4].sum() - gm.loc[1].sum())
coord = gm[[c for c in gm.columns if c != "T_reason_ms"]]
print(f"stage-sum check: N1 {gm.loc[1].sum():,.0f} vs "
      f"{hw[hw.Node_Count==1]['T_decision_ms'].mean():,.0f} | "
      f"N4 {gm.loc[4].sum():,.0f} vs {hw[hw.Node_Count==4]['T_decision_ms'].mean():,.0f}")
print(f"growth split N1->N4: coordination +{coord.loc[4].sum()-coord.loc[1].sum():.0f} ms, "
      f"reasoning +{gm.loc[4,'T_reason_ms']-gm.loc[1,'T_reason_ms']:.0f} ms, "
      f"total +{tot_growth:.0f} ms")
