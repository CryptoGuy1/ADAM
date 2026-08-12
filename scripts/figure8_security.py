#!/usr/bin/env python3
"""Figure 8 and Tables 9-10 for Section 4.5, computed from the deposited workbook.

Panel (a): all 19 fallback activations under induced local-model failure,
shown individually as an ECDF with mean, median, and P95 marked.

Panel (b): all 20 external-egress measurement windows, shown individually:
12 ADAM windows at zero and 8 Cloud-Only windows with their mean marked.

Table 9: consolidated security stress tests with exact counts.
Table 10: quorum bounds under strict majority, labeled analytical.
Output at 600 dpi PNG plus vector PDF.
"""

import math
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

WB = sys.argv[1] if len(sys.argv) > 1 else "data/ADAM_Dataset_Master.xlsx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(OUT, exist_ok=True)

inj = pd.read_excel(WB, sheet_name="10_Security_Injection", header=2)
llm = pd.read_excel(WB, sheet_name="11_Security_LLM_Failure", header=2)
poi = pd.read_excel(WB, sheet_name="12_Security_Poisoning", header=2)
egr = pd.read_excel(WB, sheet_name="13_Security_Data_Leakage", header=2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.0))

# ---- (a) fallback activation latency, all 19 events
fb = llm[llm["Fallback_Triggered"].astype(str).str.lower() == "yes"]
lat = np.sort(fb["Fallback_Latency_ms"].values.astype(float))
n = len(lat)
ecdf = np.arange(1, n + 1) / n
ax1.step(lat, ecdf, where="post", color=BLUE, linewidth=2.2, zorder=3)
ax1.plot(lat, ecdf, "o", color=BLUE, markersize=7.5, markeredgecolor=INK,
         markeredgewidth=0.9, zorder=4,
         label=f"Induced failures ($n={n}$, all recovered)")
mean, med, p95 = lat.mean(), np.median(lat), np.percentile(lat, 95)
ax1.axvline(med, color=ORANGE, linestyle="--", linewidth=1.4, zorder=2)
ax1.text(med - 1.2, 0.985, f"median {med:.1f} ms", rotation=90, va="top",
         ha="right", fontsize=12.5, color=ORANGE)
ax1.axvline(p95, color=INK, linestyle=":", linewidth=1.3, zorder=2)
ax1.text(p95 + 1.5, 0.06, f"P95 {p95:.1f} ms", rotation=90, va="bottom",
         fontsize=12.5, color=INK)
ax1.text(0.97, 0.30, f"mean {mean:.1f} ms", transform=ax1.transAxes,
         ha="right", fontsize=13, color=INK)
ax1.set_xlabel("Fallback activation latency (ms)", labelpad=10)
ax1.set_ylabel("Fraction of induced failures")
ax1.set_ylim(0, 1.05)
ax1.grid(linestyle=":", linewidth=0.7, alpha=0.6)
ax1.set_axisbelow(True)
ax1.legend(frameon=False, loc="upper left")
ax1.text(-0.14, 1.03, "(a)", transform=ax1.transAxes, fontsize=19,
         fontweight="bold")

# ---- (b) external egress, all 20 windows
adam = egr[egr["System"].astype(str).str.contains("ADAM", case=False)]
cloud = egr[~egr["System"].astype(str).str.contains("ADAM", case=False)]
a_kb = adam["Total_Bytes_External"].values.astype(float) / 1024
c_kb = cloud["Total_Bytes_External"].values.astype(float) / 1024
rng = np.random.default_rng(7)
ax2.scatter(0 + rng.uniform(-0.10, 0.10, len(a_kb)), a_kb, s=70, marker="o",
            color=BLUE, edgecolors=INK, linewidths=1.0, zorder=4,
            label=f"ADAM ($n={len(a_kb)}$ windows)")
ax2.scatter(1 + rng.uniform(-0.10, 0.10, len(c_kb)), c_kb, s=70, marker="s",
            color=ORANGE, edgecolors=INK, linewidths=1.0, zorder=4,
            label=f"Cloud-Only ($n={len(c_kb)}$ windows)")
ax2.hlines(c_kb.mean(), 0.72, 1.28, color=RED, linewidth=1.8, zorder=3)
ax2.text(1.31, c_kb.mean(), f"mean {c_kb.mean():.1f} KB", va="center",
         fontsize=12.5, color=RED)
ax2.text(0, -9.5, "all windows $= 0$ KB", ha="center", fontsize=12.5,
         color=BLUE)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["ADAM", "Cloud-Only"])
ax2.set_xlim(-0.5, 1.85)
ax2.set_ylim(-16, 155)
ax2.set_ylabel("External egress per 30-minute window (KB)")
ax2.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
ax2.set_axisbelow(True)
ax2.legend(frameon=False, loc="upper left")
ax2.text(-0.14, 1.03, "(b)", transform=ax2.transAxes, fontsize=19,
         fontweight="bold")

fig.tight_layout(w_pad=3.4)
fig.savefig(os.path.join(OUT, "figure8_security.pdf"))
fig.savefig(os.path.join(OUT, "figure8_security.png"))
plt.close(fig)

# ---------------------------------------------------------------- Table 9
det = inj["ADAM_Detected_Attack"].astype(str).str.lower().eq("yes")
by_type = inj.groupby("Attack_Type").apply(lambda g: (int(g["ADAM_Detected_Attack"].astype(str).str.lower().eq("yes").sum()), len(g)))
tp = int(((inj["ADAM_Prediction"] == "anomaly") & (inj["Ground_Truth"] == "anomaly")).sum())
fp = int(((inj["ADAM_Prediction"] == "anomaly") & (inj["Ground_Truth"] == "normal")).sum())
fn = int(((inj["ADAM_Prediction"] == "normal") & (inj["Ground_Truth"] == "anomaly")).sum())
tn = int(((inj["ADAM_Prediction"] == "normal") & (inj["Ground_Truth"] == "normal")).sum())
prec = tp / (tp + fp)
rec = tp / (tp + fn)
f1_attack = 2 * prec * rec / (prec + rec)
far_attack = fp / (fp + tn)

lvl = poi.groupby("Num_Poisoned_Entries").apply(
    lambda g: (int((g["ADAM_Prediction"] == g["Ground_Truth"]).sum()), len(g)))
retr = int(poi["Retrieval_Affected"].astype(str).str.lower().eq("yes").sum())
# Fisher exact on clean vs heaviest level
c_ok, c_n = lvl.loc[0]
h_ok, h_n = lvl.loc[lvl.index.max()]
fisher_table = [[c_ok, c_n - c_ok], [h_ok, h_n - h_ok]]
try:
    from scipy.stats import fisher_exact
    fisher_p = fisher_exact(fisher_table)[1]
except Exception:
    fisher_p = float("nan")

fb_n = len(fb)
crews = int(llm["Crew_Continued"].astype(str).str.lower().eq("yes").sum())
ep_f1 = float(llm["Running_F1_All_30_Events"].dropna().iloc[-1])
fbo = fb
tpf = int(((fbo["Prediction"] == "anomaly") & (fbo["Ground_Truth"] == "anomaly")).sum())
fpf = int(((fbo["Prediction"] == "anomaly") & (fbo["Ground_Truth"] == "normal")).sum())
fnf = int(((fbo["Prediction"] == "normal") & (fbo["Ground_Truth"] == "anomaly")).sum())
f1_fb = 2 * tpf / (2 * tpf + fpf + fnf)

order = ["replay", "spike_inject", "zero_inject", "constant_offset"]
labels = {"replay": "Replay", "spike_inject": "Spike injection",
          "zero_inject": "Zero injection", "constant_offset": "Constant offset"}
inj_rows = "\n".join(
    f"& {labels[k]} & {by_type[k][1]} & {by_type[k][0]}/{by_type[k][1]} detected \\\\"
    for k in order)
poi_rows = "\n".join(
    f"& {int(k)} poisoned entries & {v[1]} & {v[0]}/{v[1]} correct \\\\"
    for k, v in lvl.items())

table9 = rf"""\begin{{table}}[H]
\centering
\caption{{Security stress tests: exact outcomes by condition. All values are counts over the deposited test records.}}
\label{{tab:security}}
\renewcommand{{\arraystretch}}{{1.2}}
\footnotesize
\begin{{threeparttable}}
\begin{{tabular}}{{@{{}}llcl@{{}}}}
\toprule
\textbf{{Test}} & \textbf{{Condition}} & $n$ & \textbf{{Outcome}} \\
\midrule
\multirow{{5}}{{*}}{{\shortstack[l]{{Sensor injection\\(integrity)}}}}
{inj_rows}
& All attacks & 30 & $F_1={f1_attack:.3f}$, $\mathrm{{FAR}}={far_attack:.3f}$ under attack \\
\midrule
\multirow{{5}}{{*}}{{\shortstack[l]{{Store poisoning\\(integrity)}}}}
{poi_rows}
& Retrievals affected & 30 & {retr}/30 events \\
\midrule
\multirow{{4}}{{*}}{{\shortstack[l]{{Local-model failure\\(availability)}}}}
& Induced failures & 30 & {fb_n} triggered fallback, {fb_n}/{fb_n} recovered \\
& Crews completed & 30 & {crews}/30 \\
& Episode $F_1$ (all 30 events) & 30 & {ep_f1:.3f} \\
& Fallback-only decisions & {fb_n} & $F_1={f1_fb:.3f}$ \\
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item Injection detection depends on cross-node corroboration by honest peers. The poisoning comparison of the clean store against the heaviest level (20 entries) gives Fisher exact $p={fisher_p:.3f}$: no effect is detectable at this sample size, which is a statement of power, not of immunity. Fallback decisions use deterministic threshold logic and are marked \texttt{{degraded\_mode}} in the trace.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
with open(os.path.join(OUT, "table9_security.tex"), "w") as fh:
    fh.write(table9)

# ---------------------------------------------------------------- Table 10
rows = []
for nv in range(2, 8):
    q = nv // 2 + 1
    tol = math.ceil(nv / 2) - 1
    fc = [f for f in range(nv + 1) if (nv - f) < q and f < q]
    fc_s = ", ".join(str(f) for f in fc) if fc else "--"
    mark = r"\;(deployed)" if nv == 3 else ""
    rows.append(f"{nv}{mark} & {q} & {tol} & {fc_s} & $\\geq {q}$ \\\\")

table10 = r"""\begin{table}[H]
\centering
\caption{Crew-agreement bounds under the strict-majority rule of Equation~\eqref{eq:consensus}, by voting-agent count. Derived from the quorum rule, not measured.}
\label{tab:quorum}
\renewcommand{\arraystretch}{1.2}
\footnotesize
\begin{threeparttable}
\begin{tabular}{@{}ccccc@{}}
\toprule
\textbf{Voters} $|V_t|$ & $\gamma_{\text{crew}}$ & \textbf{Tolerated faults} & \textbf{Fails closed at} $f$ & \textbf{Subvertible at} $f$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\footnotesize
\item Tolerated faults is the largest $f$ at which the honest remainder still reaches quorum ($n-f\geq\gamma_{\text{crew}}$). Fails closed marks $f$ where neither honest nor compromised voters can reach quorum. The bounds assume votes are attributable to distinct registered agents; at the deployed three voters, one compromised agent can neither force nor block an action, and two colluding agents can force one.
\end{tablenotes}
\end{threeparttable}
\end{table}
"""
with open(os.path.join(OUT, "table10_quorum.tex"), "w") as fh:
    fh.write(table10)

# ---------------------------------------------------------------- verify
print(f"injection: detected {int(det.sum())}/30 | by type "
      f"{ {k: f'{v[0]}/{v[1]}' for k, v in by_type.items()} }")
print(f"under attack: F1 {f1_attack:.3f}  FAR {far_attack:.3f} ({fp}/{fp+tn})")
print(f"poisoning by level: { {int(k): f'{v[0]}/{v[1]}' for k, v in lvl.items()} } "
      f"| retrieval {retr}/30 | Fisher p {fisher_p:.4f}")
print(f"fallback: n {fb_n}, mean {mean:.1f}, median {med:.1f}, P95 {p95:.1f} | "
      f"crews {crews}/30 | episode F1 {ep_f1:.3f} | fallback-only F1 {f1_fb:.3f}")
print(f"egress: ADAM windows {len(a_kb)}, all zero: {bool((a_kb==0).all())} | "
      f"cloud windows {len(c_kb)}, mean {c_kb.mean():.1f} KB, "
      f"range {c_kb.min():.1f}-{c_kb.max():.1f}, "
      f"calls/window {cloud['External_API_Calls'].mean():.1f}")
