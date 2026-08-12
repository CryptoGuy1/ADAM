#!/usr/bin/env python3
"""
make_figures.py - regenerate Figures 3 and 4 from the deposited workbook.

Every value plotted is read from ADAM_Dataset_Master.xlsx at run time. No
number is typed into this script, so a figure cannot disagree with the dataset.

Output is vector PDF, which is what MDPI wants: it stays sharp at any zoom and
at any print size, unlike a raster image. A 600 dpi PNG is written alongside for
drafts and slides.

    pip install pandas matplotlib openpyxl
    python scripts/make_manuscript_figures.py data/ADAM_Dataset_Master.xlsx figures/
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Type sized for a full-width MDPI figure. At \textwidth these render close to
# body-text size, which is the point: figure text should not be smaller than the
# prose around it.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,   # embed TrueType, not Type 3: required by many venues
    "ps.fonttype": 42,
})

# Okabe-Ito, the reference palette for color-vision deficiency. These three
# hues stay separable under deuteranopia, protanopia and tritanopia, which
# together cover the large majority of CVD, and they also survive greyscale
# conversion because their luminances differ.
BLUE = "#0072B2"   # ADAM (Full)
ORANGE = "#E69F00" # baselines
GREEN = "#009E73"  # ablations
INK = "#1a1a1a"    # text and axes

# Sequential fill for the confusion matrices. cividis is built so that viewers
# with CVD perceive approximately the same ordering as viewers without.
CMAP = "cividis"

# Single-line names matching Table 5, rotated on the axis. Stacked two-line
# labels collide at nine categories.
LABEL = {
    "ADAM_Full": "ADAM (Full)",
    "Static_Threshold": "Static-Threshold",
    "Random_Forest": "Random Forest",
    "Cloud_Only": "Cloud-Only",
    "SingleAgent": "Single-Agent",
    "ADAM_NoAgg": "ADAM-No-Aggregator",
    "ADAM_NoLLM": "ADAM-No-LLM",
    "ADAM_NoBlockchain": "ADAM-No-Blockchain",
    "ADAM_NoWeaviate": "ADAM-No-Weaviate",
}
ORDER = list(LABEL)


def load(path: str) -> pd.DataFrame:
    t = pd.read_excel(path, sheet_name="03_D1_Trial_Results")
    for c in ("TP", "FP", "TN", "FN", "Precision", "Recall", "F1", "FAR",
              "T_decision_ms"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t


def stats(t: pd.DataFrame, system: str) -> dict:
    g = t[t.System == system]
    return {
        # Mean counts per trial, as displayed in the matrices.
        "TN": g.TN.mean(), "FP": g.FP.mean(), "FN": g.FN.mean(), "TP": g.TP.mean(),
        # Means of the per-trial metrics, matching Table 5.
        "F1": g.F1.mean(), "FAR": g.FAR.mean(),
        "F1_sd": g.F1.std(ddof=1), "FAR_sd": g.FAR.std(ddof=1),
        "lat": g.T_decision_ms.mean() / 1000.0,
        "lat_sd": g.T_decision_ms.std(ddof=1) / 1000.0,
    }


def figure3(t: pd.DataFrame, out: str) -> None:
    """Confusion matrices for ADAM and the three comparators shown in the paper."""
    shown = ["ADAM_Full", "Static_Threshold", "Random_Forest", "SingleAgent"]
    titles = ["ADAM (Full)", "Static Threshold", "Random Forest", "Single Agent"]

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 5.0))
    for ax, sysname, title in zip(axes, shown, titles):
        s = stats(t, sysname)
        m = np.array([[s["TN"], s["FP"]], [s["FN"], s["TP"]]])
        # Shade by row proportion: each row sums to its class total, so the
        # diagonal reads as recall and specificity directly.
        shade = m / m.sum(axis=1, keepdims=True)
        im = ax.imshow(shade, cmap=CMAP, vmin=0.0, vmax=1.0)

        cmap = matplotlib.colormaps[CMAP]
        for i in range(2):
            for j in range(2):
                r, g, b, _ = cmap(shade[i, j])
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
                ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center",
                        fontsize=24, fontweight="bold",
                        color="#ffffff" if lum < 0.55 else "#111111")

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomaly"], fontsize=15)
        ax.set_yticklabels(["Normal", "Anomaly"], fontsize=15, rotation=90, va="center")
        ax.set_xlabel("Predicted", fontsize=16, labelpad=8)
        if ax is axes[0]:
            ax.set_ylabel("Actual", fontsize=16, labelpad=8)
        ax.set_title(f"{title}\n$F_1$={s['F1']:.3f}   FAR={s['FAR']:.3f}",
                     fontsize=16, pad=12)
        for sp in ax.spines.values():
            sp.set_linewidth(1.0)
        ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=axes, orientation="horizontal",
                        fraction=0.035, pad=0.16, aspect=55)
    cbar.set_label("Proportion of the true class", fontsize=15, labelpad=8)
    cbar.ax.tick_params(labelsize=13)
    fig.savefig(os.path.join(out, "figure3_confusion_matrices.pdf"))
    fig.savefig(os.path.join(out, "figure3_confusion_matrices.png"))
    plt.close(fig)


def figure4(t: pd.DataFrame, out: str) -> None:
    """Latency and F1 across every evaluated system."""
    lat = [stats(t, s)["lat"] for s in ORDER]
    lat_sd = [stats(t, s)["lat_sd"] for s in ORDER]
    f1 = [stats(t, s)["F1"] for s in ORDER]
    f1_sd = [stats(t, s)["F1_sd"] for s in ORDER]

    # ADAM, then baselines, then ablations.
    colors = [BLUE] + [ORANGE] * 4 + [GREEN] * 4
    x = np.arange(len(ORDER))
    labels = [LABEL[s] for s in ORDER]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5, 7.4))

    ax1.bar(x, lat, yerr=lat_sd, color=colors, edgecolor=INK, linewidth=0.9,
            capsize=5, width=0.66, error_kw={"elinewidth": 1.2})
    for xi, v in zip(x, lat):
        ax1.text(xi, v + max(lat) * 0.035, f"{v:.2f}", ha="center",
                 fontsize=15, fontweight="bold")
    ax1.set_ylabel("Decision latency (s)", labelpad=10)
    ax1.set_ylim(0, max(lat) * 1.22)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=38, ha="right", rotation_mode="anchor")
    ax1.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
    ax1.set_axisbelow(True)
    ax1.text(-0.055, 1.04, "(a)", transform=ax1.transAxes,
             fontsize=19, fontweight="bold")

    ax2.bar(x, f1, yerr=f1_sd, color=colors, edgecolor=INK, linewidth=0.9,
            capsize=5, width=0.66, error_kw={"elinewidth": 1.2})
    for xi, v in zip(x, f1):
        ax2.text(xi, v + 0.006, f"{v:.3f}", ha="center",
                 fontsize=15, fontweight="bold")
    ax2.set_ylabel("$F_1$ score", labelpad=10)
    ax2.set_ylim(0.70, 0.95)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=38, ha="right", rotation_mode="anchor")
    ax2.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
    ax2.set_axisbelow(True)
    ax2.text(-0.055, 1.04, "(b)", transform=ax2.transAxes,
             fontsize=19, fontweight="bold")

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=BLUE, ec=INK, lw=0.9),
        plt.Rectangle((0, 0), 1, 1, fc=ORANGE, ec=INK, lw=0.9),
        plt.Rectangle((0, 0), 1, 1, fc=GREEN, ec=INK, lw=0.9),
    ]
    fig.legend(handles, ["ADAM (Full)", "Baselines", "Ablations"],
               loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=3.0)
    fig.savefig(os.path.join(out, "figure4_latency_utility.pdf"))
    fig.savefig(os.path.join(out, "figure4_latency_utility.png"))
    plt.close(fig)


def figure5(path: str, out: str) -> None:
    """Stage decomposition of the decision pipeline over the completed events."""
    co = pd.read_excel(path, sheet_name="05_D2_Coordination_Log")
    ok = co["Success"].astype(str).str.lower().eq("yes")
    num = lambda c: pd.to_numeric(co[c], errors="coerce")[ok]

    stages = [
        ("T_form_ms", "Crew formation\n$T_{\\mathrm{form}}$"),
        ("T_aggregate_ms", "Aggregation\n$T_{\\mathrm{agg}}$"),
        ("T_reason_ms", "On-device reasoning\n$T_{\\mathrm{reason}}$"),
        ("T_validate_ms", "Governance validation\n$T_{\\mathrm{gov}}$"),
        ("T_weaviate_ms", "Semantic memory\n$T_{\\mathrm{weav}}$"),
        ("T_blockchain_ms", "Blockchain commit\n$T_{\\mathrm{bc}}$"),
    ]
    means = [num(c).mean() for c, _ in stages]
    sds = [num(c).std(ddof=1) for c, _ in stages]
    labels = [lab for _, lab in stages]
    total = sum(means)
    n_ok = int(ok.sum())

    # Reasoning is the finding; crew formation is the next largest; the rest
    # are the coordination and governance overhead the paper argues is small.
    cols = [ORANGE, GREEN, BLUE, GREEN, GREEN, GREEN]

    fig, ax = plt.subplots(figsize=(14.0, 6.6))
    y = np.arange(len(stages))[::-1]
    ax.barh(y, means, xerr=sds, color=cols, edgecolor=INK, linewidth=1.0,
            height=0.64, capsize=6, error_kw={"elinewidth": 1.3})
    for yi, m, sd in zip(y, means, sds):
        ax.text(m * 1.14 + sd, yi, f"{m:,.0f} ms   ({m/total:.1%})",
                va="center", fontsize=16, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=15)
    ax.set_xscale("log")
    ax.set_xlim(100, total * 5.5)
    ax.set_xlabel("Mean stage latency (ms, log scale)", labelpad=12)
    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=15)

    fig.tight_layout()
    fig.savefig(os.path.join(out, "figure5_coordination_latency.pdf"))
    fig.savefig(os.path.join(out, "figure5_coordination_latency.png"))
    plt.close(fig)

    print(f"\nFigure 5, over {n_ok} completed events:")
    for (c, _), m, sd in zip(stages, means, sds):
        print(f"  {c:<18}{m:>10.1f} ms  +/- {sd:>6.1f}   {m/total:>6.1%}")
    print(f"  {'total':<18}{total:>10.1f} ms  = {total/1000:.2f} s")


def figure6(path: str, out: str) -> None:
    """Operating-state profile: what the per-system table cannot show.

    Table 6 already compares systems, so this figure reports the deployment's
    state profile instead - the evidence behind the C2 argument that heavy
    resource use is confined to short inference windows.
    """
    r = pd.read_excel(path, sheet_name="07_D2_Resource_Log")
    num = lambda c, g: pd.to_numeric(g[c], errors="coerce")

    states = ["idle", "monitoring", "crew_active"]
    nice = ["Idle", "Monitoring", "Crew active"]
    sub = [("Weaviate_Bytes", "Semantic memory", ORANGE),
           ("Blockchain_Bytes", "Blockchain", GREEN),
           ("LLM_Bytes", "Local inference", BLUE)]

    bw, cpu, cpu_sd, ram, ram_sd, n_state = {}, [], [], [], [], []
    for st in states:
        g = r[r["State"].astype(str) == st]
        bw[st] = [(num(c, g) / 1024).mean() for c, _, _ in sub]
        cpu.append(num("CPU_Peak_%", g).mean())
        cpu_sd.append(num("CPU_Peak_%", g).std(ddof=1))
        ram.append(num("RAM_MB", g).mean())
        ram_sd.append(num("RAM_MB", g).std(ddof=1))
        n_state.append(len(g))

    x = np.arange(len(states))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 6.6))

    # (a) network traffic by state, decomposed by the subsystem that generates it
    bottom = np.zeros(len(states))
    for k, (col, lab, c) in enumerate(sub):
        vals = np.array([bw[st][k] for st in states])
        ax1.bar(x, vals, bottom=bottom, color=c, edgecolor=INK, linewidth=0.9,
                width=0.58, label=lab)
        for xi, v, b in zip(x, vals, bottom):
            if v > 2.0:
                ax1.text(xi, b + v / 2, f"{v:.1f}", ha="center", va="center",
                         fontsize=14, fontweight="bold",
                         color="white" if c == BLUE else INK)
        bottom += vals
    for xi, tot in zip(x, bottom):
        ax1.text(xi, tot + 1.1, f"{tot:.1f} KB", ha="center",
                 fontsize=15, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(nice)
    ax1.set_ylabel("Network traffic per node (KB/min)", labelpad=10)
    ax1.set_ylim(0, bottom.max() * 1.22)
    ax1.legend(fontsize=14, frameon=False, loc="upper left")
    ax1.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
    ax1.set_axisbelow(True)
    ax1.text(-0.11, 1.03, "(a)", transform=ax1.transAxes, fontsize=19,
             fontweight="bold")

    # (b) compute profile. Axis colors identify which series belongs to which
    # scale, so the legend does not need "left"/"right" labels.
    w = 0.36
    ax2.bar(x - w / 2, cpu, yerr=cpu_sd, width=w, color=BLUE,
            edgecolor=INK, linewidth=0.9, capsize=5)
    ax2.set_ylabel("Peak CPU utilization (%)", labelpad=10, color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax2.spines["left"].set_color(BLUE)
    cpu_top = 118.0
    ax2.set_ylim(0, cpu_top)
    for xi, v in zip(x - w / 2, cpu):
        ax2.text(xi, v + 3.0, f"{v:.1f}", ha="center", fontsize=15,
                 fontweight="bold")

    ax3 = ax2.twinx()
    ax3.bar(x + w / 2, ram, yerr=ram_sd, width=w, color=ORANGE,
            edgecolor=INK, linewidth=0.9, capsize=5)
    ram_top = max(ram) * 1.42
    ax3.set_ylabel("Working set (MB)", labelpad=12, color=ORANGE)
    ax3.tick_params(axis="y", colors=ORANGE)
    ax3.spines["right"].set_color(ORANGE)
    ax3.spines["left"].set_color(BLUE)
    ax3.set_ylim(0, ram_top)
    for xi, v in zip(x + w / 2, ram):
        ax3.text(xi, v + ram_top * 0.022, f"{v:,.0f}", ha="center", fontsize=15,
                 fontweight="bold")

    # The C2 reference line is drawn on the upper axes at the height that
    # corresponds to 80% on the CPU scale, so it passes over both bar series
    # rather than being clipped by whichever is drawn last.
    ax3.axhline(ram_top * (80.0 / cpu_top), color=INK, linestyle="--",
                linewidth=1.4, zorder=6)
    ax3.text(-0.46, ram_top * (80.0 / cpu_top) + ram_top * 0.015,
             "C2 budget 80%", ha="left", va="bottom", fontsize=12,
             color="#4d4d4d", zorder=6)

    ax2.set_xlim(-0.62, len(states) - 0.38)
    ax3.set_xlim(-0.62, len(states) - 0.38)
    ax2.set_xticks(x); ax2.set_xticklabels(nice)
    ax2.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
    ax2.set_axisbelow(True)
    ax2.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=BLUE, ec=INK, lw=0.9),
                        plt.Rectangle((0, 0), 1, 1, fc=ORANGE, ec=INK, lw=0.9)],
               labels=["Peak CPU", "Working set"],
               fontsize=14, frameon=False, loc="upper left")
    ax2.text(-0.11, 1.03, "(b)", transform=ax2.transAxes, fontsize=19,
             fontweight="bold")

    fig.tight_layout(w_pad=4.0)
    fig.savefig(os.path.join(out, "figure6_resource_profile.pdf"))
    fig.savefig(os.path.join(out, "figure6_resource_profile.png"))
    plt.close(fig)

    print("\nFigure 6, by operating state:")
    for st, nm, c, rm, nn in zip(states, nice, cpu, ram, n_state):
        print(f"  {nm:<13}n={nn:<4} CPU {c:5.1f}%   RAM {rm:6.0f} MB   "
              f"traffic {sum(bw[st]):5.2f} KB/min")


def figure7(path: str, out: str) -> None:
    """Node-count scaling under fixed load.

    N = 1-4 are physical Raspberry Pi 5 measurements; N = 6-16 come from the
    Python scale-out model, which is validated against the matched N = 1-4
    hardware runs before use. Load is held at the reference configuration
    throughout (4 concurrent events, 8 sensor streams, 30,000 vectors), so
    latency changes attribute to node count. R-squared is computed over the
    raw replicate observations, not over the group means - averaging
    replicates before fitting discards the within-level variance the model is
    meant to explain and inflates the statistic toward 1.
    """
    from scipy.optimize import curve_fit

    s = pd.read_excel(path, sheet_name="08_Scalability_Log")
    mode = s["Run_Mode"].astype(str).str.strip().str.upper()
    n = pd.to_numeric(s["Node_Count"], errors="coerce")
    t = pd.to_numeric(s["T_decision_ms"], errors="coerce")

    f = lambda N, t0, a_, b_: t0 + a_ * N ** b_

    def level_stats(mask):
        g = s[mask].groupby(n[mask])["T_decision_ms"].agg(["mean", "std"])
        return g.index.values.astype(float), g["mean"].values, g["std"].values

    def fit(mask, p0):
        xr, yr = n[mask].values.astype(float), t[mask].values.astype(float)
        x, y, _ = level_stats(mask)
        pw, _cov = curve_fit(f, x, y, p0=p0, maxfev=60000)
        r2 = 1 - np.sum((yr - f(xr, *pw)) ** 2) / np.sum((yr - yr.mean()) ** 2)
        return pw, r2

    hw_mask = mode.eq("HARDWARE")
    sim_mask = mode.eq("PYTHON_SIMULATION")
    eligible = s["Fit_Eligible"].astype(str).str.strip().str.lower().eq("yes")
    so_mask = eligible & (n >= 4)

    pw_hw, r2_hw = fit(hw_mask, p0=[17000, 200, 1.4])
    pw_so, r2_so = fit(so_mask, p0=[14000, 3000, 0.3])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2))

    # ---- (a) hardware domain with simulator validation overlay, N = 1-4
    xh, yh, sh = level_stats(hw_mask)
    xs, ys, ss = level_stats(sim_mask)
    fine = np.linspace(xh.min(), xh.max(), 200)
    ax1.plot(fine, f(fine, *pw_hw) / 1000, "-", color=ORANGE, linewidth=2.2,
             zorder=2,
             label=f"$T(N)={pw_hw[0]:,.0f}+{pw_hw[1]:.0f}N^{{{pw_hw[2]:.2f}}}$")
    ax1.errorbar(xh, yh / 1000, yerr=sh / 1000, fmt="o", color=BLUE,
                 markersize=10, markeredgecolor=INK, markeredgewidth=1.1,
                 capsize=5, linewidth=0, elinewidth=1.5, zorder=3,
                 label="Hardware (Pi 5)")
    ax1.errorbar(xs, ys / 1000, yerr=ss / 1000, fmt="s", color="none",
                 markersize=9, markeredgecolor=INK, markeredgewidth=1.3,
                 capsize=4, linewidth=0, elinewidth=1.2, ecolor=INK, zorder=3,
                 label="Scale-out model (validation)")
    ax1.set_xlabel("Nodes $N$ (hardware domain)", labelpad=10)
    ax1.set_ylabel("Decision latency (s)", labelpad=10)
    ax1.set_xticks([1, 2, 3, 4])
    ax1.grid(linestyle=":", linewidth=0.7, alpha=0.65)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=12, frameon=False, loc="upper left")
    ax1.text(0.97, 0.06, f"$R^2={r2_hw:.3f}$", transform=ax1.transAxes,
             ha="right", fontsize=15, color="#4d4d4d")
    ax1.text(-0.13, 1.03, "(a)", transform=ax1.transAxes, fontsize=19,
             fontweight="bold")

    # ---- (b) scale-out domain, N = 4-16
    xo, yo, so_ = level_stats(so_mask)
    fineo = np.linspace(xo.min(), xo.max(), 200)
    ax2.plot(fineo, f(fineo, *pw_so) / 1000, "-", color=ORANGE, linewidth=2.2,
             zorder=2,
             label=f"$T(N)={pw_so[0]:,.0f}+{pw_so[1]:,.0f}N^{{{pw_so[2]:.2f}}}$")
    ax2.errorbar(xo, yo / 1000, yerr=so_ / 1000, fmt="s", color=BLUE,
                 markersize=10, markeredgecolor=INK, markeredgewidth=1.1,
                 capsize=5, linewidth=0, elinewidth=1.5, zorder=3,
                 label="Scale-out model")
    ax2.set_xlabel("Effective nodes $N$ (scale-out domain)", labelpad=10)
    ax2.set_ylabel("Decision latency (s)", labelpad=10)
    ax2.set_xticks([4, 6, 8, 12, 16])
    ax2.grid(linestyle=":", linewidth=0.7, alpha=0.65)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=12, frameon=False, loc="upper left")
    ax2.text(0.97, 0.06, f"$R^2={r2_so:.3f}$", transform=ax2.transAxes,
             ha="right", fontsize=15, color="#4d4d4d")
    ax2.text(-0.13, 1.03, "(b)", transform=ax2.transAxes, fontsize=19,
             fontweight="bold")

    fig.tight_layout(w_pad=4.0)
    fig.savefig(os.path.join(out, "figure7_scalability.pdf"))
    fig.savefig(os.path.join(out, "figure7_scalability.png"))
    plt.close(fig)

    print("\nFigure 7  (R-squared over raw replicates)")
    print(f"  hardware  N=1-4 : T0={pw_hw[0]:,.0f}  alpha={pw_hw[1]:.1f}  "
          f"beta={pw_hw[2]:.3f}  R^2={r2_hw:.3f}")
    print(f"  scale-out N=4-16: T0={pw_so[0]:,.0f}  alpha={pw_so[1]:,.1f}  "
          f"beta={pw_so[2]:.3f}  R^2={r2_so:.3f}")


def figure8(path: str, out: str) -> None:
    """Security and resilience across the three CIA dimensions.

    Each panel reports something Table 9 cannot: which attack types evade
    detection, how tightly fallback activation clusters, and the absolute
    separation in external egress. Store poisoning is omitted deliberately -
    at 7-8 events per level the effect is not distinguishable from noise
    (Fisher exact p = 0.47), so it is reported as a null result in the table
    rather than drawn as a trend.
    """
    inj = pd.read_excel(path, sheet_name="10_Security_Injection", header=2)
    llm = pd.read_excel(path, sheet_name="11_Security_LLM_Failure", header=2)
    leak = pd.read_excel(path, sheet_name="13_Security_Data_Leakage", header=2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 6.0))

    # ---- (a) detection by attack type
    inj["det"] = inj["ADAM_Detected_Attack"].astype(str).str.strip().eq("Yes")
    grp = inj.groupby("Attack_Type")["det"].agg(["sum", "size"])
    grp = grp.sort_values("size", ascending=True)
    names = [str(i).replace("_", " ").title() for i in grp.index]
    hit = grp["sum"].values.astype(float)
    miss = (grp["size"] - grp["sum"]).values.astype(float)
    y = np.arange(len(names))

    ax1.barh(y, hit, color=BLUE, edgecolor=INK, linewidth=0.9, height=0.62,
             label="Detected")
    ax1.barh(y, miss, left=hit, color=ORANGE, edgecolor=INK, linewidth=0.9,
             height=0.62, label="Missed")
    for yi, h, m in zip(y, hit, miss):
        ax1.text(h + m + 0.22, yi, f"{h:.0f}/{h+m:.0f}", va="center",
                 fontsize=16, fontweight="bold")
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=15)
    ax1.set_xlabel("Injection events", labelpad=10)
    ax1.set_xlim(0, max(hit + miss) * 1.42)
    ax1.set_xticks(np.arange(0, int(max(hit + miss)) + 2, 2))
    ax1.legend(fontsize=14, frameon=False, loc="lower right")
    ax1.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.65)
    ax1.set_axisbelow(True)
    ax1.text(-0.20, 1.04, "(a)", transform=ax1.transAxes, fontsize=19,
             fontweight="bold")

    # ---- (b) fallback activation latency
    fb = llm[llm["Fallback_Triggered"].astype(str).str.strip().eq("Yes")]
    lat = pd.to_numeric(fb["Fallback_Latency_ms"], errors="coerce").dropna()
    counts, _, _ = ax2.hist(lat, bins=8, color=BLUE, edgecolor=INK, linewidth=1.0)
    ax2.axvline(lat.mean(), color=ORANGE, linestyle="-", linewidth=2.4,
                zorder=5, label=f"Mean {lat.mean():.1f} ms")
    ax2.axvline(lat.quantile(0.95), color=INK, linestyle="--", linewidth=1.8,
                zorder=5, label=f"P95 {lat.quantile(0.95):.1f} ms")
    ax2.set_xlabel("Fallback activation latency (ms)", labelpad=10)
    ax2.set_ylabel("Induced failures", labelpad=10)
    # Headroom above the tallest bar: a smaller legend alone would still sit on
    # a bar that reaches the top of the axis.
    ax2.set_ylim(0, counts.max() * 1.45)
    ax2.legend(fontsize=12, frameon=False, loc="upper left")
    ax2.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
    ax2.set_axisbelow(True)
    ax2.text(-0.20, 1.04, "(b)", transform=ax2.transAxes, fontsize=19,
             fontweight="bold")

    # ---- (c) external egress per measurement window
    order = ["ADAM_Full", "Cloud_Only"]
    disp = ["ADAM (Full)", "Cloud-Only"]
    cols = [BLUE, ORANGE]
    rng = np.random.default_rng(0)
    for k, (sysname, c) in enumerate(zip(order, cols)):
        g = leak[leak["System"].astype(str) == sysname]
        v = pd.to_numeric(g["Total_Bytes_External"], errors="coerce") / 1024.0
        jitter = rng.uniform(-0.10, 0.10, size=len(v))
        ax3.scatter(np.full(len(v), k) + jitter, v, s=170, color=c,
                    edgecolor=INK, linewidth=1.1, zorder=3, alpha=0.9)
        ax3.text(k, v.max() + max(6.0, v.max() * 0.09),
                 f"{v.mean():,.0f} KB" if v.mean() > 0 else "0 KB",
                 ha="center", fontsize=16, fontweight="bold")
        ax3.text(k, -14, f"$n$={len(v)}", ha="center", fontsize=14,
                 color="#4d4d4d")
    ax3.set_xticks(range(len(order))); ax3.set_xticklabels(disp, fontsize=15)
    ax3.set_xlim(-0.6, len(order) - 0.4)
    ax3.set_ylim(-22, 165)
    ax3.set_ylabel("External egress per 30 min window (KB)", labelpad=10)
    ax3.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.65)
    ax3.set_axisbelow(True)
    ax3.text(-0.22, 1.04, "(c)", transform=ax3.transAxes, fontsize=19,
             fontweight="bold")

    fig.tight_layout(w_pad=4.5)
    fig.savefig(os.path.join(out, "figure8_security.pdf"))
    fig.savefig(os.path.join(out, "figure8_security.png"))
    plt.close(fig)

    print("\nFigure 8")
    print(f"  (a) detection {int(hit.sum())}/{int((hit+miss).sum())} overall; "
          f"by type " + ", ".join(f"{n} {h:.0f}/{h+m:.0f}"
                                  for n, h, m in zip(names, hit, miss)))
    print(f"  (b) fallback n={len(lat)}  mean {lat.mean():.1f}  median "
          f"{lat.median():.1f}  P95 {lat.quantile(.95):.1f} ms")
    for sysname in order:
        g = leak[leak["System"].astype(str) == sysname]
        v = pd.to_numeric(g["Total_Bytes_External"], errors="coerce") / 1024.0
        print(f"  (c) {sysname:<12} n={len(v)}  mean {v.mean():,.1f} KB  "
              f"range {v.min():,.1f}-{v.max():,.1f}")


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "data/ADAM_Dataset_Master.xlsx"
    out = sys.argv[2] if len(sys.argv) > 2 else "figures"
    os.makedirs(out, exist_ok=True)
    t = load(src)
    figure3(t, out)
    figure4(t, out)
    figure5(src, out)
    figure6(src, out)
    figure7(src, out)
    figure8(src, out)

    print("values plotted, read from the workbook:\n")
    print(f"{'System':<20}{'TN':>8}{'FP':>7}{'FN':>7}{'TP':>7}{'F1':>8}{'FAR':>8}{'lat s':>8}")
    for s in ORDER:
        v = stats(t, s)
        print(f"{s:<20}{v['TN']:>8.1f}{v['FP']:>7.1f}{v['FN']:>7.1f}{v['TP']:>7.1f}"
              f"{v['F1']:>8.3f}{v['FAR']:>8.3f}{v['lat']:>8.2f}")
    print(f"\nwrote PDF and 600 dpi PNG to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
