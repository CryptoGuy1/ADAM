"""
analysis.make_figures
=====================

Renders the manuscript figures from the JSON artifacts the experiment harnesses
write. Figures are generated from results, never from hard-coded numbers, so a
figure cannot show something the run did not produce.

    Figure 3  confusion matrices, ADAM vs baselines      <- results/trials/
    Figure 4  per-trial F1 distributions                 <- results/trials/
    Figure 5  per-stage latency decomposition            <- results/deployment/
    Figure 8  latency against concurrency, with the fit  <- results/deployment/
    Figure 9  conflict sensitivity and flip thresholds   <- results/conflict/

Usage
-----
    python -m analysis.make_figures --results results/ --out results/figures/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Figures are greyscale-safe: some journals print figures in mono.
PALETTE = ["#1b1b1b", "#4a4a4a", "#7a7a7a", "#a8a8a8", "#c9c9c9"]
DPI = 300


def _load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        logger.warning("missing %s; skipping the figures that need it", path)
        return None
    with open(path, "r") as fh:
        return json.load(fh)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )
    return plt


# ---------------------------------------------------------------------------


def figure_f1_by_system(table5: List[Dict[str, Any]], out: str) -> Optional[str]:
    """Figure 4: per-system F1 with standard deviation across trials."""
    plt = _plt()
    rows = [r for r in table5 if r.get("f1") is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r["f1"], reverse=True)

    labels = [str(r["system"]).replace("_", " ") for r in rows]
    vals = [r["f1"] for r in rows]
    errs = [r.get("f1_sd", 0.0) for r in rows]
    colors = ["#1b1b1b" if r["system"] == "adam_full" else "#8a8a8a" for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(range(len(rows)), vals, yerr=errs, color=colors, capsize=3, width=0.68)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    path = os.path.join(out, "figure4_f1_by_system.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_stage_latency(summary: Dict[str, Any], out: str) -> Optional[str]:
    """Figure 5: where the decision budget goes."""
    plt = _plt()
    stages = summary.get("stages")
    if not stages:
        return None

    names = list(stages)
    means = [stages[s]["mean_ms"] for s in names]
    shares = [stages[s]["share"] for s in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    ax1.barh(range(len(names)), means, color="#4a4a4a", height=0.6)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    ax1.invert_yaxis()
    ax1.set_xlabel("mean latency (ms)")
    ax1.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax1.set_axisbelow(True)

    left = 0.0
    for i, (n, s) in enumerate(zip(names, shares)):
        ax2.barh([0], [s], left=[left], color=PALETTE[i % len(PALETTE)],
                 edgecolor="white", linewidth=0.6, height=0.5, label=n)
        if s > 0.04:
            ax2.text(left + s / 2, 0, f"{s:.0%}", ha="center", va="center",
                     fontsize=7, color="white" if i < 2 else "black")
        left += s
    ax2.set_xlim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("share of end-to-end latency")
    ax2.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25),
               frameon=False)

    path = os.path.join(out, "figure5_stage_latency.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_scalability(scal: Dict[str, Any], out: str) -> Optional[str]:
    """Latency against participating node count, with the power fit."""
    plt = _plt()
    rows = scal.get("measurements")
    if not rows:
        return None

    xs = [r["node_count"] for r in rows]
    ys = [r["mean_ms"] / 1000.0 for r in rows]
    es = [r.get("sd_ms", 0.0) / 1000.0 for r in rows]
    fit = scal.get("fit", {})

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.errorbar(xs, ys, yerr=es, fmt="o", color="#1b1b1b", markersize=4,
                capsize=3, linewidth=0.8, label="measured")

    if fit.get("alpha", 0) > 0:
        fine = [xs[0] + i * (xs[-1] - xs[0]) / 100.0 for i in range(101)]
        pred = [
            (fit["alpha"] * (x ** fit["beta"]) + fit["T0"]) / 1000.0 for x in fine
        ]
        ax.plot(fine, pred, "--", color="#7a7a7a", linewidth=1.0,
                label=f"$T(N)={fit['T0']:.0f}+{fit['alpha']:.0f}N^{{{fit['beta']:.2f}}}$")

    ax.set_xlabel("participating nodes")
    ax.set_ylabel("end-to-end latency (s)")
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False)

    if not scal.get("fit_usable", True):
        ax.text(0.5, 0.95, "fit not usable: see fit_warnings", transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color="#8a2020")

    path = os.path.join(out, "figure8_scalability.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_conflict(conflict: Dict[str, Any], out: str) -> Optional[str]:
    """Figure 9: agreement against lambda_1, and the flip-threshold spread."""
    plt = _plt()
    regimes = conflict.get("regimes", {})
    if not regimes:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for i, (name, res) in enumerate(regimes.items()):
        curve = res.get("agreement_curve", {})
        xs = sorted(float(k) for k in curve)
        ys = [curve[str(x)] if str(x) in curve else curve[f"{x:.4f}"] for x in xs]
        ax1.plot(xs, ys, color=PALETTE[i], linewidth=1.2, label=name)

    lam = conflict.get("configured_lambda_1", 0.7)
    ax1.axvline(lam, color="#8a2020", linestyle="--", linewidth=0.8)
    ax1.text(lam, 0.02, f" $\\lambda_1={lam}$", fontsize=7, color="#8a2020")
    ax1.set_xlabel("$\\lambda_1$ (severity weight)")
    ax1.set_ylabel("decision agreement")
    ax1.set_ylim(0, 1.02)
    ax1.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=7, frameon=False, loc="lower right")

    flips_path = os.path.join(os.path.dirname(out), "conflict", "flip_thresholds_window.json")
    flips: List[float] = []
    if os.path.exists(flips_path):
        with open(flips_path) as fh:
            flips = json.load(fh)
    if flips:
        ax2.hist(flips, bins=40, color="#4a4a4a", edgecolor="white", linewidth=0.4)
        ax2.axvline(lam, color="#8a2020", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("flip threshold $\\lambda_1^*$")
        ax2.set_ylabel("contested pairs")
        ax2.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax2.set_axisbelow(True)
    else:
        ax2.text(0.5, 0.5, "no flip-threshold data", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])

    path = os.path.join(out, "figure9_conflict_sensitivity.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_confusion(table5: List[Dict[str, Any]], out: str) -> Optional[str]:
    """Figure 3: precision/recall/FAR comparison across systems."""
    plt = _plt()
    rows = [r for r in table5 if r.get("f1") is not None][:6]
    if not rows:
        return None

    metrics = ["precision", "recall", "far"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    width = 0.26
    for i, m in enumerate(metrics):
        xs = [j + i * width for j in range(len(rows))]
        ax.bar(xs, [r[m] for r in rows], width=width, color=PALETTE[i],
               label=m.upper() if m == "far" else m)

    ax.set_xticks([j + width for j in range(len(rows))])
    ax.set_xticklabels([str(r["system"]).replace("_", "\n") for r in rows], fontsize=7)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, frameon=False, ncol=3)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    path = os.path.join(out, "figure3_detection_metrics.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="render manuscript figures")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    written: List[str] = []

    trials = _load(os.path.join(args.results, "trials", "table5.json"))
    if trials:
        if trials.get("dataset", {}).get("simulated"):
            logger.warning(
                "trial results came from SIMULATED data; figures are not "
                "manuscript reproductions"
            )
        t5 = trials.get("table5", [])
        for fn in (figure_confusion, figure_f1_by_system):
            p = fn(t5, args.out)
            if p:
                written.append(p)

    dep = _load(os.path.join(args.results, "deployment", "deployment_summary.json"))
    if dep:
        p = figure_stage_latency(dep, args.out)
        if p:
            written.append(p)

    scal = _load(os.path.join(args.results, "deployment", "scalability.json"))
    if scal:
        p = figure_scalability(scal, args.out)
        if p:
            written.append(p)

    conf = _load(os.path.join(args.results, "conflict", "conflict_sweep.json"))
    if conf:
        p = figure_conflict(conf, args.out)
        if p:
            written.append(p)

    if not written:
        logger.error("no figures written; run the experiment harnesses first")
        return 1
    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
