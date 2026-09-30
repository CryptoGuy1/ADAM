#!/usr/bin/env python3
"""Generate the source-linked publication resource figure from V14.

Panel (a) uses all 909 retained 60-s resource windows.  CPU values are the
recorded per-window node peaks.  Panel (b) labels the measured per-node memory
means; its stacked segments are the reconciled estimates stored in
``16_Memory_Budget`` and are therefore not process-level RSS measurements.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WB = ROOT / "data" / "ADAM_Dataset_Master_v14_reconciled.xlsx"
EXPECTED_COUNTS = {"idle": 256, "monitoring": 434, "crew_active": 219}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workbook", nargs="?", type=Path, default=DEFAULT_WB)
    p.add_argument("outdir", nargs="?", type=Path, default=ROOT / "results" / "publication_figures")
    args = p.parse_args()

    r = pd.read_excel(args.workbook, sheet_name="07_D2_Resource_Log")
    r = r.dropna(subset=["Timestamp", "State", "CPU_Peak_%", "RAM_MB"]).copy()
    counts = r["State"].astype(str).value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise SystemExit(f"resource state counts changed: {counts}")
    if len(r) != 909:
        raise SystemExit(f"expected 909 resource windows, got {len(r)}")

    # pandas may decode Excel serial timestamps as datetimes.  Convert either
    # representation to elapsed hours without relying on the encoded dtype.
    ts = r["Timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts):
        dt = pd.to_datetime(ts, errors="raise")
        hours = (dt - dt.min()).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    else:
        t = pd.to_numeric(ts, errors="raise").to_numpy(dtype=float)
        hours = (t - t.min()) * 24.0
    if not (58.0 <= float(np.nanmax(hours)) <= 59.0):
        raise SystemExit(
            f"unexpected resource sampling span: {float(np.nanmax(hours)):.3f} h"
        )
    cpu = pd.to_numeric(r["CPU_Peak_%"], errors="raise").to_numpy(dtype=float)
    mem = pd.to_numeric(r["RAM_MB"], errors="raise").to_numpy(dtype=float)
    states = r["State"].astype(str).to_numpy()
    noninf = float(cpu[states != "crew_active"].mean())

    budget = pd.read_excel(args.workbook, sheet_name="16_Memory_Budget", header=None)
    # Component rows are identified by the Basis column, not fixed numeric offsets.
    components = []
    for _, row in budget.iterrows():
        if len(row) < 5 or str(row.iloc[4]).strip() not in {
            "Budget component",
            "Resident image after load (file is 815 MB on disk)",
            "Active-minus-monitoring residual",
        }:
            continue
        try:
            mb = float(row.iloc[1])
        except (TypeError, ValueError):
            continue
        components.append((str(row.iloc[0]).strip(), mb, str(row.iloc[3]).strip()))
    if len(components) != 8:
        raise SystemExit(f"expected 8 reconciled memory components, got {len(components)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 6.4), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})
    markers = {"idle": "o", "monitoring": "s", "crew_active": "^"}
    labels = {"idle": "Idle", "monitoring": "Monitoring", "crew_active": "Crew active"}
    for st in ("idle", "monitoring", "crew_active"):
        m = states == st
        ax1.scatter(hours[m], cpu[m], s=14, marker=markers[st], alpha=0.78,
                    label=f"{labels[st]} ($n={m.sum()}$)")
    ax1.axhline(80, linestyle="--", linewidth=1.2)
    ax1.axhline(noninf, linestyle=":", linewidth=1.2)
    ax1.text(0.23, 0.80, "C2: 80% criterion for non-inference windows",
             transform=ax1.transAxes, fontsize=9)
    ax1.text(0.23, 0.16, f"Mean sampled non-inference window peak: {noninf:.1f}%",
             transform=ax1.transAxes, fontsize=9)
    ax1.set_xlabel("Deployment time (hours from first sample)")
    ax1.set_ylabel("Per-window peak CPU (%)")
    ax1.set_ylim(0, 104)
    ax1.legend(frameon=False, loc="center left")
    ax1.grid(alpha=0.2)
    ax1.text(-0.08, 1.02, "(a)", transform=ax1.transAxes, fontweight="bold", fontsize=14)

    state_component_counts = {"idle": 3, "monitoring": 6, "crew_active": 8}
    display_name = {
        "Base OS, kernel, background services": "OS/background",
        "Sensor acquisition and experiment logging": "Sensor/logging",
        "ADAM core services": "ADAM services",
        "Weaviate process and vector index cache": "Weaviate",
        "Fides Innova PoA client and transaction buffers": "PoA client",
        "Monitoring/orchestration working buffers": "Monitoring buffers",
        "Gemma 3 1B (Q4_K_M) resident model image": "Gemma model",
        "Ollama runtime, KV cache, allocator overhead": "Ollama/KV cache",
    }
    state_order = ("idle", "monitoring", "crew_active")
    bottoms = np.zeros(3, dtype=float)
    for component_index, (name, mb, _included) in enumerate(components):
        heights = np.asarray([
            mb if component_index < state_component_counts[st] else 0.0
            for st in state_order
        ], dtype=float)
        ax2.bar(
            [0, 1, 2], heights, bottom=bottoms, width=0.62,
            edgecolor="black", linewidth=0.5, label=display_name.get(name, name),
        )
        bottoms += heights
    for i, st in enumerate(state_order):
        observed = float(mem[states == st].mean())
        ax2.text(i, bottoms[i] + 60, f"{observed:,.0f} MB\nmeasured total",
                 ha="center", va="bottom", fontsize=9)
    ax2.set_xticks([0, 1, 2], ["Idle", "Monitoring", "Crew active"])
    ax2.set_ylabel("Per-node memory (MB)")
    ax2.set_ylim(0, 4650)
    ax2.grid(axis="y", alpha=0.2)
    ax2.text(0.03, 0.98, "Stacked segments: reconciled estimates",
             transform=ax2.transAxes, ha="left", va="top", fontsize=9)
    ax2.legend(
        frameon=False, fontsize=7.2, ncol=2, loc="upper center",
        bbox_to_anchor=(0.50, -0.12), columnspacing=1.0, handletextpad=0.5,
    )
    ax2.text(-0.10, 1.02, "(b)", transform=ax2.transAxes, fontweight="bold", fontsize=14)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.outdir / f"figure8_resources.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"PASS: {len(r)} windows; non-inference mean peak={noninf:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
