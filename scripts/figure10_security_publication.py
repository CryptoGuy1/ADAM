#!/usr/bin/env python3
"""Generate the source-linked publication security figure from V14 records."""
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


def _linear_percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="linear"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workbook", nargs="?", type=Path, default=DEFAULT_WB)
    p.add_argument("outdir", nargs="?", type=Path, default=ROOT / "results" / "publication_figures")
    args = p.parse_args()

    failure = pd.read_excel(args.workbook, sheet_name="11_Security_LLM_Failure", header=2)
    failure = failure.dropna(subset=["Event_ID"])
    triggered = failure[failure["Fallback_Triggered"].astype(str).str.strip().str.lower().eq("yes")]
    if len(failure) != 30 or len(triggered) != 19:
        raise SystemExit(f"security fallback cohort changed: total={len(failure)} triggered={len(triggered)}")
    lat = pd.to_numeric(triggered["Fallback_Latency_ms"], errors="raise").to_numpy(dtype=float)

    egress = pd.read_excel(args.workbook, sheet_name="13_Security_Data_Leakage", header=2)
    egress = egress.dropna(subset=["Measurement_ID"])
    local = egress[egress["System"].astype(str).eq("ADAM_LLM")]
    cloud = egress[egress["System"].astype(str).isin(["Cloud_Only", "Cloud-Only"])]
    if len(local) != 12 or len(cloud) != 8:
        raise SystemExit(f"egress cohort changed: local={len(local)} cloud={len(cloud)}")
    local_kb = pd.to_numeric(local["Total_Bytes_External"], errors="raise").to_numpy(dtype=float) / 1024.0
    cloud_kb = pd.to_numeric(cloud["Total_Bytes_External"], errors="raise").to_numpy(dtype=float) / 1024.0

    xs = np.sort(lat)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    median = float(np.median(xs))
    p95 = _linear_percentile(xs, .95)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.7), layout="constrained")
    ax.step(xs, ys, where="post", linewidth=1.8)
    ax.scatter(xs, ys, s=24)
    ax.axvline(median, linestyle="--", linewidth=1.1)
    ax.axvline(p95, linestyle=":", linewidth=1.1)
    ax.text(.97, .06, f"Median {median:.1f} ms; P95 {p95:.1f} ms",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    ax.set_xlabel("Fallback activation latency (ms)")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_ylim(0, 1.04)
    ax.set_title(f"(a) Fallback-triggered events ($n={len(xs)}$)")
    ax.grid(alpha=.2)

    for i, data in enumerate((local_kb, cloud_kb)):
        jitter = np.linspace(-.1, .1, len(data))
        bx.scatter(np.full(len(data), i) + jitter, data, s=28)
        bx.hlines(float(np.mean(data)), i - .20, i + .20, linewidth=1.6)
    bx.set_xticks([0, 1], ["ADAM_LLM", "Cloud-Only"])
    bx.set_ylabel("External inference KB per 30-min window")
    bx.set_title(f"(b) Instrumented windows ({len(local_kb)} local; {len(cloud_kb)} cloud)")
    bx.grid(alpha=.2, axis="y")

    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.outdir / f"figure10_security_publication.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"PASS: fallback mean={lat.mean():.3f} ms median={median:.3f} ms p95={p95:.3f} ms; cloud mean={cloud_kb.mean():.3f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
