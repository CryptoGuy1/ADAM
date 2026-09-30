#!/usr/bin/env python3
"""Recompute and export the manuscript's 35 paired trial-level comparisons."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from analysis.revised_statistics import all_families

DEFAULT_WORKBOOK = ROOT / "data" / "ADAM_Dataset_Master_v14_reconciled.xlsx"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workbook", nargs="?", type=Path, default=DEFAULT_WORKBOOK)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "revised_statistical_tests.csv")
    args = p.parse_args()

    trials = pd.read_excel(args.workbook, sheet_name="03_D1_Trial_Results")
    swap = pd.read_excel(args.workbook, sheet_name="15_Swap_Study_Trials", header=1)
    degraded = pd.read_excel(args.workbook, sheet_name="14_Degraded_Conditions", header=1)
    rows = [r.to_dict() for r in all_families(trials, swap, degraded)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2) + "\n")

    counts = {}
    for row in rows:
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    print(f"Wrote {len(rows)} paired comparisons to {args.out}")
    print("Family counts:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
