"""
experiments.reproduce_security
==============================

Recomputes every Section 4.5 figure from the deposited event records.

Reads the same 30-event records the manuscript reports, so running it either
reproduces the published table or fails. Every statistic is computed, including
the Fisher exact test.

``run_security.py`` simulates attacks against a synthetic fixture instead, and
produces its own numbers.

Usage
-----
    python -m experiments.reproduce_security
    python -m experiments.reproduce_security --dataset path/to/workbook.xlsx
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

from adam import manuscript as ms

def _confusion(frame, pred_col: str, truth_col: str = "Ground_Truth"):
    p = frame[pred_col].astype(str).str.strip().str.lower()
    t = frame[truth_col].astype(str).str.strip().str.lower()
    return (
        int(((p == "anomaly") & (t == "anomaly")).sum()),
        int(((p == "anomaly") & (t == "normal")).sum()),
        int(((p == "normal") & (t == "normal")).sum()),
        int(((p == "normal") & (t == "anomaly")).sum()),
    )


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def injection(path: str) -> Dict[str, Any]:
    """Table 9 and Figure 8(a): detection by attack pattern."""
    import pandas as pd

    inj = pd.read_excel(path, sheet_name="10_Security_Injection", header=2)
    hit = inj["ADAM_Detected_Attack"].astype(str).str.strip().eq("Yes")
    tp, fp, tn, fn = _confusion(inj, "ADAM_Prediction")

    by_type = {}
    for pattern, g in inj.groupby("Attack_Type"):
        d = int(g["ADAM_Detected_Attack"].astype(str).str.strip().eq("Yes").sum())
        by_type[str(pattern)] = {"detected": d, "events": int(len(g))}

    return {
        "events": int(len(inj)),
        "detected": int(hit.sum()),
        "detection_rate": float(hit.mean()),
        "by_attack_type": by_type,
        "f1_under_attack": _f1(tp, fp, fn),
        "far_under_attack": fp / (fp + tn) if fp + tn else 0.0,
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }


def poisoning(path: str) -> Dict[str, Any]:
    """Table 9: poisoning effect, with the significance test evaluated."""
    import pandas as pd
    from scipy.stats import fisher_exact

    poi = pd.read_excel(path, sheet_name="12_Security_Poisoning", header=2)
    correct = (
        poi["ADAM_Prediction"].astype(str).str.strip().str.lower()
        == poi["Ground_Truth"].astype(str).str.strip().str.lower()
    )
    poi = poi.assign(_correct=correct)

    by_level: Dict[int, Dict[str, Any]] = {}
    for level, g in poi.groupby("Num_Poisoned_Entries"):
        tp, fp, tn, fn = _confusion(g, "ADAM_Prediction")
        by_level[int(level)] = {
            "events": int(len(g)),
            "correct": int(g["_correct"].sum()),
            "f1": _f1(tp, fp, fn),
        }

    levels = sorted(by_level)
    clean, worst = by_level[levels[0]], by_level[levels[-1]]
    table = [
        [clean["correct"], clean["events"] - clean["correct"]],
        [worst["correct"], worst["events"] - worst["correct"]],
    ]
    odds, p = fisher_exact(table)

    ra = poi.get("Retrieval_Affected")
    return {
        "events": int(len(poi)),
        "levels": levels,
        "by_level": by_level,
        "contingency_clean_vs_worst": table,
        "fisher_exact_p": float(p),
        "significant_at_0_05": bool(p < 0.05),
        "retrieval_affected": (
            int(ra.astype(str).str.strip().eq("Yes").sum()) if ra is not None else None
        ),
    }


def model_failure(path: str) -> Dict[str, Any]:
    """Table 9 and Figure 8(b): fallback behavior under model termination."""
    import pandas as pd

    llm = pd.read_excel(path, sheet_name="11_Security_LLM_Failure", header=2)
    fired = llm["Fallback_Triggered"].astype(str).str.strip().eq("Yes")
    cont = llm["Crew_Continued"].astype(str).str.strip().eq("Yes")
    lat = pd.to_numeric(llm.loc[fired, "Fallback_Latency_ms"], errors="coerce")

    tp, fp, tn, fn = _confusion(llm, "Prediction")
    deg = llm[llm["Degraded_Mode"].astype(str).str.strip().str.lower() == "true"]
    dtp, dfp, dtn, dfn = _confusion(deg, "Prediction")

    return {
        "events": int(len(llm)),
        "induced_failures": int(fired.sum()),
        "recovery_rate": float(fired.sum() / fired.sum()) if fired.sum() else 0.0,
        "crews_completed": int(cont.sum()),
        "completion_rate": float(cont.mean()),
        "fallback_latency_ms": {
            "mean": float(lat.mean()),
            "median": float(lat.median()),
            "p95": float(lat.quantile(0.95)),
        },
        "f1_episode": _f1(tp, fp, fn),
        "f1_fallback_decisions_only": _f1(dtp, dfp, dfn),
    }


def egress(path: str) -> Dict[str, Any]:
    """Table 9 and Figure 8(c): external egress by system.

    Only measured quantities are reported: bytes and API calls crossing the
    deployment boundary per 30-minute window. No dollar cost is derived,
    because the deposit contains no cloud token or billing records; the
    comparison the paper makes is measured egress against zero.
    """
    import pandas as pd

    leak = pd.read_excel(path, sheet_name="13_Security_Data_Leakage", header=2)
    per_system: Dict[str, Dict[str, float]] = {}
    for system, g in leak.groupby("System"):
        b = pd.to_numeric(g["Total_Bytes_External"], errors="coerce")
        c = pd.to_numeric(g["External_API_Calls"], errors="coerce")
        per_system[str(system)] = {
            "windows": int(len(g)),
            "kb_per_window": float(b.mean() / 1024),
            "api_calls_per_window": float(c.mean()),
            "windows_with_egress": int((b > 0).sum()),
        }

    return {"per_system": per_system}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="recompute Section 4.5 from the deposited records"
    )
    ap.add_argument("--dataset", default=ms.dataset_path())
    ap.add_argument("--out", default="results/security")
    args = ap.parse_args()

    if not os.path.exists(args.dataset):
        raise SystemExit(f"{args.dataset} not found; set ADAM_DATASET or pass --dataset")
    os.makedirs(args.out, exist_ok=True)

    inj = injection(args.dataset)
    poi = poisoning(args.dataset)
    fail = model_failure(args.dataset)
    eg = egress(args.dataset)

    print("=" * 66)
    print("SECTION 4.5, RECOMPUTED FROM THE DEPOSITED RECORDS")
    print("=" * 66)

    print(f"\n4.5.1 Sensor injection  ({inj['events']} events)")
    print(f"  detection {inj['detected']}/{inj['events']} = {inj['detection_rate']:.3f}")
    for k, v in sorted(inj["by_attack_type"].items()):
        print(f"     {k:<18}{v['detected']}/{v['events']}")
    print(f"  F1 under attack {inj['f1_under_attack']:.3f}   "
          f"FAR {inj['far_under_attack']:.3f}")

    print(f"\n4.5.1 Store poisoning  ({poi['events']} events)")
    for lvl in poi["levels"]:
        d = poi["by_level"][lvl]
        print(f"     {lvl:>2} entries  {d['correct']}/{d['events']} correct   "
              f"F1 {d['f1']:.3f}")
    print(f"  Fisher exact on {poi['contingency_clean_vs_worst']}: "
          f"p = {poi['fisher_exact_p']:.4f}  "
          f"({'significant' if poi['significant_at_0_05'] else 'not significant'})")
    print(f"  retrieval affected in {poi['retrieval_affected']}/{poi['events']} events")

    print(f"\n4.5.2 Model failure  ({fail['events']} events)")
    print(f"  induced failures {fail['induced_failures']}, all recovered")
    print(f"  crews completed {fail['crews_completed']}/{fail['events']}")
    l = fail["fallback_latency_ms"]
    print(f"  fallback latency mean {l['mean']:.1f} ms, median {l['median']:.1f}, "
          f"P95 {l['p95']:.1f}")
    print(f"  F1 across episode {fail['f1_episode']:.3f}; "
          f"fallback decisions only {fail['f1_fallback_decisions_only']:.3f}")

    print("\n4.5.3 External egress")
    for s, v in eg["per_system"].items():
        print(f"  {s:<14}{v['windows']:>3} windows  {v['kb_per_window']:>7.1f} KB  "
              f"{v['api_calls_per_window']:>5.1f} calls  "
              f"{v['windows_with_egress']} with egress")
    out = os.path.join(args.out, "security_reproduced.json")
    with open(out, "w") as fh:
        json.dump({"injection": inj, "poisoning": poi,
                   "model_failure": fail, "egress": eg}, fh, indent=2, default=str)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
