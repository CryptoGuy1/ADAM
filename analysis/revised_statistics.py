"""Recompute the three exploratory paired-comparison families in MDPI-15.

The V14 workbook retains the per-trial outcomes needed for all 35 comparisons,
but its legacy ``04_D1_Statistical_Tests`` sheet predates the two contextual
fitted comparators and therefore contains only eight main-benchmark rows.  This
module treats the retained trial outcomes as the source records and recomputes
the exact Wilcoxon tests and within-family Holm adjustments directly.

No event-level independence is assumed.  Zero paired differences are removed
before the signed-rank test, matching the manuscript.  Bootstrap confidence
intervals are percentile intervals over paired *trial* differences.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon

SEED = 42
N_BOOTSTRAP = 10_000


@dataclass(frozen=True)
class PairedTestRow:
    family: str
    comparison: str
    reference: str
    comparator: str
    n_total: int
    n_effective: int
    statistic_w: float
    p_exact: float
    p_holm: float
    mean_difference: float
    ci_lower: float
    ci_upper: float
    trials_favoring_reference: int
    ties: int

    def to_dict(self) -> dict:
        return asdict(self)


def _holm(p_values: Mapping[str, float]) -> Dict[str, float]:
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    running = 0.0
    out: Dict[str, float] = {}
    for rank, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * float(p)))
        out[key] = running
    return out


def _paired_test(
    reference: Sequence[float], comparator: Sequence[float], *,
    bootstrap_seed: int = SEED,
) -> dict:
    ref = np.asarray(reference, dtype=float)
    other = np.asarray(comparator, dtype=float)
    if ref.shape != other.shape or ref.ndim != 1 or len(ref) == 0:
        raise ValueError("paired samples must be non-empty one-dimensional arrays of equal length")
    if not np.isfinite(ref).all() or not np.isfinite(other).all():
        raise ValueError("paired samples contain non-finite values")

    diffs = ref - other
    nonzero = diffs[np.abs(diffs) > 1e-12]
    if len(nonzero):
        result = wilcoxon(
            nonzero,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="exact",
        )
        statistic = float(result.statistic)
        p_exact = float(result.pvalue)
    else:
        statistic, p_exact = 0.0, 1.0

    rng = np.random.default_rng(bootstrap_seed)
    # Vectorized paired-trial bootstrap: 10,000 x n is tiny here (n=10) and
    # avoids a Python loop during the full 35-comparison release audit.
    indices = rng.integers(0, len(diffs), size=(N_BOOTSTRAP, len(diffs)))
    means = diffs[indices].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])

    return {
        "n_total": int(len(diffs)),
        "n_effective": int(len(nonzero)),
        "statistic_w": statistic,
        "p_exact": p_exact,
        "mean_difference": float(diffs.mean()),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "trials_favoring_reference": int((diffs > 1e-12).sum()),
        "ties": int((np.abs(diffs) <= 1e-12).sum()),
    }


def _trial_map(frame, system_col: str, trial_col: str, value_col: str) -> Dict[str, Dict[int, float]]:
    out: Dict[str, Dict[int, float]] = {}
    for _, row in frame.iterrows():
        system = str(row[system_col]).strip()
        if not system or system.lower() == "nan":
            continue
        try:
            trial = int(row[trial_col])
            value = float(row[value_col])
        except (TypeError, ValueError):
            continue
        out.setdefault(system, {})[trial] = value
    return out


def _paired_values(a: Mapping[int, float], b: Mapping[int, float]) -> Tuple[List[float], List[float]]:
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("comparison has no shared trials")
    return [a[t] for t in shared], [b[t] for t in shared]


def _finalize(rows: List[dict], family: str) -> List[PairedTestRow]:
    adjusted = _holm({r["comparison"]: r["p_exact"] for r in rows})
    return [
        PairedTestRow(
            family=family,
            comparison=r["comparison"],
            reference=r["reference"],
            comparator=r["comparator"],
            p_holm=float(adjusted[r["comparison"]]),
            n_total=r["n_total"],
            n_effective=r["n_effective"],
            statistic_w=r["statistic_w"],
            p_exact=r["p_exact"],
            mean_difference=r["mean_difference"],
            ci_lower=r["ci_lower"],
            ci_upper=r["ci_upper"],
            trials_favoring_reference=r["trials_favoring_reference"],
            ties=r["ties"],
        )
        for r in rows
    ]


MAIN_COMPARATORS = (
    ("ADAM_vs_Static", "Static_Threshold"),
    ("ADAM_vs_RF_Raw", "Random_Forest"),
    ("ADAM_vs_RF_Fused", "RandomForest_Fused"),
    ("ADAM_vs_GBM_Fused", "GBM_Fused"),
    ("ADAM_vs_Cloud", "Cloud_Only"),
    ("ADAM_vs_SingleAgent", "SingleAgent"),
    ("ADAM_vs_NoAgg", "ADAM_NoAgg"),
    ("ADAM_vs_NoLLM", "ADAM_NoLLM"),
    ("ADAM_vs_NoBlockchain", "ADAM_NoBlockchain"),
    ("ADAM_vs_NoWeaviate", "ADAM_NoWeaviate"),
)


def main_benchmark_family(trials) -> List[PairedTestRow]:
    by_system = _trial_map(trials, "System", "Trial", "F1")
    ref_name = "ADAM_LLM"
    if ref_name not in by_system:
        raise ValueError("ADAM_LLM trial outcomes are missing")
    rows: List[dict] = []
    for comparison, system in MAIN_COMPARATORS:
        if system not in by_system:
            raise ValueError(f"missing main-benchmark system {system}")
        ref, other = _paired_values(by_system[ref_name], by_system[system])
        stat = _paired_test(ref, other)
        rows.append({"comparison": comparison, "reference": ref_name, "comparator": system, **stat})
    return _finalize(rows, "Main benchmark")


SUBSTITUTION_PAIRS = (
    ("Static threshold", "Static_Threshold", "ADAM_NoLLM", "trials"),
    ("LLM (Gemma 3 1B)", "SingleAgent", "ADAM_LLM", "trials"),
    ("Logistic regression", "LogReg_Fused", "ADAM_LogReg", "swap"),
    ("Random forest", "RandomForest_Fused", "ADAM_RF", "swap"),
    ("Gradient boosting", "GBM_Fused", "ADAM_GBM", "swap"),
)


def substitution_family(trials, swap_trials) -> List[PairedTestRow]:
    base = _trial_map(trials, "System", "Trial", "F1")
    swap = _trial_map(swap_trials, "System", "Trial", "F1")
    rows: List[dict] = []
    for label, standalone, crew, crew_source in SUBSTITUTION_PAIRS:
        if standalone not in base:
            raise ValueError(f"missing standalone substitution system {standalone}")
        crew_map = base if crew_source == "trials" else swap
        if crew not in crew_map:
            raise ValueError(f"missing in-crew substitution system {crew}")
        # Manuscript delta is in-crew minus standalone, so the crew is reference.
        ref, other = _paired_values(crew_map[crew], base[standalone])
        stat = _paired_test(ref, other)
        rows.append({"comparison": label, "reference": crew, "comparator": standalone, **stat})
    return _finalize(rows, "Decision-Agent substitution")


DEGRADED_CONDITIONS = ("Mild Drift", "Strong Drift", "Noise", "One-Node Dropout")
DEGRADED_COMPARATORS = (
    "Static_Threshold",
    "RandomForest_Rich",
    "GBM_Rich",
    "Single_Agent",
    "ADAM_GBM",
)


def degraded_family(degraded) -> List[PairedTestRow]:
    rows: List[dict] = []
    for condition in DEGRADED_CONDITIONS:
        cell = degraded[degraded["Condition"].astype(str).eq(condition)]
        by_system = _trial_map(cell, "System", "Trial", "F1_Score")
        if "ADAM_LLM" not in by_system:
            raise ValueError(f"missing ADAM_LLM for degraded condition {condition}")
        for system in DEGRADED_COMPARATORS:
            if system not in by_system:
                raise ValueError(f"missing {system} for degraded condition {condition}")
            ref, other = _paired_values(by_system["ADAM_LLM"], by_system[system])
            stat = _paired_test(ref, other)
            rows.append({
                "comparison": f"{condition}: ADAM_LLM vs {system}",
                "reference": "ADAM_LLM",
                "comparator": system,
                **stat,
            })
    return _finalize(rows, "Degraded conditions")


def all_families(trials, swap_trials, degraded) -> List[PairedTestRow]:
    return main_benchmark_family(trials) + substitution_family(trials, swap_trials) + degraded_family(degraded)


__all__ = [
    "PairedTestRow",
    "MAIN_COMPARATORS",
    "SUBSTITUTION_PAIRS",
    "DEGRADED_CONDITIONS",
    "DEGRADED_COMPARATORS",
    "main_benchmark_family",
    "substitution_family",
    "degraded_family",
    "all_families",
]
