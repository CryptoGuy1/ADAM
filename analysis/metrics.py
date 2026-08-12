"""
analysis.metrics
================

Detection metrics and the trial-level statistical comparison of Table 5.

Why the tests are trial-level
-----------------------------
Section 3.4.5: "Because events within a trial form a continuous,
non-independent exposure sequence, statistical comparison is performed at the
trial level rather than by pooling individual events." Pooling 2,000 correlated
events and running a test that assumes independence would produce p-values that
are meaningless and, worse, uniformly tiny. Each trial contributes one paired
F1 observation, giving n = 10.

With n = 10, the Wilcoxon signed-rank test has a floor: when all ten
differences share a sign, the two-sided exact p is 2/2^10 * ... = 0.001953,
which rounds to the 0.002 appearing throughout Table 5. That floor is a
property of the sample size, not a measure of effect magnitude, and
:func:`compare_systems` reports the effect size alongside it so the two are not
confused.

Because eight systems are compared against the same reference, the Holm
step-down procedure controls the family-wise error rate (Section 3.4.5). Both
the exact and the adjusted p-value are reported: the adjustment changes two
conclusions, so reporting only the raw value would overstate what the trials
establish.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from adam.config import ALPHA
from adam.schemas import LabeledEvent, Prediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confusion matrix and derived metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def far(self) -> float:
        """False alarm rate FP/(FP+TN). Section 3.4.5."""
        d = self.fp + self.tn
        return self.fp / d if d else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "far": self.far,
            "accuracy": self.accuracy,
        }

    def __add__(self, other: "ConfusionMatrix") -> "ConfusionMatrix":
        return ConfusionMatrix(
            self.tp + other.tp,
            self.fp + other.fp,
            self.tn + other.tn,
            self.fn + other.fn,
        )


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> ConfusionMatrix:
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} labels but {len(y_pred)} predictions")
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 0:
            tn += 1
        else:
            fn += 1
    return ConfusionMatrix(tp, fp, tn, fn)


# ---------------------------------------------------------------------------
# Per-trial scoring
# ---------------------------------------------------------------------------


@dataclass
class SystemScores:
    """One system's per-trial results and their aggregate."""

    system: str
    per_trial: Dict[int, ConfusionMatrix] = field(default_factory=dict)
    latencies_ms: List[float] = field(default_factory=list)
    degraded_count: int = 0
    api_cost_usd: float = 0.0

    @property
    def trial_ids(self) -> List[int]:
        return sorted(self.per_trial)

    def f1_series(self) -> List[float]:
        """Paired per-trial F1 scores - the unit of analysis for the tests."""
        return [self.per_trial[t].f1 for t in self.trial_ids]

    def series(self, metric: str) -> List[float]:
        return [getattr(self.per_trial[t], metric) for t in self.trial_ids]

    def mean_sd(self, metric: str) -> Tuple[float, float]:
        vals = self.series(metric)
        if not vals:
            return 0.0, 0.0
        mean = statistics.fmean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, sd

    @property
    def pooled(self) -> ConfusionMatrix:
        """Confusion matrix aggregated over all trials, as in Figure 3."""
        total = ConfusionMatrix(0, 0, 0, 0)
        for cm in self.per_trial.values():
            total = total + cm
        return total

    @property
    def mean_latency_ms(self) -> float:
        return statistics.fmean(self.latencies_ms) if self.latencies_ms else 0.0

    def summary_row(self) -> Dict[str, object]:
        """One row of Table 5."""
        p_m, p_s = self.mean_sd("precision")
        r_m, r_s = self.mean_sd("recall")
        f_m, f_s = self.mean_sd("f1")
        a_m, a_s = self.mean_sd("far")
        return {
            "system": self.system,
            "precision": p_m,
            "precision_sd": p_s,
            "recall": r_m,
            "recall_sd": r_s,
            "f1": f_m,
            "f1_sd": f_s,
            "far": a_m,
            "far_sd": a_s,
            "mean_latency_s": self.mean_latency_ms / 1000.0,
            "degraded_events": self.degraded_count,
            "api_cost_usd": self.api_cost_usd,
        }


def score_system(
    system: str,
    events: Sequence[LabeledEvent],
    predictions: Sequence[Prediction],
) -> SystemScores:
    """Group predictions by trial and compute per-trial confusion matrices."""
    by_key = {(p.trial_id, p.event_index): p for p in predictions}
    missing = [
        (e.trial_id, e.event_index)
        for e in events
        if (e.trial_id, e.event_index) not in by_key
    ]
    if missing:
        raise ValueError(
            f"{system}: {len(missing)} events have no prediction "
            f"(first few: {missing[:5]})"
        )

    scores = SystemScores(system=system)
    by_trial: Dict[int, Tuple[List[int], List[int]]] = {}
    for e in events:
        p = by_key[(e.trial_id, e.event_index)]
        yt, yp = by_trial.setdefault(e.trial_id, ([], []))
        yt.append(e.label)
        yp.append(p.predicted)
        scores.latencies_ms.append(p.latency_ms)
        scores.degraded_count += int(p.degraded_mode)
        scores.api_cost_usd += p.api_cost_usd

    for trial_id, (yt, yp) in by_trial.items():
        scores.per_trial[trial_id] = confusion(yt, yp)
    return scores


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Paired comparison of one system against the reference (ADAM Full)."""

    system: str
    reference: str
    n_pairs: int
    median_difference: float
    mean_difference: float
    n_favoring_reference: int
    wilcoxon_p: Optional[float]
    sign_test_p: Optional[float]
    effect_size_r: Optional[float]
    at_significance_boundary: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _sign_test_p(n_positive: int, n_total: int) -> float:
    """Exact two-sided binomial sign test at p = 0.5."""
    if n_total == 0:
        return 1.0

    def comb(n: int, k: int) -> int:
        return math.comb(n, k)

    k = min(n_positive, n_total - n_positive)
    tail = sum(comb(n_total, i) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail / (2**n_total))


def compare_systems(
    reference: SystemScores,
    other: SystemScores,
    alpha: float = 0.05,
    boundary_margin: float = 0.02,
) -> ComparisonResult:
    """Wilcoxon signed-rank on paired per-trial F1, plus a sign test.

    The sign test is reported whenever the Wilcoxon p falls within
    ``boundary_margin`` of ``alpha`` - Section 3.4.5's "robustness check where a
    result falls near the significance boundary." For Cloud-Only this is the
    difference between a nominally significant 0.041 and a sign test at 0.11,
    which is why the manuscript describes the two systems as statistically
    close rather than distinguishable.
    """
    shared = sorted(set(reference.trial_ids) & set(other.trial_ids))
    if not shared:
        raise ValueError(
            f"no shared trials between {reference.system} and {other.system}"
        )

    ref_f1 = [reference.per_trial[t].f1 for t in shared]
    oth_f1 = [other.per_trial[t].f1 for t in shared]
    diffs = [r - o for r, o in zip(ref_f1, oth_f1)]
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    n_favor_ref = sum(1 for d in nonzero if d > 0)

    wilcoxon_p: Optional[float] = None
    effect_r: Optional[float] = None
    if nonzero:
        try:
            from scipy.stats import wilcoxon

            stat, wilcoxon_p = wilcoxon(ref_f1, oth_f1, zero_method="wilcox")
            # Effect size r = Z / sqrt(N), from the normal approximation to W.
            n = len(nonzero)
            mean_w = n * (n + 1) / 4.0
            sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
            if sd_w > 0:
                z = (float(stat) - mean_w) / sd_w
                effect_r = abs(z) / math.sqrt(n)
        except ImportError:  # pragma: no cover
            logger.warning("scipy unavailable; Wilcoxon p not computed")

    sign_p = _sign_test_p(n_favor_ref, len(nonzero)) if nonzero else None

    at_boundary = (
        wilcoxon_p is not None and abs(wilcoxon_p - alpha) <= boundary_margin
    )

    return ComparisonResult(
        system=other.system,
        reference=reference.system,
        n_pairs=len(shared),
        median_difference=statistics.median(diffs) if diffs else 0.0,
        mean_difference=statistics.fmean(diffs) if diffs else 0.0,
        n_favoring_reference=n_favor_ref,
        wilcoxon_p=wilcoxon_p,
        sign_test_p=sign_p,
        effect_size_r=effect_r,
        at_significance_boundary=at_boundary,
    )


def holm_adjust(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down adjustment over a family of comparisons.

    Sorts ascending, scales each by the number of hypotheses still untested, and
    enforces monotonicity so an adjusted value never falls below one ranked
    before it. Less conservative than Bonferroni at the same error guarantee.

    Returns adjusted values keyed as the input. Values are capped at 1.
    """
    items = sorted(
        ((k, v) for k, v in p_values.items() if v is not None),
        key=lambda kv: kv[1],
    )
    m = len(items)
    out: Dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(items):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)   # monotonic in rank
        out[key] = running
    for k, v in p_values.items():
        if v is None:
            out[k] = None
    return out


def wilcoxon_floor(n_trials: int) -> float:
    """Smallest attainable two-sided Wilcoxon p at this sample size.

    At n = 10 this is 0.001953, the 0.002 floor in Table 5. Exposed so the
    reporting layer can mark p-values that have hit it rather than presenting
    them as a measured magnitude.
    """
    return 2.0 / (2**n_trials)


def build_table5(
    scores: Dict[str, SystemScores],
    reference_key: str = "adam_full",
    order: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    """Assemble Table 5: per-system metrics with p-values against ADAM Full."""
    if reference_key not in scores:
        raise KeyError(f"reference system {reference_key!r} not among {list(scores)}")
    ref = scores[reference_key]
    keys = list(order) if order else [reference_key] + [
        k for k in scores if k != reference_key
    ]

    raw: Dict[str, float] = {}
    detail: Dict[str, ComparisonResult] = {}
    for key in keys:
        if key == reference_key or key not in scores:
            continue
        cmp = compare_systems(ref, scores[key])
        detail[key] = cmp
        raw[key] = cmp.wilcoxon_p
    adjusted = holm_adjust(raw)

    rows: List[Dict[str, object]] = []
    for key in keys:
        if key not in scores:
            continue
        row = scores[key].summary_row()
        if key == reference_key:
            row["p_exact"] = None
            row["p_holm"] = None
            row["significant"] = None
        else:
            cmp = detail[key]
            row["p_exact"] = cmp.wilcoxon_p
            row["p_holm"] = adjusted.get(key)
            row["significant"] = (
                adjusted.get(key) is not None and adjusted[key] < ALPHA
            )
            row["median_diff"] = cmp.median_difference
            row["trials_favoring_adam"] = cmp.n_favoring_reference
        rows.append(row)
    return rows


def format_table5(rows: Sequence[Dict[str, object]], n_trials: int = 10) -> str:
    """Render Table 5 as fixed-width text."""
    floor = wilcoxon_floor(n_trials)
    head = (
        f"{'System':<22}{'Prec.':>16}{'Recall':>16}{'F1':>16}{'FAR':>16}"
        f"{'p_exact':>10}{'p_Holm':>9}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        p = r.get("p_exact")
        ph = r.get("p_holm")
        if p is None:
            p_str, ph_str = "--", "--"
        else:
            p_str = f"{p:.3f}*" if abs(p - floor) < 1e-9 else f"{p:.3f}"
            ph_str = f"{ph:.3f}" if ph is not None else "--"
        lines.append(
            f"{str(r['system']):<22}"
            f"{r['precision']:>10.3f} ±{r['precision_sd']:<4.3f}"
            f"{r['recall']:>10.3f} ±{r['recall_sd']:<4.3f}"
            f"{r['f1']:>10.3f} ±{r['f1_sd']:<4.3f}"
            f"{r['far']:>10.3f} ±{r['far_sd']:<4.3f}"
            f"{p_str:>10}{ph_str:>9}"
        )
    lines.append("")
    lines.append(
        f"* p_exact is at the Wilcoxon floor for n={n_trials} ({floor:.6f}); "
        f"every trial favors the reference. The floor reflects sample size, "
        f"not effect magnitude."
    )
    lines.append(
        "p_Holm applies the Holm step-down correction across the comparison "
        "family."
    )
    return "\n".join(lines)


__all__ = [
    "ConfusionMatrix",
    "confusion",
    "SystemScores",
    "score_system",
    "ComparisonResult",
    "compare_systems",
    "wilcoxon_floor",
    "build_table5",
    "format_table5",
]
