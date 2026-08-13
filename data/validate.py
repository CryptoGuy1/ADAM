"""
data.validate
=============

Integrity checks that a D1 dataset must pass before any metric is computed.

Why
---
Table 5 reports Static Threshold at F1 = 0.790. That number is obtainable only
if the ground-truth labels are **not** a deterministic function of the MQ-4
reading crossing the screening threshold. If they are, the static-threshold
baseline reduces to reproducing the labelling rule and scores F1 = 1.000, the
Random Forest converges on the same boundary, and every reported margin over
those baselines collapses.

:func:`assert_labels_independent` is called
by :func:`data.loader.load_trials` on every load and raises
:class:`DegenerateLabelsError` when the labels are recoverable from the
screening rule. It is not optional and cannot be silenced by a flag - the only
way past it is a dataset that does not have the defect.

What a sound D1 looks like
--------------------------
Labels come from the NDIR reference analyzer (Table 4, +/-1% FS),
which is a physically separate instrument from the MQ-4 units under evaluation.
The MQ-4 readings carry drift, noise, and cross-sensitivity; the reference does
not. Disagreement between them is the phenomenon the paper is about. Concretely,
a sound dataset shows:

  * MQ-4 above threshold on some reference-negative events (drift-driven false
    positives - the 0.166 FAR of the static baseline)
  * MQ-4 below threshold on some reference-positive events (missed detections)
  * agreement between threshold rule and label well below 100%
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from adam.config import THRESHOLD_PPM
from adam.mechanisms import trigger

logger = logging.getLogger(__name__)


class DatasetIntegrityError(ValueError):
    """Base class for dataset defects that invalidate reported metrics."""


class DegenerateLabelsError(DatasetIntegrityError):
    """Labels are recoverable from the screening rule.

    Every baseline margin in Table 5 is void under this condition.
    """


@dataclass
class LabelDiagnostics:
    """Evidence for or against label determinism."""

    n_events: int
    n_positive: int
    n_negative: int

    #: Fraction of events where trigger(ppm) == label.
    threshold_agreement: float

    #: Reference-negative events whose MQ-4 reading crossed threshold.
    false_positives_available: int

    #: Reference-positive events whose MQ-4 reading did not cross threshold.
    false_negatives_available: int

    #: Implied F1 of the static-threshold rule against these labels.
    implied_static_f1: float

    #: Implied false alarm rate of the static rule.
    implied_static_far: float

    def is_degenerate(self, tolerance: float = 0.995) -> bool:
        """True when the screening rule reproduces the labels almost exactly."""
        return self.threshold_agreement >= tolerance

    def summary(self) -> str:
        return (
            f"{self.n_events} events ({self.n_positive} positive, "
            f"{self.n_negative} negative)\n"
            f"  threshold/label agreement : {self.threshold_agreement:.4f}\n"
            f"  drift-driven FPs available: {self.false_positives_available}\n"
            f"  missed detections available: {self.false_negatives_available}\n"
            f"  implied static-threshold F1 : {self.implied_static_f1:.4f}\n"
            f"  implied static-threshold FAR: {self.implied_static_far:.4f}"
        )


def diagnose_labels(
    readings_ppm: Sequence[float],
    labels: Sequence[int],
    threshold_ppm: float = THRESHOLD_PPM,
) -> LabelDiagnostics:
    """Measure how far the labels are from being the screening rule itself."""
    if len(readings_ppm) != len(labels):
        raise DatasetIntegrityError(
            f"{len(readings_ppm)} readings but {len(labels)} labels"
        )
    if not readings_ppm:
        raise DatasetIntegrityError("empty dataset")

    tp = fp = tn = fn = 0
    agree = 0
    for ppm, label in zip(readings_ppm, labels):
        pred = trigger(ppm, threshold_ppm)
        if pred == label:
            agree += 1
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    n = len(labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    far = fp / (fp + tn) if (fp + tn) else 0.0

    return LabelDiagnostics(
        n_events=n,
        n_positive=sum(labels),
        n_negative=n - sum(labels),
        threshold_agreement=agree / n,
        false_positives_available=fp,
        false_negatives_available=fn,
        implied_static_f1=f1,
        implied_static_far=far,
    )


def assert_labels_independent(
    readings_ppm: Sequence[float],
    labels: Sequence[int],
    threshold_ppm: float = THRESHOLD_PPM,
    tolerance: float = 0.995,
) -> LabelDiagnostics:
    """Raise unless the labels carry information beyond the screening rule.

    Raises
    ------
    DegenerateLabelsError
        When threshold agreement meets or exceeds ``tolerance``, or when the
        dataset contains no drift-driven false positives and no missed
        detections at all.
    """
    diag = diagnose_labels(readings_ppm, labels, threshold_ppm)

    if diag.is_degenerate(tolerance):
        raise DegenerateLabelsError(
            "Ground-truth labels are a deterministic function of the "
            f"{threshold_ppm:.0f} ppm screening rule "
            f"(agreement {diag.threshold_agreement:.4f} >= {tolerance}).\n\n"
            f"{diag.summary()}\n\n"
            "Under these labels the Static Threshold baseline scores "
            f"F1 = {diag.implied_static_f1:.3f}, not the 0.790 reported in "
            "Table 5, and every margin over the rule- and tree-based baselines "
            "is void.\n\n"
            "Labels must come from the NDIR reference analyzer "
            "(Table 4), independently of the MQ-4 readings under evaluation. "
            "If this dataset was derived by thresholding the MQ-4 channel, it "
            "is not a valid evaluation set - re-derive the labels from the "
            "reference channel and re-deposit."
        )

    if diag.false_positives_available == 0 and diag.false_negatives_available == 0:
        raise DegenerateLabelsError(
            "The screening rule makes no errors against these labels: no "
            "drift-driven false positives and no missed detections.\n\n"
            f"{diag.summary()}\n\n"
            "A dataset in which fixed-threshold detection is perfect cannot "
            "demonstrate the failure mode the paper is about."
        )

    logger.info(
        "label independence OK: threshold agreement %.4f, implied static F1 %.3f",
        diag.threshold_agreement,
        diag.implied_static_f1,
    )
    return diag


def check_trial_structure(
    trial_sizes: Dict[int, int],
    expected_trials: int,
    expected_per_trial: int,
) -> List[str]:
    """Verify D1 has the shape Section 3.4.3 describes."""
    problems: List[str] = []
    if len(trial_sizes) != expected_trials:
        problems.append(
            f"found {len(trial_sizes)} trials, Section 3.4.3 specifies {expected_trials}"
        )
    for tid, size in sorted(trial_sizes.items()):
        if size != expected_per_trial:
            problems.append(
                f"trial {tid} holds {size} events, Section 3.4.3 specifies "
                f"{expected_per_trial}"
            )
    return problems


def check_class_balance(labels: Sequence[int], min_minority: float = 0.05) -> List[str]:
    """Warn on a class balance too skewed for F1 to be informative."""
    problems: List[str] = []
    if not labels:
        return ["no labels"]
    pos = sum(labels) / len(labels)
    if pos < min_minority or pos > 1 - min_minority:
        problems.append(
            f"class balance is {pos:.1%} positive; F1 comparisons become "
            f"unstable below {min_minority:.0%} minority representation"
        )
    return problems


def check_reference_consistency(
    reference_ppm: Sequence[float],
    labels: Sequence[int],
    reference_threshold_ppm: float = THRESHOLD_PPM,
    tolerance: float = 0.90,
) -> List[str]:
    """Verify labels track the reference channel rather than something else.

    Labels should be *largely* explained by the reference sensor - that is what
    makes them ground truth. The complement of the guard above: labels must be
    independent of the MQ-4 channel but consistent with the reference channel.
    """
    problems: List[str] = []
    if not reference_ppm or len(reference_ppm) != len(labels):
        return ["reference channel missing or misaligned; cannot verify labels"]
    agree = sum(
        1
        for ref, lab in zip(reference_ppm, labels)
        if trigger(ref, reference_threshold_ppm) == lab
    )
    frac = agree / len(labels)
    if frac < tolerance:
        problems.append(
            f"labels agree with the reference channel only {frac:.1%} of the "
            f"time (expected >= {tolerance:.0%}). Labels should be derived from "
            f"the reference sensor; this suggests a third, undocumented source."
        )
    return problems


def diagnose_all_channels(
    channels: Dict[str, Sequence[float]],
    labels: Sequence[int],
    threshold_ppm: float = THRESHOLD_PPM,
) -> Dict[str, LabelDiagnostics]:
    """Diagnose every sensor channel, not just the one the baseline uses.

    The guard below rejects a dataset whose labels are recoverable from the
    channel under evaluation. That check passes on the raw channel at 0.81
    agreement, but the calibrated channel reaches 0.96 - close enough that a
    fixed threshold on it scores F1 = 0.955, above the full system. A dataset
    can therefore satisfy the guard on one channel while another channel it
    contains is nearly the answer key.
    """
    return {
        name: diagnose_labels(values, labels, threshold_ppm)
        for name, values in channels.items()
    }


__all__ = [
    "DatasetIntegrityError",
    "diagnose_all_channels",
    "DegenerateLabelsError",
    "LabelDiagnostics",
    "diagnose_labels",
    "assert_labels_independent",
    "check_trial_structure",
    "check_class_balance",
    "check_reference_consistency",
]
