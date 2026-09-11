"""Fold-local sensor-fusion calibration utilities for sensitivity analysis.

The reported manuscript benchmark uses the fixed inverse-variance weights
preserved in the deposited D1 records; those weights were estimated once from
the labeled calibration data and then held fixed. This module provides the
stricter leave-one-trial-out recalibration requested for sensitivity analysis: it
estimates each node's residual variance on the nine training trials only, then
applies those weights unchanged to the held-out trial.

Do not silently substitute these fold-local weights for the reported benchmark.
The repository exposes the choice explicitly so reported-result reproduction and
methodological sensitivity analysis remain distinguishable.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from adam.schemas import LabeledEvent, SensorReading


def estimate_error_variances(
    train_events: Sequence[LabeledEvent],
    *,
    ddof: int = 1,
) -> Dict[str, float]:
    residuals: Dict[str, List[float]] = {}
    for event in train_events:
        reference = float(event.reference_ppm)
        for reading in event.readings:
            residuals.setdefault(reading.node_id, []).append(
                float(reading.methane_ppm) - reference
            )

    if not residuals:
        raise ValueError("cannot calibrate fusion weights from an empty training fold")

    variances: Dict[str, float] = {}
    for node_id, values in residuals.items():
        if len(values) <= ddof:
            raise ValueError(
                f"node {node_id} has only {len(values)} calibration residuals; "
                f"need more than ddof={ddof}"
            )
        var = float(np.var(np.asarray(values, dtype=float), ddof=ddof))
        if not np.isfinite(var) or var <= 0:
            raise ValueError(f"node {node_id} has invalid residual variance {var}")
        variances[node_id] = var
    return variances


def apply_error_variances(
    events: Sequence[LabeledEvent],
    variances: Mapping[str, float],
) -> List[LabeledEvent]:
    """Return copies whose only changed sensor field is ``error_variance``."""
    out: List[LabeledEvent] = []
    for event in events:
        new_readings: List[SensorReading] = []
        for reading in event.readings:
            if reading.node_id not in variances:
                raise ValueError(
                    f"no training-fold variance available for node {reading.node_id}"
                )
            new_readings.append(
                replace(reading, error_variance=float(variances[reading.node_id]))
            )
        out.append(replace(event, readings=tuple(new_readings)))
    return out


def calibrate_fold(
    train_events: Sequence[LabeledEvent],
    test_events: Sequence[LabeledEvent],
) -> Tuple[List[LabeledEvent], List[LabeledEvent], Dict[str, float]]:
    """Estimate on training events, then freeze the same weights for both arms."""
    variances = estimate_error_variances(train_events)
    return (
        apply_error_variances(train_events, variances),
        apply_error_variances(test_events, variances),
        variances,
    )


__all__ = [
    "estimate_error_variances",
    "apply_error_variances",
    "calibrate_fold",
]
