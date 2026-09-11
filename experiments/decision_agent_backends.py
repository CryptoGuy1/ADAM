#!/usr/bin/env python3
"""
Decision-Agent backends used by the ADAM substitution study.

These are planned Decision-Agent configurations, not failure fallbacks.  Each
backend receives the same event-time evidence assembled by the crew and returns
the same structured InferenceResult consumed by the existing voting,
governance, persistence, and audit path.

The fitted Decision-Agent variants use the eight features stated in the
manuscript.  Training is performed outside this module by the study harness
under leave-one-trial-out splitting.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adam.config import (
    CRITICAL_THRESHOLD_PPM,
    RF_PARAMS,
    SEED,
    THRESHOLD_PPM,
    WARNING_THRESHOLD_PPM,
)
from adam.llm.client import InferenceResult
from adam.mechanisms import FusionResult
from adam.schemas import CrewEvent, DecisionObject, LabeledEvent
from baselines.systems import BASELINE_WINDOW, fused_context_matrix


DECISION_AGENT_VARIANTS = (
    "adam_logreg",
    "adam_rf",
    "adam_gbm",
)


def decision_feature_vector(
    *,
    raw_ppm: float,
    fused_ppm: float,
    dispersion_ppm: float,
    baseline_window: Sequence[float],
    threshold_ppm: float = THRESHOLD_PPM,
) -> np.ndarray:
    """Build the eight fitted Decision-Agent features for one event.

    The baseline is causal: only values supplied in ``baseline_window`` are
    used, and the harness supplies at most the preceding six readings from the
    same trial.  The current event is not inserted here.
    """
    history = list(baseline_window[-BASELINE_WINDOW:])
    baseline_mean = statistics.fmean(history) if history else float(raw_ppm)
    denom = baseline_mean if abs(baseline_mean) > 1e-9 else 1e-9

    return np.asarray(
        [
            float(raw_ppm),
            (float(raw_ppm) - float(threshold_ppm)) / float(threshold_ppm),
            float(float(raw_ppm) >= float(threshold_ppm)),
            float(fused_ppm),
            float(dispersion_ppm),
            float(baseline_mean),
            float(fused_ppm) / denom,
            float(fused_ppm) - float(baseline_mean),
        ],
        dtype=float,
    )


def _severity_for(
    fused_ppm: float,
    is_anomaly: bool,
    threshold_ppm: float,
) -> str:
    """Map classifier output to the documented governance concentration bands."""
    del threshold_ppm  # classification threshold is separate from severity policy
    if not is_anomaly:
        return "NONE"
    if float(fused_ppm) >= CRITICAL_THRESHOLD_PPM:
        return "CRITICAL"
    if float(fused_ppm) >= WARNING_THRESHOLD_PPM:
        return "HIGH"
    return "LOW"


def _action_for(severity: str) -> str:
    return {
        "NONE": "continue monitoring",
        "LOW": "monitor",
        "MODERATE": "raise alert",
        "HIGH": "dispatch inspection",
        "CRITICAL": "escalate",
    }[severity]


def decision_from_probability(
    *,
    probability_anomaly: float,
    fused_ppm: float,
    dispersion_ppm: float,
    variant: str,
    threshold_ppm: float,
) -> DecisionObject:
    """Map a fitted classifier output to ADAM's seven-field decision schema."""
    p = min(1.0, max(0.0, float(probability_anomaly)))
    is_anomaly = p >= 0.5
    confidence = p if is_anomaly else 1.0 - p
    severity = _severity_for(fused_ppm, is_anomaly, threshold_ppm)

    return DecisionObject(
        classification="ANOMALY" if is_anomaly else "NORMAL",
        confidence=float(confidence),
        severity=severity,
        reasoning=(
            f"{variant} classified the event from the configured "
            "Decision-Agent feature vector."
        ),
        recommended_action=_action_for(severity),
        contributing_factors=[
            f"fused concentration {float(fused_ppm):.1f} ppm",
            f"cross-node dispersion {float(dispersion_ppm):.1f} ppm",
            f"{variant} anomaly probability {p:.4f}",
        ],
        requires_human_review=bool(
            confidence < 0.60
            or severity == "CRITICAL"
            or float(dispersion_ppm) > 0.5 * max(float(fused_ppm), 1e-9)
        ),
        degraded_mode=False,
    )


class FittedDecisionBackend:
    """Base class for fitted Decision-Agent substitutions."""

    name = "fitted"

    def __init__(self, threshold_ppm: float = THRESHOLD_PPM):
        self.threshold_ppm = threshold_ppm
        self.model: Optional[Any] = None

    def _make_model(self) -> Any:
        raise NotImplementedError

    def fit(self, train: Sequence[LabeledEvent]) -> "FittedDecisionBackend":
        ordered, X = fused_context_matrix(
            train,
            threshold_ppm=self.threshold_ppm,
        )
        y = np.asarray([e.label for e in ordered], dtype=int)
        self.model = self._make_model()
        self.model.fit(np.asarray(X, dtype=float), y)
        return self

    def reason(
        self,
        *,
        event: CrewEvent,
        fusion: FusionResult,
        node_readings: Sequence[Dict[str, Any]],
        baseline_window: Sequence[float],
        history: Sequence[Dict[str, Any]],
        deadline_s: float,
        threshold_ppm: float,
    ) -> InferenceResult:
        if self.model is None:
            raise RuntimeError(f"{self.name} backend has not been fitted")

        started = time.perf_counter()
        x = decision_feature_vector(
            raw_ppm=event.trigger_ppm,
            fused_ppm=fusion.fused_ppm,
            dispersion_ppm=fusion.dispersion_ppm,
            baseline_window=baseline_window,
            threshold_ppm=threshold_ppm,
        )
        p = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
        decision = decision_from_probability(
            probability_anomaly=p,
            fused_ppm=fusion.fused_ppm,
            dispersion_ppm=fusion.dispersion_ppm,
            variant=self.name,
            threshold_ppm=threshold_ppm,
        )
        return InferenceResult(
            decision=decision,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            raw_output="",
            repair_attempted=False,
            fell_back=False,
        )


class LogisticRegressionDecisionBackend(FittedDecisionBackend):
    name = "ADAM_LogReg"

    def _make_model(self) -> Any:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        )


class RandomForestDecisionBackend(FittedDecisionBackend):
    name = "ADAM_RF"

    def _make_model(self) -> Any:
        return RandomForestClassifier(**RF_PARAMS)


class GradientBoostingDecisionBackend(FittedDecisionBackend):
    name = "ADAM_GBM"

    def _make_model(self) -> Any:
        return GradientBoostingClassifier(random_state=SEED)


BACKEND_FACTORIES = {
    "adam_logreg": LogisticRegressionDecisionBackend,
    "adam_rf": RandomForestDecisionBackend,
    "adam_gbm": GradientBoostingDecisionBackend,
}


def make_backend(
    name: str,
    threshold_ppm: float = THRESHOLD_PPM,
) -> FittedDecisionBackend:
    try:
        factory = BACKEND_FACTORIES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown Decision-Agent backend {name!r}; "
            f"choose from {sorted(BACKEND_FACTORIES)}"
        ) from exc
    return factory(threshold_ppm=threshold_ppm)


__all__ = [
    "DECISION_AGENT_VARIANTS",
    "decision_feature_vector",
    "decision_from_probability",
    "FittedDecisionBackend",
    "LogisticRegressionDecisionBackend",
    "RandomForestDecisionBackend",
    "GradientBoostingDecisionBackend",
    "BACKEND_FACTORIES",
    "make_backend",
]
