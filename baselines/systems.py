"""
baselines.systems
=================

Comparator implementations used by the labeled D1 benchmark.

    Static Threshold         fixed 1,000 ppm rule on the raw MQ-4 channel
    Random Forest (raw)      three raw-input features, LOTO splitting
    Random Forest (fused)    fused/context feature set, LOTO splitting
    Gradient Boosting        fused/context feature set, LOTO splitting
    Cloud-Only               remote GPT-4o-mini through the OpenAI API
    Single-Agent             on-device reasoning without crew coordination

Every comparator implements :class:`BaselineSystem`, so the harness scores them
through one code path and no system gets a bespoke evaluation.

Ground truth never reaches a comparator: events arrive via
``LabeledEvent.agent_view()``, which strips the reference channel.

On the feature sets
-------------------
The raw Random Forest receives only the triggering-node concentration, its
normalized distance from the screening threshold, and a binary threshold
indicator.

The fitted contextual baselines and the fitted Decision-Agent variants use the
same eight-feature representation: raw concentration, normalized threshold
distance, threshold indicator, fused concentration, dispersion, causal baseline
mean, fused-to-baseline ratio, and fused-minus-baseline difference. This is the
representation preserved by the deposited D1 event records and reproduces the
reported fitted-model trial results. All temporal features use the same causal
within-trial six-reading baseline construction.
"""

from __future__ import annotations

import logging
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from adam.config import (
    ADAMConfig,
    CLOUD_MODEL,
    DEFAULT_CONFIG,
    RF_PARAMS,
    THRESHOLD_PPM,
)
from adam.mechanisms import fuse_readings, trigger
from adam.schemas import LabeledEvent, Prediction, SensorReading
from adam.telemetry import EGRESS

logger = logging.getLogger(__name__)


class BaselineSystem:
    """Interface every evaluated system implements."""

    name: str = "base"

    def fit(self, train: Sequence[LabeledEvent]) -> None:
        """Optional training hook. Rule-based systems ignore it."""

    def predict(self, event: LabeledEvent) -> Prediction:
        raise NotImplementedError

    def predict_all(self, events: Sequence[LabeledEvent]) -> List[Prediction]:
        return [self.predict(e) for e in events]


# ---------------------------------------------------------------------------
# Static Threshold
# ---------------------------------------------------------------------------


class StaticThreshold(BaselineSystem):
    """The fixed screening rule applied directly to MQ-4 readings.

    Section 4.1 reports F1 = 0.790 at FAR = 0.165, the mean of the per-trial
    rates; pooled over all 2,000 events the rate is 182/1,100 = 0.166. The high
    false-alarm rate
    reflecting sensitivity to drift and changing background conditions. This is
    the failure mode the rest of the system exists to address.
    """

    name = "static_threshold"

    def __init__(self, threshold_ppm: float = THRESHOLD_PPM):
        self.threshold_ppm = threshold_ppm

    def predict(self, event: LabeledEvent) -> Prediction:
        t0 = time.perf_counter()
        pred = trigger(event.primary.methane_ppm, self.threshold_ppm)
        return Prediction(
            system=self.name,
            trial_id=event.trial_id,
            event_index=event.event_index,
            predicted=pred,
            confidence=1.0 if pred else 0.0,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------


def _features_basic(event: LabeledEvent, threshold_ppm: float) -> List[float]:
    """Three features: the lightweight per-node classifier of Section 3.4.4."""
    ppm = event.primary.methane_ppm
    return [
        ppm,
        (ppm - threshold_ppm) / threshold_ppm,  # normalized threshold distance
        float(ppm >= threshold_ppm),  # binary threshold indicator
    ]


class RandomForestBaseline(BaselineSystem):
    """scikit-learn Random Forest, evaluated under leave-one-trial-out.

    LOTO matters: events within a trial form a continuous, non-independent
    exposure sequence (Section 3.4.5), so a random split leaks the test trial's
    exposure profile into training and inflates the score. Each fold trains on
    nine trials and tests on the held-out one.

    Hyperparameters are fixed across folds (Section 3.4.4): n_estimators=100,
    max_depth=5, random_state=42.
    """

    name = "random_forest_raw"

    def __init__(self, threshold_ppm: float = THRESHOLD_PPM):
        self.threshold_ppm = threshold_ppm
        self._model: Optional[Any] = None
        self._baseline_mean: float = 0.0

    def _featurize(self, event: LabeledEvent) -> List[float]:
        return _features_basic(event, self.threshold_ppm)

    def fit(self, train: Sequence[LabeledEvent]) -> None:
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("scikit-learn is required for the RF baseline") from exc

        # The temporal baseline is estimated from sub-threshold training events
        # only, so it never sees a test-trial reading.
        sub = [
            e.primary.methane_ppm
            for e in train
            if e.primary.methane_ppm < self.threshold_ppm
        ]
        self._baseline_mean = statistics.fmean(sub) if sub else 0.0

        X = [self._featurize(e) for e in train]
        y = [e.label for e in train]
        self._model = RandomForestClassifier(**RF_PARAMS)
        self._model.fit(X, y)

    def predict(self, event: LabeledEvent) -> Prediction:
        if self._model is None:
            raise RuntimeError(f"{self.name} has not been fitted")
        t0 = time.perf_counter()
        x = [self._featurize(event)]
        pred = int(self._model.predict(x)[0])
        proba = self._model.predict_proba(x)[0]
        conf = float(proba[1]) if len(proba) > 1 else float(proba[0])
        return Prediction(
            system=self.name,
            trial_id=event.trial_id,
            event_index=event.event_index,
            predicted=pred,
            confidence=conf,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )



# ---------------------------------------------------------------------------
# Shared fused-context feature set
# ---------------------------------------------------------------------------

BASELINE_WINDOW: int = 6

CONTEXTUAL_FEATURE_NAMES: Tuple[str, ...] = (
    "raw_ppm",
    "threshold_distance",
    "threshold_indicator",
    "fused_ppm",
    "dispersion_ppm",
    "baseline_mean",
    "fused_to_baseline_ratio",
    "fused_minus_baseline",
)

# Public aliases retained for compatibility with earlier analysis scripts.
# Both refer to the same eight-feature contextual representation in the
# revised manuscript and deposited master workbook.
DECISION_AGENT_FEATURE_NAMES: Tuple[str, ...] = CONTEXTUAL_FEATURE_NAMES
FUSED_BASELINE_FEATURE_NAMES: Tuple[str, ...] = CONTEXTUAL_FEATURE_NAMES


def fused_context_features(
    event: LabeledEvent,
    prior_raw_ppm: Sequence[float],
    threshold_ppm: float = THRESHOLD_PPM,
) -> List[float]:
    """Construct the eight fitted-model features without label leakage.

    ``prior_raw_ppm`` must contain only measurements that precede ``event`` in
    the same trial. The current reading is appended by the caller *after*
    feature construction.

    The first event of a trial has no preceding baseline samples, so its current
    raw reading initializes the baseline value. Subsequent events use at most
    the preceding six raw readings.
    """
    raw_ppm = float(event.primary.methane_ppm)
    fusion = fuse_readings(event.readings)

    history = list(prior_raw_ppm[-BASELINE_WINDOW:])
    baseline_mean = (
        statistics.fmean(history)
        if history
        else raw_ppm
    )

    denominator = (
        baseline_mean
        if abs(baseline_mean) > 1e-9
        else 1e-9
    )

    return [
        raw_ppm,
        (raw_ppm - threshold_ppm) / threshold_ppm,
        float(raw_ppm >= threshold_ppm),
        float(fusion.fused_ppm),
        float(fusion.dispersion_ppm),
        float(baseline_mean),
        float(fusion.fused_ppm) / denominator,
        float(fusion.fused_ppm) - float(baseline_mean),
    ]


def fused_context_matrix(
    events: Sequence[LabeledEvent],
    threshold_ppm: float = THRESHOLD_PPM,
) -> Tuple[List[LabeledEvent], List[List[float]]]:
    """Return events and the eight Decision-Agent features in trial-time order.

    Baseline history is reset at every trial boundary. No label or NDIR
    reference value enters the feature vector.
    """
    ordered = sorted(
        events,
        key=lambda e: (e.trial_id, e.timestamp, e.event_index),
    )

    history_by_trial: Dict[int, List[float]] = {}
    X: List[List[float]] = []

    for event in ordered:
        history = history_by_trial.setdefault(event.trial_id, [])

        X.append(
            fused_context_features(
                event,
                prior_raw_ppm=history,
                threshold_ppm=threshold_ppm,
            )
        )

        # Update only AFTER featurizing the current event.
        history.append(float(event.primary.methane_ppm))

    return ordered, X


def fused_baseline_matrix(
    events: Sequence[LabeledEvent],
    threshold_ppm: float = THRESHOLD_PPM,
) -> Tuple[List[LabeledEvent], List[List[float]]]:
    """Compatibility wrapper for the eight-feature contextual matrix.

    The function name is retained because older scripts imported it. The
    revised manuscript and deposited trial records show that the reported
    Random Forest (fused/contextual) and Gradient Boosting (fused/contextual)
    results use the same eight-feature contextual representation as the fitted
    Decision-Agent substitutions, not a three-feature subset.
    """
    return fused_context_matrix(events, threshold_ppm=threshold_ppm)


# ---------------------------------------------------------------------------
# Random Forest (fused context)
# ---------------------------------------------------------------------------


class RandomForestFusedBaseline(BaselineSystem):
    """Random Forest over the eight-feature contextual representation.

    The learner uses the same Random Forest hyperparameters as the raw-input
    model. Leave-one-trial-out splitting is provided by the experiment harness.
    """

    name = "random_forest_fused"

    def __init__(self, threshold_ppm: float = THRESHOLD_PPM):
        self.threshold_ppm = threshold_ppm
        self._model: Optional[Any] = None

    def fit(self, train: Sequence[LabeledEvent]) -> None:
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "scikit-learn is required for the fused RF baseline"
            ) from exc

        ordered, X = fused_baseline_matrix(
            train,
            threshold_ppm=self.threshold_ppm,
        )
        y = [event.label for event in ordered]

        self._model = RandomForestClassifier(**RF_PARAMS)
        self._model.fit(X, y)

    def predict_all(
        self,
        events: Sequence[LabeledEvent],
    ) -> List[Prediction]:
        if self._model is None:
            raise RuntimeError(f"{self.name} has not been fitted")

        ordered, X = fused_baseline_matrix(
            events,
            threshold_ppm=self.threshold_ppm,
        )

        out: List[Prediction] = []

        for event, features in zip(ordered, X):
            t0 = time.perf_counter()

            pred = int(self._model.predict([features])[0])
            proba = self._model.predict_proba([features])[0]
            confidence = (
                float(proba[1])
                if len(proba) > 1
                else float(proba[0])
            )

            out.append(
                Prediction(
                    system=self.name,
                    trial_id=event.trial_id,
                    event_index=event.event_index,
                    predicted=pred,
                    confidence=confidence,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
            )

        return out

    def predict(self, event: LabeledEvent) -> Prediction:
        raise RuntimeError(
            "random_forest_fused requires trial-ordered predict_all() "
            "so the six-reading baseline remains causal"
        )


# ---------------------------------------------------------------------------
# Gradient Boosting (fused context)
# ---------------------------------------------------------------------------


class GradientBoostingFusedBaseline(BaselineSystem):
    """Gradient Boosting over the eight-feature contextual representation."""

    name = "gradient_boosting_fused"

    def __init__(self, threshold_ppm: float = THRESHOLD_PPM):
        self.threshold_ppm = threshold_ppm
        self._model: Optional[Any] = None

    def fit(self, train: Sequence[LabeledEvent]) -> None:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "scikit-learn is required for the fused GBM baseline"
            ) from exc

        ordered, X = fused_baseline_matrix(
            train,
            threshold_ppm=self.threshold_ppm,
        )
        y = [event.label for event in ordered]

        self._model = GradientBoostingClassifier(
            random_state=42,
        )
        self._model.fit(X, y)

    def predict_all(
        self,
        events: Sequence[LabeledEvent],
    ) -> List[Prediction]:
        if self._model is None:
            raise RuntimeError(f"{self.name} has not been fitted")

        ordered, X = fused_baseline_matrix(
            events,
            threshold_ppm=self.threshold_ppm,
        )

        out: List[Prediction] = []

        for event, features in zip(ordered, X):
            t0 = time.perf_counter()

            pred = int(self._model.predict([features])[0])
            proba = self._model.predict_proba([features])[0]
            confidence = (
                float(proba[1])
                if len(proba) > 1
                else float(proba[0])
            )

            out.append(
                Prediction(
                    system=self.name,
                    trial_id=event.trial_id,
                    event_index=event.event_index,
                    predicted=pred,
                    confidence=confidence,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
            )

        return out

    def predict(self, event: LabeledEvent) -> Prediction:
        raise RuntimeError(
            "gradient_boosting_fused requires trial-ordered predict_all() "
            "so the six-reading baseline remains causal"
        )


def loto_predictions(
    system_factory: Any,
    events: Sequence[LabeledEvent],
) -> List[Prediction]:
    """Leave-one-trial-out evaluation. Section 3.4.4.

    For each fold, a *fresh* model is constructed, trained on the other trials,
    and used to predict the held-out trial. Reusing one instance across folds
    would carry the previous fold's fitted state.
    """
    trial_ids = sorted({e.trial_id for e in events})
    out: List[Prediction] = []
    for held_out in trial_ids:
        train = [e for e in events if e.trial_id != held_out]
        test = [e for e in events if e.trial_id == held_out]
        model = system_factory()
        model.fit(train)
        out.extend(model.predict_all(test))
    return out


# ---------------------------------------------------------------------------
# Cloud-Only
# ---------------------------------------------------------------------------


class CloudOnly(BaselineSystem):
    """Remote GPT-4o-mini through the OpenAI API. Section 3.4.4.

    Receives the same structured input fields as ADAM's Decision Agent, so the
    comparison isolates *where* inference runs rather than what it is shown.

    This is the only component in the repository that makes an external call.
    It records to :data:`adam.telemetry.EGRESS`, which is how Section 4.5.3's
    zero-egress claim for ADAM is verified rather than asserted: the ledger is
    checked after an ADAM run and must be empty.
    """

    name = "cloud_only"

    def __init__(
        self,
        model: str = CLOUD_MODEL,
        api_key: Optional[str] = None,
        threshold_ppm: float = THRESHOLD_PPM,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.threshold_ppm = threshold_ppm
        self._client: Optional[Any] = None
        self._baseline: List[float] = []

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "openai is required for the Cloud-Only baseline. "
                    "`pip install openai` and set OPENAI_API_KEY."
                ) from exc
            if not self.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. The Cloud-Only baseline needs it; "
                    "ADAM itself does not."
                )
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def fit(self, train: Sequence[LabeledEvent]) -> None:
        sub = [
            e.primary.methane_ppm
            for e in train
            if e.primary.methane_ppm < self.threshold_ppm
        ]
        self._baseline = sub[-30:]

    def predict(self, event: LabeledEvent) -> Prediction:
        from adam.llm.client import extract_json
        from adam.llm.prompt import build_system_prompt, build_user_prompt
        from adam.schemas import DecisionObject, SchemaViolation

        client = self._ensure_client()
        fusion = fuse_readings(list(event.readings))
        user_prompt = build_user_prompt(
            trigger_ppm=event.primary.methane_ppm,
            trigger_node=event.primary.node_id,
            fused_ppm=fusion.fused_ppm,
            node_readings=event.agent_view(),
            baseline_window=self._baseline,
            history=[],
            dispersion_ppm=fusion.dispersion_ppm,
        )
        system_prompt = build_system_prompt(self.threshold_ppm)

        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            text = resp.choices[0].message.content or ""
            decision = DecisionObject.from_model_json(extract_json(text))
            predicted = 1 if decision.is_anomaly else 0
            confidence = decision.confidence
        except (SchemaViolation, ValueError, KeyError) as exc:
            logger.warning("cloud baseline off-schema, falling back: %s", exc)
            predicted = trigger(fusion.fused_ppm, self.threshold_ppm)
            confidence = 0.5
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # The egress ledger records measured bytes only. No dollar figure is
        # attached anywhere in the codebase: the deposit contains no cloud
        # token or billing records, and the comparison the evaluation makes is
        # measured external egress against zero.
        EGRESS.record(
            destination="api.openai.com",
            n_bytes=len(system_prompt) + len(user_prompt),
        )

        return Prediction(
            system=self.name,
            trial_id=event.trial_id,
            event_index=event.event_index,
            predicted=predicted,
            confidence=confidence,
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Single-Agent
# ---------------------------------------------------------------------------


class SingleAgent(BaselineSystem):
    """On-device reasoning without crew coordination. Section 3.4.4.

    One monolithic agent screens, reasons, and decides. No cross-node
    aggregation, no role-specific checks, no semantic memory, no agreement
    validation. Section 4.1 reports F1 = 0.855, and the gap to full ADAM is the
    evidence that the crew workflow adds value beyond a single local LLM call.
    """

    name = "single_agent"

    def __init__(
        self,
        config: ADAMConfig = DEFAULT_CONFIG,
        client: Optional[Any] = None,
    ):
        self.config = config
        if client is None and config.enable_llm:
            from adam.llm.client import OllamaClient

            client = OllamaClient(
                model=config.ollama_model,
                host=config.ollama_host,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )
        self.client = client
        self._baseline: List[float] = []

    def fit(self, train: Sequence[LabeledEvent]) -> None:
        sub = [
            e.primary.methane_ppm
            for e in train
            if e.primary.methane_ppm < self.config.threshold_ppm
        ]
        self._baseline = sub[-30:]

    def predict(self, event: LabeledEvent) -> Prediction:
        from adam.llm.client import deterministic_fallback
        from adam.llm.prompt import build_user_prompt

        t0 = time.perf_counter()
        local = event.primary

        if self.client is None:
            decision = deterministic_fallback(
                local.methane_ppm, self.config.threshold_ppm, reason="no LLM configured"
            )
        else:
            user_prompt = build_user_prompt(
                trigger_ppm=local.methane_ppm,
                trigger_node=local.node_id,
                # No aggregation: the single agent sees only its own reading.
                fused_ppm=local.methane_ppm,
                node_readings=[local.redacted()],
                baseline_window=self._baseline,
                history=[],  # no semantic memory
            )
            result = self.client.decide(
                user_prompt=user_prompt,
                fused_ppm=local.methane_ppm,
                deadline_s=self.config.decision_deadline_s,
                threshold_ppm=self.config.threshold_ppm,
            )
            decision = result.decision

        return Prediction(
            system=self.name,
            trial_id=event.trial_id,
            event_index=event.event_index,
            predicted=1 if decision.is_anomaly else 0,
            confidence=decision.confidence,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            degraded_mode=decision.degraded_mode,
        )


__all__ = [
    "BaselineSystem",
    "StaticThreshold",
    "RandomForestBaseline",
    "RandomForestFusedBaseline",
    "GradientBoostingFusedBaseline",
    "CONTEXTUAL_FEATURE_NAMES",
    "FUSED_BASELINE_FEATURE_NAMES",
    "DECISION_AGENT_FEATURE_NAMES",
    "BASELINE_WINDOW",
    "fused_context_features",
    "fused_context_matrix",
    "fused_baseline_matrix",
    "CloudOnly",
    "SingleAgent",
    "loto_predictions",
]
