"""
baselines.systems
=================

The four comparators of Section 3.4.4.

    Static Threshold  fixed 1,000 ppm rule applied to raw MQ-4 readings
    Random Forest     scikit-learn, three features, LOTO splitting
    Cloud-Only        remote GPT-4o-mini through the OpenAI API
    Single-Agent      on-device reasoning without crew coordination

Every comparator implements :class:`BaselineSystem`, so the harness scores them
through one code path and no system gets a bespoke evaluation.

Ground truth never reaches a comparator: events arrive via
``LabeledEvent.agent_view()``, which strips the reference channel.

On the feature set
------------------
Section 3.4.4 specifies three features: the current reading, its normalized
distance from the screening threshold, and a binary threshold indicator. This is
a lightweight per-node classifier by design - the comparison it supports is
against single-node rule-based detection, not against a feature-rich model.
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

    Section 4.1 reports F1 = 0.790 at FAR = 0.166 - the high false-alarm rate
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

    name = "random_forest"

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
    "CloudOnly",
    "SingleAgent",
    "loto_predictions",
]
