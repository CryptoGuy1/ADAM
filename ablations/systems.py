"""
ablations.systems
=================

Full ADAM as an evaluable system, plus the four ablations of Section 3.4.4.

    ADAM-No-Aggregator  removes cross-node aggregation; local evidence only
    ADAM-No-LLM         disables model reasoning; deterministic logic after trigger
    ADAM-No-Blockchain  removes governance validation and ledger logging
    ADAM-No-Weaviate    disables semantic-memory retrieval

Each is a *configuration* of the same runtime, not a reimplementation. That is
the point of the design: an ablation that reimplemented the pipeline could
differ from full ADAM for reasons other than the ablated component, and the
attribution in Section 4.1 would not follow.

The one deliberate asymmetry is No-Blockchain. Removing governance removes the
validation gate, so an action executes on quorum alone. Section 4.1 reports
detection essentially unchanged (0.889 vs 0.896) - as intended, since the
governance layer supplies accountability rather than accuracy.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from adam.config import ADAMConfig, DEFAULT_CONFIG
from adam.crew import ADAMNode
from adam.governance.chain import LocalValidator, NullChainClient
from adam.memory.store import InMemoryStore
from adam.schemas import LabeledEvent, Prediction, SensorReading

from baselines.systems import BaselineSystem

logger = logging.getLogger(__name__)


class ADAMSystem(BaselineSystem):
    """Full ADAM, wrapped for scoring against labeled trials.

    Each labeled event is replayed through the real coordination pipeline:
    trigger, crew formation, fusion, retrieval, reasoning, votes, validation,
    persistence, dissolution. Nothing is short-circuited for evaluation.

    Two evaluation modes exist because two D1 runs are deposited. Under
    ``eval_mode="gated"`` an event that does not trigger never forms a crew
    (Section 3.1.1: routine sensing stays lightweight) and is scored NORMAL
    without invoking the model; this is the deployed behaviour and reproduces
    D1_RawTrigger_Log. Under ``eval_mode="full_pipeline"`` every event runs
    the complete crew workflow regardless of the trigger, which is how the
    nine-system benchmark of Table 5 was produced; it reproduces the
    ADAM_Full predictions in 06A_Event_Predictions. The live deployment
    runner is gated unconditionally.
    """

    name = "adam_full"

    def __init__(
        self,
        config: Optional[ADAMConfig] = None,
        memory: Optional[Any] = None,
        chain: Optional[Any] = None,
        validator: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        node_id: str = "node-eval",
        seed_memory: bool = True,
    ):
        self.config = config or DEFAULT_CONFIG
        self.memory = memory if memory is not None else InMemoryStore()
        self.chain = chain if chain is not None else NullChainClient()
        self.validator = validator if validator is not None else LocalValidator()
        self.seed_memory = seed_memory

        self.node = ADAMNode(
            node_id=node_id,
            config=self.config,
            memory=self.memory if self.config.enable_weaviate else None,
            chain=self.chain if self.config.enable_blockchain else None,
            validator=self.validator if self.config.enable_blockchain else None,
            llm_client=llm_client,
        )
        self.traces: List[Any] = []

    def fit(self, train: Sequence[LabeledEvent]) -> None:
        """Warm the Sensor Agent's baseline and seed semantic memory.

        Only sub-threshold training readings feed the baseline, and only
        training-fold events seed memory - a held-out trial is never visible.
        """
        for e in train:
            if e.primary.methane_ppm < self.config.threshold_ppm:
                self.node.sensor.observe(e.primary)

        if self.seed_memory and self.config.enable_weaviate:
            from adam.schemas import DecisionObject, EventTrace
            from adam.mechanisms import fuse_readings

            for e in train[:: max(1, len(train) // 200)]:
                fusion = fuse_readings(list(e.readings))
                trace = EventTrace(
                    event_id=f"seed-{e.trial_id}-{e.event_index}",
                    timestamp=e.timestamp,
                    trigger_node=e.primary.node_id,
                    trigger_ppm=e.primary.methane_ppm,
                    fused_ppm=fusion.fused_ppm,
                    decision=DecisionObject(
                        classification="ANOMALY" if e.label == 1 else "NORMAL",
                        confidence=0.9,
                        severity="HIGH" if e.label == 1 else "NONE",
                        reasoning="historical training-fold outcome",
                        recommended_action="raise alert" if e.label else "monitor",
                        contributing_factors=["training fold"],
                        requires_human_review=False,
                    ),
                    governance_valid=True,
                    final_action="raise alert" if e.label else "monitor",
                    governance_reason=(
                        "confirmed release" if e.label else "no release found"
                    ),
                )
                self.memory.persist_trace(trace)

    def predict(self, event: LabeledEvent) -> Prediction:
        t0 = time.perf_counter()
        local = event.primary

        triggered = self.node.sensor.observe(local) == 1

        # Gated mode is the deployment semantics: a sub-threshold reading never
        # forms a crew and is scored NORMAL on the fast path. Full-pipeline
        # mode is the benchmark semantics: every labeled event is replayed
        # through the complete crew workflow so that all systems classify the
        # same 2,000 events under identical conditions. The deposited runs are
        # D1_RawTrigger_Log (gated) and 06A_Event_Predictions (full pipeline).
        if self.config.eval_mode == "gated" and not triggered:
            return Prediction(
                system=self.name,
                trial_id=event.trial_id,
                event_index=event.event_index,
                predicted=0,
                confidence=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        crew_event = self.node.sensor.publish_trigger(local)
        try:
            trace = self.node.handle_event(
                crew_event, list(event.readings), sample_resources=False
            )
        except Exception:
            logger.exception("coordination failed for %s", crew_event.event_id)
            return Prediction(
                system=self.name,
                trial_id=event.trial_id,
                event_index=event.event_index,
                predicted=1,  # fail toward the safe side for a triggered event
                confidence=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                degraded_mode=True,
            )

        self.traces.append(trace)
        decision = trace.decision
        return Prediction(
            system=self.name,
            trial_id=event.trial_id,
            event_index=event.event_index,
            predicted=1 if (decision and decision.is_anomaly) else 0,
            confidence=decision.confidence if decision else 0.0,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            degraded_mode=trace.degraded_mode,
        )


# ---------------------------------------------------------------------------
# Ablation factories
# ---------------------------------------------------------------------------


def _ablate(base: ADAMConfig, name: str, **flags: bool) -> ADAMConfig:
    return base.with_(**flags)


def make_no_aggregator(base: Optional[ADAMConfig] = None, **kw: Any) -> ADAMSystem:
    """ADAM-No-Aggregator: cross-node fusion removed, local evidence only.

    Section 4.1: F1 falls to 0.869. The Aggregator is also the primary defense
    against sensor injection (Section 4.5.1), so this configuration is the one
    an attacker would most like to face.
    """
    cfg = _ablate(base or DEFAULT_CONFIG, "no_aggregator", enable_aggregator=False)
    sys = ADAMSystem(config=cfg, **kw)
    sys.name = "no_aggregator"
    return sys


def make_no_llm(base: Optional[ADAMConfig] = None, **kw: Any) -> ADAMSystem:
    """ADAM-No-LLM: reasoning disabled, deterministic logic after triggering.

    Section 4.1: the largest single drop, to F1 = 0.840 - which is what
    establishes that crew coordination alone does not account for ADAM's
    improvement over rule-based monitoring.
    """
    cfg = _ablate(base or DEFAULT_CONFIG, "no_llm", enable_llm=False)
    kw.setdefault("llm_client", None)
    sys = ADAMSystem(config=cfg, **kw)
    sys.name = "no_llm"
    return sys


def make_no_blockchain(base: Optional[ADAMConfig] = None, **kw: Any) -> ADAMSystem:
    """ADAM-No-Blockchain: governance validation and ledger logging removed.

    Section 4.1: F1 = 0.889, essentially unchanged. The near-zero accuracy
    effect is the intended result for an accountability mechanism, not evidence
    that the layer is unnecessary (Section 5.1).
    """
    cfg = _ablate(base or DEFAULT_CONFIG, "no_blockchain", enable_blockchain=False)
    sys = ADAMSystem(config=cfg, **kw)
    sys.name = "no_blockchain"
    return sys


def make_no_weaviate(base: Optional[ADAMConfig] = None, **kw: Any) -> ADAMSystem:
    """ADAM-No-Weaviate: semantic-memory retrieval disabled.

    Section 4.1: F1 falls to 0.868. The rest of the pipeline is preserved, so
    the drop isolates the contribution of h_past in Equation (3).
    """
    cfg = _ablate(base or DEFAULT_CONFIG, "no_weaviate", enable_weaviate=False)
    kw.setdefault("seed_memory", False)
    sys = ADAMSystem(config=cfg, **kw)
    sys.name = "no_weaviate"
    return sys


ABLATION_FACTORIES = {
    "no_aggregator": make_no_aggregator,
    "no_llm": make_no_llm,
    "no_blockchain": make_no_blockchain,
    "no_weaviate": make_no_weaviate,
}


__all__ = [
    "ADAMSystem",
    "make_no_aggregator",
    "make_no_llm",
    "make_no_blockchain",
    "make_no_weaviate",
    "ABLATION_FACTORIES",
]
