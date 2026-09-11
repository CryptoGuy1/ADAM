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

    The reference benchmark replays every labeled event through the complete
    crew workflow. Deployment semantics are derived afterward from those frozen
    benchmark predictions: above-gate predictions are preserved exactly and
    below-gate events are assigned NORMAL without a second model execution.
    """

    name = "adam_llm"

    def __init__(
        self,
        config: Optional[ADAMConfig] = None,
        memory: Optional[Any] = None,
        chain: Optional[Any] = None,
        validator: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        decision_backend: Optional[Any] = None,
        node_id: str = "node-eval",
    ):
        self.config = config or DEFAULT_CONFIG
        self.memory = memory if memory is not None else InMemoryStore()
        self.chain = chain if chain is not None else NullChainClient()
        self.validator = validator if validator is not None else LocalValidator()

        self.node = ADAMNode(
            node_id=node_id,
            config=self.config,
            memory=self.memory if self.config.enable_weaviate else None,
            chain=self.chain if self.config.enable_blockchain else None,
            validator=self.validator if self.config.enable_blockchain else None,
            llm_client=llm_client,
            decision_backend=decision_backend,
        )
        self.traces: List[Any] = []

    def fit(self, train: Sequence[LabeledEvent]) -> None:
        """ADAM has no supervised fit step in the reference benchmark.

        The experiment harness still rebuilds the stateful runtime per held-out
        trial. Sensor history and semantic memory then accumulate only from
        events already processed in that held-out replay; training-fold labels
        and measurements are not used to warm either state.
        """
        return None

    def predict(self, event: LabeledEvent) -> Prediction:
        t0 = time.perf_counter()
        local = event.primary

        # Snapshot temporal context before ingesting the current reading.
        baseline_before = self.node.sensor.baseline
        triggered = self.node.sensor.observe(local) == 1

        # Full-pipeline mode is the benchmark semantics. The legacy gated branch
        # is retained only for non-manuscript compatibility; revised manuscript
        # deployment semantics must be derived from frozen benchmark outputs.
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
                crew_event,
                list(event.readings),
                sample_resources=False,
                baseline_window=baseline_before,
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
    cfg = _ablate(base or DEFAULT_CONFIG, "adam_no_aggregator", enable_aggregator=False)
    sys = ADAMSystem(config=cfg, **kw)
    sys.name = "adam_no_aggregator"
    return sys


def make_no_llm(base: Optional[ADAMConfig] = None, **kw: Any) -> ADAMSystem:
    """ADAM-No-LLM: reasoning disabled, deterministic logic after triggering.

    Section 4.1: the largest single drop, to F1 = 0.840 - which is what
    establishes that crew coordination alone does not account for ADAM's
    improvement over rule-based monitoring.
    """
    cfg = _ablate(base or DEFAULT_CONFIG, "adam_no_llm", enable_llm=False)
    kw.setdefault("llm_client", None)
    sys = ADAMSystem(config=cfg, **kw)
    sys.name = "adam_no_llm"
    return sys


def make_no_blockchain(base: Optional[ADAMConfig] = None, **kw: Any) -> ADAMSystem:
    """ADAM-No-Blockchain: governance validation and ledger logging removed.

    Section 4.1: F1 = 0.889, essentially unchanged. The near-zero accuracy
    effect is the intended result for an accountability mechanism, not evidence
    that the layer is unnecessary (Section 5.1).
    """
    cfg = _ablate(base or DEFAULT_CONFIG, "adam_no_blockchain", enable_blockchain=False)
    sys = ADAMSystem(config=cfg, **kw)
    sys.name = "adam_no_blockchain"
    return sys


def make_no_weaviate(base: Optional[ADAMConfig] = None, **kw: Any) -> ADAMSystem:
    """ADAM-No-Weaviate: semantic-memory retrieval disabled.

    Section 4.1: F1 falls to 0.868. The rest of the pipeline is preserved, so
    the drop isolates the contribution of h_past in Equation (3).
    """
    cfg = _ablate(base or DEFAULT_CONFIG, "adam_no_weaviate", enable_weaviate=False)
    sys = ADAMSystem(config=cfg, **kw)
    sys.name = "adam_no_weaviate"
    return sys


ABLATION_FACTORIES = {
    "adam_no_aggregator": make_no_aggregator,
    "adam_no_llm": make_no_llm,
    "adam_no_blockchain": make_no_blockchain,
    "adam_no_weaviate": make_no_weaviate,
}


__all__ = [
    "ADAMSystem",
    "make_no_aggregator",
    "make_no_llm",
    "make_no_blockchain",
    "make_no_weaviate",
    "ABLATION_FACTORIES",
]
