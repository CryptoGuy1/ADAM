"""
adam.crew
=========

Event-driven crew formation and the coordination protocol of Algorithm 1.

A crew is created when an event needs collaborative interpretation and
dissolved once the event resolves (Section 3.1.2). Nothing persistent is kept
between events beyond the Sensor Agents' rolling baselines and whatever the
memory and governance layers wrote.

Algorithm 1 mapping
-------------------
    sensor screening -> event trigger -> crew formation -> aggregation
    -> semantic retrieval -> initial Decision Agent output
    -> independent Sensor/Aggregator/Decision classification votes
    -> strict majority or UNRESOLVED (withhold action)
    -> class-aligned severity/action -> policy validation
    -> required store acknowledgments -> action release or withholding
    -> crew dissolution.

Historical measurements are not inferred from these reference execution paths.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .agents.roles import (
    AggregatorAgent,
    CoordinatorAgent,
    DecisionAgent,
    SensorAgent,
    ValidationOutcome,
)
from .config import ADAMConfig, DEFAULT_CONFIG, SAME_EVENT_WINDOW_S, SEVERITY_SCORES, quorum
from .coordination import UNRESOLVED, align_decision_to_crew
from .llm.client import OllamaClient
from .mechanisms import Candidate, FusionResult, resolve_conflict
from .schemas import (
    CrewEvent,
    DecisionObject,
    EventTrace,
    ResourceCounters,
    SensorReading,
    StageLatencies,
)
from .telemetry import ResourceSampler, StageTimer

logger = logging.getLogger(__name__)


class CrewFormationError(RuntimeError):
    """Raised when too few agents are available to satisfy constraint C4."""


@dataclass
class Crew:
    """A transient, event-scoped team of role-specialized agents."""

    event_id: str
    sensor: SensorAgent
    aggregator: Optional[AggregatorAgent]
    decision: DecisionAgent
    coordinator: CoordinatorAgent
    config: ADAMConfig = field(default_factory=lambda: DEFAULT_CONFIG)
    formed_at: float = field(default_factory=time.time)
    dissolved: bool = False

    @property
    def members(self) -> List[Any]:
        """All agents in the crew, including the non-voting Coordinator."""
        out: List[Any] = [self.sensor]
        if self.aggregator is not None:
            out.append(self.aggregator)
        out += [self.decision, self.coordinator]
        return out

    @property
    def voters(self) -> List[Any]:
        """Agents that cast a ballot. The Coordinator tallies, so is excluded."""
        out: List[Any] = [self.sensor]
        if self.aggregator is not None:
            out.append(self.aggregator)
        out.append(self.decision)
        return out

    @property
    def size(self) -> int:
        """Agents instantiated for this event, including the Coordinator."""
        return len(self.members)

    @property
    def voter_count(self) -> int:
        """Ballots available to the strict-majority classification rule.

        This, not :attr:`size`, is what quorum is computed over. The
        Coordinator tallies and does not vote (Section 3.2), so a four-agent
        crew supplies three ballots and the deployed threshold is
        quorum(3) = 2: any two of the three role-specific checks must agree.
        Computing quorum over :attr:`size` instead would require 3 of 4 and
        let the tallying agent's presence change the threshold.
        """
        return len(self.voters)

    def dissolve(self) -> None:
        self.dissolved = True


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class ADAMNode:
    """One edge node with local logic for all four ADAM roles.

    The Sensor role screens continuously. When this node triggers an event,
    Aggregator, Decision, and Coordinator role objects are instantiated locally
    for that event. Peer nodes contribute measurements through the shared state
    service; roles are not discovered or negotiated across physical nodes.
    """

    def __init__(
        self,
        node_id: str,
        config: ADAMConfig = DEFAULT_CONFIG,
        memory: Optional[Any] = None,
        chain: Optional[Any] = None,
        validator: Optional[Any] = None,
        llm_client: Optional[OllamaClient] = None,
        decision_backend: Optional[Any] = None,
    ):
        self.node_id = node_id
        self.config = config
        self.memory = memory
        self.chain = chain
        self.validator = validator

        if llm_client is None and config.enable_llm and decision_backend is None:
            llm_client = OllamaClient(
                model=config.ollama_model,
                host=config.ollama_host,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )
        self.llm_client = llm_client
        self.decision_backend = decision_backend

        self.sensor = SensorAgent(f"{node_id}-sensor", node_id, config)
        self._pending: List[Tuple[CrewEvent, DecisionObject]] = []

    # -- crew lifecycle ----------------------------------------------------

    def form_crew(self, event: CrewEvent, available_roles: Optional[Sequence[str]] = None) -> Crew:
        """Instantiate the fixed role set for one event.

        Role definitions are local and fixed rather than discovered across
        physical nodes. The Aggregator may be absent in the degraded
        configuration; Sensor, Decision, and Coordinator are required.
        """
        roles = set(available_roles) if available_roles is not None else set(("sensor", "aggregator", "decision", "coordinator"))

        if not self.config.enable_aggregator:
            roles.discard("aggregator")

        required_roles = {"sensor", "decision", "coordinator"}
        missing_roles = required_roles - roles
        if missing_roles:
            raise CrewFormationError(
                f"cannot form a crew without required roles {sorted(missing_roles)}; "
                f"available: {sorted(roles)}"
            )

        aggregator = (
            AggregatorAgent(f"{self.node_id}-aggregator", self.node_id, self.config)
            if "aggregator" in roles
            else None
        )
        decision = DecisionAgent(
            f"{self.node_id}-decision",
            self.node_id,
            self.config,
            client=self.llm_client,
            backend=self.decision_backend,
        )
        coordinator = CoordinatorAgent(
            f"{self.node_id}-coordinator", self.node_id, self.config, validator=self.validator
        )

        crew = Crew(
            event_id=event.event_id,
            sensor=self.sensor,
            aggregator=aggregator,
            decision=decision,
            coordinator=coordinator,
            config=self.config,
        )

        if crew.size < self.config.min_crew_size:
            raise CrewFormationError(
                f"crew of {crew.size} is below C_min={self.config.min_crew_size} "
                f"(constraint C4)"
            )

        event.crew_members = [a.agent_id for a in crew.members]
        logger.info(
            "crew %s formed with %d agents: %s",
            event.event_id,
            crew.size,
            [a.role for a in crew.members],
        )
        return crew

    # -- Algorithm 1 -------------------------------------------------------

    def handle_event(
        self,
        event: CrewEvent,
        peer_readings: Sequence[SensorReading],
        sample_resources: bool = True,
        baseline_window: Optional[Sequence[float]] = None,
    ) -> EventTrace:
        """Run one full coordination episode. Algorithm 1 lines 6-21.

        Every stage is timed separately so that Figure 5's decomposition falls
        out of the trace rather than being reconstructed afterwards.
        """
        # Freeze temporal context at event start. The current reading may update
        # sensor state for future events, but must not contribute to its own baseline.
        event_baseline = (
            list(baseline_window)
            if baseline_window is not None
            else self.sensor.baseline
        )

        timer = StageTimer()
        sampler = ResourceSampler() if sample_resources else None
        if sampler:
            sampler.start()

        required_stores: List[str] = []
        if self.chain is not None and self.config.enable_blockchain:
            required_stores.append("blockchain")
        if self.memory is not None and self.config.enable_weaviate:
            required_stores.append("weaviate")

        trace = EventTrace(
            event_id=event.event_id,
            timestamp=event.timestamp,
            trigger_node=event.trigger_node,
            trigger_ppm=event.trigger_ppm,
            required_stores=required_stores,
        )

        # -- crew formation (T_form)
        with timer.stage("T_form"):
            crew = self.form_crew(event)
            if self.memory is not None and self.config.enable_weaviate:
                self.memory.publish_trigger(event)
        trace.crew_size = crew.size
        trace.voter_count = crew.voter_count

        try:
            # -- aggregation (T_agg), Equation (2)
            with timer.stage("T_agg"):
                if crew.aggregator is not None:
                    fusion = crew.aggregator.aggregate(peer_readings)
                else:
                    local = next(
                        (r for r in peer_readings if r.node_id == event.trigger_node),
                        peer_readings[0],
                    )
                    fusion = AggregatorAgent.local_only(local)
            trace.fused_ppm = fusion.fused_ppm
            trace.contributing_nodes = list(fusion.contributing_nodes)

            # -- semantic memory retrieval (T_weav), h_past
            with timer.stage("T_weav"):
                history: List[Dict[str, Any]] = []
                if self.memory is not None and self.config.enable_weaviate:
                    history = self.memory.retrieve(
                        fused_ppm=fusion.fused_ppm,
                        k=self.config.semantic_memory_k,
                        cutoff_timestamp=event.timestamp,
                    )

            # -- local reasoning (T_reason). C1 is a shared
            # end-to-end budget, not a per-stage timeout. If earlier stages
            # consume it, do not grant the reasoner an artificial grace period.
            spent_s = timer.total_ms / 1000.0
            remaining = self.config.decision_deadline_s - spent_s
            if remaining <= 0.0:
                trace.failure_stage = "deadline_before_reasoning"
                trace.latencies = timer.to_stage_latencies()
                if sampler:
                    trace.resources = sampler.stop()
                return trace

            with timer.stage("T_reason"):
                inference = crew.decision.reason(
                    event=event,
                    fusion=fusion,
                    node_readings=[r.redacted() for r in peer_readings],
                    baseline_window=event_baseline,
                    history=history,
                    deadline_s=remaining,
                )
            decision = inference.decision
            trace.local_decision = decision
            trace.initial_decision = decision
            trace.model_confidence = decision.confidence
            trace.initial_classification = decision.classification
            trace.decision = decision
            trace.degraded_mode = decision.degraded_mode

            # A reasoner that consumes the remaining shared budget cannot be
            # followed by voting/governance while still satisfying C1. Preserve
            # its output in the trace, but fail closed before action validation.
            if timer.total_ms / 1000.0 >= self.config.decision_deadline_s:
                trace.failure_stage = "local_reasoning_deadline"
                trace.latencies = timer.to_stage_latencies()
                if sampler:
                    trace.resources = sampler.stop()
                return trace

            # -- classification votes and validation (T_gov)
            with timer.stage("T_gov"):
                decision, source_event_id = self._arbitrate_pending(event, decision)
                if source_event_id is not None:
                    trace.conflict_resolved = True
                    trace.conflict_source_event_id = source_event_id
                    trace.initial_decision = decision
                    trace.initial_classification = decision.classification
                    trace.model_confidence = decision.confidence
                    trace.degraded_mode = decision.degraded_mode
                    trace.decision = decision
                context = {
                    "trigger_ppm": event.trigger_ppm,
                    "trigger_node": event.trigger_node,
                    "fused_ppm": fusion.fused_ppm,
                    "dispersion_ppm": fusion.dispersion_ppm,
                    "baseline_window": list(event_baseline),
                }
                crew.coordinator.collect_votes(event, crew.voters, decision, context)
                final_class, support, required = crew.coordinator.tally(event, crew.voter_count)
                trace.classification_votes = dict(event.votes)
                trace.vote_errors = dict(event.vote_errors)
                trace.final_classification = final_class
                trace.quorum_required = required
                trace.quorum_achieved = support
                if final_class == UNRESOLVED:
                    # Neither NORMAL nor ANOMALY has quorum. No policy call,
                    # ledger write, or action release may follow.
                    trace.failure_stage = "classification_quorum"
                    trace.governance_reason = "classification quorum not reached"
                else:
                    trace.crew_agreement_fraction = support / crew.voter_count
                    trace.crew_support = trace.crew_agreement_fraction
                    decision, trace.confidence_source = align_decision_to_crew(
                        decision, final_class, fusion.fused_ppm,
                        support, crew.voter_count,
                    )
                    trace.decision = decision
                    outcome = crew.coordinator.validate(event, decision, crew.voter_count)
                    trace.governance_valid = outcome.governance_valid
                    trace.governance_reason = outcome.reason

            if final_class == UNRESOLVED:
                trace.latencies = timer.to_stage_latencies()
                if sampler:
                    trace.resources = sampler.stop()
                return trace

            # -- action selection (Algorithm 1 line 17); execution is withheld
            # until the trace commits, so nothing is marked executed here.
            if outcome.approved:
                trace.final_action = decision.recommended_action
            else:
                # Governance rejection withholds the action. Conflict
                # arbitration, when needed, occurred before crew voting.
                trace.final_action = self._safe_fallback_action(event, decision)
            trace.executed = False

            # -- persistence (T_bc), Algorithm 1 line 22. Commit precedes
            # execution: an event whose trace cannot be committed within the
            # deadline withholds its action rather than acting unaudited
            # (Section 3.2).
            with timer.stage("T_bc"):
                if self.chain is not None and self.config.enable_blockchain:
                    tx = self.chain.log_decision(event, decision, trace.final_action, outcome)
                    trace.blockchain_tx = tx
                    trace.persisted_chain = tx is not None
                if self.memory is not None and self.config.enable_weaviate:
                    trace.persisted_weaviate = self.memory.persist_trace(trace)

            # -- execution (Algorithm 1 lines 23-27). Released only when the
            # crew approved, every enabled store acknowledged the commit, and
            # the end-to-end budget still holds.
            if outcome.approved:
                acknowledged = self._commit_acknowledged(trace)
                on_time = timer.total_ms / 1000.0 <= self.config.decision_deadline_s
                if acknowledged and on_time:
                    trace.executed = True
                else:
                    trace.failure_stage = (
                        "blockchain_commit" if not acknowledged else "deadline"
                    )
                    logger.warning(
                        "event %s approved but withheld: acknowledged=%s on_time=%s",
                        event.event_id, acknowledged, on_time,
                    )
            else:
                trace.failure_stage = "governance_rejected"

        finally:
            crew.dissolve()
            if self.memory is not None and self.config.enable_weaviate:
                # CrewEvent is ephemeral: cleared on dissolution (Section 3.1.2).
                self.memory.clear_crew_event(event.event_id)

        trace.latencies = timer.to_stage_latencies()
        if sampler:
            trace.resources = sampler.stop()

        if not trace.within_deadline(self.config.decision_deadline_s):
            logger.warning(
                "event %s took %.1fs, exceeding the %.0fs deadline (C1)",
                event.event_id,
                trace.latencies.total_s,
                self.config.decision_deadline_s,
            )
        return trace

    def _safe_fallback_action(
        self, event: CrewEvent, decision: DecisionObject
    ) -> str:
        """Withheld action recorded after policy rejection, never executed."""
        return "WITHHELD: validation failed, escalated for operator review"

    def _commit_acknowledged(self, trace: EventTrace) -> bool:
        """True when every enabled audit store acknowledged the trace commit.

        Ablations disable a store deliberately (Section 3.4.5), so a disabled
        backend cannot withhold execution; only an enabled backend that fails
        to acknowledge does.
        """
        if self.chain is not None and self.config.enable_blockchain:
            if not trace.persisted_chain:
                return False
        if self.memory is not None and self.config.enable_weaviate:
            if not trace.persisted_weaviate:
                return False
        return True

    # -- Conflict resolution ------------------------------------------------

    def _arbitrate_pending(
        self, event: CrewEvent, decision: DecisionObject
    ) -> tuple[DecisionObject, Optional[str]]:
        """Select a concurrent recommendation before classification voting.

        Only recommendations for the same location and event window compete.
        The selected decision still needs crew quorum and policy approval.
        """
        competitors = [
            (pending_event, pending_decision)
            for pending_event, pending_decision in self._pending
            if pending_event.event_id != event.event_id
            and pending_event.location == event.location
            and abs(pending_event.timestamp - event.timestamp) <= SAME_EVENT_WINDOW_S
            and pending_decision.recommended_action != decision.recommended_action
        ]
        if not competitors:
            return decision, None

        options = [(event, decision), *competitors]
        candidates = [
            Candidate(
                action=proposal.recommended_action,
                severity=proposal.severity_score,
                timestamp=source.timestamp,
                event_id=source.event_id,
            )
            for source, proposal in options
        ]
        winner = resolve_conflict(candidates)
        selected = next(
            proposal for source, proposal in options if source.event_id == winner.event_id
        )
        return selected, winner.event_id

    def register_pending(self, event: CrewEvent, decision: DecisionObject) -> None:
        """Register a concurrent unresolved recommendation for conflict resolution.

        Eligible recommendations from the same location and event window are
        ranked before the current event's crew votes. Callers clear this list
        when the concurrent event window has closed.
        """
        self._pending.append((event, decision))

    def clear_pending(self) -> None:
        self._pending.clear()


__all__ = ["Crew", "ADAMNode", "CrewFormationError"]
