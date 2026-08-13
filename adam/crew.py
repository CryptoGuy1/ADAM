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
    line 6      SensorAgent.publish_trigger        -> event e_t
    line 7-9    Crew.form                          -> C_t, N_t
    line 10     AggregatorAgent.aggregate          -> m_bar_t   (Eq. 2)
    line 11     SemanticMemory.retrieve            -> h_past
    line 12     DecisionAgent.reason               -> d_t       (Eq. 3)
    line 13     d_t.recommended_action             -> a_t
    line 14     CoordinatorAgent.collect_votes     -> q_t
    line 15     CoordinatorAgent.validate          -> V, quorum (Eq. 4)
    line 18     resolve_conflict                   -> a*        (Eq. 5)
    line 20     persist                            -> chain + Weaviate
    line 21     Crew.dissolve
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
from .config import ADAMConfig, DEFAULT_CONFIG, SEVERITY_SCORES, quorum
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
        """Ballots available to Equation (4).

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
    """One edge node: hosts all four agent processes and forms crews on demand.

    Section 3.4.1: "Each node hosts the Sensor, Aggregator, Decision, and
    Coordinator agent processes, so it can join event-specific crews by role
    availability."
    """

    def __init__(
        self,
        node_id: str,
        config: ADAMConfig = DEFAULT_CONFIG,
        memory: Optional[Any] = None,
        chain: Optional[Any] = None,
        validator: Optional[Any] = None,
        llm_client: Optional[OllamaClient] = None,
    ):
        self.node_id = node_id
        self.config = config
        self.memory = memory
        self.chain = chain
        self.validator = validator

        if llm_client is None and config.enable_llm:
            llm_client = OllamaClient(
                model=config.ollama_model,
                host=config.ollama_host,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )
        self.llm_client = llm_client

        self.sensor = SensorAgent(f"{node_id}-sensor", node_id, config)
        self._pending: List[Tuple[CrewEvent, DecisionObject]] = []

    # -- crew lifecycle ----------------------------------------------------

    def form_crew(self, event: CrewEvent, available_roles: Optional[Sequence[str]] = None) -> Crew:
        """Assemble an event-specific crew. Algorithm 1 line 8.

        Agents self-select by role availability (Section 3.1.2). Constraint C4
        permits degraded operation down to two agents; below that the node
        refuses to form a crew rather than acting on unvalidated reasoning.
        """
        roles = set(available_roles) if available_roles is not None else set(("sensor", "aggregator", "decision", "coordinator"))

        if not self.config.enable_aggregator:
            roles.discard("aggregator")

        if "sensor" not in roles or "coordinator" not in roles:
            raise CrewFormationError(
                f"cannot form a crew without both sensor and coordinator roles; "
                f"available: {sorted(roles)}"
            )

        aggregator = (
            AggregatorAgent(f"{self.node_id}-aggregator", self.node_id, self.config)
            if "aggregator" in roles
            else None
        )
        decision = DecisionAgent(
            f"{self.node_id}-decision", self.node_id, self.config, client=self.llm_client
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
    ) -> EventTrace:
        """Run one full coordination episode. Algorithm 1 lines 6-21.

        Every stage is timed separately so that Figure 5's decomposition falls
        out of the trace rather than being reconstructed afterwards.
        """
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
                        fused_ppm=fusion.fused_ppm, k=self.config.semantic_memory_k
                    )

            # -- local reasoning (T_reason), Equation (3)
            with timer.stage("T_reason"):
                spent_s = timer.total_ms / 1000.0
                remaining = max(self.config.decision_deadline_s - spent_s, 0.5)
                inference = crew.decision.reason(
                    event=event,
                    fusion=fusion,
                    node_readings=[r.redacted() for r in peer_readings],
                    baseline_window=crew.sensor.baseline,
                    history=history,
                    deadline_s=remaining,
                )
            decision = inference.decision
            trace.decision = decision
            trace.degraded_mode = decision.degraded_mode

            # -- votes and validation (T_gov), Equation (4)
            with timer.stage("T_gov"):
                context = {
                    "trigger_ppm": event.trigger_ppm,
                    "trigger_node": event.trigger_node,
                    "fused_ppm": fusion.fused_ppm,
                    "dispersion_ppm": fusion.dispersion_ppm,
                }
                crew.coordinator.collect_votes(event, crew.voters, decision, context)
                outcome = crew.coordinator.validate(event, decision, crew.voter_count)

            trace.governance_valid = outcome.governance_valid
            trace.quorum_required = outcome.quorum_required
            trace.quorum_achieved = outcome.quorum_achieved
            trace.governance_reason = outcome.reason

            # -- action selection (Algorithm 1 line 17); execution is withheld
            # until the trace commits, so nothing is marked executed here.
            if outcome.approved:
                trace.final_action = decision.recommended_action
            else:
                # A rejection is not a conflict: Equation (5) arbitrates between
                # competing concurrent recommendations, of which there is only
                # one here. The safe fallback records what would have been done
                # without executing it.
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
        """Action recorded when governance or quorum rejects the proposal.

        Distinct from Equation (5): no competing recommendation exists on this
        path, so there is nothing to arbitrate. The action is recorded for the
        audit trace only and is never executed.
        """
        return self._resolve(event, decision)

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

    # -- Equation (5) ------------------------------------------------------

    def _resolve(self, event: CrewEvent, decision: DecisionObject) -> str:
        """Deterministic conflict resolution. Algorithm 1 line 18, Equation (5).

        Reached when governance validation fails or competing recommendations
        arise. Section 4.6 notes this never fired across the 459 deployment
        events, so it is exercised by the synthetic sweep rather than the field
        data - but it must be present and correct for a larger deployment.

        With a single candidate and no pending competitors, the safe outcome is
        to withhold the action and defer to review rather than execute an
        unvalidated recommendation.
        """
        now = time.time()
        candidates = [
            Candidate(
                action=decision.recommended_action,
                severity=decision.severity_score,
                timestamp=event.timestamp,
                event_id=event.event_id,
            )
        ]
        for pending_event, pending_decision in self._pending:
            candidates.append(
                Candidate(
                    action=pending_decision.recommended_action,
                    severity=pending_decision.severity_score,
                    timestamp=pending_event.timestamp,
                    event_id=pending_event.event_id,
                )
            )

        if len(candidates) == 1:
            return "WITHHELD: validation failed, escalated for operator review"

        winner = resolve_conflict(
            candidates,
            t_now=now,
            lambda_severity=self.config.lambda_severity,
            lambda_recency=self.config.lambda_recency,
            normalization=self.config.conflict_normalization,
        )
        return winner.action

    def register_pending(self, event: CrewEvent, decision: DecisionObject) -> None:
        """Register a concurrent unresolved recommendation for Equation (5).

        Used by the multi-crew concurrency harness. In the four-node deployment
        this list stayed empty, which is why Section 4.6 reports the rule as
        never triggered.
        """
        self._pending.append((event, decision))

    def clear_pending(self) -> None:
        self._pending.clear()


__all__ = ["Crew", "ADAMNode", "CrewFormationError"]
