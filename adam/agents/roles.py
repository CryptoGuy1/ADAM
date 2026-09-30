"""
adam.agents.roles
=================

The four role-specialized agents of Section 3.1.2.

    Sensor       monitors local streams, screens, publishes the trigger
    Aggregator   retrieves cross-node evidence, fuses it (Equation 2)
    Decision     prompts the local model, returns d_t (Equation 3)
    Coordinator  tallies class quorum, validates policy, executes, verifies trace

Two properties matter for the manuscript's claims and are enforced here rather
than left to convention:

1. **The Coordinator is a crew-instance role, not a global controller.**
   Section 3.1.2 is explicit on this point. Each crew constructs its own, and
   it holds no state across events.

2. **The Coordinator does not cast or alter ballots.** It tallies independent
   NORMAL/ANOMALY classifications and validates a class-aligned action only
   when strict majority is met. :meth:`CoordinatorAgent.tally` reads the vote
   map but never supplies another vote.

Voting
------
Each agent casts its own NORMAL/ANOMALY classification, not an approval of the
Decision Agent's action. Sensor uses local observations, Aggregator uses fused
peer evidence, and Decision uses its proposed class. The Coordinator counts but
does not vote, and a two-voter split withholds action.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from ..config import ADAMConfig, DEFAULT_CONFIG, SEVERITY_SCORES
from ..llm.client import InferenceResult, OllamaClient, deterministic_fallback
from ..llm.prompt import build_user_prompt
from ..mechanisms import FusionResult, fuse_readings, trigger
from ..schemas import CrewEvent, DecisionObject, SensorReading

logger = logging.getLogger(__name__)


class Agent:
    """Common base. Agents are cheap objects created per crew."""

    role: str = "base"

    def __init__(self, agent_id: str, node_id: str, config: ADAMConfig = DEFAULT_CONFIG):
        self.agent_id = agent_id
        self.node_id = node_id
        self.config = config

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.agent_id}@{self.node_id}>"

    def vote(self, decision: DecisionObject, context: Dict[str, Any]) -> str:
        """Return an independent NORMAL/ANOMALY classification."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Sensor Agent
# ---------------------------------------------------------------------------


class SensorAgent(Agent):
    """Monitors the local stream, screens, and publishes crew-formation triggers.

    Holds a rolling baseline window, which serves two purposes: it is the
    ``{m_{t-k:t}}`` temporal context of the Decision-Agent input, and it is this agent's
    independent evidence when voting.
    """

    role = "sensor"

    def __init__(
        self,
        agent_id: str,
        node_id: str,
        config: ADAMConfig = DEFAULT_CONFIG,
        baseline_window: int = 30,
    ):
        super().__init__(agent_id, node_id, config)
        self.baseline_window = baseline_window
        self._history: List[float] = []

    def observe(self, reading: SensorReading) -> int:
        """Ingest one sample and evaluate tau. Equation (1).

        The baseline is updated only on sub-threshold readings, so a sustained
        release does not drag the baseline up behind it and mask itself.
        """
        t = trigger(reading.methane_ppm, self.config.threshold_ppm)
        if t == 0:
            self._history.append(reading.methane_ppm)
            if len(self._history) > self.baseline_window:
                self._history.pop(0)
        return t

    @property
    def baseline(self) -> List[float]:
        return list(self._history)

    @property
    def baseline_mean(self) -> float:
        return statistics.fmean(self._history) if self._history else float("nan")

    def publish_trigger(self, reading: SensorReading, location: str = "") -> CrewEvent:
        """Create the event that opens crew formation. Algorithm 1, line 6."""
        return CrewEvent(
            event_id=CrewEvent.new_id(),
            trigger_node=reading.node_id,
            trigger_ppm=reading.methane_ppm,
            timestamp=reading.timestamp,
            location=location or reading.node_id,
            severity_hint=self._severity_hint(reading.methane_ppm),
        )

    def _severity_hint(self, ppm: float) -> str:
        ratio = ppm / self.config.threshold_ppm if self.config.threshold_ppm else 0.0
        if ratio >= 5.0:
            return "CRITICAL"
        if ratio >= 2.0:
            return "HIGH"
        if ratio >= 1.5:
            return "MODERATE"
        return "LOW"

    def vote(self, decision: DecisionObject, context: Dict[str, Any]) -> str:
        """Independent local class: screen crossing or >=2x prior baseline.

        Only pre-event baseline values are used, never the NDIR reference or
        the Decision Agent's proposed class. A missing baseline does not
        automatically imply an infinite departure.
        """
        local_ppm = float(context["trigger_ppm"])
        triggered = trigger(local_ppm, self.config.threshold_ppm) == 1
        history = context.get("baseline_window")
        base = statistics.fmean(history) if history else self.baseline_mean
        departure = local_ppm / base if base > 0 and base == base else 0.0
        return "ANOMALY" if triggered or departure >= 2.0 else "NORMAL"


# ---------------------------------------------------------------------------
# Aggregator Agent
# ---------------------------------------------------------------------------


class AggregatorAgent(Agent):
    """Retrieves cross-node evidence and composes the fused snapshot.

    This is the agent the pre-revision codebase lacked: it was named
    "validator" and performed schema checks rather than Equation (2) fusion.
    Its contribution can be assessed through the corresponding ablation.
    """

    role = "aggregator"

    #: Weighted z-score past which a node is flagged as disagreeing.
    OUTLIER_Z: float = 1.5
    # With three contributing nodes the maximum attainable standardized
    # deviation is sqrt(2)=1.414. Use a testable 1.25 threshold on the
    # three-node dropout path rather than raising or disabling corroboration.
    DROPOUT_OUTLIER_Z: float = 1.25

    def __init__(self, agent_id: str, node_id: str, config: ADAMConfig = DEFAULT_CONFIG):
        super().__init__(agent_id, node_id, config)
        self._last: Optional[FusionResult] = None

    def aggregate(self, readings: Sequence[SensorReading]) -> FusionResult:
        """Fuse active-node readings. Equation (2), Algorithm 1 line 10.

        When the aggregator is disabled (ADAM-No-Aggregator), the crew calls
        :meth:`local_only` instead.
        """
        threshold = self.DROPOUT_OUTLIER_Z if len(readings) == 3 else self.OUTLIER_Z
        result = fuse_readings(readings, outlier_z=threshold)
        self._last = result
        if result.outliers:
            logger.info(
                "%s flags disagreeing nodes %s (dispersion %.1f ppm)",
                self.agent_id,
                result.outliers,
                result.dispersion_ppm,
            )
        return result

    @staticmethod
    def local_only(reading: SensorReading) -> FusionResult:
        """Degenerate 'fusion' over a single node. ADAM-No-Aggregator path."""
        return FusionResult(
            fused_ppm=reading.methane_ppm,
            weights={reading.node_id: 1.0},
            contributing_nodes=(reading.node_id,),
            dispersion_ppm=0.0,
            outliers=(),
        )

    def vote(self, decision: DecisionObject, context: Dict[str, Any]) -> str:
        """Independent cross-node classification using the fused measurement.

        A single high triggering node cannot force this vote when its peers
        lower the fused measurement below the screening operating point.
        """
        fused = float(context["fused_ppm"])
        return "ANOMALY" if trigger(fused, self.config.threshold_ppm) else "NORMAL"


# ---------------------------------------------------------------------------
# Decision Agent
# ---------------------------------------------------------------------------


class DecisionBackend(Protocol):
    """Interchangeable reasoner occupying the ADAM Decision-Agent slot.

    A backend receives the same event-time evidence assembled by the crew and
    returns the same structured ``InferenceResult`` consumed by voting,
    governance, persistence, and audit logic.
    """

    name: str

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
        ...


class DecisionAgent(Agent):
    """Hosts the configured reasoner and returns the structured decision d_t.

    Gemma is the default backend in ADAM_LLM. The same slot can host a fitted
    model or deterministic reasoner while the surrounding crew, voting,
    governance, persistence, and audit path remain unchanged.
    """

    role = "decision"

    def __init__(
        self,
        agent_id: str,
        node_id: str,
        config: ADAMConfig = DEFAULT_CONFIG,
        client: Optional[OllamaClient] = None,
        backend: Optional[DecisionBackend] = None,
    ):
        super().__init__(agent_id, node_id, config)
        self.client = client
        self.backend = backend
        self._last: Optional[DecisionObject] = None

    def reason(
        self,
        event: CrewEvent,
        fusion: FusionResult,
        node_readings: Sequence[Dict[str, Any]],
        baseline_window: Sequence[float],
        history: Sequence[Dict[str, Any]],
        deadline_s: Optional[float] = None,
    ) -> InferenceResult:
        """Generate the structured decision object for the event.

        With ``enable_llm=False`` (ADAM-No-LLM) this reverts to deterministic
        threshold logic without invoking the model at all - the ablation is a
        genuine bypass, not a suppressed result.
        """
        deadline = deadline_s if deadline_s is not None else self.config.decision_deadline_s

        # A configured backend occupies the same Decision-Agent slot as Gemma.
        # Substituted reasoners are planned configurations, not failure modes.
        if self.backend is not None:
            result = self.backend.reason(
                event=event,
                fusion=fusion,
                node_readings=node_readings,
                baseline_window=baseline_window,
                history=history,
                deadline_s=deadline,
                threshold_ppm=self.config.threshold_ppm,
            )
            self._last = result.decision
            return result

        if not self.config.enable_llm or self.client is None:
            ablated = not self.config.enable_llm
            reason = (
                "LLM reasoning disabled (ADAM-No-LLM ablation)"
                if ablated
                else "no inference client configured"
            )
            t0 = time.perf_counter()
            decision = deterministic_fallback(
                fusion.fused_ppm, self.config.threshold_ppm, reason=reason
            )
            # degraded_mode marks *runtime failure* of a model that was meant to
            # run (Section 4.5.2), not a configuration in which no model was
            # ever going to run. Flagging the ablation as degraded would inflate
            # the fallback rate reported for the full system.
            decision.degraded_mode = not ablated
            self._last = decision
            return InferenceResult(
                decision=decision,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                fell_back=True,
            )

        user_prompt = build_user_prompt(
            trigger_ppm=event.trigger_ppm,
            trigger_node=event.trigger_node,
            fused_ppm=fusion.fused_ppm,
            node_readings=list(node_readings),
            baseline_window=list(baseline_window),
            history=list(history),
            dispersion_ppm=fusion.dispersion_ppm,
            outlier_nodes=list(fusion.outliers),
        )
        result = self.client.decide(
            user_prompt=user_prompt,
            fused_ppm=fusion.fused_ppm,
            deadline_s=deadline,
            threshold_ppm=self.config.threshold_ppm,
        )
        self._last = result.decision
        return result

    def vote(self, decision: DecisionObject, context: Dict[str, Any]) -> str:
        """Vote the reasoner's initial class; confidence is checked by policy."""
        return decision.classification


# ---------------------------------------------------------------------------
# Coordinator Agent
# ---------------------------------------------------------------------------


@dataclass
class ValidationOutcome:
    """Result of the Coordinator's combined quorum and policy check."""

    approved: bool
    quorum_required: int
    quorum_achieved: int
    governance_valid: bool
    reason: str = ""


class CoordinatorAgent(Agent):
    """Checks agreement, validates policy, executes, verifies trace completeness.

    Scoped to one crew instance. Section 3.1.2: "The Coordinator Agent is a role
    within a specific crew instance, not a global controller."
    """

    role = "coordinator"

    def __init__(
        self,
        agent_id: str,
        node_id: str,
        config: ADAMConfig = DEFAULT_CONFIG,
        validator: Optional[Any] = None,
    ):
        super().__init__(agent_id, node_id, config)
        self.validator = validator  # GovernanceValidator or None

    def collect_votes(
        self,
        event: CrewEvent,
        voters: Sequence[Agent],
        decision: DecisionObject,
        context: Dict[str, Any],
    ) -> int:
        """Collect independent class votes; a failed ballot is not NORMAL."""
        event.expected_voter_count = len(voters)
        for agent in voters:
            try:
                classification = agent.vote(decision, context)
                event.record_vote(agent.agent_id, classification)
            except Exception as exc:
                logger.exception("%s failed to cast a class vote", agent.agent_id)
                event.record_vote_error(agent.agent_id, type(exc).__name__)
        return len(event.votes)

    def tally(self, event: CrewEvent, voter_count: int) -> tuple[str, int, int]:
        """Apply the strict class majority without consulting governance."""
        event.expected_voter_count = voter_count
        classification, agreement, required = event.tally(voter_count)
        event.final_classification = classification
        return classification, agreement, required

    def validate(
        self,
        event: CrewEvent,
        decision: DecisionObject,
        voter_count: int,
    ) -> ValidationOutcome:
        """Only a quorate, class-aligned decision may reach policy validation."""
        classification, achieved, required = self.tally(event, voter_count)
        if classification == "UNRESOLVED":
            return ValidationOutcome(
                approved=False, quorum_required=required,
                quorum_achieved=achieved, governance_valid=False,
                reason="classification quorum not reached; action withheld",
            )
        if decision.classification != classification:
            return ValidationOutcome(
                approved=False, quorum_required=required,
                quorum_achieved=achieved, governance_valid=False,
                reason="decision class differs from crew majority; action withheld",
            )
        if self.config.enable_blockchain and self.validator is not None:
            gov_valid, gov_reason = self.validator.validate(decision, event)
        else:
            gov_valid, gov_reason = True, "governance disabled (ADAM-No-Blockchain)"
        return ValidationOutcome(
            approved=bool(gov_valid), quorum_required=required,
            quorum_achieved=achieved, governance_valid=bool(gov_valid),
            reason=gov_reason if gov_valid else f"policy rejected: {gov_reason}",
        )

    def vote(self, decision: DecisionObject, context: Dict[str, Any]) -> bool:
        """The Coordinator tallies rather than votes.

        Section 3.2 names three voting roles - Sensor, Aggregator, Decision -
        and assigns the Coordinator the counting role. Including its own ballot
        would let the tallying agent tip its own quorum.
        """
        raise NotImplementedError(
            "The Coordinator counts votes and does not cast one (Section 3.2). "
            "Pass only Sensor, Aggregator, and Decision agents to collect_votes()."
        )


__all__ = [
    "Agent",
    "SensorAgent",
    "AggregatorAgent",
    "DecisionAgent",
    "DecisionBackend",
    "CoordinatorAgent",
    "ValidationOutcome",
]
