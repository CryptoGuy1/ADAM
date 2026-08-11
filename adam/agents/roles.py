"""
adam.agents.roles
=================

The four role-specialized agents of Section 3.1.2.

    Sensor       monitors local streams, screens, publishes the trigger
    Aggregator   retrieves cross-node evidence, fuses it (Equation 2)
    Decision     prompts the local model, returns d_t (Equation 3)
    Coordinator  checks quorum (Equation 4), validates policy, executes, verifies trace

Two properties matter for the manuscript's claims and are enforced here rather
than left to convention:

1. **The Coordinator is a crew-instance role, not a global controller.**
   Section 3.1.2 is explicit on this point. Each crew constructs its own, and
   it holds no state across events.

2. **The Coordinator does not override votes.** Section 3.2 states it "counts
   approvals, checks the governance validator, and executes the action only
   when the quorum condition holds." :meth:`CoordinatorAgent.tally` therefore
   only reads the vote map.

Voting
------
Each agent votes on the proposed action using its own role-specific evidence
(Section 3.2): the Sensor checks consistency with the local trigger, the
Aggregator with cross-node context, the Decision agent with its own semantic
interpretation. Independent evidence is what makes the vote meaningful; if all
three simply echoed the Decision Agent, Equation (4) would be theatre.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

    def vote(self, decision: DecisionObject, context: Dict[str, Any]) -> bool:
        """Binary approval vote v_i(a_t). Overridden per role."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Sensor Agent
# ---------------------------------------------------------------------------


class SensorAgent(Agent):
    """Monitors the local stream, screens, and publishes crew-formation triggers.

    Holds a rolling baseline window, which serves two purposes: it is the
    ``{m_{t-k:t}}`` temporal context of Equation (3), and it is this agent's
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

    def vote(self, decision: DecisionObject, context: Dict[str, Any]) -> bool:
        """Approve when the decision is consistent with the local trigger.

        Evidence: did this node's own reading cross threshold, and does the
        classification agree with that? A NORMAL call on a reading far above
        baseline is refused; so is an ANOMALY call on a reading that never
        triggered locally and shows no baseline departure.
        """
        local_ppm = float(context.get("trigger_ppm", 0.0))
        triggered = trigger(local_ppm, self.config.threshold_ppm) == 1
        base = self.baseline_mean
        departure = (local_ppm / base) if base and base > 0 else float("inf")

        if decision.is_anomaly:
            return triggered or departure >= 2.0
        # NORMAL: refuse if the local evidence is strongly against it.
        if triggered and departure >= 3.0:
            logger.info(
                "%s refuses NORMAL: local %.0f ppm is %.1fx baseline",
                self.agent_id,
                local_ppm,
                departure,
            )
            return False
        return True


# ---------------------------------------------------------------------------
# Aggregator Agent
# ---------------------------------------------------------------------------


class AggregatorAgent(Agent):
    """Retrieves cross-node evidence and composes the fused snapshot.

    This is the agent the pre-revision codebase lacked: it was named
    "validator" and performed schema checks rather than Equation (2) fusion.
    Section 4.1 attributes 2.7 F1 points to it (0.896 vs 0.869 without).
    """

    role = "aggregator"

    #: Weighted z-score past which a node is flagged as disagreeing. This is the
    #: cross-node corroboration behind the 90.0% attack detection rate
    #: (27 of 30 injections) in Section 4.5.1.
    OUTLIER_Z: float = 1.5

    def __init__(self, agent_id: str, node_id: str, config: ADAMConfig = DEFAULT_CONFIG):
        super().__init__(agent_id, node_id, config)
        self._last: Optional[FusionResult] = None

    def aggregate(self, readings: Sequence[SensorReading]) -> FusionResult:
        """Fuse active-node readings. Equation (2), Algorithm 1 line 10.

        When the aggregator is disabled (ADAM-No-Aggregator), the crew calls
        :meth:`local_only` instead.
        """
        result = fuse_readings(readings, outlier_z=self.OUTLIER_Z)
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

    def vote(self, decision: DecisionObject, context: Dict[str, Any]) -> bool:
        """Approve when the decision is consistent with cross-node context.

        Evidence: does the fused estimate support the classification, and did
        the triggering node stand alone? A single node reading high while its
        peers read background is the signature of the injection attacks in
        Section 4.5.1, and this agent refuses to endorse an ANOMALY on that
        basis alone.
        """
        fused = float(context.get("fused_ppm", 0.0))
        trigger_node = context.get("trigger_node", "")
        fused_triggers = trigger(fused, self.config.threshold_ppm) == 1

        if self._last is not None and trigger_node in self._last.outliers:
            if decision.is_anomaly and not fused_triggers:
                logger.info(
                    "%s refuses ANOMALY: trigger node %s disagrees with peers",
                    self.agent_id,
                    trigger_node,
                )
                return False

        if decision.is_anomaly:
            return fused_triggers or fused >= 0.75 * self.config.threshold_ppm
        return not fused_triggers or fused < 1.5 * self.config.threshold_ppm


# ---------------------------------------------------------------------------
# Decision Agent
# ---------------------------------------------------------------------------


class DecisionAgent(Agent):
    """Builds the prompt, invokes the local model, returns d_t.

    Section 4.2 attributes 81.4% of the decision budget to this agent's
    inference call, and Section 5.3 identifies it as the sole optimization
    lever. It is the only agent that touches the model.
    """

    role = "decision"

    def __init__(
        self,
        agent_id: str,
        node_id: str,
        config: ADAMConfig = DEFAULT_CONFIG,
        client: Optional[OllamaClient] = None,
    ):
        super().__init__(agent_id, node_id, config)
        self.client = client
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
        """Generate d_t. Equation (3), Algorithm 1 line 12.

        With ``enable_llm=False`` (ADAM-No-LLM) this reverts to deterministic
        threshold logic without invoking the model at all - the ablation is a
        genuine bypass, not a suppressed result.
        """
        deadline = deadline_s if deadline_s is not None else self.config.decision_deadline_s

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

    def vote(self, decision: DecisionObject, context: Dict[str, Any]) -> bool:
        """Approve when the action is consistent with its own interpretation.

        Evidence: the semantic interpretation this agent produced. It abstains
        from endorsing low-confidence calls, and refuses when the recommended
        action contradicts the classification it assigned.
        """
        if decision.confidence < 0.35:
            logger.info(
                "%s refuses: confidence %.2f below floor", self.agent_id, decision.confidence
            )
            return False
        if decision.degraded_mode and decision.is_anomaly:
            # A fallback anomaly is a threshold comparison, not an
            # interpretation. Still approvable - availability matters - but the
            # trace records that no reasoning stood behind it.
            return True
        action = decision.recommended_action.lower()
        contradicts = decision.is_anomaly and any(
            phrase in action for phrase in ("no action", "ignore", "continue monitoring")
        )
        return not contradicts


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
        """Poll each voting agent and record its ballot. Algorithm 1 line 14.

        Recorded through :meth:`CrewEvent.record_vote`, which rejects duplicate
        ballots from one agent - the attributability assumption the Table 8
        bounds rest on.
        """
        for agent in voters:
            try:
                approve = agent.vote(decision, context)
            except Exception:
                logger.exception("%s vote raised; recording as refusal", agent.agent_id)
                approve = False
            event.record_vote(agent.agent_id, approve)
        return event.approvals

    def validate(
        self,
        event: CrewEvent,
        decision: DecisionObject,
        voter_count: int,
    ) -> ValidationOutcome:
        """Combined quorum and governance check. Algorithm 1 line 15.

        Both conditions must hold: ``V(d_t, S_t) = 1 AND q_t >= gamma_crew``.

        ``voter_count`` is the number of agents that cast a ballot, which
        excludes this Coordinator. Passing the full crew size would compute a
        quorum the crew can exceed, overstating its fault tolerance.
        """
        from ..config import quorum as quorum_threshold

        required = quorum_threshold(voter_count)
        achieved = event.approvals

        if self.config.enable_blockchain and self.validator is not None:
            gov_valid, gov_reason = self.validator.validate(decision, event)
        else:
            # ADAM-No-Blockchain: policy validation and ledger logging removed.
            # Section 4.1 reports detection essentially unchanged (0.889 vs
            # 0.896), which is the intended result - the layer supplies
            # accountability, not accuracy.
            gov_valid, gov_reason = True, "governance disabled (ADAM-No-Blockchain)"

        approved = gov_valid and achieved >= required
        if not approved:
            reason = (
                f"quorum {achieved}/{required} not met"
                if achieved < required
                else f"policy rejected: {gov_reason}"
            )
        else:
            reason = gov_reason or "quorum and policy satisfied"

        return ValidationOutcome(
            approved=approved,
            quorum_required=required,
            quorum_achieved=achieved,
            governance_valid=gov_valid,
            reason=reason,
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
    "CoordinatorAgent",
    "ValidationOutcome",
]
