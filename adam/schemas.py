"""
adam.schemas
============

Typed data contracts shared by the runtime, the comparators, and the analysis
scripts. These are deliberately plain dataclasses rather than pydantic models:
the audit trace has to serialize identically on a Raspberry Pi and in an
offline analysis run, and a reviewer should be able to read a trace file
without installing the package.

The central object is :class:`DecisionObject` - the seven-field structure the
Decision Agent must emit. :class:`EventTrace` records the audit tuple and, for
new/reference runs, explicit acknowledgments from the enabled audit stores.
An end-to-end completion count is not an independent dual-store persistence
measurement.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import (
    CLASSIFICATION_VALUES,
    DECISION_SCHEMA_FIELDS,
    SEVERITY_LEVELS,
    SEVERITY_SCORES,
)


# ---------------------------------------------------------------------------
# Sensing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorReading:
    """One MQ-4 sample from one node.

    ``reference_ppm`` carries the NDIR reference ground truth where available
    (labeled trials only). It is never visible to any agent - it exists solely
    for scoring, and the loader in ``data/loader.py`` enforces that separation.
    """

    node_id: str
    timestamp: float
    methane_ppm: float
    temperature_c: Optional[float] = None
    humidity_pct: Optional[float] = None
    reference_ppm: Optional[float] = None
    error_variance: Optional[float] = None

    @property
    def weight(self) -> float:
        """Inverse-variance fusion weight w_i = 1 / sigma_i^2. Equation (2).

        sigma_i^2 is the node's sensor error variance, estimated from the
        residuals of its raw readings against the co-located NDIR reference
        over the labeled trials. On the deployed testbed the four variances
        are closely matched, so the weights are near uniform; the mechanism
        matters under heterogeneous or degraded hardware.
        """
        if self.error_variance is None or self.error_variance <= 0:
            raise ValueError(
                f"node {self.node_id} has no positive error variance; "
                f"Equation (2) weights are undefined"
            )
        return 1.0 / self.error_variance

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def redacted(self) -> Dict[str, Any]:
        """Agent-visible view. Strips the reference label.

        Any path that feeds a reading to an agent goes through this. It is the
        mechanism that prevents ground-truth leakage into the decision loop.
        """
        d = asdict(self)
        d.pop("reference_ppm", None)
        return d


# ---------------------------------------------------------------------------
# Decision object  (Section 3.4.2)
# ---------------------------------------------------------------------------


class SchemaViolation(ValueError):
    """Raised when model output does not satisfy the seven-field schema."""


@dataclass
class DecisionObject:
    """The structured recommendation d_t produced by the Decision Agent.

    The manuscript defines d_t as the Decision-Agent output over the fused
    estimate, recent temporal context, and retrieved semantic memory. Section
    3.4.2 fixes the seven fields below.
    """

    classification: str
    confidence: float
    severity: str
    reasoning: str
    recommended_action: str
    contributing_factors: List[str]
    requires_human_review: bool

    #: True when this object came from deterministic fallback rather than the
    #: model. Written into the audit record - Section 3.4.2 and 4.5.2.
    degraded_mode: bool = False

    #: Set when a format-repair retry was needed. Not part of the emitted
    #: schema; retained for reliability accounting.
    repair_attempted: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.classification not in CLASSIFICATION_VALUES:
            raise SchemaViolation(
                f"classification must be one of {CLASSIFICATION_VALUES}; got "
                f"{self.classification!r}"
            )
        if not isinstance(self.confidence, (int, float)) or math.isnan(self.confidence):
            raise SchemaViolation(f"confidence must be numeric; got {self.confidence!r}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise SchemaViolation(
                f"confidence must lie in [0, 1]; got {self.confidence}"
            )
        if self.severity not in SEVERITY_LEVELS:
            raise SchemaViolation(
                f"severity must be one of {SEVERITY_LEVELS}; got {self.severity!r}"
            )
        if not isinstance(self.reasoning, str):
            raise SchemaViolation("reasoning must be a string")
        if not isinstance(self.recommended_action, str) or not self.recommended_action:
            raise SchemaViolation("recommended_action must be a non-empty string")
        if not isinstance(self.contributing_factors, list) or not all(
            isinstance(x, str) for x in self.contributing_factors
        ):
            raise SchemaViolation("contributing_factors must be a list of strings")
        if not isinstance(self.requires_human_review, bool):
            raise SchemaViolation("requires_human_review must be a boolean")

    @property
    def is_anomaly(self) -> bool:
        return self.classification == "ANOMALY"

    @property
    def severity_score(self) -> float:
        """Numeric severity used by conflict resolution."""
        return SEVERITY_SCORES[self.severity]

    @classmethod
    def from_model_json(cls, payload: Dict[str, Any]) -> "DecisionObject":
        """Build from parsed model output, rejecting anything off-schema.

        Coercion is deliberately narrow: string booleans and out-of-range
        confidences are common small-model failure modes and are repaired here,
        but a missing or unrecognized field raises so that the caller can issue
        the single format-repair retry Section 3.4.2 allows.
        """
        missing = [f for f in DECISION_SCHEMA_FIELDS if f not in payload]
        if missing:
            raise SchemaViolation(f"model output missing fields: {missing}")

        classification = str(payload["classification"]).strip().upper()
        severity = str(payload["severity"]).strip().upper()

        try:
            confidence = float(payload["confidence"])
        except (TypeError, ValueError) as exc:
            raise SchemaViolation(
                f"confidence not coercible to float: {payload['confidence']!r}"
            ) from exc
        confidence = min(1.0, max(0.0, confidence))

        raw_review = payload["requires_human_review"]
        if isinstance(raw_review, str):
            requires_review = raw_review.strip().lower() in ("true", "yes", "1")
        else:
            requires_review = bool(raw_review)

        factors = payload["contributing_factors"]
        if isinstance(factors, str):
            factors = [factors]
        factors = [str(x) for x in (factors or [])]

        return cls(
            classification=classification,
            confidence=confidence,
            severity=severity,
            reasoning=str(payload["reasoning"]),
            recommended_action=str(payload["recommended_action"]),
            contributing_factors=factors,
            requires_human_review=requires_review,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_schema_json(self) -> str:
        """Serialize only the seven schema fields, for prompt exemplars."""
        return json.dumps(
            {f: getattr(self, f) for f in DECISION_SCHEMA_FIELDS}, indent=2
        )


# ---------------------------------------------------------------------------
# Crew coordination
# ---------------------------------------------------------------------------


@dataclass
class CrewEvent:
    """Ephemeral coordination state for an ACTIVE event.

    Backed by the Weaviate ``CrewEvent`` class and cleared on crew dissolution
    (Section 3.1.2). Holds the trigger, crew membership, and votes.
    """

    event_id: str
    trigger_node: str
    trigger_ppm: float
    timestamp: float
    location: str = ""
    severity_hint: str = "UNKNOWN"
    crew_members: List[str] = field(default_factory=list)
    votes: Dict[str, int] = field(default_factory=dict)
    vote_errors: Dict[str, str] = field(default_factory=dict)
    expected_voter_count: int = 0
    final_classification: Optional[str] = None

    @staticmethod
    def new_id() -> str:
        return f"evt-{uuid.uuid4().hex[:12]}"

    def record_vote(self, agent_id: str, classification: str | int) -> None:
        """Record one NORMAL/ANOMALY class vote; reject duplicate ballots.

        Boolean approval/refusal is deliberately rejected; a refusal to approve
        a proposal is not equivalent to voting NORMAL.
        """
        from .coordination import normalize_class_vote

        if agent_id in self.votes or agent_id in self.vote_errors:
            raise ValueError(f"agent {agent_id} has already voted on event {self.event_id}")
        self.votes[agent_id] = normalize_class_vote(classification)

    def record_vote_error(self, agent_id: str, reason: str) -> None:
        """Retain a missing ballot as an error, never substitute NORMAL."""
        if agent_id in self.votes or agent_id in self.vote_errors:
            raise ValueError(f"agent {agent_id} has already voted on event {self.event_id}")
        self.vote_errors[agent_id] = reason

    def tally(self, expected_voters: Optional[int] = None) -> tuple[str, int, int]:
        """Return (final class, winning-vote count, required quorum)."""
        from .coordination import majority_class

        n = expected_voters or self.expected_voter_count or len(self.votes) + len(self.vote_errors)
        return majority_class(self.votes, n)

    @property
    def approvals(self) -> int:
        """Legacy contract ABI field: votes agreeing with the final class.

        NOT the count of ANOMALY votes; NORMAL majorities also require quorum.
        Only useful after a successful classification tally.
        """
        cls, support, _ = self.tally()
        return support if cls != "UNRESOLVED" else 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StageLatencies:
    """The six deployment-stage latencies, in milliseconds."""

    T_form: float = 0.0
    T_agg: float = 0.0
    T_reason: float = 0.0
    T_gov: float = 0.0
    T_weav: float = 0.0
    T_bc: float = 0.0

    @property
    def total_ms(self) -> float:
        """End-to-end decision latency across the six recorded stages."""
        return self.T_form + self.T_agg + self.T_reason + self.T_gov + self.T_weav + self.T_bc

    @property
    def total_s(self) -> float:
        return self.total_ms / 1000.0

    def share(self) -> Dict[str, float]:
        """Fractional contribution of each stage, as in Figure 5."""
        total = self.total_ms
        if total <= 0:
            return {k: 0.0 for k in asdict(self)}
        return {k: v / total for k, v in asdict(self).items()}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceCounters:
    """Per-node resource and network counters sampled during an event."""

    cpu_peak_pct: float = 0.0
    cpu_sustained_pct: float = 0.0
    memory_mb: float = 0.0
    bandwidth_kbps: float = 0.0
    external_bytes: int = 0
    api_cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventTrace:
    """The persistent audit record for a resolved event.

    The audit trace records the tuple <trigger, fused context, inference output,
    policy-validation result, final action>. :meth:`is_complete` is an explicit
    commit predicate for current traces. It must not be used to reinterpret an
    end-to-end completion count as an independently measured persistence rate.
    """

    event_id: str
    timestamp: float

    # <trigger>
    trigger_node: str = ""
    trigger_ppm: float = 0.0

    # <fused context>
    fused_ppm: Optional[float] = None
    contributing_nodes: List[str] = field(default_factory=list)

    # <inference output>. The original model output is preserved separately.
    decision: Optional[DecisionObject] = None
    initial_decision: Optional[DecisionObject] = None
    # Current event's reasoner output when a concurrent recommendation wins.
    local_decision: Optional[DecisionObject] = None
    initial_classification: Optional[str] = None
    final_classification: Optional[str] = None
    classification_votes: Dict[str, int] = field(default_factory=dict)
    vote_errors: Dict[str, str] = field(default_factory=dict)
    crew_agreement_fraction: Optional[float] = None
    # Model score refers to the initial Decision-Agent class, even on a flip.
    model_confidence: Optional[float] = None
    # Fraction of eligible voters supporting the final class; not a probability.
    crew_support: Optional[float] = None
    confidence_source: str = "decision_agent"

    # <policy validation result>
    governance_valid: Optional[bool] = None
    quorum_required: Optional[int] = None
    quorum_achieved: Optional[int] = None
    governance_reason: str = ""

    # <final action>
    final_action: Optional[str] = None
    executed: bool = False
    conflict_resolved: bool = False
    conflict_source_event_id: Optional[str] = None

    # Operational metadata
    latencies: StageLatencies = field(default_factory=StageLatencies)
    resources: ResourceCounters = field(default_factory=ResourceCounters)
    degraded_mode: bool = False
    crew_size: int = 0
    voter_count: int = 0
    blockchain_tx: Optional[str] = None
    persisted_weaviate: bool = False
    persisted_chain: bool = False
    #: Stage at which the pipeline withheld execution, when it did. Mirrors the
    #: failure taxonomy reported for the deployment events (Section 4.2).
    failure_stage: Optional[str] = None
    #: Audit stores this configuration was required to commit to. Ablations
    #: disable a store deliberately (Section 3.4.5), so a disabled backend is
    #: not required and cannot make a trace incomplete.
    required_stores: List[str] = field(default_factory=list)

    def is_complete(self) -> bool:
        """True when all five tuple elements were recorded and committed.

        This is the trace-persistence predicate of Section 3.4.3: the tuple
        must be recorded across every audit store this configuration requires,
        not merely one of them. Ablations that disable a store list only the
        stores that remain, so a deliberately disabled backend cannot make a
        trace incomplete.
        """
        tuple_recorded = (
            bool(self.trigger_node)
            and self.fused_ppm is not None
            and self.decision is not None
            and self.governance_valid is not None
            and self.final_action is not None
        )
        if not tuple_recorded:
            return False
        committed = {
            "blockchain": self.persisted_chain,
            "weaviate": self.persisted_weaviate,
        }
        return all(committed.get(s, False) for s in self.required_stores)

    def within_deadline(self, deadline_s: float) -> bool:
        """True when T_decision met constraint C1."""
        return self.latencies.total_s <= deadline_s

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["complete"] = self.is_complete()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EventTrace":
        d = dict(d)
        d.pop("complete", None)
        if isinstance(d.get("latencies"), dict):
            d["latencies"] = StageLatencies(**d["latencies"])
        if isinstance(d.get("resources"), dict):
            d["resources"] = ResourceCounters(**d["resources"])
        for field_name in ("decision", "initial_decision", "local_decision"):
            dec = d.get(field_name)
            if isinstance(dec, dict):
                dec = dict(dec)
                degraded = dec.pop("degraded_mode", False)
                repaired = dec.pop("repair_attempted", False)
                obj = DecisionObject(**{k: dec[k] for k in DECISION_SCHEMA_FIELDS})
                obj.degraded_mode = degraded
                obj.repair_attempted = repaired
                d[field_name] = obj
        return cls(**d)


# ---------------------------------------------------------------------------
# Labeled evaluation records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabeledEvent:
    """One scored event from a D1 trial.

    ``label`` is the reference-sensor ground truth. It MUST be derived from the
    NDIR reference, never from the MQ-4 reading crossing a threshold
    - see ``data/validate.py``, which refuses to load a dataset whose labels are
    a deterministic function of the screening rule.
    """

    trial_id: int
    event_index: int
    timestamp: float
    readings: Tuple[SensorReading, ...]
    label: int  # 1 = anomaly, 0 = normal
    reference_ppm: float

    @property
    def primary(self) -> SensorReading:
        """The triggering node's reading."""
        return self.readings[0]

    def agent_view(self) -> List[Dict[str, Any]]:
        """Label-free view handed to any system under evaluation."""
        return [r.redacted() for r in self.readings]


@dataclass
class Prediction:
    """One system's output on one labeled event."""

    system: str
    trial_id: int
    event_index: int
    predicted: int
    confidence: float = 0.0
    latency_ms: float = 0.0
    degraded_mode: bool = False
    api_cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "SensorReading",
    "DecisionObject",
    "SchemaViolation",
    "CrewEvent",
    "StageLatencies",
    "ResourceCounters",
    "EventTrace",
    "LabeledEvent",
    "Prediction",
]
