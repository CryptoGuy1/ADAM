"""Reconstructed manuscript protocol: independent class votes and fail-closed quorum.

These unit tests exercise new reference logic; they do not validate any
historical measured F1 or physical-deployment traces.
"""

from __future__ import annotations

from itertools import product

import pytest

from adam.agents.roles import AggregatorAgent, CoordinatorAgent, DecisionAgent, SensorAgent
from adam.config import ADAMConfig
from adam.coordination import (
    ANOMALY, NORMAL, UNRESOLVED, align_decision_to_crew, majority_class,
    normalize_class_vote,
)
from adam.crew import ADAMNode
from adam.governance.chain import InMemoryChainClient, LocalValidator
from adam.memory.store import InMemoryStore
from adam.schemas import CrewEvent, DecisionObject, EventTrace, SensorReading


def decision(classification: str, *, confidence: float = 0.91) -> DecisionObject:
    is_anomaly = classification == ANOMALY
    return DecisionObject(
        classification=classification,
        confidence=confidence,
        severity="HIGH" if is_anomaly else "NONE",
        reasoning="model interpretation must stay available in original output",
        recommended_action="raise alert" if is_anomaly else "continue monitoring",
        contributing_factors=["primary input"],
        requires_human_review=False,
    )


@pytest.mark.parametrize("combination", list(product([NORMAL, ANOMALY], repeat=3)))
def test_every_nominal_three_voter_combination_has_a_class(combination):
    values = {name: int(cls == ANOMALY) for name, cls in zip("SAD", combination)}
    result, support, required = majority_class(values, 3)
    assert required == 2
    assert support >= required
    assert result == (ANOMALY if combination.count(ANOMALY) >= 2 else NORMAL)


@pytest.mark.parametrize(
    "votes, expected",
    [
        ((NORMAL, NORMAL), NORMAL),
        ((ANOMALY, ANOMALY), ANOMALY),
        ((NORMAL, ANOMALY), UNRESOLVED),
        ((ANOMALY, NORMAL), UNRESOLVED),
    ],
)
def test_two_voter_configuration_requires_matching_classes(votes, expected):
    cls, count, threshold = majority_class(dict(enumerate(int(v == ANOMALY) for v in votes)), 2)
    assert cls == expected
    assert threshold == 2
    assert count == (1 if expected == UNRESOLVED else 2)


def test_missing_ballot_does_not_shrink_formed_majority():
    assert majority_class({"sensor": 1}, 3)[0] == UNRESOLVED
    assert majority_class({"sensor": 1, "decision": 1}, 3)[0] == ANOMALY
    assert majority_class({"sensor": 0, "decision": 0}, 3)[0] == NORMAL
    assert majority_class({"sensor": 0}, 2)[0] == UNRESOLVED


@pytest.mark.parametrize("vote", [True, False, None, "yes", "approve", -1, 2, 0.0, 1.0])
def test_ambiguous_or_invalid_approval_values_are_rejected(vote):
    with pytest.raises(ValueError, match="classification vote"):
        normalize_class_vote(vote)


def test_event_rejects_duplicate_and_preserves_vote_errors():
    event = CrewEvent("evt-test", "N1", 1500, 1.0, expected_voter_count=3)
    event.record_vote("S", NORMAL)
    event.record_vote("A", ANOMALY)
    event.record_vote_error("D", "InferenceUnavailable")
    assert event.tally() == (UNRESOLVED, 1, 2)
    assert event.vote_errors == {"D": "InferenceUnavailable"}
    with pytest.raises(ValueError, match="already voted"):
        event.record_vote("S", ANOMALY)
    with pytest.raises(ValueError, match="already voted"):
        event.record_vote("D", NORMAL)


def test_reconcile_flips_class_action_and_severity_both_directions():
    original = decision(NORMAL, confidence=0.93)
    result, source = align_decision_to_crew(original, ANOMALY, 1450, 2, 3)
    assert original.classification == NORMAL and original.confidence == 0.93
    assert result.classification == ANOMALY
    assert result.severity == "LOW" and result.recommended_action == "raise alert"
    assert result.confidence == original.confidence
    assert source == "decision_agent_initial_class"
    opposite, src = align_decision_to_crew(decision(ANOMALY), NORMAL, 1450, 2, 3)
    assert opposite.classification == NORMAL
    assert opposite.severity == "NONE"
    assert opposite.recommended_action == "continue monitoring"
    assert opposite.confidence == pytest.approx(0.91)
    assert src == "decision_agent_initial_class"
    unchanged, src = align_decision_to_crew(original, NORMAL, 1450, 2, 3)
    assert unchanged.confidence == original.confidence
    assert src == "decision_agent"


def test_critical_anomaly_flip_requests_human_review():
    final, _ = align_decision_to_crew(decision(NORMAL), ANOMALY, 5500, 2, 3)
    assert final.severity == "CRITICAL"
    assert final.recommended_action == "escalate"
    assert final.requires_human_review


def test_r2_uses_initial_score_and_not_crew_support_after_flip():
    from adam.governance.chain import LocalValidator

    event = CrewEvent("evt-flip", "N1", 1450, 1.0, expected_voter_count=3)
    event.record_vote("sensor", ANOMALY)
    event.record_vote("aggregator", ANOMALY)
    event.record_vote("decision", NORMAL)
    initial = decision(NORMAL, confidence=0.20)
    final, source = align_decision_to_crew(initial, ANOMALY, 1450, 2, 3)
    assert source == "decision_agent_initial_class"
    assert final.confidence == initial.confidence == 0.20
    assert 2 / 3 > LocalValidator().policy.min_confidence
    valid, reason = LocalValidator().validate(final, event)
    assert not valid and "confidence" in reason


def test_missing_vote_is_not_a_normal_ballot():
    event = CrewEvent("evt-missing", "N1", 1450, 1.0, expected_voter_count=3)
    event.record_vote("sensor", NORMAL)
    event.record_vote_error("aggregator", "Unavailable")
    event.record_vote("decision", ANOMALY)
    assert event.tally() == (UNRESOLVED, 1, 2)


def test_three_node_dropout_uses_explicit_reference_outlier_threshold():
    from adam.agents.roles import AggregatorAgent
    from adam.mechanisms import max_detectable_z

    assert AggregatorAgent.OUTLIER_Z == 1.5
    assert AggregatorAgent.DROPOUT_OUTLIER_Z == 1.25
    assert AggregatorAgent.DROPOUT_OUTLIER_Z < max_detectable_z(3) < AggregatorAgent.OUTLIER_Z
    agent = AggregatorAgent("N1-aggregator", "N1")
    readings = [SensorReading(f"N{i}", 1.0, ppm, error_variance=100.0)
                for i, ppm in enumerate((800, 800, 1700), 1)]
    assert agent.aggregate(readings).outliers == ("N3",)


@pytest.mark.parametrize(
    "sensor_vote, aggregator_vote, decision_vote, expected",
    [
        (ANOMALY, ANOMALY, NORMAL, ANOMALY),
        (NORMAL, NORMAL, ANOMALY, NORMAL),
        (NORMAL, ANOMALY, ANOMALY, ANOMALY),
        (NORMAL, NORMAL, NORMAL, NORMAL),
    ],
)
def test_runtime_uses_majority_not_initial_decision(monkeypatch, sensor_vote, aggregator_vote, decision_vote, expected):
    monkeypatch.setattr(SensorAgent, "vote", lambda self, d, c: sensor_vote)
    monkeypatch.setattr(AggregatorAgent, "vote", lambda self, d, c: aggregator_vote)
    monkeypatch.setattr(DecisionAgent, "vote", lambda self, d, c: decision_vote)
    node = ADAMNode(
        "N1", config=ADAMConfig(enable_llm=False),
        memory=InMemoryStore(), chain=InMemoryChainClient(), validator=LocalValidator(),
    )
    reading = SensorReading("N1", 1.0, 1450, error_variance=100.0)
    ev = node.sensor.publish_trigger(reading)
    trace = node.handle_event(ev, [reading], sample_resources=False)
    assert trace.initial_classification == ANOMALY  # fallback model class
    assert trace.initial_decision.classification == ANOMALY
    assert trace.final_classification == expected
    assert trace.decision.classification == expected
    assert trace.model_confidence == trace.initial_decision.confidence
    assert trace.decision.confidence == trace.model_confidence
    assert trace.crew_support == pytest.approx(trace.quorum_achieved / trace.voter_count)
    assert trace.confidence_source == (
        "decision_agent" if expected == trace.initial_classification
        else "decision_agent_initial_class"
    )
    assert trace.quorum_required == 2 and trace.quorum_achieved >= 2
    assert trace.classification_votes == {
        "N1-sensor": int(sensor_vote == ANOMALY),
        "N1-aggregator": int(aggregator_vote == ANOMALY),
        "N1-decision": int(decision_vote == ANOMALY),
    }
    assert trace.governance_valid is True
    assert trace.executed is True
    assert trace.is_complete() is True
    assert EventTrace.from_dict(trace.to_dict()).to_dict() == trace.to_dict()


def test_concurrent_conflict_is_selected_before_votes_and_policy(monkeypatch):
    seen = []
    monkeypatch.setattr(SensorAgent, "vote", lambda self, d, c: ANOMALY)
    monkeypatch.setattr(AggregatorAgent, "vote", lambda self, d, c: ANOMALY)
    def decision_vote(self, d, context):
        seen.append(d.recommended_action)
        return d.classification
    monkeypatch.setattr(DecisionAgent, "vote", decision_vote)
    chain = InMemoryChainClient()
    memory = InMemoryStore()
    node = ADAMNode(
        "N1", config=ADAMConfig(enable_llm=False),
        memory=memory, chain=chain, validator=LocalValidator(),
    )
    reading = SensorReading("N1", 10.0, 1450, error_variance=100.0)
    event = node.sensor.publish_trigger(reading)
    pending_event = CrewEvent("evt-other", "N1", 1450, 9.0, location=event.location)
    pending = decision(ANOMALY)
    pending.recommended_action = "dispatch inspection"
    node.register_pending(pending_event, pending)

    trace = node.handle_event(event, [reading], sample_resources=False)
    assert seen == ["dispatch inspection"]
    assert trace.conflict_resolved and trace.conflict_source_event_id == pending_event.event_id
    assert trace.local_decision.recommended_action != trace.initial_decision.recommended_action
    assert trace.initial_decision.recommended_action == pending.recommended_action
    assert trace.model_confidence == pending.confidence
    assert trace.final_action == pending.recommended_action and trace.executed
    assert chain.records[0]["decision"]["recommended_action"] == pending.recommended_action
    stored = memory.retrieve(trace.fused_ppm)[0]
    assert stored["conflict_source_event_id"] == pending_event.event_id
    assert stored["local_action"] == trace.local_decision.recommended_action
    assert EventTrace.from_dict(trace.to_dict()).to_dict() == trace.to_dict()


def test_unrelated_pending_recommendation_cannot_change_event_decision():
    node = ADAMNode("N1", config=ADAMConfig(enable_llm=False))
    current = CrewEvent("evt-current", "N1", 1450, 10.0, location="zone-a")
    unrelated = CrewEvent("evt-other", "N2", 1450, 9.0, location="zone-b")
    proposal = decision(ANOMALY)
    higher = decision(ANOMALY)
    higher.recommended_action = "dispatch inspection"
    node.register_pending(unrelated, higher)
    selected, source = node._arbitrate_pending(current, proposal)
    assert selected is proposal and source is None


def test_conflict_winner_still_cannot_release_rejected_action(monkeypatch):
    monkeypatch.setattr(SensorAgent, "vote", lambda self, d, c: ANOMALY)
    monkeypatch.setattr(AggregatorAgent, "vote", lambda self, d, c: ANOMALY)
    monkeypatch.setattr(DecisionAgent, "vote", lambda self, d, c: ANOMALY)
    node = ADAMNode(
        "N1", config=ADAMConfig(enable_llm=False),
        memory=InMemoryStore(), chain=InMemoryChainClient(), validator=LocalValidator(),
    )
    reading = SensorReading("N1", 10.0, 1450, error_variance=100.0)
    event = node.sensor.publish_trigger(reading)
    pending_event = CrewEvent("evt-rejected", "N1", 1450, 9.0, location=event.location)
    pending = decision(ANOMALY, confidence=0.2)
    pending.recommended_action = "dispatch inspection"
    node.register_pending(pending_event, pending)

    trace = node.handle_event(event, [reading], sample_resources=False)
    assert trace.conflict_resolved
    assert trace.model_confidence == pending.confidence
    assert trace.governance_valid is False
    assert trace.final_action.startswith("WITHHELD")
    assert not trace.executed


def test_split_degraded_runtime_never_calls_validator_or_chain(monkeypatch):
    monkeypatch.setattr(SensorAgent, "vote", lambda self, d, c: ANOMALY)
    monkeypatch.setattr(DecisionAgent, "vote", lambda self, d, c: NORMAL)

    class NoValidator:
        def validate(self, *_):
            raise AssertionError("policy validation cannot run without quorum")

    class NoChain:
        def log_decision(self, *_):
            raise AssertionError("no classification quorum cannot log an approved action")

    node = ADAMNode(
        "N1", config=ADAMConfig(enable_llm=False, enable_aggregator=False),
        validator=NoValidator(), chain=NoChain(), memory=InMemoryStore(),
    )
    reading = SensorReading("N1", 1.0, 1450, error_variance=100.0)
    ev = node.sensor.publish_trigger(reading)
    trace = node.handle_event(ev, [reading], sample_resources=False)
    assert trace.final_classification == UNRESOLVED
    assert trace.failure_stage == "classification_quorum"
    assert trace.governance_valid is None
    assert trace.executed is False
    assert trace.final_action is None
    assert trace.quorum_required == 2 and trace.quorum_achieved == 1
    assert trace.model_confidence == trace.initial_decision.confidence
    assert trace.crew_support is None
    assert trace.persisted_chain is False
    assert ev.final_classification == UNRESOLVED


def test_normal_majority_onchain_passes_winning_vote_count_not_anomaly_sum():
    from adam.governance.chain import FidesInnovaClient

    class Call:
        def call(self):
            return True, "policy satisfied"

    class Functions:
        args = None
        def validateDecision(self, *args):
            self.args = args
            return Call()

    class Contract:
        functions = Functions()

    client = FidesInnovaClient()
    client._w3 = object()
    contract = Contract()
    client._load_contract = lambda _: contract
    event = CrewEvent("evt-abcdef12", "N1", 1450, 1.0, expected_voter_count=3)
    event.record_vote("sensor", NORMAL)
    event.record_vote("aggregator", NORMAL)
    event.record_vote("decision", ANOMALY)
    valid, _ = client.validate(decision(NORMAL), event)
    assert valid
    assert contract.functions.args[7] == 3
    assert contract.functions.args[8] == 2  # the NORMAL majority
    rejected, _ = client.validate(decision(ANOMALY), event)
    assert not rejected  # cannot validate a class contradicted by crew majority


def test_onchain_arguments_keep_initial_score_separate_from_support():
    from adam.governance.chain import FidesInnovaClient

    class Call:
        def call(self):
            return False, "confidence below policy floor"

    class Functions:
        args = None
        def validateDecision(self, *args):
            self.args = args
            return Call()

    class Contract:
        functions = Functions()

    event = CrewEvent("evt-abcdef12", "N1", 1450, 1.0, expected_voter_count=3)
    event.record_vote("sensor", ANOMALY)
    event.record_vote("aggregator", ANOMALY)
    event.record_vote("decision", NORMAL)
    final, _ = align_decision_to_crew(decision(NORMAL, confidence=0.20), ANOMALY, 1450, 2, 3)
    client = FidesInnovaClient()
    client._w3 = object()
    contract = Contract()
    client._load_contract = lambda _: contract
    client.validate(final, event)
    assert contract.functions.args[1] == 20  # initial Decision-Agent score, scaled
    assert contract.functions.args[8] == 2   # final-class supporting votes


def test_role_votes_use_their_own_evidence_and_not_proposal():
    cfg = ADAMConfig()
    sensor = SensorAgent("N1-sensor", "N1", cfg)
    aggregator = AggregatorAgent("N1-aggregator", "N1", cfg)
    reasoner = DecisionAgent("N1-decision", "N1", cfg)
    ctx = {"trigger_ppm": 1200.0, "baseline_window": [750.0] * 6, "fused_ppm": 820.0}
    proposed_normal = decision(NORMAL)
    proposed_anomaly = decision(ANOMALY)
    assert sensor.vote(proposed_normal, ctx) == sensor.vote(proposed_anomaly, ctx) == ANOMALY
    assert aggregator.vote(proposed_normal, ctx) == aggregator.vote(proposed_anomaly, ctx) == NORMAL
    assert reasoner.vote(proposed_normal, ctx) == NORMAL
    assert reasoner.vote(proposed_anomaly, ctx) == ANOMALY


def test_coordinator_does_not_turn_failing_ballot_into_normal(monkeypatch):
    monkeypatch.setattr(SensorAgent, "vote", lambda self, d, c: ANOMALY)
    monkeypatch.setattr(AggregatorAgent, "vote", lambda self, d, c: (_ for _ in ()).throw(RuntimeError("lost")))
    monkeypatch.setattr(DecisionAgent, "vote", lambda self, d, c: NORMAL)
    node = ADAMNode(
        "N1", config=ADAMConfig(enable_llm=False),
        memory=InMemoryStore(), chain=InMemoryChainClient(), validator=LocalValidator(),
    )
    reading = SensorReading("N1", 1.0, 1400, error_variance=100.0)
    trace = node.handle_event(node.sensor.publish_trigger(reading), [reading], sample_resources=False)
    assert trace.final_classification == UNRESOLVED
    assert trace.classification_votes == {"N1-sensor": 1, "N1-decision": 0}
    assert trace.vote_errors == {"N1-aggregator": "RuntimeError"}
    assert trace.governance_valid is None
    assert not trace.executed
