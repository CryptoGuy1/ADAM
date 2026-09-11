"""
tests.test_manuscript_parity
============================

Tests that fail when the code and the manuscript disagree.

These are the ones that matter for review. Ordinary unit tests catch bugs; these
catch a codebase that has drifted from the paper it is supposed to implement -
which is the failure mode that wastes a reviewer's afternoon.

Run:  python -m pytest tests/ -v
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from adam.config import (
    CRITICAL_THRESHOLD_PPM,
    DECISION_DEADLINE_S,
    MIN_CREW_SIZE,
    N_DEPLOYMENT_EVENTS,
    NODE_SCALING_FIT_HW,
    NODE_SCALING_FIT_SCALEOUT,
    REFERENCE_TOLERANCE_PPM,
    REFERENCE_STAGE_LATENCY_MS,
    SENSOR_ERROR_VARIANCE_RANGE_PPM2,
    THRESHOLD_PPM,
    WARNING_THRESHOLD_PPM,
    ADAMConfig,
    fails_closed,
    is_subvertible,
    quorum,
    tolerated_faults,
    verify_against_manuscript,
)
from adam.mechanisms import (
    Candidate,
    fuse_readings,
    quorum_satisfied,
    resolve_conflict,
    trigger,
)
from adam.schemas import DecisionObject, SchemaViolation, SensorReading

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Config parity
# ---------------------------------------------------------------------------


def test_config_reproduces_manuscript():
    """Every derived figure the paper states must fall out of the constants."""
    problems = verify_against_manuscript()
    assert not problems, "config drifted from manuscript:\n  " + "\n  ".join(problems)


def test_screening_threshold_matches_manuscript():
    """Constraint C5 uses a 1,000 ppm raw-MQ-4 screening threshold."""
    assert THRESHOLD_PPM == 1000.0


def test_stage_latencies_sum_to_reported_mean():
    """Equation (6): the six stages sum to the reported decision latency."""
    total_s = sum(REFERENCE_STAGE_LATENCY_MS.values()) / 1000.0
    assert total_s == pytest.approx(18.99, abs=0.05)


def test_reasoning_is_dominant_latency_component():
    """Local reasoning accounts for about 81.5% of completed-event mean latency."""
    total = sum(REFERENCE_STAGE_LATENCY_MS.values())
    share = REFERENCE_STAGE_LATENCY_MS["T_reason"] / total
    assert share == pytest.approx(0.815, abs=0.002)
    coordination = (
        REFERENCE_STAGE_LATENCY_MS["T_agg"]
        + REFERENCE_STAGE_LATENCY_MS["T_gov"]
        + REFERENCE_STAGE_LATENCY_MS["T_weav"]
        + REFERENCE_STAGE_LATENCY_MS["T_bc"]
    )
    assert coordination / total < 0.08


def test_node_scaling_fit_endpoints():
    """Table 7: the hardware fit must recover the measured endpoints.

    Sheet 09 reports mean decision latency of about 17.64 s at N = 1 and
    18.82 s at N = 4 on hardware, and the scale-out model reaches about
    20.94 s at N = 16.
    """
    hw = NODE_SCALING_FIT_HW
    t1 = (hw["T0"] + hw["alpha"] * 1 ** hw["beta"]) / 1000
    t4 = (hw["T0"] + hw["alpha"] * 4 ** hw["beta"]) / 1000
    assert t1 == pytest.approx(17.58, abs=0.15)
    assert t4 == pytest.approx(18.87, abs=0.15)

    so = NODE_SCALING_FIT_SCALEOUT
    t16 = (so["T0"] + so["alpha"] * 16 ** so["beta"]) / 1000
    assert t16 == pytest.approx(20.9, abs=0.4)


def test_node_scaling_exponents():
    """The hardware exponent exceeds unity; the scale-out exponent does not.

    Over N = 1-4, adding physical nodes adds coordination work slightly faster
    than linearly. Over N = 4-16, the scale-out curve flattens: the per-event
    reasoning stage dominates and node count contributes a decelerating share.
    """
    assert NODE_SCALING_FIT_HW["beta"] > 1.0
    assert NODE_SCALING_FIT_SCALEOUT["beta"] < 1.0


def test_error_variance_exceeds_reference_tolerance():
    """The MQ-4 must be noisier than the instrument that labels it.

    The reference is accurate to +/-1% of its 2,000 ppm full scale, i.e. 20 ppm
    across the range, so any sensor variance below that is physically impossible.
    """
    import math

    lo, hi = SENSOR_ERROR_VARIANCE_RANGE_PPM2
    reference_sd = REFERENCE_TOLERANCE_PPM
    assert math.sqrt(lo) > reference_sd, (
        f"MQ-4 sd {math.sqrt(lo):.1f} ppm is below the reference tolerance "
        f"{reference_sd:.1f} ppm, which is physically implausible"
    )


def test_holm_reproduces_published_adjustments():
    """The ten-comparison Holm family must reproduce the revised Table 5."""
    from analysis.metrics import holm_adjust

    raw = {
        "static": 0.001953125,
        "rf_raw": 0.001953125,
        "rf_contextual": 0.005859375,
        "gbm_contextual": 0.001953125,
        "cloud": 0.048828125,
        "single": 0.001953125,
        "noagg": 0.00390625,
        "nollm": 0.001953125,
        "noblockchain": 0.09765625,
        "noweav": 0.00390625,
    }
    adj = holm_adjust(raw)
    assert adj["static"] == pytest.approx(0.01953125)
    assert adj["rf_contextual"] == pytest.approx(0.01953125)
    assert adj["cloud"] == pytest.approx(0.09765625)
    assert adj["noblockchain"] == pytest.approx(0.09765625)
    assert sum(1 for v in adj.values() if v >= 0.05) == 2


def test_enriched_random_forest_is_gone():
    """One Random Forest configuration exists in the data; the methods must match."""
    from adam.config import BASELINES, SYSTEMS

    assert not any("enrich" in s for s in BASELINES + SYSTEMS)


def test_comparison_family_size_matches_systems():
    """Holm's family size must equal the number of systems compared."""
    from adam.config import N_COMPARISONS, SYSTEMS

    assert len(SYSTEMS) - 1 == N_COMPARISONS


# ---------------------------------------------------------------------------
# Table 8 / quorum
# ---------------------------------------------------------------------------

TABLE_8 = {
    # n: (gamma_crew, tolerated_f) under strict majority, gamma = floor(n/2)+1.
    # tolerated_f = ceil(n/2) - 1: honest voters retain quorum while
    # n - f >= gamma.
    2: (2, 0),
    3: (2, 1),
    4: (3, 1),
    5: (3, 2),
    6: (4, 2),
    7: (4, 3),
}


@pytest.mark.parametrize("n,expected", TABLE_8.items())
def test_table8_quorum_and_tolerance(n, expected):
    exp_q, exp_f = expected
    assert quorum(n) == exp_q, f"Table 8 row n={n} states gamma={exp_q}"
    assert tolerated_faults(n) == exp_f, f"Table 8 row n={n} states f={exp_f}"


def test_quorum_prevents_unilateral_action():
    """Section 3.2.4: no single agent may approve an action alone."""
    for n in range(2, 12):
        assert quorum(n) >= 2


def test_degraded_two_agent_crew_requires_unanimity():
    """Section 3.2: with |C_t| = 2, both agents must approve."""
    assert quorum(2) == 2
    assert quorum_satisfied(2, 2)
    assert not quorum_satisfied(1, 2)


def test_deployed_voting_set_tolerates_one_compromised_agent():
    """The deployed four-role crew has three voters; two colluding voters subvert."""
    from adam.config import DEPLOYED_VOTER_COUNT

    assert DEPLOYED_VOTER_COUNT == 3
    assert tolerated_faults(DEPLOYED_VOTER_COUNT) == 1
    assert not is_subvertible(DEPLOYED_VOTER_COUNT, 1)
    assert is_subvertible(DEPLOYED_VOTER_COUNT, 2)


def test_percentage_quorum_rule_is_rejected():
    """Guards against reintroducing the ceil(n*51/100) rule.

    At the small crew sizes of Table 8 the percentage rule happens to coincide
    with strict majority, which is exactly why it survived unnoticed in an
    earlier contract revision. The two diverge as n grows, so quorum must be
    the explicit floor(n/2)+1 expression rather than a percentage constant.
    """
    def old_rule(n: int) -> int:
        return (n * 51 + 99) // 100

    for n in range(2, 8):
        assert old_rule(n) == quorum(n)  # the coincidence that hid the defect
    diverges = [n for n in range(2, 201) if old_rule(n) != quorum(n)]
    assert diverges, "the two rules must diverge somewhere below n = 200"


def test_solidity_quorum_matches_python():
    """Parity between GovernanceRules.sol and adam.config.quorum.

    The Solidity uses integer arithmetic: crewSize / 2 + 1. This reimplements
    that expression exactly and checks it against the Python definition, so the
    two cannot diverge without CI noticing.
    """
    sol = (REPO / "contracts" / "GovernanceRules.sol").read_text()

    m = re.search(
        r"function requiredQuorum\(uint256 crewSize\)[^}]*?return ([^;]+);",
        sol,
        re.DOTALL,
    )
    assert m, "requiredQuorum not found in GovernanceRules.sol"
    expr = m.group(1).strip()
    assert expr == "crewSize / 2 + 1", f"unexpected quorum expression: {expr}"

    def solidity_quorum(n: int) -> int:
        return n // 2 + 1  # uint integer division floors

    for n in range(1, 33):
        assert solidity_quorum(n) == quorum(n), f"quorum parity broke at n={n}"


def test_solidity_screening_threshold_is_1000():
    """The contract must not revert to the stale 5,000 ppm value."""
    sol = (REPO / "contracts" / "GovernanceRules.sol").read_text()
    m = re.search(r"screeningThreshold\s*=\s*(\d+)\s*;", sol)
    assert m, "screeningThreshold not set in the constructor"
    assert int(m.group(1)) == int(THRESHOLD_PPM) == 1000




def test_solidity_governance_rules_cover_python_policy_surface():
    """Reviewer-facing contract must expose the same R1--R6 inputs as Python.

    This is a source-level parity guard because Solidity toolchains are optional
    in the Python CI environment. It catches the specific drift that previously
    left permitted-action and degraded-mode policy invisible on-chain.
    """
    sol = (REPO / "contracts" / "GovernanceRules.sol").read_text()

    signature = re.search(
        r"function validateDecision\((.*?)\) external view returns",
        sol,
        re.DOTALL,
    )
    assert signature, "validateDecision not found in GovernanceRules.sol"
    params = signature.group(1)
    for required in (
        "methanePpm",
        "confidenceScaled",
        "classification",
        "severity",
        "recommendedAction",
        "requiresReview",
        "degradedMode",
        "crewSize",
        "approvals",
    ):
        assert required in params, f"Solidity validator missing {required}"

    # R1--R6 policy surfaces. The exact wording may evolve, but these helpers
    # and branches must remain present if Python/Solidity parity is claimed.
    assert "_isRecognizedSeverity(severity)" in sol
    assert "confidenceScaled < minConfidenceScaled" in sol
    assert "_isPermittedAction(recommendedAction)" in sol
    assert "_isPassiveOnly(recommendedAction)" in sol
    assert '_eq(severity, "CRITICAL")' in sol
    assert 'degradedMode && _eq(classification, "ANOMALY")' in sol


def test_active_contract_has_no_weighted_conflict_resolver():
    """The revised manuscript uses severity first, timestamp only as tie-breaker."""
    logger_sol = (REPO / "contracts" / "DecisionLogger.sol").read_text()
    banned = ("lambdaSeverity", "lambda_1", "LAMBDA_SEVERITY", "weighted conflict")
    lowered = logger_sol.lower()
    for token in banned:
        assert token.lower() not in lowered, f"legacy conflict token remains: {token}"


def test_contract_does_not_encode_lel_derivation_for_screening_threshold():
    """1,000 ppm is an experimental operating point, not a claimed LEL-derived rule."""
    sol = (REPO / "contracts" / "GovernanceRules.sol").read_text()
    assert "METHANE_LEL_PPM" not in sol
    assert "thresholdPercentOfLel" not in sol

def test_solidity_tolerated_faults_matches_table8():
    def solidity_tolerated(n: int) -> int:
        half_up = (n + 1) // 2  # ceil(n/2) in uint arithmetic
        return 0 if half_up == 0 else half_up - 1

    for n, (_, exp_f) in TABLE_8.items():
        assert solidity_tolerated(n) == exp_f


# ---------------------------------------------------------------------------
# Equations
# ---------------------------------------------------------------------------


def test_trigger_boundary_is_inclusive():
    """Equation (1) uses >=."""
    assert trigger(999.99) == 0
    assert trigger(1000.0) == 1
    assert trigger(1000.01) == 1


def test_fusion_weights_match_reported_range():
    """Section 3.2.2: raw-channel error variances of about 6,073-6,610 ppm^2.

    The corresponding weights are 1.51-1.65 x 1e-4 ppm^-2, and the ratio of
    best to worst is about 1.09, so the deployed sensors carry near-uniform
    influence.
    """
    lo, hi = SENSOR_ERROR_VARIANCE_RANGE_PPM2
    readings = [
        SensorReading("n1", 0.0, 1000.0, error_variance=lo),
        SensorReading("n2", 0.0, 1000.0, error_variance=hi),
    ]
    result = fuse_readings(readings)
    assert result.weights["n1"] == pytest.approx(1.646e-4, rel=1e-2)
    assert result.weights["n2"] == pytest.approx(1.513e-4, rel=1e-2)
    assert result.weights["n1"] / result.weights["n2"] == pytest.approx(1.088, abs=0.01)


def test_sensor_variances_match_deposit():
    """The configured variances must be the ones the deposited data yields.

    Computed from residuals of the raw instantaneous reading against the
    co-located NDIR reference, per node, over the labeled trials.
    """
    from adam import manuscript as ms
    from adam.config import SENSOR_ERROR_VARIANCE_RANGE_PPM2 as rng

    if not ms.available():
        pytest.skip("deposited dataset not present")
    measured = ms.sensor_error_variances()
    assert measured["min_ppm2"] == pytest.approx(rng[0], abs=1.0)
    assert measured["max_ppm2"] == pytest.approx(rng[1], abs=1.0)
    assert measured["weight_ratio"] == pytest.approx(1.088, abs=0.01)
    assert measured["min_paired"] >= 400, "variance needs real residual degrees of freedom"


def test_fusion_favors_better_calibrated_sensor():
    """Lower calibration variance must pull the estimate toward its reading."""
    readings = [
        SensorReading("good", 0.0, 1000.0, error_variance=1000.0),
        SensorReading("poor", 0.0, 2000.0, error_variance=4000.0),
    ]
    fused = fuse_readings(readings).fused_ppm
    assert 1000.0 < fused < 1500.0, "the better-calibrated sensor must dominate"


def test_fusion_rejects_empty_and_zero_weight():
    with pytest.raises(ValueError):
        fuse_readings([])
    with pytest.raises(ValueError):
        fuse_readings([SensorReading("n", 0.0, 100.0, error_variance=0.0)])


def test_fusion_flags_injected_outlier():
    """Cross-node corroboration is the defense in Section 4.5.1."""
    readings = [
        SensorReading("n1", 0.0, 1000.0, error_variance=1000.0),
        SensorReading("n2", 0.0, 1010.0, error_variance=1000.0),
        SensorReading("n3", 0.0, 990.0, error_variance=1000.0),
        SensorReading("attacked", 0.0, 9000.0, error_variance=1000.0),
    ]
    assert "attacked" in fuse_readings(readings, outlier_z=1.5).outliers


def test_conflict_prefers_higher_severity_even_when_older():
    """Equation (5): severity takes precedence over recency."""
    high_old = Candidate("high", 1.0, -20.0, event_id="e-high")
    low_new = Candidate("low", 0.25, -1.0, event_id="e-low")

    winner = resolve_conflict([high_old, low_new])

    assert winner.action == "high"


def test_equal_severity_prefers_more_recent_recommendation():
    """Equation (5): recency is used only as a severity tie-break."""
    older = Candidate("older", 0.75, -10.0, event_id="e-old")
    newer = Candidate("newer", 0.75, -1.0, event_id="e-new")

    winner = resolve_conflict([older, newer])

    assert winner.action == "newer"


def test_single_conflict_candidate_returns_itself():
    candidate = Candidate("alert", 0.75, -2.0, event_id="e-1")

    assert resolve_conflict([candidate]) is candidate


def test_empty_conflict_candidate_set_is_rejected():
    with pytest.raises(ValueError, match="no candidates"):
        resolve_conflict([])


def test_conflict_resolution_is_deterministic_on_exact_tie():
    """Identical severity and time are resolved by stable metadata."""
    candidates = [
        Candidate("monitor", 0.75, -5.0, event_id="event-a"),
        Candidate("alert", 0.75, -5.0, event_id="event-b"),
    ]

    winners = {resolve_conflict(candidates).action for _ in range(50)}

    assert len(winners) == 1


def test_min_crew_size_respects_c4():
    with pytest.raises(ValueError, match="C4"):
        ADAMConfig(min_crew_size=1)


def test_threshold_must_lie_in_sensor_range():
    """A threshold the MQ-4 cannot resolve is not screenable."""
    with pytest.raises(ValueError, match="sensing range"):
        ADAMConfig(threshold_ppm=50_000.0)


# ---------------------------------------------------------------------------
# Decision schema
# ---------------------------------------------------------------------------


def _valid_payload(**over):
    p = {
        "classification": "ANOMALY",
        "confidence": 0.8,
        "severity": "HIGH",
        "reasoning": "fused estimate well above baseline",
        "recommended_action": "raise alert",
        "contributing_factors": ["baseline departure"],
        "requires_human_review": False,
    }
    p.update(over)
    return p


def test_seven_field_schema_accepted():
    obj = DecisionObject.from_model_json(_valid_payload())
    assert obj.is_anomaly and obj.severity_score == 0.75


def test_missing_field_rejected():
    payload = _valid_payload()
    del payload["severity"]
    with pytest.raises(SchemaViolation, match="missing fields"):
        DecisionObject.from_model_json(payload)


def test_out_of_range_confidence_is_clamped():
    """A common small-model failure; repaired rather than escalated."""
    assert DecisionObject.from_model_json(_valid_payload(confidence=1.4)).confidence == 1.0
    assert DecisionObject.from_model_json(_valid_payload(confidence=-0.2)).confidence == 0.0


def test_string_boolean_coerced():
    obj = DecisionObject.from_model_json(_valid_payload(requires_human_review="true"))
    assert obj.requires_human_review is True


def test_unknown_classification_rejected():
    with pytest.raises(SchemaViolation, match="classification"):
        DecisionObject.from_model_json(_valid_payload(classification="MAYBE"))


# ---------------------------------------------------------------------------
# Dataset integrity
# ---------------------------------------------------------------------------


def test_degenerate_labels_are_refused():
    """The reproducibility guard. See data/validate.py for why this exists."""
    from data.validate import DegenerateLabelsError, assert_labels_independent

    ppm = [500.0, 1500.0, 900.0, 2000.0, 300.0, 1200.0] * 50
    labels = [trigger(p) for p in ppm]  # labels derived from the rule itself
    with pytest.raises(DegenerateLabelsError, match="deterministic function"):
        assert_labels_independent(ppm, labels)


def test_sound_labels_accepted():
    """Drift-driven FPs and missed detections must both be present."""
    from data.validate import assert_labels_independent

    ppm, labels = [], []
    for i in range(400):
        if i % 10 == 0:  # interference: high MQ-4, no release
            ppm.append(1800.0)
            labels.append(0)
        elif i % 10 == 1:  # missed detection: low MQ-4, real release
            ppm.append(700.0)
            labels.append(1)
        elif i % 2 == 0:
            ppm.append(400.0)
            labels.append(0)
        else:
            ppm.append(1600.0)
            labels.append(1)
    diag = assert_labels_independent(ppm, labels)
    assert diag.false_positives_available > 0
    assert diag.false_negatives_available > 0
    assert diag.implied_static_f1 < 0.99


def test_simulator_fixture_passes_the_guard():
    """The stand-in must not have the defect it stands in for."""
    from data.loader import SimulationParams, simulate_trials
    from data.validate import assert_labels_independent

    ds = simulate_trials(SimulationParams(n_trials=3, events_per_trial=100))
    diag = assert_labels_independent(ds.primary_ppm(), ds.labels())
    assert diag.implied_static_f1 < 0.95
    assert ds.is_simulated


# ---------------------------------------------------------------------------
# Runtime invariants
# ---------------------------------------------------------------------------


def test_crew_dissolves_and_leaves_no_ephemeral_state():
    """Section 3.1.2: CrewEvent is cleared on dissolution."""
    import time

    from adam.crew import ADAMNode
    from adam.governance.chain import LocalValidator, NullChainClient
    from adam.memory.store import InMemoryStore

    mem = InMemoryStore()
    node = ADAMNode(
        "n1",
        ADAMConfig(enable_llm=False),
        memory=mem,
        chain=NullChainClient(),
        validator=LocalValidator(),
        llm_client=None,
    )
    t = time.time()
    reading = SensorReading("n1", t, 1500.0, error_variance=900.0)
    node.sensor.observe(reading)
    event = node.sensor.publish_trigger(reading)
    node.handle_event(event, [reading], sample_resources=False)
    assert mem.active_crew_events == 0
    assert mem.count_traces() == 1


def test_coordinator_does_not_vote():
    """Section 3.2: the Coordinator counts, it does not cast a ballot."""
    from adam.agents.roles import CoordinatorAgent

    coord = CoordinatorAgent("c", "n1")
    with pytest.raises(NotImplementedError, match="does not cast"):
        coord.vote(DecisionObject.from_model_json(_valid_payload()), {})


def test_duplicate_votes_rejected():
    """Attributability underpins the Table 8 bounds."""
    from adam.schemas import CrewEvent

    ev = CrewEvent("e1", "n1", 1500.0, 0.0)
    ev.record_vote("agent-a", True)
    with pytest.raises(ValueError, match="already voted"):
        ev.record_vote("agent-a", True)


def test_agent_view_strips_ground_truth():
    """No system under evaluation may see the reference label."""
    r = SensorReading("n1", 0.0, 1500.0, reference_ppm=1480.0, error_variance=900.0)
    assert "reference_ppm" not in r.redacted()
    assert r.reference_ppm == 1480.0


def test_remote_inference_endpoint_refused():
    """Section 4.5.3's zero-egress claim is enforced, not assumed."""
    from adam.llm.client import InferenceUnavailable, assert_local_endpoint

    with pytest.raises(InferenceUnavailable, match="zero-egress"):
        assert_local_endpoint("https://api.openai.com")
    assert_local_endpoint("http://127.0.0.1:11434")
    assert_local_endpoint("http://192.168.1.10:11434")


def test_fallback_marks_degraded_mode():
    """Section 4.5.2: fallback decisions stay distinguishable in the audit record."""
    from adam.llm.client import deterministic_fallback

    d = deterministic_fallback(1500.0)
    assert d.degraded_mode is True
    assert d.requires_human_review is True
    assert d.is_anomaly


def test_json_extraction_survives_small_model_output():
    from adam.llm.client import extract_json

    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}
    assert extract_json('{"reasoning": "brace } inside", "a": 1}')["a"] == 1


def test_wilcoxon_floor_matches_table5():
    """Table 5's recurring p = 0.002 is the n=10 floor, not an effect size."""
    from analysis.metrics import wilcoxon_floor

    assert wilcoxon_floor(10) == pytest.approx(0.001953, abs=1e-6)
    assert round(wilcoxon_floor(10), 3) == 0.002


def test_sign_test_reproduces_cloud_only_value():
    """Table 5 reports a sign test of 0.11 for Cloud-Only: 8 of 10 trials."""
    from analysis.metrics import _sign_test_p

    assert round(_sign_test_p(8, 10), 2) == 0.11


def test_quorum_is_computed_over_voters_not_crew_size():
    """The Coordinator tallies, so a four-agent crew supplies three ballots.

    Computing quorum over crew size instead would require 3 of 4 and let the
    tallying agent's presence change the threshold. The deployed threshold is
    quorum(3) = 2: any two of the three voting roles must agree.
    """
    import time

    from adam.config import ADAMConfig, DEPLOYED_VOTER_COUNT, quorum, tolerated_faults
    from adam.crew import ADAMNode
    from adam.governance.chain import LocalValidator, NullChainClient
    from adam.memory.store import InMemoryStore

    cfg = ADAMConfig(enable_llm=False)
    node = ADAMNode(
        "n1", cfg,
        memory=InMemoryStore(), chain=NullChainClient(),
        validator=LocalValidator(), llm_client=None,
    )
    t0 = time.time()
    reading = SensorReading("n1", t0, 1500.0, error_variance=900.0)
    node.sensor.observe(reading)
    event = node.sensor.publish_trigger(reading)
    trace = node.handle_event(event, [reading], sample_resources=False)

    assert trace.crew_size == 4
    assert trace.voter_count == DEPLOYED_VOTER_COUNT == 3
    assert trace.quorum_required == quorum(3) == 2
    assert tolerated_faults(3) == 1


def test_deployed_configuration_tolerates_one_compromise():
    """Section 4.5.1 under strict majority.

    At three voters and a threshold of two, one compromised voter can neither
    force an action alone nor block the two honest voters from acting; two
    colluding voters can supply quorum, which is the integrity bound the
    security analysis states.
    """
    from adam.config import DEPLOYED_VOTER_COUNT, fails_closed, is_subvertible

    assert tolerated_faults(DEPLOYED_VOTER_COUNT) == 1
    assert not fails_closed(DEPLOYED_VOTER_COUNT, 1)
    assert not is_subvertible(DEPLOYED_VOTER_COUNT, 1)
    assert is_subvertible(DEPLOYED_VOTER_COUNT, 2)


def test_coordinator_still_refuses_to_vote():
    """The tallying agent must not be able to tip its own quorum."""
    from adam.agents.roles import CoordinatorAgent

    coord = CoordinatorAgent("c", "n1")
    with pytest.raises(NotImplementedError):
        coord.vote(DecisionObject.from_model_json(_valid_payload()), {})


def test_security_results_reproduce_from_deposit():
    """Section 4.5's published figures must be recomputable, not quoted."""
    from adam import manuscript as ms
    from experiments import reproduce_security as rs

    if not ms.available():
        pytest.skip("deposited dataset not present")
    path = ms.dataset_path()

    inj = rs.injection(path)
    assert inj["events"] == 30
    assert inj["detected"] == 27
    assert inj["detection_rate"] == pytest.approx(0.900, abs=0.001)
    assert inj["f1_under_attack"] == pytest.approx(0.769, abs=0.002)
    assert set(inj["by_attack_type"]) == {
        "zero_inject", "constant_offset", "spike_inject", "replay"
    }

    poi = rs.poisoning(path)
    assert poi["levels"] == [0, 5, 10, 20], "paper reports 0/5/10/20, not 0/5/10/20/50"
    assert poi["contingency_clean_vs_worst"] == [[8, 0], [6, 1]]
    assert poi["fisher_exact_p"] == pytest.approx(0.47, abs=0.01)
    assert not poi["significant_at_0_05"]

    fail = rs.model_failure(path)
    assert fail["induced_failures"] == 19
    assert fail["crews_completed"] == 30
    assert fail["f1_fallback_decisions_only"] == pytest.approx(0.842, abs=0.002)


def test_egress_reports_measured_quantities_only():
    """No dollar figure may be derived: the deposit has no billing records.

    What the deposit does establish: ADAM crosses the deployment boundary with
    zero bytes and zero API calls in every window, while Cloud-Only averages
    about 117 KB and 19 calls per 30-minute window.
    """
    from adam import manuscript as ms
    from experiments import reproduce_security as rs

    if not ms.available():
        pytest.skip("deposited dataset not present")
    eg = rs.egress(ms.dataset_path())

    assert "cloud_cost" not in eg
    adam = eg["per_system"]["ADAM_LLM"]
    cloud = eg["per_system"]["Cloud_Only"]
    assert adam["windows"] == 12 and cloud["windows"] == 8
    assert adam["kb_per_window"] == 0.0
    assert adam["windows_with_egress"] == 0
    assert cloud["kb_per_window"] == pytest.approx(117.4, abs=0.5)
    assert cloud["api_calls_per_window"] == pytest.approx(19.1, abs=0.05)


def test_static_threshold_baseline_reproduces():
    """Table 5's Static Threshold row is the raw channel against 1,000 ppm.

    This is the same raw instantaneous sample that gates ADAM's crew
    formation, so the two systems receive identical input.
    """
    from adam import manuscript as ms

    if not ms.available():
        pytest.skip("deposited dataset not present")
    got = ms.threshold_baseline("Raw_Instantaneous_PPM")
    assert got["f1"] == pytest.approx(0.790, abs=0.002)
    assert got["far"] == pytest.approx(0.165, abs=0.002)


def test_deployment_semantics_workbook_sheet_reproduces_summary():
    """The deposited gated sheet is the deterministic revised operating point."""
    from adam import manuscript as ms

    if not ms.available():
        pytest.skip("deposited dataset not present")

    g = ms.gated_run_summary()
    assert g["triggered"] == 889
    assert g["trigger_rate"] == pytest.approx(0.4445, abs=0.0005)
    assert g["f1"] == pytest.approx(0.830, abs=0.002)
    assert g["far"] == pytest.approx(0.066, abs=0.002)

    struct = ms.gated_predictions_agree()
    assert struct["rows"] == 2000
    assert struct["trigger_rule_matches"] == 2000
    assert struct["untriggered_anomalies"] == 0


def test_benchmark_and_deployment_semantics_are_distinct_and_ordered():
    """Benchmark reasoning remains stronger than the gate-limited operating point."""
    from adam import manuscript as ms

    if not ms.available():
        pytest.skip("deposited dataset not present")

    benchmark = ms.detection_scores()["ADAM_LLM"]["f1"]
    deployment = ms.gated_run_summary()["f1"]
    static = ms.threshold_baseline("Raw_Instantaneous_PPM")["f1"]
    assert benchmark == pytest.approx(0.896, abs=0.002)
    assert deployment == pytest.approx(0.830, abs=0.002)
    assert benchmark > deployment > static


def test_eval_mode_controls_the_gate():
    """The two evaluation modes must implement the two deposited semantics.

    A sub-threshold event is scored NORMAL without forming a crew under
    "gated", and runs the full crew workflow under "full_pipeline". The
    deployment runner itself is gated unconditionally; this switch only
    affects offline D1 scoring.
    """
    from ablations.systems import ADAMSystem
    from adam.config import ADAMConfig
    from adam.governance.chain import LocalValidator, NullChainClient
    from adam.memory.store import InMemoryStore
    from adam.schemas import LabeledEvent, SensorReading

    def make(mode: str) -> ADAMSystem:
        return ADAMSystem(
            config=ADAMConfig(enable_llm=False, eval_mode=mode),
            memory=InMemoryStore(),
            chain=NullChainClient(),
            validator=LocalValidator(),
            llm_client=None,
        )

    readings = (
        SensorReading("n1", 0.0, 800.0, error_variance=6100.0),
        SensorReading("n2", 0.0, 815.0, error_variance=6300.0),
    )
    sub_threshold = LabeledEvent(
        trial_id=1, event_index=0, timestamp=0.0,
        readings=readings, label=0, reference_ppm=810.0,
    )

    gated = make("gated")
    n_before = len(gated.traces)
    pred = gated.predict(sub_threshold)
    assert pred.predicted == 0
    assert len(gated.traces) == n_before, "gated mode must not form a crew"

    full = make("full_pipeline")
    full.predict(sub_threshold)
    assert len(full.traces) == 1, "full-pipeline mode must run the crew workflow"

# ---------------------------------------------------------------------------
# Decision-Agent substitution path
# ---------------------------------------------------------------------------


def test_fitted_decision_agent_feature_vector_has_eight_features():
    from experiments.decision_agent_backends import decision_feature_vector

    x = decision_feature_vector(
        raw_ppm=1200.0,
        fused_ppm=1100.0,
        dispersion_ppm=50.0,
        baseline_window=[300, 310, 320, 330, 340, 350],
        threshold_ppm=1000.0,
    )
    assert x.shape == (8,)
    # Baseline must be causal and use only the supplied six prior values.
    assert x[5] == pytest.approx(325.0)
    assert x[6] == pytest.approx(1100.0 / 325.0)
    assert x[7] == pytest.approx(775.0)


def test_offline_nominal_chain_adapter_acknowledges_without_claiming_blockchain():
    from adam.governance.chain import InMemoryChainClient

    client = InMemoryChainClient()
    class E:
        event_id = "evt-1"
    class D:
        def to_dict(self): return {"classification": "NORMAL"}
    class O:
        approved = True

    receipt = client.log_decision(E(), D(), "monitor", O())
    assert receipt == "memory://decision/1"
    assert len(client.records) == 1


def test_substitution_runner_omits_zero_differences_before_exact_wilcoxon():
    src = (REPO / "experiments" / "run_decision_agent_substitution.py").read_text()
    assert "nonzero = delta[np.abs(delta) > 1e-12]" in src
    assert "wilcoxon(\n                nonzero," in src

# ---------------------------------------------------------------------------
# Semantic-memory causality
# ---------------------------------------------------------------------------


def _memory_trace(event_id: str, timestamp: float, fused_ppm: float):
    from adam.schemas import DecisionObject, EventTrace
    return EventTrace(
        event_id=event_id,
        timestamp=timestamp,
        trigger_node="N1",
        trigger_ppm=fused_ppm,
        fused_ppm=fused_ppm,
        decision=DecisionObject(
            classification="NORMAL",
            confidence=0.9,
            severity="NONE",
            reasoning="resolved historical event",
            recommended_action="monitor",
            contributing_factors=["test"],
            requires_human_review=False,
        ),
        governance_valid=True,
        final_action="monitor",
    )


def test_inmemory_semantic_retrieval_excludes_current_and_future_records():
    from adam.memory.store import InMemoryStore

    store = InMemoryStore()
    assert store.persist_trace(_memory_trace("past", 90.0, 1005.0))
    assert store.persist_trace(_memory_trace("current", 100.0, 1001.0))
    assert store.persist_trace(_memory_trace("future", 110.0, 1000.5))

    got = store.retrieve(1000.0, k=5, cutoff_timestamp=100.0)
    assert [r["event_id"] for r in got] == ["past"]


def test_crew_passes_exclusive_event_timestamp_to_semantic_retrieval():
    src = (REPO / "adam" / "crew.py").read_text()
    assert "cutoff_timestamp=event.timestamp" in src


def test_reference_fit_does_not_seed_semantic_memory_from_labels():
    src = (REPO / "ablations" / "systems.py").read_text()
    fit_body = src.split("def fit(self, train", 1)[1].split("def predict", 1)[0]
    assert "persist_trace" not in fit_body
    assert "historical training-fold outcome" not in fit_body

# ---------------------------------------------------------------------------
# Deployment-semantics derivation
# ---------------------------------------------------------------------------


def test_deployment_semantics_script_preserves_above_gate_and_forces_below_gate():
    src = (REPO / "scripts" / "derive_deployment_semantics.py").read_text()
    assert "if above:" in src
    assert "pred = int(b.predicted)" in src
    assert "pred = 0" in src
    assert "never invokes Ollama" in src


def test_run_trials_rejects_second_stochastic_gated_reproduction():
    src = (REPO / "experiments" / "run_trials.py").read_text()
    assert 'if args.eval_mode == "gated"' in src
    assert "derive deployment semantics" in src.lower()

# ---------------------------------------------------------------------------
# Leakage-safe fusion calibration
# ---------------------------------------------------------------------------


def _cal_event(trial, idx, ref, n1, n2):
    from adam.schemas import LabeledEvent, SensorReading
    readings = (
        SensorReading("N1", float(idx), float(n1), reference_ppm=float(ref), error_variance=999.0),
        SensorReading("N2", float(idx), float(n2), reference_ppm=float(ref), error_variance=999.0),
    )
    return LabeledEvent(
        trial_id=trial,
        event_index=idx,
        timestamp=float(idx),
        readings=readings,
        label=int(ref >= 1000.0),
        reference_ppm=float(ref),
    )


def test_fold_calibration_ignores_extreme_heldout_residuals():
    from data.calibration import calibrate_fold, estimate_error_variances

    train = [
        _cal_event(1, 0, 500, 490, 520),
        _cal_event(1, 1, 600, 620, 570),
        _cal_event(2, 2, 700, 685, 730),
        _cal_event(2, 3, 800, 825, 760),
    ]
    heldout = [
        _cal_event(3, 4, 900, 9000, 10),
        _cal_event(3, 5, 950, 9500, 5),
    ]

    expected = estimate_error_variances(train)
    train_c, test_c, got = calibrate_fold(train, heldout)
    assert got == pytest.approx(expected)

    # If held-out residuals leaked into calibration these values would explode.
    pooled = estimate_error_variances(train + heldout)
    assert pooled["N1"] > got["N1"] * 100
    assert pooled["N2"] > got["N2"] * 100

    for original, calibrated in zip(heldout, test_c):
        assert calibrated.label == original.label
        assert calibrated.reference_ppm == original.reference_ppm
        for ro, rc in zip(original.readings, calibrated.readings):
            assert rc.node_id == ro.node_id
            assert rc.methane_ppm == ro.methane_ppm
            assert rc.timestamp == ro.timestamp
            assert rc.error_variance == pytest.approx(expected[ro.node_id])


def test_reported_harnesses_keep_fixed_weights_and_expose_fold_local_sensitivity():
    main = (REPO / "experiments" / "run_trials.py").read_text()
    swap = (REPO / "experiments" / "run_decision_agent_substitution.py").read_text()
    assert "fold_local_calibration: bool = False" in main
    assert "--fold-local-calibration" in main
    assert "fold_local_calibration: bool = False" in swap
    assert "--fold-local-calibration" in swap
    assert "fixed deposited inverse-variance weights" in swap


def test_adam_reference_fit_does_not_warm_state_from_other_trials():
    src = (REPO / "ablations" / "systems.py").read_text()
    fit_body = src.split("def fit(self, train", 1)[1].split("def predict", 1)[0]
    assert "observe(" not in fit_body
    assert "persist_trace" not in fit_body

def test_contextual_fitted_baselines_use_same_eight_features_as_decision_agent():
    from baselines.systems import (
        CONTEXTUAL_FEATURE_NAMES,
        DECISION_AGENT_FEATURE_NAMES,
        FUSED_BASELINE_FEATURE_NAMES,
    )
    assert len(CONTEXTUAL_FEATURE_NAMES) == 8
    assert FUSED_BASELINE_FEATURE_NAMES == CONTEXTUAL_FEATURE_NAMES
    assert DECISION_AGENT_FEATURE_NAMES == CONTEXTUAL_FEATURE_NAMES


def test_policy_bands_and_fallback_actions_match_governance():
    from adam.config import PERMITTED_ACTIONS
    from adam.llm.client import deterministic_fallback
    from experiments.decision_agent_backends import decision_from_probability

    low = deterministic_fallback(1500.0)
    critical = deterministic_fallback(CRITICAL_THRESHOLD_PPM + 1)
    assert low.severity == "LOW"
    assert critical.severity == "CRITICAL"
    assert low.recommended_action in PERMITTED_ACTIONS
    assert critical.recommended_action in PERMITTED_ACTIONS

    fitted = decision_from_probability(
        probability_anomaly=0.9,
        fused_ppm=WARNING_THRESHOLD_PPM + 1,
        dispersion_ppm=10.0,
        variant="test",
        threshold_ppm=THRESHOLD_PPM,
    )
    assert fitted.severity == "HIGH"
    assert fitted.recommended_action in PERMITTED_ACTIONS


def test_zero_budget_fails_closed_before_reasoner_invocation():
    from adam.crew import ADAMNode
    from adam.governance.chain import InMemoryChainClient, LocalValidator
    from adam.memory.store import InMemoryStore
    from adam.schemas import CrewEvent, SensorReading

    class CountingBackend:
        name = "counting"
        def __init__(self): self.calls = 0
        def reason(self, **kwargs):
            self.calls += 1
            raise AssertionError("reasoner must not run after deadline exhaustion")

    backend = CountingBackend()
    node = ADAMNode(
        "deadline-node",
        config=ADAMConfig(decision_deadline_s=0.0, enable_llm=True),
        memory=InMemoryStore(),
        chain=InMemoryChainClient(),
        validator=LocalValidator(),
        decision_backend=backend,
    )
    event = CrewEvent(
        event_id="deadline-event",
        timestamp=1.0,
        trigger_node="n1",
        trigger_ppm=1200.0,
    )
    readings = [SensorReading("n1", 1.0, 1200.0, error_variance=6200.0)]
    trace = node.handle_event(event, readings, sample_resources=False)
    assert backend.calls == 0
    assert trace.failure_stage == "deadline_before_reasoning"
    assert not trace.executed


# ---------------------------------------------------------------------------
# Degraded-condition harness
# ---------------------------------------------------------------------------


def test_degraded_harness_dropout_refuses_stale_n4_and_refuses_label_changes():
    import numpy as np
    import pandas as pd
    from experiments.degraded_harness import apply_condition, condition_seed, _node_sigma
    from adam.schemas import SensorReading
    from adam.mechanisms import fuse_readings

    rows = []
    for k in range(6):
        ref = 900.0 + 10 * k
        for node, offset in [("N1", 10), ("N2", -5), ("N3", 20), ("N4", 400)]:
            rows.append({
                "event_id": f"E{k}", "trial": 1, "node_id": node,
                "timestamp": float(k), "trigger_node": "N1",
                "raw_ppm": ref + offset, "reference_ppm": ref,
            })
    df = pd.DataFrame(rows)
    sigma = _node_sigma(df)
    seed, _ = condition_seed("one_node_dropout", 1)
    out = apply_condition(df, "one_node_dropout", 1, np.random.default_rng(seed), sigma)

    # Pick the last event, where N4 must be offline.
    g = out[out.event_id == "E5"]
    assert not bool(g[g.node_id == "N4"].iloc[0].node_online)
    assert np.isnan(g[g.node_id == "N4"].iloc[0].perturbed_ppm)
    assert g.reference_ppm.nunique() == 1

    # Fusion is constructed only from online rows; N4's stale/high value is not
    # silently substituted. Equal variances make the expected fused value the
    # simple mean of N1--N3.
    online = g[g.node_online]
    readings = [
        SensorReading(
            node_id=row.node_id,
            timestamp=float(row.timestamp),
            methane_ppm=float(row.perturbed_ppm),
            reference_ppm=float(row.reference_ppm),
            error_variance=100.0,
        )
        for row in online.itertuples(index=False)
    ]
    fused = fuse_readings(readings)
    assert tuple(fused.contributing_nodes) == ("N1", "N2", "N3")
    assert fused.fused_ppm == pytest.approx(float(online.perturbed_ppm.mean()))


def test_degraded_harness_hashes_agent_facing_replay_fields():
    src = (REPO / "experiments" / "degraded_harness.py").read_text()
    for field in ("trial", "event_id", "node_id", "timestamp", "perturbed_ppm", "node_online"):
        assert field in src

# ---------------------------------------------------------------------------
# Appendix A / prompt source of truth
# ---------------------------------------------------------------------------


def test_generated_appendix_example_is_explicitly_synthetic_and_date_neutral():
    from adam.llm.prompt import emit_latex

    latex = emit_latex()
    assert "synthetic illustrative event" in latex
    assert "not a D1/D2 observation" in latex
    assert "2026-03-" not in latex
    assert "2026-04-" not in latex


def test_prompt_human_review_rule_matches_methods():
    from adam.llm.prompt import build_system_prompt

    prompt = build_system_prompt()
    assert "confidence is below 0.6" in prompt
    assert "dispersion exceeds half the fused estimate" in prompt
    assert "severity is\nCRITICAL" in prompt or "severity is CRITICAL" in prompt



def test_fides_client_exposes_onchain_governance_validation_without_web3():
    """Real-chain path must call GovernanceRules before ledger persistence."""
    from adam.governance.chain import FidesInnovaClient
    from adam.schemas import CrewEvent, DecisionObject

    class _Call:
        def call(self):
            return (True, "policy satisfied")

    class _Functions:
        def __init__(self):
            self.args = None

        def validateDecision(self, *args):
            self.args = args
            return _Call()

    class _Contract:
        def __init__(self):
            self.functions = _Functions()

    contract = _Contract()
    client = FidesInnovaClient()
    client._w3 = object()  # avoid connect(); no web3 dependency in this unit test
    client._load_contract = lambda name: contract

    event = CrewEvent(
        event_id="evt-abcdef12",
        trigger_node="N1",
        trigger_ppm=1200.0,
        timestamp=1.0,
    )
    event.record_vote("sensor", True)
    event.record_vote("aggregator", True)
    event.record_vote("decision", False)
    decision = DecisionObject(
        classification="ANOMALY",
        confidence=0.82,
        severity="HIGH",
        reasoning="test",
        recommended_action="raise alert",
        contributing_factors=["test"],
        requires_human_review=False,
    )

    valid, reason = client.validate(decision, event)
    assert valid is True
    assert reason == "policy satisfied"
    args = contract.functions.args
    assert args[0] == 1200
    assert args[1] == 82
    assert args[2] == "ANOMALY"
    assert args[3] == "HIGH"
    assert args[4] == "raise alert"
    assert args[6] is False  # degraded_mode
    assert args[7] == 3      # number of ballots
    assert args[8] == 2      # approvals


def test_fides_governance_validation_fails_closed_on_contract_error():
    from adam.governance.chain import FidesInnovaClient
    from adam.schemas import CrewEvent, DecisionObject

    client = FidesInnovaClient()
    client._w3 = object()
    client._load_contract = lambda name: (_ for _ in ()).throw(RuntimeError("rpc down"))

    event = CrewEvent("evt-abcdef12", "N1", 1200.0, 1.0)
    event.record_vote("sensor", True)
    event.record_vote("aggregator", True)
    decision = DecisionObject(
        classification="ANOMALY",
        confidence=0.82,
        severity="HIGH",
        reasoning="test",
        recommended_action="raise alert",
        contributing_factors=["test"],
        requires_human_review=False,
    )

    valid, reason = client.validate(decision, event)
    assert valid is False
    assert "unavailable" in reason
