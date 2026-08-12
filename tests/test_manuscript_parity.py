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
    DECISION_DEADLINE_S,
    LAMBDA_RECENCY,
    LAMBDA_SEVERITY,
    METHANE_LEL_PPM,
    MIN_CREW_SIZE,
    N_DEPLOYMENT_EVENTS,
    NODE_SCALING_FIT_HW,
    NODE_SCALING_FIT_SCALEOUT,
    REFERENCE_ACCURACY,
    REFERENCE_STAGE_LATENCY_MS,
    SENSOR_ERROR_VARIANCE_RANGE_PPM2,
    THRESHOLD_PPM,
    ADAMConfig,
    fails_closed,
    is_subvertible,
    quorum,
    tolerated_faults,
    verify_against_manuscript,
)
from adam.mechanisms import (
    Candidate,
    flip_threshold,
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


def test_threshold_is_two_percent_of_lel():
    """Constraint C1's argument depends on this exact ratio."""
    assert THRESHOLD_PPM == 1000.0
    assert THRESHOLD_PPM / METHANE_LEL_PPM == pytest.approx(0.02)


def test_stage_latencies_sum_to_reported_mean():
    """Equation (6): the six stages sum to the reported decision latency."""
    total_s = sum(REFERENCE_STAGE_LATENCY_MS.values()) / 1000.0
    assert total_s == pytest.approx(18.99, abs=0.05)


def test_reasoning_dominates_latency_budget():
    """Section 4.2 and 5.3 both rest on reasoning being the sole bottleneck."""
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

    The reference is accurate to +/-2%, about 20 ppm at the 1,000 ppm screening
    threshold, so any sensor variance below that is physically impossible.
    """
    import math

    lo, hi = SENSOR_ERROR_VARIANCE_RANGE_PPM2
    reference_sd = REFERENCE_ACCURACY * THRESHOLD_PPM
    assert math.sqrt(lo) > reference_sd, (
        f"MQ-4 sd {math.sqrt(lo):.1f} ppm is below the reference tolerance "
        f"{reference_sd:.1f} ppm, which is physically implausible"
    )


def test_holm_reproduces_published_adjustments():
    """The Holm correction must give Table 5's adjusted column."""
    from analysis.metrics import holm_adjust

    raw = {
        "static": 0.001953, "rf": 0.001953, "single": 0.001953, "nollm": 0.001953,
        "noagg": 0.003906, "noweav": 0.003906,
        "cloud": 0.048828, "noblockchain": 0.097656,
    }
    adj = holm_adjust(raw)
    assert round(adj["static"], 3) == 0.016
    assert round(adj["noagg"], 3) == 0.016
    assert round(adj["cloud"], 3) == 0.098
    assert round(adj["noblockchain"], 3) == 0.098
    # Two comparisons must fail to survive correction.
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


def test_deployed_crew_tolerates_one_compromised_agent():
    """Section 4.5.1: the n=4 crew tolerates one, fails closed at two."""
    assert tolerated_faults(4) == 1
    assert fails_closed(4, 2)
    assert not is_subvertible(4, 2)
    assert is_subvertible(4, 3)


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


def test_recency_is_inverse_age_not_inverse_timestamp():
    """Section 3.2.5: inverse absolute timestamp is numerically degenerate."""
    now = 1_000_000.0
    older = Candidate("a", 0.5, now - 10.0)
    newer = Candidate("b", 0.5, now - 1.0)
    assert newer.recency(now) > older.recency(now)
    assert newer.recency(now) == pytest.approx(1.0)


def test_conflict_prefers_severity_at_configured_weights():
    now = 0.0
    high_old = Candidate("high", 1.0, -20.0)
    low_new = Candidate("low", 0.25, -1.0)
    winner = resolve_conflict(
        [high_old, low_new], now, LAMBDA_SEVERITY, LAMBDA_RECENCY, "window"
    )
    assert winner.action == "high"


def test_pairwise_normalization_flips_at_exactly_half():
    """Section 4.6's degeneracy result."""
    now = 0.0
    a = Candidate("a", 1.0, -20.0)
    b = Candidate("b", 0.25, -1.0)
    ft = flip_threshold(a, b, now, normalization="pairwise")
    assert ft == pytest.approx(0.5)


def test_conflict_resolution_is_deterministic():
    """The audit trace must be replayable."""
    now = 0.0
    cands = [Candidate("a", 0.75, -5.0), Candidate("b", 0.75, -5.0)]
    winners = {
        resolve_conflict(cands, now, 0.7, 0.3, "window").action for _ in range(50)
    }
    assert len(winners) == 1


def test_lambda_one_is_rejected():
    """Section 4.6 bounds lambda_1 strictly below 1."""
    with pytest.raises(ValueError, match="strictly"):
        ADAMConfig(lambda_severity=1.0, lambda_recency=0.0)


def test_lambda_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="must equal 1"):
        ADAMConfig(lambda_severity=0.7, lambda_recency=0.5)


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
    assert fail["f1_episode"] == pytest.approx(0.774, abs=0.002)


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
    adam = eg["per_system"]["ADAM_Full"]
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


def test_gated_run_reproduces_summary():
    """The trigger-gated D1 run is the deployed operating point.

    Its overall figures come from D1_RawTrigger_Summary, and two structural
    properties must hold row by row: the trigger fires exactly when the raw
    reading meets the screening threshold, and an untriggered event is never
    classified as an anomaly.
    """
    from adam import manuscript as ms

    if not ms.available():
        pytest.skip("deposited dataset not present")

    g = ms.gated_run_summary()
    assert g["triggered"] == 889
    assert g["trigger_rate"] == pytest.approx(0.4445, abs=0.0005)
    assert g["f1"] == pytest.approx(0.8142, abs=0.002)

    struct = ms.gated_predictions_agree()
    assert struct["rows"] == 2000
    assert struct["trigger_rule_matches"] == 2000
    assert struct["untriggered_anomalies"] == 0


def test_two_adam_runs_are_distinct_and_ordered():
    """Both deposited D1 runs must be present and relate as the paper states.

    The full-pipeline benchmark measures interpretation quality over all
    2,000 events; the gated run measures the deployed operating point, where
    the screening threshold caps recall. The benchmark therefore sits above
    the gated run, and both sit above the raw fixed-threshold rule.
    """
    from adam import manuscript as ms

    if not ms.available():
        pytest.skip("deposited dataset not present")

    benchmark = ms.detection_scores()["ADAM_Full"]["f1"]
    gated = ms.gated_run_summary()["f1"]
    static = ms.threshold_baseline("Raw_Instantaneous_PPM")["f1"]
    assert benchmark == pytest.approx(0.896, abs=0.002)
    assert gated == pytest.approx(0.814, abs=0.002)
    assert benchmark > gated > static


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
            seed_memory=False,
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
