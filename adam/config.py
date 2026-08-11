"""
adam.config
===========

Single source of truth for every numeric constant reported in the manuscript.

Every value below carries a citation to the manuscript location that fixes it.
Nothing in this codebase should hard-code an operational constant; import it
from here instead. `verify_against_manuscript()` is exercised by the test suite
so that a drift between code and paper fails CI rather than surfacing in review.

Reference:
    Nweke, B.C., Ramezan, G., Saraji, S. "Agentic Decentralized Autonomous
    Machines (ADAM): An Agentic AI Framework for Decentralized Physical
    Infrastructure Networks." Manuscript, 2026.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Operational constraints  (manuscript Section 3.3, Table 3)
# ---------------------------------------------------------------------------

#: C1 - end-to-end decision deadline, seconds. Table 3.
DECISION_DEADLINE_S: float = 30.0

#: C2 - sustained CPU budget as a fraction. Table 3. Measured as the mean over
#: the deployment cycle on a rolling five-minute average, NOT the instantaneous
#: inference-time peak.
MAX_SUSTAINED_CPU: float = 0.80

#: C2 - rolling window over which sustained CPU is averaged. Section 3.3.
CPU_ROLLING_WINDOW_S: float = 300.0

#: C4 - minimum crew size for degraded-mode operation. Table 3.
MIN_CREW_SIZE: int = 2

#: C5 - local event-screening threshold, ppm. Table 3.
#: NOTE: the pre-revision codebase used 5000 ppm here and 5000/3000 on-chain.
#: Confirmed 2026-07: the 72-hour deployment ran at 1000 ppm. The 2%-of-LEL
#: argument in C1 and Section 5.1 depends on this value.
THRESHOLD_PPM: float = 1000.0

#: Methane lower explosive limit, ppm (5% by volume). Section 1.
METHANE_LEL_PPM: float = 50_000.0

# ---------------------------------------------------------------------------
# Crew coordination  (manuscript Section 3.2)
# ---------------------------------------------------------------------------

#: The four role names that compose a full crew. Section 3.1.2.
CREW_ROLES: Tuple[str, ...] = ("sensor", "aggregator", "decision", "coordinator")

#: Roles that cast a ballot. Section 3.2: the Coordinator "counts approvals,
#: checks the governance validator, and executes the action" - it tallies and
#: does not vote, so including it would let the tallying agent tip its own
#: quorum.
VOTING_ROLES: Tuple[str, ...] = ("sensor", "aggregator", "decision")

#: Agents instantiated per crew, including the non-voting Coordinator.
DEPLOYED_CREW_SIZE: int = 4

#: Ballots available to Equation (4). This is the quantity quorum() takes, and
#: it is NOT the crew size: three voters plus a tallying Coordinator.
#:
#: At three voters the deployed threshold is quorum(3) = 2: any two of the
#: three role-specific checks (local trigger consistency, cross-node
#: consistency, semantic consistency) must agree before an action executes. A
#: single dissenting voter cannot block a decision, and no single voter can
#: approve one alone.
DEPLOYED_VOTER_COUNT: int = 3


def quorum(voter_count: int) -> int:
    """Crew-level quorum threshold over the VOTING agents.

    Implements Equation (4), strict majority:  gamma_crew = floor(n / 2) + 1

    ``n`` is the number of agents that cast a ballot, not the number
    instantiated. The Coordinator tallies rather than votes, so a four-agent
    crew supplies three ballots and the deployed threshold is quorum(3) = 2.

    This is the authoritative definition, and it matches the consensus rule
    recorded in the dataset (01_Config: "strict majority = FLOOR(n/2)+1") and
    the deployment failure notes in 05_D2_Coordination_Log. The on-chain
    ``GovernanceRules.getRequiredConsensus`` MUST agree with it for every crew
    size in Table 8; ``tests/test_manuscript_parity.py`` asserts that parity.

    A prior contract revision used ``ceil(n * 51 / 100)``, which agrees with
    the strict-majority rule at odd ``n`` but understates it at every even
    ``n`` (n=2 -> 2 vs 2 ok, n=4 -> 3 vs 3 ok; the divergence appears against
    the earlier ceil(n/2)+1 draft rule, which this revision replaces).
    """
    if voter_count < 1:
        raise ValueError(f"voter_count must be >= 1, got {voter_count}")
    return voter_count // 2 + 1


def tolerated_faults(voter_count: int) -> int:
    """Compromised voters under which the honest remainder retains quorum.

    Honest voters can still execute an action while n - f >= quorum(n), which
    gives f <= ceil(n/2) - 1. At the deployed three voters this is 1: one
    compromised or unavailable voter cannot prevent the remaining two from
    reaching the two-vote threshold. Integrity is bounded separately by
    ``is_subvertible``: quorum(n) colluding voters can force an action, which
    at three voters means two.
    """
    return max(0, math.ceil(voter_count / 2) - 1)


def is_subvertible(voter_count: int, n_compromised: int) -> bool:
    """True when compromised voters alone can supply quorum. Table 8."""
    return n_compromised >= quorum(voter_count)


def fails_closed(voter_count: int, n_compromised: int) -> bool:
    """True when quorum is unreachable, so no action executes. Table 8."""
    honest = voter_count - n_compromised
    return honest < quorum(voter_count) and not is_subvertible(
        voter_count, n_compromised
    )


# ---------------------------------------------------------------------------
# Conflict resolution  (manuscript Section 3.2.5, Equation 5)
# ---------------------------------------------------------------------------

#: Severity weight. Section 3.2.5.
LAMBDA_SEVERITY: float = 0.7

#: Recency weight. Section 3.2.5.
LAMBDA_RECENCY: float = 0.3

#: Normalization regime for Equation (5). Section 4.6 evaluates both.
#:   "window"  - min-max across the population of concurrent events; preserves
#:               the magnitude of each severity gap. Reported as the primary
#:               regime (>=99% agreement across lambda_1 in [0.532, 0.998]).
#:   "pairwise"- min-max within each conflicting pair; degenerate, since any
#:               pair normalizes to {0, 1} and the decision flips at exactly 0.5.
CONFLICT_NORMALIZATION: str = "window"

#: Guard from Section 4.6: at lambda_1 = 1 the recency term vanishes and
#: equal-severity conflicts cannot be separated.
LAMBDA_SEVERITY_MAX_EXCLUSIVE: float = 1.0

# ---------------------------------------------------------------------------
# Sensing hardware  (manuscript Table 4)
# ---------------------------------------------------------------------------

#: Number of physical edge nodes. Table 4.
N_NODES: int = 4

#: MQ-4 sensing range, ppm. Table 4.
MQ4_RANGE_PPM: Tuple[float, float] = (300.0, 10_000.0)

#: Sensor sampling rate, Hz. Table 4.
SAMPLING_RATE_HZ: float = 1.0

#: Reference (ground-truth) electrochemical sensor accuracy. Table 4.
REFERENCE_ACCURACY: float = 0.02

#: Per-sensor error-variance range, ppm^2. Section 3.2.2.
#: Estimated from residuals of each node's raw instantaneous reading against
#: the co-located NDIR reference over the labeled trials, 500 paired readings
#: per node. Weights w_i = 1/sigma_i^2 fall in [1.51, 1.65] x 1e-4 ppm^-2 and
#: the normalized weights span 0.239 to 0.260, so the four sensors carry
#: near-uniform influence (best-to-worst ratio about 1.09). The weighting
#: mechanism is retained for heterogeneous or degraded hardware, where the
#: variances would separate.
SENSOR_ERROR_VARIANCE_RANGE_PPM2: Tuple[float, float] = (6073.2, 6609.6)

# ---------------------------------------------------------------------------
# Language model  (manuscript Section 3.4.2, Table 4)
# ---------------------------------------------------------------------------

#: On-device model tag as served by Ollama. Table 4: Gemma 3 1B (INT4).
#: Confirmed 2026-07: the manuscript stands as written; the runtime is Gemma,
#: not Llama.
OLLAMA_MODEL: str = os.getenv("ADAM_OLLAMA_MODEL", "gemma3:1b")

#: Ollama HTTP endpoint. Local by construction - Section 4.5.3 reports zero
#: external egress, which only holds if this stays on-host.
OLLAMA_HOST: str = os.getenv("ADAM_OLLAMA_HOST", "http://127.0.0.1:11434")

#: Sampling temperature. Section 3.4.2.
LLM_TEMPERATURE: float = 0.1

#: Maximum response length in tokens. Section 3.4.2.
LLM_MAX_TOKENS: int = 256

#: Format-repair retries before deterministic fallback. Section 3.4.2 specifies
#: exactly one.
LLM_FORMAT_REPAIR_RETRIES: int = 1

#: The seven fields of the structured decision object. Section 3.4.2.
DECISION_SCHEMA_FIELDS: Tuple[str, ...] = (
    "classification",
    "confidence",
    "severity",
    "reasoning",
    "recommended_action",
    "contributing_factors",
    "requires_human_review",
)

#: Permitted values of the ``classification`` field. Equation (3).
CLASSIFICATION_VALUES: Tuple[str, ...] = ("ANOMALY", "NORMAL")

#: Permitted values of the ``severity`` field, ordered low to high. The conflict
#: sweep in Section 4.6 spans exactly these levels.
SEVERITY_LEVELS: Tuple[str, ...] = ("NONE", "LOW", "MODERATE", "HIGH", "CRITICAL")

#: Numeric encoding of severity for Equation (5).
SEVERITY_SCORES: Dict[str, float] = {
    "NONE": 0.0,
    "LOW": 0.25,
    "MODERATE": 0.50,
    "HIGH": 0.75,
    "CRITICAL": 1.0,
}

# ---------------------------------------------------------------------------
# Cloud comparator  (manuscript Section 3.4.4, Table 6)
# ---------------------------------------------------------------------------

#: Cloud-Only baseline model. Section 3.4.4. This is the ONLY component in the
#: repository permitted to make an external API call, and it is a comparator,
#: never part of the ADAM runtime.
CLOUD_MODEL: str = os.getenv("ADAM_CLOUD_MODEL", "gpt-4o-mini")

#: Measured external egress of the Cloud-Only comparator, from the 20-window
#: confidentiality measurement (13_Security_Data_Leakage): mean bytes and API
#: calls per 30-minute window across the 8 Cloud-Only windows. ADAM records
#: zero in both quantities across its 12 windows. No per-decision dollar cost
#: is stated anywhere in the codebase because the deposit contains no cloud
#: token or billing records to support one.
CLOUD_EGRESS_PER_WINDOW_KB: float = 117.4
CLOUD_API_CALLS_PER_WINDOW: float = 19.1

# ---------------------------------------------------------------------------
# Semantic memory  (manuscript Section 3.1.2, Table 4)
# ---------------------------------------------------------------------------

WEAVIATE_HOST: str = os.getenv("ADAM_WEAVIATE_HOST", "http://127.0.0.1:8080")

#: Ephemeral class: trigger, crew membership, votes for an ACTIVE event.
#: Cleared on crew dissolution. Section 3.1.2.
CLASS_CREW_EVENT: str = "CrewEvent"

#: Persistent class: resolved events retained for historical retrieval.
CLASS_EVENT_TRACE: str = "EventTrace"

#: Number of historical records retrieved as h_past. Equation (3).
SEMANTIC_MEMORY_K: int = 5

#: Measured trigger-publication latency per write, ms. Section 3.1.2. Retained
#: as a documented design cost (2.6% of crew-formation time), not a target.
TRIGGER_PUBLISH_LATENCY_MS: float = 55.0

# ---------------------------------------------------------------------------
# Governance layer  (manuscript Section 3.1.3, Table 4)
# ---------------------------------------------------------------------------

CHAIN_RPC_URL: str = os.getenv("ADAM_CHAIN_RPC", "http://127.0.0.1:8545")

#: Fides Innova PoA testnet chain id. Override for a local devnet.
CHAIN_ID: int = int(os.getenv("ADAM_CHAIN_ID", "706883"))

#: Deployed contract addresses, populated by scripts/deploy.js at deploy time.
CONTRACT_ADDRESSES: Dict[str, str] = {
    "GovernanceRules": os.getenv("ADAM_ADDR_GOVERNANCE", ""),
    "CrewRegistry": os.getenv("ADAM_ADDR_CREW_REGISTRY", ""),
    "DecisionLogger": os.getenv("ADAM_ADDR_DECISION_LOGGER", ""),
    "ConsensusValidator": os.getenv("ADAM_ADDR_CONSENSUS_VALIDATOR", ""),
}

#: Same-event coalescing window, seconds. Prevents one physical release from
#: forming redundant crews. Matches the C1 deadline by construction.
SAME_EVENT_WINDOW_S: int = 30

# ---------------------------------------------------------------------------
# Evaluation design  (manuscript Section 3.4.3)
# ---------------------------------------------------------------------------

#: D1 - number of labeled trials.
N_TRIALS: int = 10

#: D1 - labeled events per trial.
EVENTS_PER_TRIAL: int = 200

#: D2 - live coordination events over the 72-hour deployment. This is the
#: trace-persistence denominator.
N_DEPLOYMENT_EVENTS: int = 459

#: Events that completed end to end. Latency statistics use this denominator;
#: the remaining 13 reached the decision deadline without committing an action.
N_COMPLETED_EVENTS: int = 446

#: D2 - deployment duration, hours.
DEPLOYMENT_HOURS: float = 72.0

#: Random Forest hyperparameters, fixed across LOTO folds. Section 3.4.4.
RF_PARAMS: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
}

#: Significance level for the trial-level tests. Section 3.4.5.
ALPHA: float = 0.05

#: Eight systems are compared against one reference, so the Holm procedure
#: controls the family-wise error rate. Section 3.4.5 reports both the exact
#: and the adjusted p-value.
N_COMPARISONS: int = 8

#: Synthetic conflict pairs generated for the Section 4.6 sweep.
CONFLICT_SWEEP_N: int = 20_000

#: Global seed. Every stochastic component derives from this so that a reviewer
#: reproduces figures bit-for-bit.
SEED: int = 42

# ---------------------------------------------------------------------------
# Per-stage latency budget  (manuscript Section 4.2, Figure 5)
# ---------------------------------------------------------------------------

#: Mean per-stage latencies over the 459 deployment events, milliseconds.
#: Held here as REFERENCE values for regression-checking a reproduction run,
#: never as substitutes for measurement.
REFERENCE_STAGE_LATENCY_MS: Dict[str, float] = {
    "T_form": 2076.5,
    "T_agg": 437.2,
    "T_reason": 15473.6,
    "T_gov": 307.9,
    "T_weav": 256.0,
    "T_bc": 438.6,
}

#: Crew-formation dispersion, ms. Section 4.2.
REFERENCE_FORM_LATENCY_SD_MS: float = 181.6
REFERENCE_FORM_LATENCY_MEDIAN_MS: float = 2077.8
REFERENCE_FORM_LATENCY_P95_MS: float = 2373.1

#: Deterministic-fallback activation latency, ms. Section 4.5.2. Mean over the
#: 19 induced local-model failures (median 54.6 ms, P95 81.6 ms).
REFERENCE_FALLBACK_LATENCY_MS: float = 55.7

#: Node-scaling models, Table 7. The scalability study varies node count N
#: while holding load fixed at the reference configuration (4 concurrent
#: events, 8 sensor streams, 4 logical workers, 30,000 vectors). N = 1-4 runs
#: on physical Raspberry Pi 5 hardware; N = 6-16 is a Python scale-out model
#: validated against the matched N = 1-4 hardware runs (18 replicates per
#: level: 6 per day across 3 days). Both fits take the form
#: T(N) = T0 + alpha * N^beta and are recomputed from 08_Scalability_Log by
#: adam.manuscript; the values here are references for regression checks.
NODE_SCALING_FIT_HW: Dict[str, float] = {
    "T0": 17_379.97, "alpha": 198.7696, "beta": 1.4619,  # hardware, N = 1-4
}
NODE_SCALING_FIT_SCALEOUT: Dict[str, float] = {
    "T0": 14_261.78, "alpha": 2950.9415, "beta": 0.288,  # scale-out, N = 4-16
}

#: Simulator-validation statistics: Python scale-out model against matched
#: N = 1-4 hardware, decision latency. 09_Fitted_Models.
SIM_VALIDATION_MAPE_PCT: float = 2.373
SIM_VALIDATION_BIAS_PCT: float = -0.017

# ---------------------------------------------------------------------------
# Systems under evaluation
# ---------------------------------------------------------------------------

BASELINES: Tuple[str, ...] = (
    "static_threshold",
    "random_forest",
    "cloud_only",
    "single_agent",
)

ABLATIONS: Tuple[str, ...] = (
    "no_aggregator",
    "no_llm",
    "no_blockchain",
    "no_weaviate",
)

SYSTEMS: Tuple[str, ...] = ("adam_full",) + BASELINES + ABLATIONS


# ---------------------------------------------------------------------------
# Runtime configuration object
# ---------------------------------------------------------------------------


@dataclass
class ADAMConfig:
    """Mutable runtime view over the constants above.

    Experiments that need to vary a parameter (the conflict sweep varies
    lambda_1; the scalability harness varies concurrency) construct a modified
    copy rather than mutating module state.
    """

    threshold_ppm: float = THRESHOLD_PPM
    decision_deadline_s: float = DECISION_DEADLINE_S
    min_crew_size: int = MIN_CREW_SIZE
    max_sustained_cpu: float = MAX_SUSTAINED_CPU

    lambda_severity: float = LAMBDA_SEVERITY
    lambda_recency: float = LAMBDA_RECENCY
    conflict_normalization: str = CONFLICT_NORMALIZATION

    ollama_model: str = OLLAMA_MODEL
    ollama_host: str = OLLAMA_HOST
    llm_temperature: float = LLM_TEMPERATURE
    llm_max_tokens: int = LLM_MAX_TOKENS
    llm_format_repair_retries: int = LLM_FORMAT_REPAIR_RETRIES

    weaviate_host: str = WEAVIATE_HOST
    semantic_memory_k: int = SEMANTIC_MEMORY_K

    chain_rpc_url: str = CHAIN_RPC_URL
    contract_addresses: Dict[str, str] = field(
        default_factory=lambda: dict(CONTRACT_ADDRESSES)
    )

    # Feature switches used by the ablation harnesses.
    enable_aggregator: bool = True
    enable_llm: bool = True
    enable_blockchain: bool = True
    enable_weaviate: bool = True

    #: How labeled D1 events are scored. Two runs are deposited and each mode
    #: reproduces one of them:
    #:
    #:   "gated"          Deployment semantics. A reading below threshold_ppm
    #:                    never forms a crew and is scored NORMAL on the fast
    #:                    path; only triggered readings receive aggregation and
    #:                    reasoning. Reproduces D1_RawTrigger_Log.
    #:
    #:   "full_pipeline"  Benchmark semantics. Every labeled event is replayed
    #:                    through the complete crew pipeline regardless of the
    #:                    trigger, so all nine systems classify the same 2,000
    #:                    events under identical conditions. Reproduces the
    #:                    ADAM_Full predictions in 06A_Event_Predictions and
    #:                    the Table 5 row.
    #:
    #: The live deployment runner is gated unconditionally; this switch only
    #: affects offline scoring of D1.
    eval_mode: str = "gated"

    seed: int = SEED

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject configurations the manuscript rules out."""
        if not 0.0 < self.lambda_severity < LAMBDA_SEVERITY_MAX_EXCLUSIVE:
            raise ValueError(
                f"lambda_severity must lie strictly in (0, 1); got "
                f"{self.lambda_severity}. Section 4.6: at lambda_1 = 1 the "
                f"recency term vanishes and equal-severity conflicts cannot be "
                f"separated."
            )
        if abs(self.lambda_severity + self.lambda_recency - 1.0) > 1e-9:
            raise ValueError(
                f"lambda_severity + lambda_recency must equal 1; got "
                f"{self.lambda_severity + self.lambda_recency}"
            )
        if self.conflict_normalization not in ("window", "pairwise"):
            raise ValueError(
                f"conflict_normalization must be 'window' or 'pairwise'; got "
                f"{self.conflict_normalization!r}"
            )
        if self.eval_mode not in ("gated", "full_pipeline"):
            raise ValueError(
                f"eval_mode must be 'gated' or 'full_pipeline'; got "
                f"{self.eval_mode!r}"
            )
        if self.min_crew_size < 2:
            raise ValueError(
                f"min_crew_size must be >= 2 (constraint C4); got {self.min_crew_size}"
            )
        if self.threshold_ppm <= 0:
            raise ValueError(f"threshold_ppm must be positive; got {self.threshold_ppm}")
        if not MQ4_RANGE_PPM[0] <= self.threshold_ppm <= MQ4_RANGE_PPM[1]:
            raise ValueError(
                f"threshold_ppm {self.threshold_ppm} lies outside the MQ-4 "
                f"sensing range {MQ4_RANGE_PPM} and cannot be screened for."
            )

    def with_(self, **overrides: Any) -> "ADAMConfig":
        """Return a validated copy with the given fields replaced."""
        base = asdict(self)
        base.update(overrides)
        return ADAMConfig(**base)

    @property
    def threshold_fraction_of_lel(self) -> float:
        """Screening threshold as a fraction of the LEL. Section 3.3 (C1)."""
        return self.threshold_ppm / METHANE_LEL_PPM


DEFAULT_CONFIG = ADAMConfig()


# ---------------------------------------------------------------------------
# Manuscript parity check
# ---------------------------------------------------------------------------


def verify_against_manuscript() -> List[str]:
    """Check the constants above against the deposited dataset.

    Reference values come from :mod:`adam.manuscript`, which recomputes them
    from the workbook rather than holding a second table of literals, so the
    check fails as soon as code and data diverge.

    Returns a list of discrepancies. Empty means the constants reproduce the
    deposit. Raises nothing if the dataset is absent - the structural checks
    that need no data still run.
    """
    problems: List[str] = []

    # ---- structural checks, independent of the dataset ----------------------

    frac = THRESHOLD_PPM / METHANE_LEL_PPM
    if abs(frac - 0.02) > 1e-9:
        problems.append(
            f"threshold/LEL = {frac:.4%}, constraint C1 and Section 5.1 state 2%"
        )

    expected_table8 = {2: (2, 0), 3: (2, 1), 4: (3, 1), 5: (3, 2), 6: (4, 2), 7: (4, 3)}
    for n, (exp_q, exp_f) in expected_table8.items():
        if quorum(n) != exp_q:
            problems.append(f"quorum({n}) = {quorum(n)}, Table 8 states {exp_q}")
        if tolerated_faults(n) != exp_f:
            problems.append(
                f"tolerated_faults({n}) = {tolerated_faults(n)}, Table 8 states {exp_f}"
            )

    for n in range(2, 12):
        if quorum(n) < 2:
            problems.append(
                f"quorum({n}) = {quorum(n)} permits unilateral approval, "
                f"contradicting Section 3.2.4"
            )

    # ---- checks against the deposit -----------------------------------------

    from . import manuscript as ms

    if not ms.available():
        problems.append(
            f"NOTE: {ms.dataset_path()} not found, so measured quantities were "
            f"not verified. Structural checks passed."
        )
        return problems

    def close(label: str, got: float, want: float, tol: float) -> None:
        if abs(got - want) > tol:
            problems.append(f"{label}: code says {want}, deposit gives {got:.4g}")

    lat = ms.stage_latencies_ms()
    for stage, want in REFERENCE_STAGE_LATENCY_MS.items():
        close(f"stage latency {stage}", lat[stage], want, 0.5)

    form = ms.crew_formation_ms()
    close("crew formation sd", form["sd"], REFERENCE_FORM_LATENCY_SD_MS, 0.5)
    close("crew formation median", form["median"], REFERENCE_FORM_LATENCY_MEDIAN_MS, 0.5)
    close("crew formation p95", form["p95"], REFERENCE_FORM_LATENCY_P95_MS, 0.5)

    close("deployment events", ms.deployment_events(), N_DEPLOYMENT_EVENTS, 0)
    close("completed events", ms.completed_events(), N_COMPLETED_EVENTS, 0)

    var = ms.sensor_error_variances()
    close("sensor variance min", var["min_ppm2"], SENSOR_ERROR_VARIANCE_RANGE_PPM2[0], 1.0)
    close("sensor variance max", var["max_ppm2"], SENSOR_ERROR_VARIANCE_RANGE_PPM2[1], 1.0)

    hw = ms.node_scaling_fit("hardware")
    for key, want in NODE_SCALING_FIT_HW.items():
        close(f"node scaling (hardware) {key}", hw[key], want, max(abs(want) * 0.01, 0.01))
    so = ms.node_scaling_fit("scaleout")
    for key, want in NODE_SCALING_FIT_SCALEOUT.items():
        close(f"node scaling (scale-out) {key}", so[key], want, max(abs(want) * 0.01, 0.01))

    val = ms.simulator_validation()
    close("simulator MAPE %", val["mape_pct"], SIM_VALIDATION_MAPE_PCT, 0.05)
    close("simulator bias %", val["bias_pct"], SIM_VALIDATION_BIAS_PCT, 0.05)

    lab = ms.labeled_events()
    close("labeled events", lab["total"], N_TRIALS * EVENTS_PER_TRIAL, 0)
    close("labeled trials", lab["trials"], N_TRIALS, 0)

    # The Static Threshold row of Table 5 is the raw channel against the fixed
    # screening threshold, and must recompute from the deposit exactly.
    try:
        got = ms.threshold_baseline("Raw_Instantaneous_PPM", THRESHOLD_PPM)
        close("static threshold F1", got["f1"], 0.790, 0.002)
        close("static threshold FAR", got["far"], 0.165, 0.002)
    except ms.DatasetUnavailable as exc:
        problems.append(str(exc))

    # Both deposited ADAM runs over D1 must reconcile: the full-pipeline
    # benchmark (Table 5) and the trigger-gated deployment semantics
    # (D1_RawTrigger_Summary).
    try:
        gated = ms.gated_run_summary()
        close("gated run triggered events", gated["triggered"], 889, 0)
        close("gated run F1", gated["f1"], 0.8142, 0.002)
    except ms.DatasetUnavailable as exc:
        problems.append(str(exc))

    # Every system the code evaluates must appear in the deposit.
    deposited = set(ms.evaluated_systems())
    alias = {
        "adam_full": "ADAM_Full", "static_threshold": "Static_Threshold",
        "random_forest": "Random_Forest", "cloud_only": "Cloud_Only",
        "single_agent": "SingleAgent", "no_aggregator": "ADAM_NoAgg",
        "no_llm": "ADAM_NoLLM", "no_blockchain": "ADAM_NoBlockchain",
        "no_weaviate": "ADAM_NoWeaviate",
    }
    for key in SYSTEMS:
        want = alias.get(key)
        if want and want not in deposited:
            problems.append(f"system {key!r} is evaluated in code but absent from the deposit")

    if len(SYSTEMS) - 1 != N_COMPARISONS:
        problems.append(
            f"{len(SYSTEMS) - 1} systems are compared against the reference but "
            f"N_COMPARISONS = {N_COMPARISONS}; the Holm correction would use the "
            f"wrong family size"
        )

    return problems


if __name__ == "__main__":  # pragma: no cover
    issues = verify_against_manuscript()
    if issues:
        print("Config does NOT reproduce the manuscript:")
        for line in issues:
            print(f"  - {line}")
        raise SystemExit(1)
    print("Config reproduces every derived figure checked against the manuscript.")
