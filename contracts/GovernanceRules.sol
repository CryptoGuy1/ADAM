// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title GovernanceRules
 * @notice Encodes the safety thresholds and validation rules of ADAM's
 *         Governance Layer (manuscript Section 3.1.3).
 *
 * @dev CHANGES FROM THE PRE-REVISION CONTRACT. Two defects made the deployed
 *      contract disagree with the manuscript. Both are corrected here.
 *
 *      (1) QUORUM RULE. The previous `getRequiredConsensus` computed
 *          `ceil(n * 51 / 100)`, a percentage-of-voters rule. The
 *          manuscript specifies strict majority, `gamma_crew = floor(n/2) + 1`, which is
 *          the consensus rule recorded in the dataset (01_Config) and in the
 *          deployment failure notes of 05_D2_Coordination_Log. The two rules
 *          agree at odd n but the percentage rule understates quorum at even
 *          n (n = 2 gives 2 vs 2, n = 4 gives 3 vs 3 only because ceil
 *          rounds up; n = 6 gives 4 vs 4; the divergence appears at
 *          fractional boundaries and made quorum depend on an arbitrary
 *          percentage constant rather than on the stated rule).
 *          `requiredQuorum` now implements the strict-majority rule directly, and the
 *          voter count it takes is the number of BALLOTING agents: the
 *          Coordinator tallies and does not vote, so the deployed crew of
 *          four agents supplies three ballots and a threshold of 2.
 *
 *      (2) SCREENING THRESHOLD. The previous constructor set
 *          criticalThreshold = 5000 and warningThreshold = 3000, with no
 *          screening threshold at all, while the manuscript's constraint C5
 *          specifies the 1,000 ppm experimental operating point.
 *          `screeningThreshold` is now first-class and defaults to 1,000 ppm.
 *
 *      The quorum rule is asserted against the Python implementation by
 *      tests/test_manuscript_parity.py, which fails CI if the two diverge.
 */
contract GovernanceRules is Ownable {
    // ==================== Thresholds (ppm) ====================

    /// @notice Constraint C5: local event-screening threshold. Initiates crew
    ///         formation; NOT the final anomaly decision rule.
    uint256 public screeningThreshold;

    /// @notice Concentration above which an event is treated as critical.
    uint256 public criticalThreshold;

    /// @notice Concentration above which an event is treated as a warning.
    uint256 public warningThreshold;

    // ==================== Crew parameters ====================

    /// @notice Constraint C4: minimum crew size for degraded-mode operation.
    uint256 public minCrewSize;

    /// @notice Coalescing window for repeat events at one location, seconds.
    uint256 public sameEventWindow;

    /// @notice Confidence floor, scaled by 100 (35 == 0.35).
    uint256 public minConfidenceScaled;

    /// @notice Minimum reputation an agent needs to vote. 0 disables the check.
    uint256 public minReputationScore;

    // ==================== Events ====================

    event ThresholdUpdated(string name, uint256 oldValue, uint256 newValue);
    event CrewParameterUpdated(string name, uint256 oldValue, uint256 newValue);

    // ==================== Constructor ====================

    constructor() Ownable(msg.sender) {
        screeningThreshold = 1000; // Constraint C5: experimental screening threshold
        warningThreshold = 3000; // 3x screening
        criticalThreshold = 5000; // policy critical band
        minCrewSize = 2; // Constraint C4
        sameEventWindow = 30; // matches the C1 deadline
        minConfidenceScaled = 35; // 0.35
        minReputationScore = 0;
    }

    // ==================== Classification quorum ====================

    /**
     * @notice Crew-level quorum threshold gamma_crew = floor(n/2) + 1.
     * @param crewSize Number of VOTING agents in the crew.
     * @return Matching votes required for either final event class.
     *
     * @dev Strict majority. Solidity integer division floors, so the
     *      expression is exactly the rule recorded in the dataset
     *      (01_Config: "strict majority = FLOOR(n/2)+1"). Reverts on
     *      crewSize == 0, which is not a crew.
     *
     *      At the deployed three voters the threshold is 2: any two of the
     *      three role-specific checks must agree. For every crewSize >= 2 the
     *      threshold is at least 2, so no single agent can unilaterally
     *      determine a final class (Section 3.2.4). The minCrewSize floor of 2
     *      keeps the runtime out of the one-voter regime, where a strict
     *      majority of one would be unilateral.
     */
    function requiredQuorum(uint256 crewSize) public pure returns (uint256) {
        require(crewSize > 0, "GovernanceRules: crew size must be positive");
        return crewSize / 2 + 1;
    }

    /**
     * @notice Compromised agents tolerable while honest agents retain quorum.
     * @dev Honest voters can still execute while n - f >= requiredQuorum(n),
     *      giving f <= ceil(n/2) - 1. At three voters this is 1: one
     *      compromised voter cannot block the remaining two. Integrity is
     *      bounded separately by isSubvertible: requiredQuorum(n) colluding
     *      voters can force an action.
     */
    function toleratedFaults(uint256 crewSize) public pure returns (uint256) {
        uint256 halfUp = (crewSize + 1) / 2; // ceil(n/2)
        return halfUp == 0 ? 0 : halfUp - 1;
    }

    /**
     * @notice True when votes supporting the final class meet quorum.
     */
    function quorumSatisfied(
        uint256 classSupport,
        uint256 voterCount
    ) external pure returns (bool) {
        return classSupport >= requiredQuorum(voterCount);
    }

    /**
     * @notice True when compromised agents alone could supply quorum.
     * @dev Analytical helper for colluding-voter bounds; not a measured BFT claim.
     */
    function isSubvertible(
        uint256 crewSize,
        uint256 compromised
    ) external pure returns (bool) {
        return compromised >= requiredQuorum(crewSize);
    }

    // ==================== Classification helpers ====================

    /// @notice Constraint C5: does this reading initiate crew formation?
    function triggersScreening(uint256 methanePpm) external view returns (bool) {
        return methanePpm >= screeningThreshold;
    }

    function isCritical(uint256 methanePpm) external view returns (bool) {
        return methanePpm >= criticalThreshold;
    }

    function isWarning(uint256 methanePpm) external view returns (bool) {
        return methanePpm >= warningThreshold && methanePpm < criticalThreshold;
    }

    function isSameEventWindow(
        uint256 timestamp1,
        uint256 timestamp2
    ) external view returns (bool) {
        uint256 diff = timestamp1 > timestamp2
            ? timestamp1 - timestamp2
            : timestamp2 - timestamp1;
        return diff <= sameEventWindow;
    }

    // ==================== Validation: V(d_t, S_t) ====================

    /**
     * @notice On-chain policy check, mirroring adam.governance.chain.LocalValidator.
     * @param methanePpm       Triggering concentration.
     * @param confidenceScaled Initial Decision-Agent score x 100. After a
     *                         crew class flip this is not final-class confidence.
     * @param severity         Severity label, e.g. "HIGH".
     * @param requiresReview   Whether the decision requests human review.
     * @param voterCount       Number of voting roles for this event.
     * @param classSupport     Votes matching the selected final class.
     * @return valid  Whether the action may execute.
     * @return reason Human-readable justification, recorded in the audit trace.
     */
    function validateDecision(
        uint256 methanePpm,
        uint256 confidenceScaled,
        string calldata classification,
        string calldata severity,
        string calldata recommendedAction,
        bool requiresReview,
        bool degradedMode,
        uint256 voterCount,
        uint256 classSupport
    ) external view returns (bool valid, string memory reason) {
        // Formation/quorum checks are kept here as defense in depth. The
        // Python Coordinator applies the same quorum rule before execution.
        if (voterCount < minCrewSize) {
            return (false, "voting set below minimum size (C4)");
        }
        if (classSupport < requiredQuorum(voterCount)) {
            return (false, "quorum not met");
        }

        // R1: recognized severity.
        if (!_isRecognizedSeverity(severity)) {
            return (false, "unrecognized severity");
        }

        // R2: autonomous-action confidence floor.
        if (confidenceScaled < minConfidenceScaled) {
            return (false, "confidence below policy floor");
        }

        // R3: the recommendation must contain an allowed policy action.
        if (!_isPermittedAction(recommendedAction)) {
            return (false, "action not permitted by policy");
        }

        // R4: at or above the critical concentration, passive-only actions
        // are rejected. An action containing an alert remains active even if
        // it also contains the word monitor. This mirrors LocalValidator.
        if (methanePpm >= criticalThreshold && _isPassiveOnly(recommendedAction)) {
            return (false, "passive action at critical concentration");
        }

        // R5: CRITICAL severity requires explicit human review.
        if (_eq(severity, "CRITICAL") && !requiresReview) {
            return (false, "CRITICAL severity requires human review");
        }

        // R6: degraded-mode anomalies are allowed only as explicitly marked
        // degraded decisions. `degradedMode` is subsequently persisted by
        // DecisionLogger, preventing fallback decisions from appearing as
        // full semantic-reasoning decisions in the audit record.
        if (degradedMode && _eq(classification, "ANOMALY")) {
            return (true, "approved in degraded mode; no semantic reasoning applied");
        }

        return (true, "policy satisfied");
    }

    function _eq(string memory a, string memory b) internal pure returns (bool) {
        return keccak256(bytes(a)) == keccak256(bytes(b));
    }

    function _contains(string memory haystack, string memory needle)
        internal
        pure
        returns (bool)
    {
        bytes memory h = bytes(haystack);
        bytes memory n = bytes(needle);
        if (n.length == 0 || n.length > h.length) return false;
        for (uint256 i = 0; i <= h.length - n.length; i++) {
            bool match_ = true;
            for (uint256 j = 0; j < n.length; j++) {
                bytes1 hc = h[i + j];
                bytes1 nc = n[j];
                // ASCII-only case folding is sufficient for the fixed policy
                // vocabulary used by the Decision Agent.
                if (hc >= 0x41 && hc <= 0x5A) hc = bytes1(uint8(hc) + 32);
                if (nc >= 0x41 && nc <= 0x5A) nc = bytes1(uint8(nc) + 32);
                if (hc != nc) {
                    match_ = false;
                    break;
                }
            }
            if (match_) return true;
        }
        return false;
    }

    function _isRecognizedSeverity(string memory severity) internal pure returns (bool) {
        return _eq(severity, "NONE") || _eq(severity, "LOW") ||
            _eq(severity, "MODERATE") || _eq(severity, "HIGH") ||
            _eq(severity, "CRITICAL");
    }

    function _isPermittedAction(string memory action) internal pure returns (bool) {
        return _contains(action, "monitor") ||
            _contains(action, "continue monitoring") ||
            _contains(action, "raise alert") ||
            _contains(action, "dispatch inspection") ||
            _contains(action, "escalate") ||
            _contains(action, "log only");
    }

    function _isPassiveOnly(string memory action) internal pure returns (bool) {
        bool passive = _contains(action, "monitor") || _contains(action, "log only");
        bool activeAlert = _contains(action, "alert");
        return passive && !activeAlert;
    }

    // ==================== Admin ====================

    function updateScreeningThreshold(uint256 v) external onlyOwner {
        require(v > 0 && v < warningThreshold, "GovernanceRules: threshold out of range");
        emit ThresholdUpdated("screening", screeningThreshold, v);
        screeningThreshold = v;
    }

    function updateWarningThreshold(uint256 v) external onlyOwner {
        require(v > screeningThreshold, "GovernanceRules: warning below screening");
        emit ThresholdUpdated("warning", warningThreshold, v);
        warningThreshold = v;
    }

    function updateCriticalThreshold(uint256 v) external onlyOwner {
        require(v > warningThreshold, "GovernanceRules: critical below warning");
        emit ThresholdUpdated("critical", criticalThreshold, v);
        criticalThreshold = v;
    }

    function updateMinCrewSize(uint256 v) external onlyOwner {
        require(v >= 2, "GovernanceRules: C4 requires at least 2 agents");
        emit CrewParameterUpdated("minCrewSize", minCrewSize, v);
        minCrewSize = v;
    }

    function updateSameEventWindow(uint256 v) external onlyOwner {
        emit CrewParameterUpdated("sameEventWindow", sameEventWindow, v);
        sameEventWindow = v;
    }

    function updateMinConfidence(uint256 v) external onlyOwner {
        require(v <= 100, "GovernanceRules: confidence is scaled by 100");
        emit CrewParameterUpdated("minConfidence", minConfidenceScaled, v);
        minConfidenceScaled = v;
    }
}
