// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title DecisionLogger
 * @notice Immutable audit record for validated decisions (Section 3.1.3).
 *
 * @dev The ledger's scope, stated plainly in Section 3.1.3: it "secures records
 *      once written: it establishes what was decided and on what evidence, not
 *      whether the underlying measurement was true." Nothing here validates a
 *      sensor reading. A falsified reading that passes crew validation is
 *      recorded faithfully as a falsified reading.
 *
 *      degradedMode is stored on-chain so that threshold-only fallback
 *      decisions (Section 4.5.2) stay distinguishable from full-reasoning ones
 *      in the permanent record, not merely in local logs.
 */
contract DecisionLogger {
    struct Decision {
        uint256 decisionId;
        bytes32 eventId;
        uint256 methanePpm;
        string classification;     // ANOMALY | NORMAL
        uint256 confidenceScaled;  // confidence x 100
        string severity;
        string finalAction;
        bool governanceValid;
        uint256 quorumAchieved;
        uint256 quorumRequired;
        bool degradedMode;
        uint256 loggedAt;
        address loggedBy;
    }

    struct ConflictResolution {
        uint256[] conflictingDecisions;
        uint256 resolvedDecisionId;
        uint256 lambdaSeverityScaled;  // lambda_1 x 100
        uint256 resolvedAt;
    }

    mapping(uint256 => Decision) public decisions;
    uint256 public decisionCounter;

    mapping(bytes32 => uint256[]) public eventDecisions;
    mapping(uint256 => ConflictResolution) public conflicts;
    uint256 public conflictCounter;

    event DecisionLogged(
        uint256 indexed decisionId,
        bytes32 indexed eventId,
        uint256 methanePpm,
        string classification,
        bool governanceValid,
        bool degradedMode
    );
    event ConflictResolved(
        uint256 indexed conflictId,
        uint256 resolvedDecisionId,
        uint256 lambdaSeverityScaled
    );

    /// @notice Commit a decision tuple. Algorithm 1 line 20.
    function logDecision(
        bytes32 eventId,
        uint256 methanePpm,
        string calldata classification,
        uint256 confidenceScaled,
        string calldata severity,
        string calldata finalAction,
        bool governanceValid,
        uint256 quorumAchieved,
        uint256 quorumRequired,
        bool degradedMode
    ) external returns (uint256 decisionId) {
        require(confidenceScaled <= 100, "DecisionLogger: confidence is scaled by 100");

        decisionId = ++decisionCounter;
        decisions[decisionId] = Decision({
            decisionId: decisionId,
            eventId: eventId,
            methanePpm: methanePpm,
            classification: classification,
            confidenceScaled: confidenceScaled,
            severity: severity,
            finalAction: finalAction,
            governanceValid: governanceValid,
            quorumAchieved: quorumAchieved,
            quorumRequired: quorumRequired,
            degradedMode: degradedMode,
            loggedAt: block.timestamp,
            loggedBy: msg.sender
        });
        eventDecisions[eventId].push(decisionId);

        emit DecisionLogged(
            decisionId, eventId, methanePpm, classification, governanceValid, degradedMode
        );
    }

    /// @notice Record the outcome of Equation (5). Algorithm 1 line 18.
    function logConflictResolution(
        uint256[] calldata conflictingDecisions,
        uint256 resolvedDecisionId,
        uint256 lambdaSeverityScaled
    ) external returns (uint256 conflictId) {
        require(conflictingDecisions.length >= 2, "DecisionLogger: need >= 2 candidates");
        require(decisions[resolvedDecisionId].loggedAt != 0, "DecisionLogger: unknown winner");
        require(lambdaSeverityScaled < 100, "DecisionLogger: lambda_1 must be below 1");

        conflictId = ++conflictCounter;
        conflicts[conflictId] = ConflictResolution({
            conflictingDecisions: conflictingDecisions,
            resolvedDecisionId: resolvedDecisionId,
            lambdaSeverityScaled: lambdaSeverityScaled,
            resolvedAt: block.timestamp
        });
        emit ConflictResolved(conflictId, resolvedDecisionId, lambdaSeverityScaled);
    }

    function getEventDecisions(bytes32 eventId) external view returns (uint256[] memory) {
        return eventDecisions[eventId];
    }

    function getConflictCandidates(uint256 conflictId) external view returns (uint256[] memory) {
        return conflicts[conflictId].conflictingDecisions;
    }

    /// @notice Fraction of logged decisions made in degraded mode, scaled by 10000.
    function degradedRateScaled() external view returns (uint256) {
        if (decisionCounter == 0) return 0;
        uint256 degraded = 0;
        for (uint256 i = 1; i <= decisionCounter; i++) {
            if (decisions[i].degradedMode) degraded++;
        }
        return (degraded * 10000) / decisionCounter;
    }
}
