// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./GovernanceRules.sol";
import "./CrewRegistry.sol";

/**
 * @title ConsensusValidator
 * @notice On-chain vote collection and quorum evaluation (Equation 4).
 *
 * @dev Enforces two properties the Table 8 bounds depend on:
 *        1. one ballot per agent per request (hasVoted)
 *        2. only crew members may vote (checked against CrewRegistry)
 *      Without either, quorum can be inflated by a single actor and the
 *      tolerance analysis does not hold.
 *
 *      The quorum threshold is not duplicated here; it is read from
 *      GovernanceRules.requiredQuorum so there is exactly one definition
 *      on-chain, matching adam.config.quorum off-chain.
 */
contract ConsensusValidator {
    struct Vote {
        address voter;
        bool approve;
        uint256 timestamp;
    }

    struct ConsensusRequest {
        uint256 requestId;
        uint256 crewId;
        bytes32 eventId;
        address[] eligibleVoters;
        uint256 approvals;
        uint256 rejections;
        uint256 createdAt;
        uint256 resolvedAt;
        bool reached;
    }

    GovernanceRules public immutable governance;
    CrewRegistry public immutable registry;

    mapping(uint256 => ConsensusRequest) public requests;
    mapping(uint256 => mapping(address => bool)) public hasVoted;
    mapping(uint256 => Vote[]) public votes;
    uint256 public requestCounter;

    event ConsensusRequested(uint256 indexed requestId, uint256 indexed crewId, bytes32 eventId);
    event VoteCast(uint256 indexed requestId, address indexed voter, bool approve);
    event ConsensusReached(uint256 indexed requestId, uint256 approvals, uint256 required);
    event ConsensusFailed(uint256 indexed requestId, uint256 approvals, uint256 required);

    constructor(address governanceAddress, address registryAddress) {
        governance = GovernanceRules(governanceAddress);
        registry = CrewRegistry(registryAddress);
    }

    function requestConsensus(uint256 crewId, bytes32 eventId, address[] calldata eligibleVoters)
        external
        returns (uint256 requestId)
    {
        require(eligibleVoters.length > 0, "ConsensusValidator: no eligible voters");

        requestId = ++requestCounter;
        ConsensusRequest storage r = requests[requestId];
        r.requestId = requestId;
        r.crewId = crewId;
        r.eventId = eventId;
        r.eligibleVoters = eligibleVoters;
        r.createdAt = block.timestamp;

        emit ConsensusRequested(requestId, crewId, eventId);
    }

    function castVote(uint256 requestId, bool approve) external {
        ConsensusRequest storage r = requests[requestId];
        require(r.createdAt != 0, "ConsensusValidator: unknown request");
        require(r.resolvedAt == 0, "ConsensusValidator: request already resolved");
        require(!hasVoted[requestId][msg.sender], "ConsensusValidator: agent already voted");

        bool eligible = false;
        for (uint256 i = 0; i < r.eligibleVoters.length; i++) {
            if (r.eligibleVoters[i] == msg.sender) {
                eligible = true;
                break;
            }
        }
        require(eligible, "ConsensusValidator: caller is not a crew member");

        hasVoted[requestId][msg.sender] = true;
        votes[requestId].push(Vote({voter: msg.sender, approve: approve, timestamp: block.timestamp}));
        if (approve) {
            r.approvals += 1;
        } else {
            r.rejections += 1;
        }
        emit VoteCast(requestId, msg.sender, approve);
    }

    /// @notice Evaluate Equation (4) against the crew size on record.
    function evaluate(uint256 requestId) external returns (bool reached) {
        ConsensusRequest storage r = requests[requestId];
        require(r.createdAt != 0, "ConsensusValidator: unknown request");
        require(r.resolvedAt == 0, "ConsensusValidator: request already resolved");

        uint256 crewSize = registry.getCrewSize(r.crewId);
        if (crewSize == 0) {
            crewSize = r.eligibleVoters.length;
        }
        uint256 required = governance.requiredQuorum(crewSize);

        reached = r.approvals >= required;
        r.reached = reached;
        r.resolvedAt = block.timestamp;

        if (reached) {
            emit ConsensusReached(requestId, r.approvals, required);
        } else {
            emit ConsensusFailed(requestId, r.approvals, required);
        }
    }

    function requiredFor(uint256 requestId) external view returns (uint256) {
        uint256 crewSize = registry.getCrewSize(requests[requestId].crewId);
        if (crewSize == 0) crewSize = requests[requestId].eligibleVoters.length;
        return governance.requiredQuorum(crewSize);
    }

    function voteCount(uint256 requestId) external view returns (uint256) {
        return votes[requestId].length;
    }
}
