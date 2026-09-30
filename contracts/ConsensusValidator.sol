// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./GovernanceRules.sol";
import "./CrewRegistry.sol";

/**
 * @title ConsensusValidator
 * @notice Optional on-chain collection of ADAM application-level class votes.
 *
 * @dev Ballots represent event classifications, not approval/refusal of a
 *      Decision Agent recommendation. `true` means ANOMALY and `false` means
 *      NORMAL. The Coordinator remains non-voting, so quorum is computed over
 *      the explicitly supplied eligible voting set, not over every registered
 *      crew role. A two-voter split therefore remains UNRESOLVED.
 *
 *      This contract is reference infrastructure. The Python event workflow
 *      also tallies the same class votes before policy validation; the
 *      governance contract repeats quorum checks as defense in depth.
 */
contract ConsensusValidator {
    uint8 public constant CLASS_UNRESOLVED = 0;
    uint8 public constant CLASS_NORMAL = 1;
    uint8 public constant CLASS_ANOMALY = 2;

    struct Vote {
        address voter;
        bool anomalyVote;
        uint256 timestamp;
    }

    struct ConsensusRequest {
        uint256 requestId;
        uint256 crewId;
        bytes32 eventId;
        address[] eligibleVoters;
        uint256 anomalyVotes;
        uint256 normalVotes;
        uint256 createdAt;
        uint256 resolvedAt;
        bool reached;
        uint8 finalClass;
    }

    GovernanceRules public immutable governance;
    CrewRegistry public immutable registry;

    mapping(uint256 => ConsensusRequest) public requests;
    mapping(uint256 => mapping(address => bool)) public hasVoted;
    mapping(uint256 => Vote[]) public votes;
    uint256 public requestCounter;

    event ConsensusRequested(uint256 indexed requestId, uint256 indexed crewId, bytes32 eventId);
    event VoteCast(uint256 indexed requestId, address indexed voter, bool anomalyVote);
    event ConsensusReached(
        uint256 indexed requestId,
        uint8 finalClass,
        uint256 supportingVotes,
        uint256 required
    );
    event ConsensusFailed(
        uint256 indexed requestId,
        uint256 anomalyVotes,
        uint256 normalVotes,
        uint256 required
    );

    constructor(address governanceAddress, address registryAddress) {
        governance = GovernanceRules(governanceAddress);
        registry = CrewRegistry(registryAddress);
    }

    function requestConsensus(
        uint256 crewId,
        bytes32 eventId,
        address[] calldata eligibleVoters
    ) external returns (uint256 requestId) {
        require(eligibleVoters.length >= 2, "ConsensusValidator: fewer than two voters");

        // The voting set must be a duplicate-free subset of the registered
        // crew. This prevents the non-voting Coordinator or an unrelated
        // address from being silently counted through an arbitrary list.
        address[] memory members = registry.getCrewMembers(crewId);
        require(members.length > 0, "ConsensusValidator: unknown crew");
        for (uint256 i = 0; i < eligibleVoters.length; i++) {
            bool member = false;
            for (uint256 m = 0; m < members.length; m++) {
                if (eligibleVoters[i] == members[m]) {
                    member = true;
                    break;
                }
            }
            require(member, "ConsensusValidator: voter is not a crew member");
            require(
                registry.isVotingAgent(eligibleVoters[i]),
                "ConsensusValidator: non-voting role in eligible set"
            );
            for (uint256 j = i + 1; j < eligibleVoters.length; j++) {
                require(eligibleVoters[i] != eligibleVoters[j], "ConsensusValidator: duplicate voter");
            }
        }

        requestId = ++requestCounter;
        ConsensusRequest storage r = requests[requestId];
        r.requestId = requestId;
        r.crewId = crewId;
        r.eventId = eventId;
        r.eligibleVoters = eligibleVoters;
        r.createdAt = block.timestamp;
        r.finalClass = CLASS_UNRESOLVED;

        emit ConsensusRequested(requestId, crewId, eventId);
    }

    function castVote(uint256 requestId, bool anomalyVote) external {
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
        require(eligible, "ConsensusValidator: caller is not an eligible voter");

        hasVoted[requestId][msg.sender] = true;
        votes[requestId].push(
            Vote({voter: msg.sender, anomalyVote: anomalyVote, timestamp: block.timestamp})
        );
        if (anomalyVote) {
            r.anomalyVotes += 1;
        } else {
            r.normalVotes += 1;
        }
        emit VoteCast(requestId, msg.sender, anomalyVote);
    }

    /// @notice Resolve the strict majority over the eligible voting set.
    function evaluate(uint256 requestId) external returns (bool reached) {
        ConsensusRequest storage r = requests[requestId];
        require(r.createdAt != 0, "ConsensusValidator: unknown request");
        require(r.resolvedAt == 0, "ConsensusValidator: request already resolved");

        uint256 required = governance.requiredQuorum(r.eligibleVoters.length);
        uint256 support = 0;
        if (r.anomalyVotes >= required) {
            r.finalClass = CLASS_ANOMALY;
            support = r.anomalyVotes;
            reached = true;
        } else if (r.normalVotes >= required) {
            r.finalClass = CLASS_NORMAL;
            support = r.normalVotes;
            reached = true;
        } else {
            r.finalClass = CLASS_UNRESOLVED;
            reached = false;
        }

        r.reached = reached;
        r.resolvedAt = block.timestamp;

        if (reached) {
            emit ConsensusReached(requestId, r.finalClass, support, required);
        } else {
            emit ConsensusFailed(requestId, r.anomalyVotes, r.normalVotes, required);
        }
    }

    function requiredFor(uint256 requestId) external view returns (uint256) {
        return governance.requiredQuorum(requests[requestId].eligibleVoters.length);
    }

    function voteCount(uint256 requestId) external view returns (uint256) {
        return votes[requestId].length;
    }
}
