// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title CrewRegistry
 * @notice Agent identity and crew lifecycle (manuscript Section 3.1.3).
 *
 * @dev Vote attributability rests here. Table 8's tolerance bounds assume each
 *      ballot maps to one distinct registered agent; an adversary able to
 *      register unlimited identities defeats them regardless of quorum
 *      (Section 4.5.1). MAX_AGENTS_PER_NODE is the prototype's blunt limit on
 *      that, and it is a limit, not a solution: a node with several keys can
 *      still register several agents.
 */
contract CrewRegistry is Ownable {
    struct Agent {
        address nodeAddress;
        string role;              // sensor | aggregator | decision | coordinator
        bool active;
        uint256 reputationScore;  // 0-1000
        uint256 registeredAt;
        uint256 totalDecisions;
        uint256 correctDecisions;
    }

    struct Crew {
        uint256 crewId;
        uint256 formedAt;
        uint256 dissolvedAt;      // 0 while active
        address[] members;
        uint256 methanePpm;
        bytes32 eventId;
    }

    mapping(address => Agent) public agents;
    address[] public agentList;

    mapping(uint256 => Crew) public crews;
    uint256 public crewCounter;

    mapping(address => uint256) public nodeAgentCount;
    mapping(address => uint256) public nodeActiveCrews;

    /// @dev Four roles x two concurrent crews per node.
    uint256 public constant MAX_AGENTS_PER_NODE = 8;

    event AgentRegistered(address indexed agent, address indexed node, string role);
    event AgentDeactivated(address indexed agent);
    event CrewFormed(uint256 indexed crewId, bytes32 indexed eventId, address[] members, uint256 methanePpm);
    event CrewDissolved(uint256 indexed crewId, uint256 dissolvedAt);
    event ReputationUpdated(address indexed agent, uint256 oldScore, uint256 newScore);

    modifier onlyRegistered() {
        require(agents[msg.sender].active, "CrewRegistry: caller is not an active agent");
        _;
    }

    constructor() Ownable(msg.sender) {}

    function registerAgent(address agent, address node, string calldata role) external onlyOwner {
        require(agent != address(0), "CrewRegistry: zero agent address");
        require(agents[agent].registeredAt == 0, "CrewRegistry: agent already registered");
        require(nodeAgentCount[node] < MAX_AGENTS_PER_NODE, "CrewRegistry: node agent limit reached");

        agents[agent] = Agent({
            nodeAddress: node,
            role: role,
            active: true,
            reputationScore: 500,
            registeredAt: block.timestamp,
            totalDecisions: 0,
            correctDecisions: 0
        });
        agentList.push(agent);
        nodeAgentCount[node] += 1;
        emit AgentRegistered(agent, node, role);
    }

    function deactivateAgent(address agent) external onlyOwner {
        require(agents[agent].active, "CrewRegistry: agent not active");
        agents[agent].active = false;
        emit AgentDeactivated(agent);
    }

    /// @notice Record crew formation. Algorithm 1 line 8.
    function formCrew(bytes32 eventId, address[] calldata members, uint256 methanePpm)
        external
        onlyRegistered
        returns (uint256 crewId)
    {
        require(members.length >= 2, "CrewRegistry: crew below C4 minimum");
        for (uint256 i = 0; i < members.length; i++) {
            require(agents[members[i]].active, "CrewRegistry: inactive member");
            for (uint256 j = i + 1; j < members.length; j++) {
                require(members[i] != members[j], "CrewRegistry: duplicate member");
            }
        }

        crewId = ++crewCounter;
        crews[crewId] = Crew({
            crewId: crewId,
            formedAt: block.timestamp,
            dissolvedAt: 0,
            members: members,
            methanePpm: methanePpm,
            eventId: eventId
        });
        nodeActiveCrews[agents[msg.sender].nodeAddress] += 1;
        emit CrewFormed(crewId, eventId, members, methanePpm);
    }

    /// @notice Record crew dissolution. Algorithm 1 line 21.
    function dissolveCrew(uint256 crewId) external onlyRegistered {
        Crew storage c = crews[crewId];
        require(c.formedAt != 0, "CrewRegistry: unknown crew");
        require(c.dissolvedAt == 0, "CrewRegistry: crew already dissolved");
        c.dissolvedAt = block.timestamp;

        address node = agents[msg.sender].nodeAddress;
        if (nodeActiveCrews[node] > 0) {
            nodeActiveCrews[node] -= 1;
        }
        emit CrewDissolved(crewId, block.timestamp);
    }

    function updateReputation(address agent, bool wasCorrect) external onlyOwner {
        Agent storage a = agents[agent];
        require(a.registeredAt != 0, "CrewRegistry: unknown agent");
        uint256 old = a.reputationScore;
        a.totalDecisions += 1;
        if (wasCorrect) {
            a.correctDecisions += 1;
            a.reputationScore = old + 10 > 1000 ? 1000 : old + 10;
        } else {
            a.reputationScore = old < 20 ? 0 : old - 20;
        }
        emit ReputationUpdated(agent, old, a.reputationScore);
    }

    function getCrewMembers(uint256 crewId) external view returns (address[] memory) {
        return crews[crewId].members;
    }

    function getCrewSize(uint256 crewId) external view returns (uint256) {
        return crews[crewId].members.length;
    }

    function isCrewActive(uint256 crewId) external view returns (bool) {
        return crews[crewId].formedAt != 0 && crews[crewId].dissolvedAt == 0;
    }

    function agentCount() external view returns (uint256) {
        return agentList.length;
    }
}
