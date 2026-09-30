"""Source-level guards for classification-vote semantics in reference contracts.

Hardhat is optional in the Python verification environment, so these checks
protect the specific manuscript/code contracts that previously drifted without
requiring a Solidity compiler in every audit run.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_consensus_contract_counts_both_event_classes():
    sol = (REPO / "contracts" / "ConsensusValidator.sol").read_text()
    assert "uint256 anomalyVotes" in sol
    assert "uint256 normalVotes" in sol
    assert "r.anomalyVotes >= required" in sol
    assert "r.normalVotes >= required" in sol
    assert "CLASS_UNRESOLVED" in sol
    assert "r.approvals" not in sol
    assert "r.rejections" not in sol


def test_consensus_quorum_uses_eligible_voters_not_full_crew_size():
    sol = (REPO / "contracts" / "ConsensusValidator.sol").read_text()
    assert "governance.requiredQuorum(r.eligibleVoters.length)" in sol
    # A Coordinator may be a crew member but is not one of the classification voters.
    assert "registry.getCrewSize(r.crewId)" not in sol


def test_governance_contract_receives_final_class_support():
    sol = (REPO / "contracts" / "GovernanceRules.sol").read_text()
    assert "uint256 voterCount" in sol
    assert "uint256 classSupport" in sol
    assert "classSupport < requiredQuorum(voterCount)" in sol
    assert "uint256 approvals" not in sol


def test_consensus_eligible_set_excludes_nonvoting_roles():
    consensus = (REPO / "contracts" / "ConsensusValidator.sol").read_text()
    registry = (REPO / "contracts" / "CrewRegistry.sol").read_text()
    assert "registry.isVotingAgent(eligibleVoters[i])" in consensus
    assert "function isVotingAgent(address agent)" in registry
    assert 'keccak256(bytes("coordinator"))' in registry  # role is recognized
    # isVotingAgent itself returns only sensor/aggregator/decision.
    voting_body = registry.split("function isVotingAgent(address agent)", 1)[1].split("function getCrewMembers", 1)[0]
    assert 'keccak256(bytes("sensor"))' in voting_body
    assert 'keccak256(bytes("aggregator"))' in voting_body
    assert 'keccak256(bytes("decision"))' in voting_body
    assert 'keccak256(bytes("coordinator"))' not in voting_body
