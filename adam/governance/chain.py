"""
adam.governance.chain
=====================

The Governance Layer: policy validation at event time and immutable trace
logging (Section 3.1.3).

Two separable concerns, deliberately kept apart:

    GovernanceValidator  evaluates V(d_t, S_t) - the policy check
    ChainClient          commits the validated decision to the PoA ledger

Section 3.1.3 draws the same line: "at the application layer, participating
agents vote on the proposed action ... at the ledger layer, validated decisions
are committed through the PoA blockchain, which provides immutable logging and
trace integrity rather than deciding the action itself."

And the limit, stated plainly in Section 3.1.3: "The ledger secures records
once written: it establishes what was decided and on what evidence, not whether
the underlying measurement was true."

LocalValidator mirrors the Solidity ``GovernanceRules`` exactly so the offline
harness and the on-chain deployment cannot diverge;
``tests/test_manuscript_parity.py`` checks them against each other.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

from ..config import (
    CHAIN_RPC_URL,
    CONTRACT_ADDRESSES,
    SAME_EVENT_WINDOW_S,
    SEVERITY_LEVELS,
    THRESHOLD_PPM,
    quorum,
)
from ..schemas import CrewEvent, DecisionObject, EventTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------


@dataclass
class GovernancePolicy:
    """The policy state S_t against which V(.) evaluates.

    Mirrors the storage of the Solidity ``GovernanceRules`` contract. Defaults
    match the deployed configuration confirmed for the 72-hour run.
    """

    #: Screening threshold. Contract: ``screeningThreshold``.
    screening_threshold_ppm: float = THRESHOLD_PPM

    #: Concentration above which an event is critical. 5x the screening
    #: threshold, i.e. 10% of the LEL.
    critical_threshold_ppm: float = 5000.0

    #: Concentration above which an event is a warning. 3x screening.
    warning_threshold_ppm: float = 3000.0

    #: Coalescing window for repeat events at one location, seconds.
    same_event_window_s: int = SAME_EVENT_WINDOW_S

    #: Minimum crew size. Constraint C4.
    min_crew_size: int = 2

    #: Confidence floor below which an autonomous action is refused and the
    #: event escalates to human review instead.
    min_confidence: float = 0.35

    #: Actions an agent may recommend without operator confirmation.
    permitted_actions: Tuple[str, ...] = (
        "monitor",
        "continue monitoring",
        "raise alert",
        "dispatch inspection",
        "escalate",
        "log only",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "screening_threshold_ppm": self.screening_threshold_ppm,
            "critical_threshold_ppm": self.critical_threshold_ppm,
            "warning_threshold_ppm": self.warning_threshold_ppm,
            "same_event_window_s": self.same_event_window_s,
            "min_crew_size": self.min_crew_size,
            "min_confidence": self.min_confidence,
        }


class Validator(Protocol):
    def validate(self, decision: DecisionObject, event: CrewEvent) -> Tuple[bool, str]: ...


class LocalValidator:
    """Off-chain implementation of V(d_t, S_t).

    Every rule here has a counterpart in ``contracts/GovernanceRules.sol``. The
    parity test asserts identical outcomes over a shared fixture set, so a
    reviewer running offline gets the same validation decisions as one running
    against the testnet.
    """

    def __init__(self, policy: Optional[GovernancePolicy] = None):
        self.policy = policy or GovernancePolicy()
        self._recent: List[Tuple[str, float]] = []  # (location, timestamp)

    def validate(self, decision: DecisionObject, event: CrewEvent) -> Tuple[bool, str]:
        """Return (valid, reason). Both are written into the audit trace."""
        p = self.policy

        # R1: severity must be a recognized level.
        if decision.severity not in SEVERITY_LEVELS:
            return False, f"unrecognized severity {decision.severity!r}"

        # R2: confidence floor for autonomous action.
        if decision.confidence < p.min_confidence:
            return False, (
                f"confidence {decision.confidence:.2f} below policy floor "
                f"{p.min_confidence:.2f}"
            )

        # R3: recommended action must fall within the permitted set.
        action = decision.recommended_action.lower()
        if not any(a in action for a in p.permitted_actions):
            return False, f"action not permitted by policy: {decision.recommended_action!r}"

        # R4: a critical concentration may not resolve to a passive action.
        if event.trigger_ppm >= p.critical_threshold_ppm:
            passive = ("monitor", "log only")
            if any(a in action for a in passive) and "alert" not in action:
                return False, (
                    f"passive action at critical concentration "
                    f"{event.trigger_ppm:.0f} ppm"
                )

        # R5: a CRITICAL severity call must request human review.
        if decision.severity == "CRITICAL" and not decision.requires_human_review:
            return False, "CRITICAL severity requires human review"

        # R6: degraded-mode anomalies are permitted but flagged, never silent.
        if decision.degraded_mode and decision.is_anomaly:
            self._note(event)
            return True, "approved in degraded mode; no semantic reasoning applied"

        self._note(event)
        return True, "policy satisfied"

    def _note(self, event: CrewEvent) -> None:
        now = event.timestamp
        cutoff = now - self.policy.same_event_window_s
        self._recent = [(l, t) for (l, t) in self._recent if t >= cutoff]
        self._recent.append((event.location, now))

    def is_repeat_event(self, event: CrewEvent) -> bool:
        """True when this location fired inside the coalescing window.

        Exposed for the concurrency harness: repeat events at one location are
        what produce the overlapping crew jurisdictions Equation (5) exists for.
        """
        cutoff = event.timestamp - self.policy.same_event_window_s
        return any(l == event.location and t >= cutoff for (l, t) in self._recent)


# ---------------------------------------------------------------------------
# Ledger client
# ---------------------------------------------------------------------------


class ChainClient(Protocol):
    def log_decision(
        self,
        event: CrewEvent,
        decision: DecisionObject,
        final_action: Optional[str],
        outcome: Any,
    ) -> Optional[str]: ...


class NullChainClient:
    """No-op ledger for the ADAM-No-Blockchain ablation and offline trials.

    Returns ``None`` for every write, so ``EventTrace.persisted_chain`` stays
    false and trace completeness reflects only what the memory store recorded.
    """

    def log_decision(self, event, decision, final_action, outcome) -> Optional[str]:
        return None

    def close(self) -> None:
        pass


class FidesInnovaClient:
    """web3.py client for the Fides Innova PoA testnet.

    Writes go to ``DecisionLogger.logDecision``. The transaction hash is
    recorded in the trace, giving a reviewer a direct path from a row in the
    dataset to an on-chain record.

    ``web3`` is imported lazily so the offline harness does not require it.
    """

    def __init__(
        self,
        rpc_url: str = CHAIN_RPC_URL,
        private_key: Optional[str] = None,
        addresses: Optional[Dict[str, str]] = None,
        abi_dir: str = "blockchain/artifacts",
        wait_for_receipt: bool = True,
    ):
        self.rpc_url = rpc_url
        self.private_key = private_key
        self.addresses = addresses or dict(CONTRACT_ADDRESSES)
        self.abi_dir = abi_dir
        self.wait_for_receipt = wait_for_receipt
        self._w3: Optional[Any] = None
        self._account: Optional[Any] = None
        self._contracts: Dict[str, Any] = {}

    def connect(self) -> None:
        try:
            from web3 import Web3  # type: ignore
            from web3.middleware import ExtraDataToPOAMiddleware  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "web3 is not installed. `pip install web3`, or use "
                "NullChainClient for offline runs."
            ) from exc

        w3 = Web3(Web3.HTTPProvider(self.rpc_url, request_kwargs={"timeout": 15}))
        # PoA chains carry extended extraData that the default validator rejects.
        w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        if not w3.is_connected():
            raise RuntimeError(f"cannot reach the PoA node at {self.rpc_url}")
        self._w3 = w3
        if self.private_key:
            self._account = w3.eth.account.from_key(self.private_key)
        logger.info("connected to chain id %s at %s", w3.eth.chain_id, self.rpc_url)

    def _load_contract(self, name: str) -> Any:
        if name in self._contracts:
            return self._contracts[name]
        import os

        path = os.path.join(self.abi_dir, f"{name}.json")
        with open(path, "r") as fh:
            artifact = json.load(fh)
        abi = artifact["abi"] if isinstance(artifact, dict) and "abi" in artifact else artifact
        address = self.addresses.get(name)
        if not address:
            raise RuntimeError(
                f"no deployed address for {name}; run scripts/deploy.js and "
                f"export ADAM_ADDR_* or pass addresses= explicitly"
            )
        assert self._w3 is not None
        contract = self._w3.eth.contract(
            address=self._w3.to_checksum_address(address), abi=abi
        )
        self._contracts[name] = contract
        return contract

    def log_decision(
        self,
        event: CrewEvent,
        decision: DecisionObject,
        final_action: Optional[str],
        outcome: Any,
    ) -> Optional[str]:
        """Commit the decision tuple. Algorithm 1 line 20."""
        if self._w3 is None:
            self.connect()
        assert self._w3 is not None

        try:
            logger_contract = self._load_contract("DecisionLogger")
            fn = logger_contract.functions.logDecision(
                int(event.event_id.replace("evt-", "")[:8] or "0", 16),
                int(round(event.trigger_ppm)),
                decision.classification,
                int(round(decision.confidence * 100)),
                decision.severity,
                (final_action or "")[:200],
                bool(getattr(outcome, "governance_valid", False)),
                int(getattr(outcome, "quorum_achieved", 0)),
                int(getattr(outcome, "quorum_required", 0)),
                bool(decision.degraded_mode),
            )

            if self._account is None:
                raise RuntimeError("no signing key configured for chain writes")

            tx = fn.build_transaction(
                {
                    "from": self._account.address,
                    "nonce": self._w3.eth.get_transaction_count(self._account.address),
                    "gas": 500_000,
                    "gasPrice": self._w3.eth.gas_price,
                    "chainId": self._w3.eth.chain_id,
                }
            )
            signed = self._account.sign_transaction(tx)
            tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
            if self.wait_for_receipt:
                self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=20)
            return tx_hash.hex()
        except Exception:
            logger.exception("chain write failed for event %s", event.event_id)
            return None

    def close(self) -> None:
        self._w3 = None
        self._contracts.clear()


__all__ = [
    "GovernancePolicy",
    "LocalValidator",
    "NullChainClient",
    "FidesInnovaClient",
]
