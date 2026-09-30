"""Pure classification-voting and post-vote decision-alignment rules.

Reference implementation of the revised manuscript's strict class majority.
An unresolved ballot is *not* NORMAL. An invalid/absent vote cannot be
silently counted as a NORMAL vote. These rules contain no label/reference input.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Optional, Union

from .config import CRITICAL_THRESHOLD_PPM, WARNING_THRESHOLD_PPM, quorum
from .schemas import DecisionObject

NORMAL = "NORMAL"
ANOMALY = "ANOMALY"
UNRESOLVED = "UNRESOLVED"


def normalize_class_vote(value: Union[str, int]) -> int:
    """Validate one explicit class vote; reject ambiguous Boolean approvals."""
    if type(value) is int and value in (0, 1):
        return value
    if isinstance(value, str) and value.strip().upper() in (NORMAL, ANOMALY):
        return 1 if value.strip().upper() == ANOMALY else 0
    raise ValueError(f"classification vote must be NORMAL/ANOMALY or integer 0/1, got {value!r}")


def majority_class(votes: Mapping[str, int], expected_voters: int) -> tuple[str, int, int]:
    """Return (class, agreeing votes, quorum) using the *formed* voter count.

    Missing votes do not shrink quorum. For a two-voter split, neither class
    reaches the required two matching votes; return UNRESOLVED.
    """
    if expected_voters < 2:
        raise ValueError("at least two voting roles are required")
    if len(votes) > expected_voters:
        raise ValueError("more votes than the formed voting set")
    values = [normalize_class_vote(v) for v in votes.values()]
    required = quorum(expected_voters)
    yes = sum(values)
    no = len(values) - yes
    if yes >= required:
        return ANOMALY, yes, required
    if no >= required:
        return NORMAL, no, required
    return UNRESOLVED, max(yes, no), required


def align_decision_to_crew(
    initial: DecisionObject, final_class: str, fused_ppm: float,
    agreeing_votes: int, expected_voters: int,
) -> tuple[DecisionObject, str]:
    """Produce the governed decision without modifying the initial output.

    The original model confidence is retained even when the class changes. It
    describes the initial class, never the crew-selected class. The agreement
    fraction is stored separately on EventTrace as crew_support.
    """
    if final_class not in (NORMAL, ANOMALY):
        raise ValueError("cannot align a decision without classification quorum")
    if expected_voters < 2 or agreeing_votes < quorum(expected_voters):
        raise ValueError("cannot align a decision without strict majority")
    if initial.classification == final_class:
        return replace(initial, contributing_factors=list(initial.contributing_factors)), "decision_agent"

    if final_class == NORMAL:
        severity, action = "NONE", "continue monitoring"
    elif fused_ppm >= CRITICAL_THRESHOLD_PPM:
        severity, action = "CRITICAL", "escalate"
    elif fused_ppm >= WARNING_THRESHOLD_PPM:
        severity, action = "HIGH", "dispatch inspection"
    else:
        severity, action = "LOW", "raise alert"

    final = replace(
        initial,
        classification=final_class,
        severity=severity,
        recommended_action=action,
        reasoning=(
            "Crew majority revised the initial Decision-Agent classification "
            f"from {initial.classification} to {final_class}; the original "
            "reasoning remains available in the initial decision trace."
        ),
        requires_human_review=(initial.requires_human_review or severity == "CRITICAL"),
        contributing_factors=list(initial.contributing_factors) + [
            f"crew class {final_class}: {agreeing_votes}/{expected_voters} matching votes"
        ],
    )
    return final, "decision_agent_initial_class"
