"""
adam.mechanisms
===============

The manuscript's equations as pure functions, free of I/O and agent state.

    Equation (1)  trigger detection            -> :func:`trigger`
    Equation (2)  inverse-variance fusion      -> :func:`fuse_readings`
    Equation (4)  crew-agreement validation    -> :func:`quorum_satisfied`
    Equation (5)  conflict resolution          -> :func:`resolve_conflict`
    Equation (6)  end-to-end decision latency  -> ``StageLatencies.total_ms``

Keeping these mechanisms separable allows each manuscript equation to be
tested independently of agent and service I/O.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import THRESHOLD_PPM, quorum
from .schemas import SensorReading

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Equation (1): trigger detection
# ---------------------------------------------------------------------------


def trigger(methane_ppm: float, threshold_ppm: float = THRESHOLD_PPM) -> int:
    """Local screening rule tau(m_t^i).

    Returns 1 when the reading meets or exceeds the threshold, else 0.

    This is emphatically not the anomaly decision (Section 3.2.1); it only
    initiates crew formation. Conflating the two is what produces the
    degenerate ground truth that ``data/validate.py`` screens for.
    """
    return 1 if methane_ppm >= threshold_ppm else 0


# ---------------------------------------------------------------------------
# Equation (2): reliability-weighted multi-sensor aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionResult:
    """Outcome of Equation (2) plus the diagnostics the Aggregator reports."""

    fused_ppm: float
    weights: Dict[str, float]
    contributing_nodes: Tuple[str, ...]
    dispersion_ppm: float
    outliers: Tuple[str, ...] = ()

    @property
    def n_nodes(self) -> int:
        return len(self.contributing_nodes)


def max_detectable_z(n_nodes: int) -> float:
    """Largest weighted z-score any single node can attain in a crew of n.

    The dispersion in :func:`fuse_readings` is computed over the same set that
    contains the outlier, so an extreme reading inflates the denominator it is
    measured against. For n equally weighted nodes with one arbitrarily large
    deviation, the outlier's z-score converges to sqrt(n - 1) from below:

        n = 3  ->  1.414
        n = 4  ->  1.732
        n = 5  ->  2.000
        n = 8  ->  2.646

    Consequence: on the four-node testbed, an ``outlier_z`` of 2.0 or above can
    never fire, and the cross-node corroboration defense of Section 4.5.1 is
    silently disabled. The deployed value of 1.5 sits below the 1.732 ceiling
    with little margin, which is why :func:`fuse_readings` warns when the
    configured threshold approaches it.
    """
    if n_nodes < 2:
        return 0.0
    return math.sqrt(n_nodes - 1)


def fuse_readings(
    readings: Sequence[SensorReading],
    outlier_z: Optional[float] = None,
) -> FusionResult:
    """Reliability-weighted concentration estimate.

    Implements Equation (2):

        m_bar_t = sum_i (w_i * m_t^i) / sum_i w_i,   w_i = 1 / sigma_i^2

    where sigma_i^2 is the error variance of sensor i, estimated once from the
    residuals between each node's raw readings and the co-located NDIR
    reference across the labeled trials (Section 3.2.2).

    Parameters
    ----------
    readings
        Active-node readings. Each must carry a positive calibration variance.
    outlier_z
        When set, nodes whose reading deviates from the weighted mean by more
        than this many weighted standard deviations are reported in
        ``outliers``. This is the cross-node corroboration signal behind the
        90.0% attack detection rate in Section 4.5.1. Flagging does not remove
        the reading from the estimate - the Aggregator reports, the Coordinator
        and Decision agents act.

        Must sit below :func:`max_detectable_z` for the crew size or no node
        can ever be flagged; a threshold at or above the ceiling raises.

    Raises
    ------
    ValueError
        If no readings are supplied, the weights sum to zero, or ``outlier_z``
        exceeds the ceiling for this crew size.
    """
    if not readings:
        raise ValueError("Equation (2) is undefined over an empty node set")

    if outlier_z is not None and len(readings) >= 3:
        ceiling = max_detectable_z(len(readings))
        if outlier_z >= ceiling:
            raise ValueError(
                f"outlier_z={outlier_z} is at or above the maximum attainable "
                f"weighted z-score for {len(readings)} nodes ({ceiling:.3f}). "
                f"No node could ever be flagged, silently disabling the "
                f"cross-node corroboration defense of Section 4.5.1. Use a "
                f"threshold below {ceiling:.3f}, or add nodes."
            )
        if outlier_z > 0.9 * ceiling:
            logger.warning(
                "outlier_z=%.2f is within 10%% of the %.3f ceiling for %d "
                "nodes; detection sensitivity will be poor",
                outlier_z,
                ceiling,
                len(readings),
            )

    weights = {r.node_id: r.weight for r in readings}
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("fusion weights sum to zero; check calibration variances")

    fused = sum(w * r.methane_ppm for w, r in zip(weights.values(), readings)) / total_w

    # Weighted dispersion, used both as a confidence signal for the prompt and
    # as the basis for outlier flagging.
    var = sum(
        w * (r.methane_ppm - fused) ** 2 for w, r in zip(weights.values(), readings)
    ) / total_w
    dispersion = var**0.5

    outliers: Tuple[str, ...] = ()
    if outlier_z is not None and dispersion > 0 and len(readings) >= 3:
        outliers = tuple(
            r.node_id
            for r in readings
            if abs(r.methane_ppm - fused) / dispersion > outlier_z
        )

    return FusionResult(
        fused_ppm=fused,
        weights=weights,
        contributing_nodes=tuple(r.node_id for r in readings),
        dispersion_ppm=dispersion,
        outliers=outliers,
    )


# ---------------------------------------------------------------------------
# Equation (4): crew-agreement validation
# ---------------------------------------------------------------------------


def quorum_satisfied(approvals: int, crew_size: int) -> bool:
    """True when sum_i v_i(a_t) >= gamma_crew. Equation (4).

    Delegates the threshold to :func:`adam.config.quorum` so that the runtime,
    the Table 8 generator, and the Solidity parity test share one definition.
    """
    return approvals >= quorum(crew_size)


# ---------------------------------------------------------------------------
# Equation (5): deterministic conflict resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One competing recommendation entering Equation (5)."""

    action: str
    severity: float  # numeric value mapped from the ordered severity scale
    timestamp: float
    event_id: str = ""


def resolve_conflict(candidates: Sequence[Candidate]) -> Candidate:
    """Resolve concurrent recommendations by severity, then recency.

    Implements the manuscript's lexicographic rule:

        a* = lex max(severity(a), timestamp(a))

    Higher severity always takes precedence. The more recent recommendation
    is used only when severity is equal. event_id and action provide stable
    final tie-breaks when both severity and timestamp are identical.
    """
    if not candidates:
        raise ValueError("no candidates to resolve")

    return max(
        candidates,
        key=lambda c: (
            float(c.severity),
            float(c.timestamp),
            str(c.event_id),
            str(c.action),
        ),
    )


__all__ = [
    "trigger",
    "fuse_readings",
    "max_detectable_z",
    "FusionResult",
    "quorum_satisfied",
    "Candidate",
    "resolve_conflict",
]
