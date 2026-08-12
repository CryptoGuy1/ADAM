"""
experiments.run_conflict_sweep
==============================

Reproduces the conflict-resolution sensitivity analysis of Section 4.6 and
Figure 9.

Why this is analytic rather than empirical
------------------------------------------
Equation (5) never fired during the 459 deployment events: with four nodes
there were no overlapping crew jurisdictions to arbitrate. The rule is retained
for larger deployments, so its weighting is examined by generating synthetic
conflicts rather than by measuring field behavior. Section 4.6 says exactly
this, and the distinction matters - these figures characterize a mechanism, not
an observation.

What the sweep establishes
--------------------------
1. About 40% of conflicts are *contested*: the two criteria disagree, so the
   weighting can change the outcome. The remainder resolve identically for any
   lambda_1, because one candidate dominates on both severity and recency.

2. Under **pairwise** normalization, every contested pair flips at exactly
   lambda_1 = 0.5. Min-max over two points maps both terms to {0, 1}, so the
   comparison reduces to lambda_1 vs lambda_2 and the magnitude of the severity
   gap is destroyed. Every value in (0.5, 1) yields identical decisions - which
   makes the regime stable but uninformative.

3. Under **window** normalization, each candidate is scaled against the
   population of concurrent events, preserving gap magnitude. Agreement with
   the configured lambda_1 = 0.7 stays at or above 99% across a broad interval.

4. The parameter is bounded above: at lambda_1 = 1 the recency term vanishes
   and equal-severity conflicts cannot be separated, so lambda_1 must remain
   strictly below 1.

Usage
-----
    python -m experiments.run_conflict_sweep --n 20000 --out results/conflict/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from adam.config import (
    CONFLICT_SWEEP_N,
    DECISION_DEADLINE_S,
    LAMBDA_SEVERITY,
    SEED,
    SEVERITY_SCORES,
)
from adam.mechanisms import Candidate, flip_threshold, resolve_conflict

logger = logging.getLogger(__name__)


@dataclass
class SweepResult:
    normalization: str
    n_pairs: int
    n_contested: int
    contested_fraction: float
    #: lambda_1 -> fraction of decisions unchanged relative to the configured value
    agreement_curve: Dict[float, float]
    #: Widest interval over which agreement stays >= the reporting threshold
    stable_interval: Tuple[float, float]
    agreement_threshold: float
    #: Contested conflicts resolved in favor of severity at the configured value
    severity_favored_fraction: float
    flip_thresholds: List[float]

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["agreement_curve"] = {str(k): v for k, v in self.agreement_curve.items()}
        # The full flip-threshold list is large; keep summary statistics inline
        # and write the raw values separately.
        flips = self.flip_thresholds
        d["flip_thresholds"] = {
            "n": len(flips),
            "median": statistics.median(flips) if flips else None,
            "mean": statistics.fmean(flips) if flips else None,
            "p95": (
                sorted(flips)[int(0.95 * len(flips))] if len(flips) > 20 else None
            ),
            "max": max(flips) if flips else None,
        }
        return d


def generate_conflicts(
    n: int,
    seed: int = SEED,
    window_s: float = DECISION_DEADLINE_S,
) -> List[Tuple[Candidate, Candidate]]:
    """Generate synthetic conflict pairs. Section 4.6.

    Severities are drawn from the Decision Agent schema levels; event ages are
    drawn uniformly within the 30 s decision window, which is the interval over
    which two crews could plausibly hold competing recommendations.
    """
    rng = random.Random(seed)
    levels = [v for k, v in SEVERITY_SCORES.items() if k != "NONE"]
    pairs: List[Tuple[Candidate, Candidate]] = []
    for i in range(n):
        sev_a, sev_b = rng.choice(levels), rng.choice(levels)
        age_a, age_b = rng.uniform(0.1, window_s), rng.uniform(0.1, window_s)
        pairs.append(
            (
                Candidate(f"action_a_{i}", sev_a, -age_a, f"evt-a-{i}"),
                Candidate(f"action_b_{i}", sev_b, -age_b, f"evt-b-{i}"),
            )
        )
    return pairs


def run_sweep(
    pairs: Sequence[Tuple[Candidate, Candidate]],
    normalization: str,
    configured_lambda: float = LAMBDA_SEVERITY,
    grid: Optional[Sequence[float]] = None,
    agreement_threshold: float = 0.99,
) -> SweepResult:
    """Sweep lambda_1 and record how often the selected action changes."""
    if grid is None:
        grid = [i / 500.0 for i in range(0, 501)]  # 0.000 .. 1.000 step 0.002

    t_now = 0.0
    population: List[Candidate] = [c for pair in pairs for c in pair]
    pop_arg = population if normalization == "window" else None

    # Contested pairs and their analytic flip thresholds.
    #
    # The agreement curve follows from these directly, so it is computed
    # analytically rather than by re-resolving every pair at every grid point.
    # A contested pair's winner at lambda differs from its winner at the
    # configured value exactly when lambda and configured_lambda fall on
    # opposite sides of that pair's flip threshold; an uncontested pair never
    # differs. Brute-force sweeping is O(pairs x grid x population) and takes
    # minutes at n = 20,000; this is O(pairs + grid log pairs) and exact.
    flips: List[float] = []
    contested = 0
    for pair in pairs:
        ft = flip_threshold(
            pair[0], pair[1], t_now=t_now, normalization=normalization, population=pop_arg
        )
        if ft is not None:
            contested += 1
            flips.append(ft)

    sorted_flips = sorted(flips)
    n_pairs = len(pairs)

    def _count_between(a: float, b: float) -> int:
        """Flip thresholds strictly inside the open interval (a, b)."""
        import bisect

        lo, hi = (a, b) if a <= b else (b, a)
        return bisect.bisect_left(sorted_flips, hi) - bisect.bisect_right(
            sorted_flips, lo
        )

    curve: Dict[float, float] = {}
    for lam in grid:
        lam_c = min(max(lam, 0.0), 1.0)
        changed = _count_between(configured_lambda, lam_c)
        curve[round(lam_c, 4)] = (n_pairs - changed) / n_pairs if n_pairs else 1.0

    # Widest contiguous interval at or above the reporting threshold.
    lo = hi = None
    best: Tuple[float, float] = (configured_lambda, configured_lambda)
    for lam in sorted(curve):
        if curve[lam] >= agreement_threshold:
            if lo is None:
                lo = lam
            hi = lam
        else:
            if lo is not None and hi is not None and (hi - lo) > (best[1] - best[0]):
                best = (lo, hi)
            lo = hi = None
    if lo is not None and hi is not None and (hi - lo) > (best[1] - best[0]):
        best = (lo, hi)

    # Among contested conflicts, how often does severity win at the configured
    # weight? A contested pair resolves in favor of severity precisely when
    # the configured lambda_1 sits above that pair's flip threshold.
    severity_wins = sum(1 for ft in flips if configured_lambda > ft)

    return SweepResult(
        normalization=normalization,
        n_pairs=len(pairs),
        n_contested=contested,
        contested_fraction=contested / len(pairs) if pairs else 0.0,
        agreement_curve=curve,
        stable_interval=best,
        agreement_threshold=agreement_threshold,
        severity_favored_fraction=(severity_wins / contested if contested else 0.0),
        flip_thresholds=flips,
    )


def report(result: SweepResult, configured_lambda: float = LAMBDA_SEVERITY) -> str:
    lines = [
        f"Normalization regime: {result.normalization}",
        f"  conflict pairs generated : {result.n_pairs:,}",
        f"  contested (criteria disagree): {result.n_contested:,} "
        f"({result.contested_fraction:.1%})",
        f"  resolved identically for any lambda_1: "
        f"{1 - result.contested_fraction:.1%}",
        f"  stable interval at >= {result.agreement_threshold:.0%} agreement: "
        f"[{result.stable_interval[0]:.3f}, {result.stable_interval[1]:.3f}]",
        f"  contested conflicts favoring severity at lambda_1={configured_lambda}: "
        f"{result.severity_favored_fraction:.1%}",
    ]
    if result.flip_thresholds:
        uniq = set(round(x, 6) for x in result.flip_thresholds)
        if len(uniq) == 1:
            lines.append(
                f"  every contested pair flips at exactly lambda_1 = "
                f"{next(iter(uniq)):.3f}"
            )
        else:
            lines.append(
                f"  flip thresholds: median {statistics.median(result.flip_thresholds):.3f}, "
                f"max {max(result.flip_thresholds):.3f}"
            )
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Section 4.6 conflict sensitivity sweep")
    ap.add_argument("--n", type=int, default=CONFLICT_SWEEP_N)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--lambda1", type=float, default=LAMBDA_SEVERITY)
    ap.add_argument("--out", default="results/conflict")
    args = ap.parse_args()

    if not 0.0 < args.lambda1 < 1.0:
        ap.error(
            "lambda1 must lie strictly in (0, 1): at 1 the recency term "
            "vanishes and equal-severity conflicts cannot be separated "
            "(Section 4.6)."
        )

    os.makedirs(args.out, exist_ok=True)
    pairs = generate_conflicts(args.n, seed=args.seed)

    results: Dict[str, SweepResult] = {}
    for regime in ("window", "pairwise"):
        logger.info("sweeping lambda_1 under %s normalization ...", regime)
        res = run_sweep(pairs, regime, configured_lambda=args.lambda1)
        results[regime] = res
        print()
        print(report(res, args.lambda1))

    payload = {
        "configured_lambda_1": args.lambda1,
        "seed": args.seed,
        "n_pairs": args.n,
        "regimes": {k: v.to_dict() for k, v in results.items()},
    }
    out_path = os.path.join(args.out, "conflict_sweep.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)

    flips_path = os.path.join(args.out, "flip_thresholds_window.json")
    with open(flips_path, "w") as fh:
        json.dump(results["window"].flip_thresholds, fh)

    print(f"\nwrote {out_path}")
    print(f"wrote {flips_path}  (Figure 9(b) input)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
