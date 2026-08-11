"""
experiments.run_deployment
==========================

The 72-hour continuous deployment (D2) and the concurrency sweep of Section 4.4.

Two harnesses share this module because they measure the same thing under
different load:

    run_deployment   continuous monitoring, one event at a time  -> Figure 5, Table 6
    run_scalability  node-count scaling under fixed load         -> Table 7, Figure 7

What D2 produces
----------------
459 EventTrace records with six per-stage latencies each (Equation 6), resource
counters, and completeness flags. Section 4.2's figures are readouts of these:
mean 18.99 s end-to-end, 81.5% in reasoning, 97.2% trace persistence.

On the scalability design
-------------------------
The scalability study varies the number of participating nodes N while holding
load fixed at the reference configuration (4 concurrent events, 8 sensor
streams, 30,000 vectors), so latency changes attribute to node count. N = 1-4
runs on physical Raspberry Pi 5 hardware; N = 6, 8, 12 and 16 come from a
Python scale-out model validated against the matched N = 1-4 hardware runs
(decision-latency MAPE 2.4%). Table 7 fits T(N) = T0 + alpha * N^beta to each
domain. This harness reproduces the in-process analogue of that design: each
event carries N sensor streams, so fusion, corroboration and vote collection
grow with N while the reasoning stage stays per-event.

Usage
-----
    python -m experiments.run_deployment --data data/artifacts/d1_simulated.csv \
        --events 459 --out results/deployment/
    python -m experiments.run_deployment --data data/artifacts/d1_simulated.csv \
        --scalability --node-counts 1 2 3 4 6 8 12 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from adam.config import (
    ADAMConfig,
    DEPLOYMENT_HOURS,
    N_DEPLOYMENT_EVENTS,
    NODE_SCALING_FIT_HW,
    NODE_SCALING_FIT_SCALEOUT,
    REFERENCE_STAGE_LATENCY_MS,
    SEED,
    DECISION_DEADLINE_S,
)
from adam.crew import ADAMNode
from adam.governance.chain import LocalValidator, NullChainClient
from adam.llm.client import OllamaClient
from adam.memory.store import InMemoryStore
from adam.schemas import EventTrace, SensorReading
from adam.telemetry import SustainedCPUMonitor

from data.loader import Dataset, load_trials, save_deployment_traces

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# D2: continuous deployment
# ---------------------------------------------------------------------------


def run_deployment(
    dataset: Dataset,
    config: ADAMConfig,
    n_events: int = N_DEPLOYMENT_EVENTS,
    llm_client: Optional[Any] = None,
    sample_resources: bool = True,
    seed: int = SEED,
) -> List[EventTrace]:
    """Replay triggering events through the full pipeline, collecting traces."""
    rng = random.Random(seed)
    memory = InMemoryStore()
    node = ADAMNode(
        node_id="node-01",
        config=config,
        memory=memory,
        chain=NullChainClient(),
        validator=LocalValidator(),
        llm_client=llm_client,
    )

    # Warm the baseline on sub-threshold readings, as a live node would be.
    for e in dataset.events:
        if e.primary.methane_ppm < config.threshold_ppm:
            node.sensor.observe(e.primary)

    triggering = [
        e for e in dataset.events if e.primary.methane_ppm >= config.threshold_ppm
    ]
    if not triggering:
        raise RuntimeError("no triggering events in the dataset")

    cpu_monitor = SustainedCPUMonitor()
    if sample_resources:
        cpu_monitor.start()

    traces: List[EventTrace] = []
    try:
        for i in range(n_events):
            event_src = triggering[i % len(triggering)]
            reading = event_src.primary
            crew_event = node.sensor.publish_trigger(reading)
            try:
                trace = node.handle_event(
                    crew_event, list(event_src.readings), sample_resources=sample_resources
                )
                traces.append(trace)
            except Exception:
                logger.exception("event %d failed", i)
            if (i + 1) % 50 == 0:
                logger.info("  %d/%d events", i + 1, n_events)
    finally:
        if sample_resources:
            cpu_monitor.stop()

    if sample_resources:
        logger.info(
            "sustained CPU over the run: %.1f%% (C2 budget %.0f%%, satisfied=%s)",
            cpu_monitor.sustained_pct,
            config.max_sustained_cpu * 100,
            cpu_monitor.satisfies_c2(),
        )
    return traces


def summarize_deployment(
    traces: Sequence[EventTrace],
    deadline_s: float = DECISION_DEADLINE_S,
) -> Dict[str, Any]:
    """Figure 5 and Table 6 quantities, computed from the traces."""
    if not traces:
        return {}

    stages = ("T_form", "T_agg", "T_reason", "T_gov", "T_weav", "T_bc")
    per_stage: Dict[str, List[float]] = {s: [] for s in stages}
    totals: List[float] = []
    for t in traces:
        d = t.latencies.to_dict()
        for s in stages:
            per_stage[s].append(d[s])
        totals.append(t.latencies.total_ms)

    total_mean = statistics.fmean(totals)
    stage_summary = {}
    for s in stages:
        vals = per_stage[s]
        stage_summary[s] = {
            "mean_ms": statistics.fmean(vals),
            "sd_ms": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "median_ms": statistics.median(vals),
            "share": statistics.fmean(vals) / total_mean if total_mean else 0.0,
            "reference_mean_ms": REFERENCE_STAGE_LATENCY_MS.get(s),
        }

    complete = sum(1 for t in traces if t.is_complete())
    within = sum(1 for t in traces if t.within_deadline(deadline_s))
    degraded = sum(1 for t in traces if t.degraded_mode)
    executed = sum(1 for t in traces if t.executed)

    sorted_totals = sorted(totals)
    return {
        "n_events": len(traces),
        "end_to_end_ms": {
            "mean": total_mean,
            "sd": statistics.stdev(totals) if len(totals) > 1 else 0.0,
            "median": statistics.median(totals),
            "p95": sorted_totals[int(0.95 * len(sorted_totals))],
            "max": max(totals),
        },
        "stages": stage_summary,
        "trace_persistence": complete / len(traces),
        "within_deadline": within / len(traces),
        "degraded_fraction": degraded / len(traces),
        "executed_fraction": executed / len(traces),
        "reference": {
            "mean_s": 18.9,
            "trace_persistence": 0.972,
            "reason_share": 0.814,
        },
    }


# ---------------------------------------------------------------------------
# Scalability
# ---------------------------------------------------------------------------


def fit_power_law(
    node_counts: Sequence[int], latency_ms: Sequence[float]
) -> Dict[str, float]:
    """Fit T(N) = T0 + alpha * N^beta by least squares. Table 7.

    Falls back to a two-parameter fit on log-transformed data when scipy is
    unavailable, which is adequate for reporting beta.
    """
    xs = [float(x) for x in node_counts]
    ys = [float(y) for y in latency_ms]

    try:
        import numpy as np
        from scipy.optimize import curve_fit

        def model(N, alpha, beta, T0):
            return alpha * np.power(N, beta) + T0

        p0 = [NODE_SCALING_FIT_HW["alpha"], 1.0, min(ys)]
        popt, pcov = curve_fit(
            model, np.array(xs), np.array(ys), p0=p0, maxfev=20000
        )
        alpha, beta, t0 = (float(v) for v in popt)
        pred = [alpha * (x**beta) + t0 for x in xs]
        ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
        ss_tot = sum((y - statistics.fmean(ys)) ** 2 for y in ys)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        perr = [float(v) for v in np.sqrt(np.diag(pcov))]
        return {
            "alpha": alpha,
            "beta": beta,
            "T0": t0,
            "r_squared": r2,
            "alpha_se": perr[0],
            "beta_se": perr[1],
            "T0_se": perr[2],
        }
    except ImportError:  # pragma: no cover
        import math

        base = min(ys)
        pts = [(math.log(x), math.log(max(y - base, 1e-6))) for x, y in zip(xs, ys) if x > 0]
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        num = sum((p[0] - mx) * (p[1] - my) for p in pts)
        den = sum((p[0] - mx) ** 2 for p in pts)
        beta = num / den if den else 1.0
        alpha = math.exp(my - beta * mx)
        return {"alpha": alpha, "beta": beta, "T0": base, "r_squared": float("nan")}


def run_scalability(
    dataset: Dataset,
    config: ADAMConfig,
    node_counts: Sequence[int] = (1, 2, 3, 4, 6, 8, 12, 16),
    events_per_run: int = 4,
    repeats: int = 5,
    llm_client: Optional[Any] = None,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Measure end-to-end latency against participating node count.

    Load is held at the reference configuration while N varies, mirroring the
    deposited design: each event carries N sensor streams, so fusion,
    cross-node corroboration and vote collection grow with N while the
    reasoning stage stays per-event. On the physical testbed N = 1-4 are
    hardware runs and N > 4 comes from the scale-out model; in this in-process
    harness every N is a software run, so the fitted exponents characterize
    this container and are compared against, not substituted for, the
    deposited fits.
    """
    rng = random.Random(seed)
    triggering = [
        e for e in dataset.events if e.primary.methane_ppm >= config.threshold_ppm
    ]
    if not triggering:
        raise RuntimeError("no triggering events available")

    memory = InMemoryStore()
    node = ADAMNode(
        node_id="node-scale",
        config=config,
        memory=memory,
        chain=NullChainClient(),
        validator=LocalValidator(),
        llm_client=llm_client,
    )
    for e in dataset.events:
        if e.primary.methane_ppm < config.threshold_ppm:
            node.sensor.observe(e.primary)

    def widen(readings: Sequence[SensorReading], n: int) -> List[SensorReading]:
        """Extend an event to n sensor streams by perturbed replication."""
        base = list(readings)
        out: List[SensorReading] = []
        for i in range(n):
            src = base[i % len(base)]
            jitter = rng.gauss(0.0, 12.0) if i >= len(base) else 0.0
            out.append(
                SensorReading(
                    node_id=f"N{i + 1}",
                    timestamp=src.timestamp,
                    methane_ppm=max(src.methane_ppm + jitter, 0.0),
                    error_variance=src.error_variance,
                    reference_ppm=src.reference_ppm,
                )
            )
        return out

    rows: List[Dict[str, Any]] = []
    for n_nodes in node_counts:
        samples: List[float] = []
        for _ in range(repeats):
            batch = [rng.choice(triggering) for _ in range(events_per_run)]
            t0 = time.perf_counter()
            for src in batch:
                ev = node.sensor.publish_trigger(src.primary)
                try:
                    node.handle_event(
                        ev, widen(src.readings, n_nodes), sample_resources=False
                    )
                except Exception:
                    logger.exception("scalability event failed")
            per_event_ms = (time.perf_counter() - t0) * 1000.0 / events_per_run
            samples.append(per_event_ms)
        rows.append(
            {
                "node_count": n_nodes,
                "events_per_run": events_per_run,
                "mean_ms": statistics.fmean(samples),
                "sd_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                "median_ms": statistics.median(samples),
            }
        )
        logger.info(
            "  N=%d: %.0f ms (±%.0f)", n_nodes, rows[-1]["mean_ms"], rows[-1]["sd_ms"]
        )

    fit = fit_power_law([r["node_count"] for r in rows], [r["mean_ms"] for r in rows])

    # A fit is only meaningful when there is a dominant per-event cost against
    # which node-dependent work accumulates. Without the local model the
    # per-event work is microseconds and the optimizer will happily return a
    # degenerate curve that says nothing about scaling.
    warnings: List[str] = []
    if fit["alpha"] <= 0:
        warnings.append(
            f"alpha = {fit['alpha']:.1f} is not positive: the fit is degenerate. "
            f"T(N) = T0 + alpha*N^beta presumes a positive node-dependent cost."
        )
    if rows and rows[-1]["mean_ms"] < 1000:
        warnings.append(
            f"per-event latency at N={rows[-1]['node_count']} is "
            f"{rows[-1]['mean_ms']:.1f} ms, orders below the ~19 s the "
            f"deployment measures. Inference is not running; this sweep does "
            f"not characterize the deployed system."
        )

    return {
        "measurements": rows,
        "fit": fit,
        "reference_fit_hardware": NODE_SCALING_FIT_HW,
        "reference_fit_scaleout": NODE_SCALING_FIT_SCALEOUT,
        "fit_warnings": warnings,
        "fit_usable": not warnings,
        "interpretation": (
            "On the deposited testbed the hardware exponent over N = 1-4 is "
            "about 1.46 and the scale-out curve over N = 4-16 flattens to an "
            "exponent of about 0.29: coordination work grows with node count "
            "while the per-event reasoning stage dominates the budget "
            "(Section 4.4)."
        ),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser(description="D2 deployment and scalability harnesses")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/deployment")
    ap.add_argument("--events", type=int, default=N_DEPLOYMENT_EVENTS)
    ap.add_argument("--scalability", action="store_true")
    ap.add_argument(
        "--node-counts", type=int, nargs="+",
        default=[1, 2, 3, 4, 6, 8, 12, 16],
        help="participating node counts to sweep under the fixed reference load",
    )
    ap.add_argument("--events-per-run", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-resources", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset = load_trials(args.data)
    config = ADAMConfig(enable_llm=not args.no_llm, seed=args.seed)

    llm_client = None
    if config.enable_llm:
        client = OllamaClient(
            model=config.ollama_model,
            host=config.ollama_host,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        if not client.health():
            logger.error("Ollama unreachable; pass --no-llm or start it")
            return 2
        llm_client = client

    if args.scalability:
        logger.info("node-count sweep, N in %s", args.node_counts)
        result = run_scalability(
            dataset,
            config,
            node_counts=args.node_counts,
            events_per_run=args.events_per_run,
            repeats=args.repeats,
            llm_client=llm_client,
            seed=args.seed,
        )
        fit = result["fit"]
        print()
        if not result["fit_usable"]:
            print("FIT NOT USABLE:")
            for w in result["fit_warnings"]:
                print(f"  - {w}")
            print()
        print(
            f"T(N) = {fit['T0']:.1f} + {fit['alpha']:.1f} * N^{fit['beta']:.3f} ms"
            f"   R^2 = {fit.get('r_squared', float('nan')):.4f}"
        )
        hw, so = NODE_SCALING_FIT_HW, NODE_SCALING_FIT_SCALEOUT
        print(f"deposited fits (Table 7): hardware N=1-4  T0={hw['T0']}, "
              f"alpha={hw['alpha']}, beta={hw['beta']}")
        print(f"                          scale-out N=4-16 T0={so['T0']}, "
              f"alpha={so['alpha']}, beta={so['beta']}")
        print(f"\n{result['interpretation']}")
        out = os.path.join(args.out, "scalability.json")
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {out}")
        return 0

    logger.info("replaying %d deployment events", args.events)
    traces = run_deployment(
        dataset,
        config,
        n_events=args.events,
        llm_client=llm_client,
        sample_resources=not args.no_resources,
        seed=args.seed,
    )
    save_deployment_traces(traces, os.path.join(args.out, "d2_traces.jsonl"))
    summary = summarize_deployment(traces, config.decision_deadline_s)

    print()
    e2e = summary["end_to_end_ms"]
    print(
        f"end-to-end: mean {e2e['mean']/1000:.2f} s  median {e2e['median']/1000:.2f} s"
        f"  p95 {e2e['p95']/1000:.2f} s   (manuscript: 18.9 s mean)"
    )
    print(f"{'stage':<10}{'mean ms':>12}{'share':>10}{'paper ms':>12}")
    for s, d in summary["stages"].items():
        ref = d["reference_mean_ms"]
        print(
            f"{s:<10}{d['mean_ms']:>12.1f}{d['share']:>9.1%}"
            f"{(f'{ref:.0f}' if ref else '--'):>12}"
        )
    print(
        f"\ntrace persistence {summary['trace_persistence']:.1%} "
        f"(manuscript 97.2%)   within deadline {summary['within_deadline']:.1%}"
        f"   degraded {summary['degraded_fraction']:.1%}"
    )

    out = os.path.join(args.out, "deployment_summary.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out}")
    if dataset.is_simulated:
        print(
            "\nNOTE: simulated data and no hardware. Latencies reflect this "
            "container, not a Raspberry Pi 5."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
