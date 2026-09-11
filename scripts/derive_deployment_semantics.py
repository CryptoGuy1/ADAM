#!/usr/bin/env python3
"""Derive ADAM deployment semantics from frozen benchmark predictions.

The revised manuscript defines deployment-semantics predictions as a
*deterministic transform* of the benchmark-mode ADAM_LLM predictions:

* raw trigger >= threshold: preserve the stored benchmark prediction exactly;
* raw trigger < threshold: assign NORMAL without forming a crew.

This script therefore never invokes Ollama. It prevents a second stochastic
language-model pass from changing above-gate classifications and isolates the
effect of the screening gate itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from adam.config import THRESHOLD_PPM
from adam.schemas import Prediction
from data.loader import load_trials


def load_predictions(path: Path) -> Dict[Tuple[int, int], Prediction]:
    out: Dict[Tuple[int, int], Prediction] = {}
    allowed = {
        "system", "trial_id", "event_index", "predicted", "confidence",
        "latency_ms", "degraded_mode", "api_cost_usd",
    }
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            p = Prediction(**{k: v for k, v in raw.items() if k in allowed})
            key = (int(p.trial_id), int(p.event_index))
            if key in out:
                raise ValueError(f"duplicate benchmark prediction for {key}")
            out[key] = p
    return out


def derive(
    data_path: str,
    benchmark_path: str,
    threshold_ppm: float = THRESHOLD_PPM,
) -> List[Prediction]:
    dataset = load_trials(data_path, threshold_ppm=threshold_ppm)
    benchmark = load_predictions(Path(benchmark_path))

    derived: List[Prediction] = []
    expected = {(int(e.trial_id), int(e.event_index)) for e in dataset.events}
    missing = expected - set(benchmark)
    extra = set(benchmark) - expected
    if missing or extra:
        raise ValueError(
            f"benchmark/data key mismatch: missing={len(missing)} extra={len(extra)}"
        )

    for event in dataset.events:
        key = (int(event.trial_id), int(event.event_index))
        b = benchmark[key]
        above = float(event.primary.methane_ppm) >= float(threshold_ppm)
        if above:
            # Exact identity above the gate is the defining invariant.
            pred = int(b.predicted)
            confidence = float(b.confidence)
            latency_ms = float(b.latency_ms)
            degraded = bool(b.degraded_mode)
            api_cost = float(b.api_cost_usd)
        else:
            pred = 0
            confidence = 0.0  # no Decision-Agent inference occurred
            latency_ms = 0.0
            degraded = False
            api_cost = 0.0

        derived.append(
            Prediction(
                system="adam_deployment_semantics",
                trial_id=event.trial_id,
                event_index=event.event_index,
                predicted=pred,
                confidence=confidence,
                latency_ms=latency_ms,
                degraded_mode=degraded,
                api_cost_usd=api_cost,
            )
        )

    return derived


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="D1 CSV with raw screening channel")
    ap.add_argument("--benchmark", required=True, help="stored ADAM_LLM benchmark JSONL")
    ap.add_argument("--out", required=True, help="output deployment-semantics JSONL")
    ap.add_argument("--threshold", type=float, default=THRESHOLD_PPM)
    args = ap.parse_args()

    preds = derive(args.data, args.benchmark, threshold_ppm=args.threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for p in preds:
            fh.write(json.dumps(p.to_dict()) + "\n")

    above = sum(
        1
        for e in load_trials(args.data, threshold_ppm=args.threshold).events
        if e.primary.methane_ppm >= args.threshold
    )
    print(f"wrote {len(preds)} predictions to {out}")
    print(f"above-gate benchmark predictions preserved: {above}")
    print(f"below-gate events forced NORMAL: {len(preds) - above}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
