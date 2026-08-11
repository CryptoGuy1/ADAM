"""
experiments.run_trials
======================

Scores every system over D1 and emits Table 5.

Systems evaluated
-----------------
    cloud_only, single_agent, no_aggregator, no_llm, no_blockchain, no_weaviate

All learned and memory-bearing systems use leave-one-trial-out splitting: for
each fold the system is rebuilt, fitted on nine trials, and evaluated on the
held-out one. This applies to ADAM and its ablations as well as to the Random
Forest, because ADAM's semantic memory is seeded from the training fold and
would otherwise carry the held-out trial's events.

Usage
-----
    # offline, no hardware, no API keys
    python -m experiments.run_trials --data data/artifacts/d1_simulated.csv \
        --skip cloud_only --no-llm

    # full reproduction against the deposited data
    python -m experiments.run_trials --data data/d1_labeled_trials.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from adam.config import ADAMConfig, DEFAULT_CONFIG, SEED, THRESHOLD_PPM
from adam.schemas import LabeledEvent, Prediction
from adam.telemetry import EGRESS

from ablations.systems import ADAMSystem, ABLATION_FACTORIES
from analysis.metrics import (
    SystemScores,
    build_table5,
    format_table5,
    score_system,
)
from baselines.systems import (
    CloudOnly,
    RandomForestBaseline,
    SingleAgent,
    StaticThreshold,
)
from data.loader import Dataset, load_trials

logger = logging.getLogger(__name__)

#: Systems that must be rebuilt and refitted per fold. Stateless rule-based
#: systems are exempt.
_NEEDS_LOTO = {
    "adam_full",
    "random_forest",
    "cloud_only",
    "single_agent",
    "no_aggregator",
    "no_llm",
    "no_blockchain",
    "no_weaviate",
}


def build_factories(
    config: ADAMConfig,
    llm_client: Optional[Any] = None,
) -> Dict[str, Callable[[], Any]]:
    """Construct a fresh-instance factory for each system."""
    from adam.governance.chain import LocalValidator, NullChainClient
    from adam.memory.store import InMemoryStore

    def adam_full() -> ADAMSystem:
        return ADAMSystem(
            config=config,
            memory=InMemoryStore(),
            chain=NullChainClient(),
            validator=LocalValidator(),
            llm_client=llm_client,
        )

    factories: Dict[str, Callable[[], Any]] = {
        "adam_full": adam_full,
        "static_threshold": lambda: StaticThreshold(config.threshold_ppm),
        "random_forest": lambda: RandomForestBaseline(
            threshold_ppm=config.threshold_ppm
        ),
        "cloud_only": lambda: CloudOnly(threshold_ppm=config.threshold_ppm),
        "single_agent": lambda: SingleAgent(config=config, client=llm_client),
    }

    for name, make in ABLATION_FACTORIES.items():
        def _factory(make=make) -> ADAMSystem:
            return make(
                config,
                memory=InMemoryStore(),
                chain=NullChainClient(),
                validator=LocalValidator(),
                llm_client=llm_client,
            )

        factories[name] = _factory

    return factories


def evaluate_system(
    name: str,
    factory: Callable[[], Any],
    dataset: Dataset,
) -> List[Prediction]:
    """Evaluate one system, applying LOTO where the system carries state."""
    events = dataset.events
    if name not in _NEEDS_LOTO:
        system = factory()
        system.fit(events)
        return system.predict_all(events)

    preds: List[Prediction] = []
    for held_out in dataset.trial_ids:
        train = [e for e in events if e.trial_id != held_out]
        test = [e for e in events if e.trial_id == held_out]
        system = factory()
        system.fit(train)
        preds.extend(system.predict_all(test))
        logger.debug("%s: fold %d done (%d test events)", name, held_out, len(test))
    return preds


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s"
    )
    ap = argparse.ArgumentParser(description="Score all systems over D1 (Table 5)")
    ap.add_argument("--data", required=True, help="path to the D1 CSV")
    ap.add_argument("--out", default="results/trials")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="systems to skip, e.g. cloud_only when no API key is available",
    )
    ap.add_argument(
        "--only", nargs="*", default=None, help="evaluate only these systems"
    )
    ap.add_argument(
        "--no-llm",
        action="store_true",
        help="run without Ollama; ADAM falls back to deterministic logic and "
        "the reported figures are NOT comparable to the manuscript",
    )
    ap.add_argument("--threshold", type=float, default=THRESHOLD_PPM)
    ap.add_argument(
        "--eval-mode",
        choices=("gated", "full_pipeline"),
        default="full_pipeline",
        help="how ADAM and its ablations score D1. 'full_pipeline' replays "
        "every event through the complete crew workflow and corresponds to "
        "the nine-system benchmark of Table 5 (06A_Event_Predictions). "
        "'gated' applies the deployment semantics, where sub-threshold "
        "readings never form a crew, and corresponds to D1_RawTrigger_Log.",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # -- load, with the integrity gate
    dataset = load_trials(args.data, threshold_ppm=args.threshold)
    if dataset.is_simulated:
        logger.warning(
            "dataset is SIMULATED. Results exercise the pipeline but do not "
            "reproduce the manuscript. Use the deposited data for that."
        )

    config = ADAMConfig(
        threshold_ppm=args.threshold,
        enable_llm=not args.no_llm,
        eval_mode=args.eval_mode,
        seed=args.seed,
    )

    llm_client = None
    if config.enable_llm:
        from adam.llm.client import OllamaClient

        client = OllamaClient(
            model=config.ollama_model,
            host=config.ollama_host,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
        )
        if not client.health():
            logger.error(
                "Ollama is not reachable at %s with model %s.\n"
                "  Start it:  ollama serve\n"
                "  Pull it:   ollama pull %s\n"
                "Or pass --no-llm to run the non-LLM systems only.",
                config.ollama_host,
                config.ollama_model,
                config.ollama_model,
            )
            return 2
        llm_client = client
        logger.info("on-device model ready: %s", config.ollama_model)

    factories = build_factories(config, llm_client)

    selected = list(args.only) if args.only else list(factories)
    selected = [s for s in selected if s not in set(args.skip)]
    if config.enable_llm is False:
        for needs_llm in ("cloud_only",):
            if needs_llm in selected and not os.getenv("OPENAI_API_KEY"):
                logger.warning("skipping %s: OPENAI_API_KEY not set", needs_llm)
                selected.remove(needs_llm)

    unknown = [s for s in selected if s not in factories]
    if unknown:
        ap.error(f"unknown systems: {unknown}. Available: {sorted(factories)}")

    EGRESS.reset()
    scores: Dict[str, SystemScores] = {}
    timings: Dict[str, float] = {}

    for name in selected:
        logger.info("evaluating %s ...", name)
        t0 = time.perf_counter()
        try:
            preds = evaluate_system(name, factories[name], dataset)
        except Exception as exc:
            logger.error("%s failed: %s", name, exc)
            continue
        timings[name] = time.perf_counter() - t0
        scores[name] = score_system(name, dataset.events, preds)
        m, s = scores[name].mean_sd("f1")
        logger.info("  %s: F1 = %.3f ± %.3f  (%.1fs)", name, m, s, timings[name])

        with open(os.path.join(args.out, f"predictions_{name}.jsonl"), "w") as fh:
            for p in preds:
                fh.write(json.dumps(p.to_dict()) + "\n")

    if "adam_full" not in scores:
        logger.error("adam_full was not evaluated; Table 5 needs it as reference")
        return 1

    # -- zero-egress check (Section 4.5.3)
    if "cloud_only" not in scores:
        summary = EGRESS.summary()
        if summary["external_bytes"] > 0:
            logger.error(
                "ADAM run recorded %d bytes of external egress to %s. "
                "Section 4.5.3 claims zero.",
                summary["external_bytes"],
                summary["destinations"],
            )
            return 1
        logger.info("zero external egress confirmed for the ADAM run")

    order = [
        "adam_full",
        "static_threshold",
        "random_forest",
        "cloud_only",
        "single_agent",
        "no_aggregator",
        "no_llm",
        "no_blockchain",
        "no_weaviate",
    ]
    rows = build_table5(scores, reference_key="adam_full", order=order)

    print()
    print(format_table5(rows, n_trials=len(dataset.trial_ids)))
    print()

    payload = {
        "dataset": {
            "path": args.data,
            "source": dataset.source,
            "n_events": len(dataset.events),
            "n_trials": len(dataset.trial_ids),
            "simulated": dataset.is_simulated,
        },
        "config": {
            "threshold_ppm": config.threshold_ppm,
            "model": config.ollama_model if config.enable_llm else None,
            "llm_enabled": config.enable_llm,
            "seed": args.seed,
        },
        "table5": rows,
        "runtime_s": timings,
        "egress": EGRESS.summary(),
        "caveat": (
            "SIMULATED DATA - exercises the pipeline, does not reproduce the "
            "manuscript." if dataset.is_simulated else None
        ),
    }
    out_path = os.path.join(args.out, "table5.json")
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"wrote {out_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
