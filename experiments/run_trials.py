"""
experiments.run_trials
======================

Scores every system over D1 and emits Table 5.

Main benchmark
--------------
ADAM_LLM is evaluated against ten comparator configurations: raw-input and
fused-input baselines, Cloud-Only, Single Agent, and four ADAM component
ablations.

Stateful systems are rebuilt for each held-out trial. Supervised classifiers
are fitted on the other nine trials. ADAM itself is not fitted from labels; its
semantic memory contains only traces resolved earlier in the held-out replay.

Usage
-----
    # offline, no hardware, no API keys
    python -m experiments.run_trials --data data/artifacts/d1_simulated.csv \
        --skip cloud_only --no-llm

    # reference rerun against an explicit event-level dataset
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
    GradientBoostingFusedBaseline,
    RandomForestBaseline,
    RandomForestFusedBaseline,
    SingleAgent,
    StaticThreshold,
)
from data.calibration import calibrate_fold
from data.loader import Dataset, load_trials

logger = logging.getLogger(__name__)

#: Systems that must be rebuilt and refitted per fold. Stateless rule-based
#: systems are exempt.
_NEEDS_LOTO = {
    "adam_llm",
    "random_forest_raw",
    "random_forest_fused",
    "gradient_boosting_fused",
    "cloud_only",
    "single_agent",
    "adam_no_aggregator",
    "adam_no_llm",
    "adam_no_blockchain",
    "adam_no_weaviate",
}


def build_factories(
    config: ADAMConfig,
    llm_client: Optional[Any] = None,
) -> Dict[str, Callable[[], Any]]:
    """Construct a fresh-instance factory for each system."""
    from adam.governance.chain import LocalValidator, NullChainClient
    from adam.memory.store import InMemoryStore

    def adam_llm() -> ADAMSystem:
        return ADAMSystem(
            config=config,
            memory=InMemoryStore(),
            chain=NullChainClient(),
            validator=LocalValidator(),
            llm_client=llm_client,
        )

    factories: Dict[str, Callable[[], Any]] = {
        "adam_llm": adam_llm,
        "static_threshold": lambda: StaticThreshold(config.threshold_ppm),
        "random_forest_raw": lambda: RandomForestBaseline(
            threshold_ppm=config.threshold_ppm
        ),
        "random_forest_fused": lambda: RandomForestFusedBaseline(
            threshold_ppm=config.threshold_ppm
        ),
        "gradient_boosting_fused": lambda: GradientBoostingFusedBaseline(
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
    *,
    fold_local_calibration: bool = False,
) -> List[Prediction]:
    """Evaluate one system under leave-one-trial-out state isolation.

    By default, events retain the fixed sensor error variances stored in the
    deposited D1 records, reproducing the reported benchmark. Set
    ``fold_local_calibration=True`` for the stricter sensitivity analysis that
    re-estimates fusion variances on the nine training trials of each fold.
    """
    events = dataset.events
    if name not in _NEEDS_LOTO:
        system = factory()
        system.fit(events)
        return system.predict_all(events)

    preds: List[Prediction] = []
    for held_out in dataset.trial_ids:
        raw_train = [e for e in events if e.trial_id != held_out]
        raw_test = [e for e in events if e.trial_id == held_out]
        if fold_local_calibration:
            train, test, _fold_variances = calibrate_fold(raw_train, raw_test)
        else:
            # Reported benchmark: use the fixed error variances embedded in the
            # deposited D1 records. The manuscript discloses the resulting
            # indirect held-out influence and treats fold-local recalibration as
            # a sensitivity analysis rather than rewriting historical evidence.
            train, test = raw_train, raw_test
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
        "--fold-local-calibration",
        action="store_true",
        help="sensitivity analysis only: re-estimate sensor fusion variances on "
        "the nine training trials of each fold instead of using the fixed "
        "deposited weights used by the reported benchmark",
    )
    ap.add_argument(
        "--eval-mode",
        choices=("gated", "full_pipeline"),
        default="full_pipeline",
        help="benchmark runs use full_pipeline. The legacy 'gated' option is "
        "rejected for manuscript reproduction; derive deployment semantics "
        "from frozen benchmark predictions with "
        "scripts/derive_deployment_semantics.py.",
    )
    args = ap.parse_args()

    if args.eval_mode == "gated":
        ap.error(
            "A second gated LLM execution is not a valid revised-manuscript "
            "reproduction. Run full_pipeline once, then derive deployment "
            "semantics with scripts/derive_deployment_semantics.py."
        )

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

    fusion_dependent = {
        "adam_llm", "random_forest_fused", "gradient_boosting_fused",
        "cloud_only", "adam_no_aggregator", "adam_no_llm",
        "adam_no_blockchain", "adam_no_weaviate",
    }
    if set(selected) & fusion_dependent:
        dataset.require_multinode("fusion-dependent D1 benchmark", minimum=4)

    EGRESS.reset()
    scores: Dict[str, SystemScores] = {}
    timings: Dict[str, float] = {}

    for name in selected:
        logger.info("evaluating %s ...", name)
        t0 = time.perf_counter()
        try:
            preds = evaluate_system(
                name,
                factories[name],
                dataset,
                fold_local_calibration=args.fold_local_calibration,
            )
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

    if "adam_llm" not in scores:
        logger.error("adam_llm was not evaluated; the benchmark table needs it as reference")
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
        "adam_llm",
        "static_threshold",
        "random_forest_raw",
        "random_forest_fused",
        "gradient_boosting_fused",
        "cloud_only",
        "single_agent",
        "adam_no_aggregator",
        "adam_no_llm",
        "adam_no_blockchain",
        "adam_no_weaviate",
    ]
    rows = build_table5(scores, reference_key="adam_llm", order=order)

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
