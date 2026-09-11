#!/usr/bin/env python3
"""
Decision-Agent Substitution Study.

The study has two arms for each reasoner:

  standalone       the reasoner outside ADAM
  in_crew          the corresponding reasoner in the ADAM Decision-Agent slot

For the two existing manuscript configurations (Static Threshold / ADAM-No-LLM
and Single Agent / ADAM_LLM), this script can reuse prediction JSONL files
emitted by experiments.run_trials instead of re-running a stochastic LLM.

For Logistic Regression, Random Forest, and Gradient Boosting, both the
standalone and in-crew arms use the same eight-feature contextual representation
preserved by the deposited D1 records. Both arms use leave-one-trial-out fitting
and the same held-out trials; the surrounding crew execution context is what
changes.

The in-crew fitted variants call ADAMNode.handle_event directly.  There is no
special decide_with() path: formation, fusion, retrieval, voting, governance,
persistence, and trace construction use the normal runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adam.config import ADAMConfig, RF_PARAMS, SEED, THRESHOLD_PPM
from adam.crew import ADAMNode
from adam.governance.chain import InMemoryChainClient, LocalValidator
from adam.memory.store import InMemoryStore
from adam.schemas import LabeledEvent, Prediction
from analysis.metrics import holm_adjust, score_system
from baselines.systems import (
    BASELINE_WINDOW,
    CONTEXTUAL_FEATURE_NAMES,
    fused_context_matrix,
)
from data.calibration import calibrate_fold
from data.loader import Dataset, load_trials
from experiments.decision_agent_backends import (
    DECISION_AGENT_VARIANTS,
    make_backend,
)


PAIR_LABELS = {
    "static": ("Static Threshold", "ADAM-No-LLM"),
    "llm": ("Gemma 3 1B", "ADAM_LLM"),
    "logreg": ("Logistic Regression", "ADAM_LogReg"),
    "rf": ("Random Forest", "ADAM_RF"),
    "gbm": ("Gradient Boosting", "ADAM_GBM"),
}

MAIN_PREDICTION_FILES = {
    "static": (
        ("predictions_static_threshold.jsonl",),
        ("predictions_adam_no_llm.jsonl", "predictions_no_llm.jsonl"),
    ),
    "llm": (
        ("predictions_single_agent.jsonl",),
        ("predictions_adam_llm.jsonl", "predictions_adam_full.jsonl"),
    ),
}


def _prediction_from_dict(d: Dict[str, Any]) -> Prediction:
    allowed = {
        "system",
        "trial_id",
        "event_index",
        "predicted",
        "confidence",
        "latency_ms",
        "degraded_mode",
        "api_cost_usd",
    }
    return Prediction(**{k: v for k, v in d.items() if k in allowed})


def load_prediction_jsonl(path: Path) -> List[Prediction]:
    out: List[Prediction] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(_prediction_from_dict(json.loads(line)))
    return out


def _model_for(name: str) -> Any:
    if name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        )
    if name == "rf":
        return RandomForestClassifier(**RF_PARAMS)
    if name == "gbm":
        return GradientBoostingClassifier(random_state=SEED)
    raise KeyError(name)


def standalone_fitted_predictions(
    dataset: Dataset,
    name: str,
    *,
    fold_local_calibration: bool = False,
) -> List[Prediction]:
    """LOTO standalone fitted model using the eight contextual features."""
    out: List[Prediction] = []

    for held_out in dataset.trial_ids:
        raw_train = [e for e in dataset.events if e.trial_id != held_out]
        raw_test = [e for e in dataset.events if e.trial_id == held_out]
        if fold_local_calibration:
            train, test, _fold_variances = calibrate_fold(raw_train, raw_test)
        else:
            train, test = raw_train, raw_test

        ordered_train, X_train = fused_context_matrix(
            train,
            threshold_ppm=THRESHOLD_PPM,
        )
        ordered_test, X_test = fused_context_matrix(
            test,
            threshold_ppm=THRESHOLD_PPM,
        )
        y_train = np.asarray([e.label for e in ordered_train], dtype=int)

        model = _model_for(name)
        model.fit(np.asarray(X_train, dtype=float), y_train)

        probs = model.predict_proba(np.asarray(X_test, dtype=float))[:, 1]
        preds = (probs >= 0.5).astype(int)

        for event, pred, prob in zip(ordered_test, preds, probs):
            conf = float(prob if pred == 1 else 1.0 - prob)
            out.append(
                Prediction(
                    system=f"{name}_standalone",
                    trial_id=event.trial_id,
                    event_index=event.event_index,
                    predicted=int(pred),
                    confidence=conf,
                    latency_ms=0.0,
                    degraded_mode=False,
                )
            )

    return out


def in_crew_fitted_predictions(
    dataset: Dataset,
    name: str,
    *,
    fold_local_calibration: bool = False,
) -> List[Prediction]:
    """LOTO fitted Decision Agent executed through the normal ADAM crew path."""
    backend_name = {
        "logreg": "adam_logreg",
        "rf": "adam_rf",
        "gbm": "adam_gbm",
    }[name]

    out: List[Prediction] = []

    for held_out in dataset.trial_ids:
        raw_train = [e for e in dataset.events if e.trial_id != held_out]
        raw_test = [e for e in dataset.events if e.trial_id == held_out]
        if fold_local_calibration:
            train, calibrated_test, _fold_variances = calibrate_fold(raw_train, raw_test)
        else:
            train, calibrated_test = raw_train, raw_test
        test = sorted(
            calibrated_test,
            key=lambda e: (e.timestamp, e.event_index),
        )

        backend = make_backend(
            backend_name,
            threshold_ppm=THRESHOLD_PPM,
        ).fit(train)

        # The substituted backend is supplied explicitly, so no Ollama call is
        # made. InMemoryChainClient acknowledges the normal persistence call so
        # nominal offline events are not mislabeled as ledger failures; it is
        # not used to make blockchain performance claims.
        config = ADAMConfig(
            threshold_ppm=THRESHOLD_PPM,
            eval_mode="full_pipeline",
        )
        node = ADAMNode(
            node_id=f"node-sub-{held_out}",
            config=config,
            memory=InMemoryStore(),
            chain=InMemoryChainClient(),
            validator=LocalValidator(),
            decision_backend=backend,
        )

        prior_raw: List[float] = []

        for event in test:
            baseline_before = list(prior_raw[-BASELINE_WINDOW:])

            # Preserve the normal sensor-state transition. Benchmark mode still
            # evaluates every event even when the raw screen does not trigger.
            node.sensor.observe(event.primary)
            crew_event = node.sensor.publish_trigger(event.primary)
            trace = node.handle_event(
                crew_event,
                list(event.readings),
                sample_resources=False,
                baseline_window=baseline_before,
            )

            decision = trace.decision
            if decision is None:
                raise RuntimeError(
                    f"{backend.name}: no DecisionObject for "
                    f"trial={event.trial_id}, event={event.event_index}"
                )

            out.append(
                Prediction(
                    system=backend_name,
                    trial_id=event.trial_id,
                    event_index=event.event_index,
                    predicted=1 if decision.is_anomaly else 0,
                    confidence=float(decision.confidence),
                    latency_ms=float(trace.latencies.total_ms),
                    degraded_mode=bool(trace.degraded_mode),
                )
            )

            # Current event becomes eligible only for the next event's
            # six-reading fitted-model baseline.
            prior_raw.append(float(event.primary.methane_ppm))

    return out


def bootstrap_ci(
    diffs: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = SEED,
) -> Tuple[float, float]:
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        means[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_summary(
    *,
    label: str,
    standalone: Sequence[Prediction],
    in_crew: Sequence[Prediction],
    dataset: Dataset,
) -> Dict[str, Any]:
    s0 = score_system(f"{label}_standalone", dataset.events, standalone)
    s1 = score_system(f"{label}_in_crew", dataset.events, in_crew)

    shared = sorted(set(s0.trial_ids) & set(s1.trial_ids))
    alone = np.asarray([s0.per_trial[t].f1 for t in shared], dtype=float)
    crew = np.asarray([s1.per_trial[t].f1 for t in shared], dtype=float)
    delta = crew - alone

    nonzero = delta[np.abs(delta) > 1e-12]
    if len(nonzero):
        # Methods: zero-difference trial pairs are omitted before the exact
        # signed-rank calculation. Passing the unfiltered arrays with
        # ``zero_method=wilcox`` can make SciPy fall back from an exact p-value.
        p_exact = float(
            wilcoxon(
                nonzero,
                np.zeros_like(nonzero),
                alternative="two-sided",
                zero_method="wilcox",
                method="exact",
            ).pvalue
        )
    else:
        p_exact = 1.0

    lo, hi = bootstrap_ci(delta)
    return {
        "key": label,
        "alone_f1": float(alone.mean()),
        "in_crew_f1": float(crew.mean()),
        "delta_f1": float(delta.mean()),
        "ci95_low": lo,
        "ci95_high": hi,
        "trials_favoring_crew": int((delta > 0).sum()),
        "ties": int((np.abs(delta) <= 1e-12).sum()),
        "n_trials": len(shared),
        "p_exact": p_exact,
        "standalone_trial_f1": alone.tolist(),
        "in_crew_trial_f1": crew.tolist(),
    }


def write_jsonl(path: Path, predictions: Sequence[Prediction]) -> None:
    with path.open("w") as fh:
        for p in predictions:
            fh.write(json.dumps(p.to_dict()) + "\n")


def _first_existing(results_dir: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        path = results_dir / name
        if path.exists():
            return path
    return None


def _load_main_pair(
    results_dir: Path,
    key: str,
) -> Optional[Tuple[List[Prediction], List[Prediction]]]:
    a_candidates, b_candidates = MAIN_PREDICTION_FILES[key]
    a = _first_existing(results_dir, a_candidates)
    b = _first_existing(results_dir, b_candidates)
    if a is None or b is None:
        return None
    return load_prediction_jsonl(a), load_prediction_jsonl(b)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the Decision-Agent Substitution Study"
    )
    ap.add_argument("--data", required=True, help="D1 CSV used by run_trials.py")
    ap.add_argument(
        "--out",
        default="results/decision_agent_substitution",
    )
    ap.add_argument(
        "--main-results",
        default="results/trials",
        help=(
            "directory containing main-benchmark prediction JSONL files; "
            "used to reuse Static/ADAM-No-LLM and Single-Agent/ADAM_LLM "
            "without a second stochastic LLM run"
        ),
    )
    ap.add_argument(
        "--fold-local-calibration",
        action="store_true",
        help="sensitivity analysis only: re-estimate fusion variances on the "
        "nine training trials instead of using the fixed deposited weights",
    )
    ap.add_argument(
        "--fitted-only",
        action="store_true",
        help="run only Logistic Regression, RF, and GBM pairs",
    )
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = load_trials(args.data, threshold_ppm=THRESHOLD_PPM)
    dataset.require_multinode("Decision-Agent substitution study", minimum=4)

    pairs: Dict[str, Tuple[List[Prediction], List[Prediction]]] = {}

    if not args.fitted_only:
        main_dir = Path(args.main_results)
        for key in ("static", "llm"):
            loaded = _load_main_pair(main_dir, key)
            if loaded is not None:
                pairs[key] = loaded
            else:
                print(
                    f"NOTE: {key} pair not loaded; expected "
                    f"one candidate from each of {MAIN_PREDICTION_FILES[key]} in {main_dir}"
                )

    for key in ("logreg", "rf", "gbm"):
        print(f"evaluating {key}: standalone")
        standalone = standalone_fitted_predictions(
            dataset, key, fold_local_calibration=args.fold_local_calibration
        )

        print(f"evaluating {key}: in crew")
        in_crew = in_crew_fitted_predictions(
            dataset, key, fold_local_calibration=args.fold_local_calibration
        )

        pairs[key] = (standalone, in_crew)
        write_jsonl(outdir / f"predictions_{key}_standalone.jsonl", standalone)
        write_jsonl(outdir / f"predictions_{key}_in_crew.jsonl", in_crew)

    rows: List[Dict[str, Any]] = []
    raw_p: Dict[str, float] = {}

    for key, (standalone, in_crew) in pairs.items():
        row = paired_summary(
            label=key,
            standalone=standalone,
            in_crew=in_crew,
            dataset=dataset,
        )
        rows.append(row)
        raw_p[key] = float(row["p_exact"])

    # The manuscript defines a five-comparison substitution family. Apply Holm
    # across five only when all five reasoner pairs are present. Otherwise the
    # partial run is clearly marked and no manuscript-family pHolm is claimed.
    full_family = set(pairs) == {"static", "llm", "logreg", "rf", "gbm"}
    adjusted = holm_adjust(raw_p) if full_family else {}

    for row in rows:
        row["p_holm"] = adjusted.get(row["key"]) if full_family else None
        standalone_name, crew_name = PAIR_LABELS[row["key"]]
        row["decision_agent"] = standalone_name
        row["crew_configuration"] = crew_name

    rows.sort(key=lambda r: ("static", "llm", "logreg", "rf", "gbm").index(r["key"]))

    payload = {
        "study": "Decision-Agent Substitution Study",
        "seed": SEED,
        "threshold_ppm": THRESHOLD_PPM,
        "baseline_window": BASELINE_WINDOW,
        "fit_protocol": "leave-one-trial-out",
        "contextual_features": list(CONTEXTUAL_FEATURE_NAMES),
        "fitted_standalone_features": list(CONTEXTUAL_FEATURE_NAMES),
        "fitted_in_crew_features": list(CONTEXTUAL_FEATURE_NAMES),
        "fusion_calibration": (
            "fold-local training-only sensitivity"
            if args.fold_local_calibration
            else "fixed deposited inverse-variance weights (reported benchmark)"
        ),
        "full_five_comparison_family": full_family,
        "rows": rows,
    }

    with (outdir / "substitution_results.json").open("w") as fh:
        json.dump(payload, fh, indent=2)

    print("\nDecision-Agent substitution summary")
    print("-" * 102)
    print(
        f"{'Decision Agent':<22}{'Alone F1':>11}{'In-crew F1':>13}"
        f"{'Delta':>10}{'95% CI':>24}{'Trials':>10}{'p_exact':>10}{'p_Holm':>10}"
    )
    for r in rows:
        ci = f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]"
        trials = f"{r['trials_favoring_crew']}/{r['n_trials']}"
        ph = "--" if r["p_holm"] is None else f"{r['p_holm']:.3f}"
        print(
            f"{r['decision_agent']:<22}"
            f"{r['alone_f1']:>11.3f}"
            f"{r['in_crew_f1']:>13.3f}"
            f"{r['delta_f1']:>+10.3f}"
            f"{ci:>24}"
            f"{trials:>10}"
            f"{r['p_exact']:>10.3f}"
            f"{ph:>10}"
        )

    print(f"\nwrote {outdir / 'substitution_results.json'}")
    if not full_family:
        print(
            "Partial family only: p_Holm intentionally omitted. "
            "Provide the main benchmark prediction files to assemble all five pairs."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
