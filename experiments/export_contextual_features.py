"""Export the actual eight-feature context from a COMPLETE four-node D1 CSV.

Fails closed for workbook primary-channel exports, which do not contain the
original concurrent per-node streams. Synthetic inputs remain tagged synthetic.
The exported features are agent-visible; the reference/label columns are audit
metadata, never inputs to model fitting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from adam.config import THRESHOLD_PPM
from baselines.systems import CONTEXTUAL_FEATURE_NAMES, fused_context_matrix
from data.loader import Dataset, load_trials
from experiments.decision_agent_backends import decision_feature_vector

AUDIT_COLUMNS = ("event_id", "trial_id", "event_index", "timestamp", "trigger_node", "label", "reference_ppm")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_features(dataset: Dataset, output: Path, input_path: Path) -> dict[str, Any]:
    dataset.require_multinode("contextual feature export", minimum=4)
    ordered, matrix = fused_context_matrix(dataset.events, threshold_ppm=THRESHOLD_PPM)
    if len(ordered) != len(matrix):
        raise ValueError("feature matrix does not cover each event exactly once")
    keys: set[tuple[int, int]] = set()
    history: dict[int, list[float]] = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*AUDIT_COLUMNS, *CONTEXTUAL_FEATURE_NAMES])
        writer.writeheader()
        for event, features in zip(ordered, matrix):
            key = (event.trial_id, event.event_index)
            if key in keys:
                raise ValueError(f"duplicate event key {key}")
            keys.add(key)
            prior = history.setdefault(event.trial_id, [])
            independent = decision_feature_vector(
                raw_ppm=event.primary.methane_ppm,
                fused_ppm=features[3], dispersion_ppm=features[4],
                baseline_window=prior, threshold_ppm=THRESHOLD_PPM,
            )
            if not np.allclose(features, independent, rtol=1e-12, atol=1e-9):
                raise ValueError(f"fitted backend/export feature mismatch at {key}")
            if not np.isfinite(features).all():
                raise ValueError(f"nonfinite feature at {key}")
            writer.writerow({
                "event_id": f"derived-trial-{event.trial_id}-event-{event.event_index}",
                "trial_id": event.trial_id, "event_index": event.event_index,
                "timestamp": event.timestamp, "trigger_node": event.primary.node_id,
                "label": event.label, "reference_ppm": event.reference_ppm,
                **dict(zip(CONTEXTUAL_FEATURE_NAMES, map(float, features))),
            })
            prior.append(float(event.primary.methane_ppm))
    manifest = {
        "schema": "adam.contextual_features.v1", "source": dataset.source,
        "historical_reproduction": False,
        "description": "Fresh derived features, not an original archived acquisition stream or the historical fitted-model feature matrix.",
        "event_id_policy": "derived trial/index ID, not historical sensor Event_ID",
        "n_events": len(keys), "n_trials": len(dataset.trial_ids),
        "n_concurrent_nodes_min": dataset.min_concurrent_nodes,
        "features": list(CONTEXTUAL_FEATURE_NAMES),
        "baseline": "preceding six primary-channel readings within trial, current excluded; first event initialized to current raw value",
        "input_path": str(input_path), "input_sha256": sha256_file(input_path),
        "output_file": output.name, "output_sha256": sha256_file(output),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="original complete per-node D1 CSV")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    result = export_features(load_trials(str(args.data)), args.out, args.data)
    print(json.dumps({k: result[k] for k in ("source", "n_events", "n_trials", "output_sha256")}, indent=2))

if __name__ == "__main__":
    main()
