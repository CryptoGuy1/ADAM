"""Reference-only exports, not historical replay or historical score validation."""
import csv
import json
from pathlib import Path

import pytest

from data.loader import Dataset, SimulationParams, save_trials, simulate_trials
from experiments.export_contextual_features import export_features, sha256_file
from experiments.run_decision_agent_substitution import in_crew_fitted_predictions


@pytest.fixture
def fixture_dataset(tmp_path):
    ds = simulate_trials(SimulationParams(n_trials=3, events_per_trial=24, seed=23))
    path = tmp_path / "synthetic_four_node.csv"
    save_trials(ds, str(path))
    return ds, path


def test_feature_export_complete_causal_and_provenance(fixture_dataset, tmp_path):
    ds, path = fixture_dataset
    dest = tmp_path / "features.csv"
    manifest = export_features(ds, dest, path)
    assert manifest["source"] == "simulated"
    assert manifest["historical_reproduction"] is False
    assert manifest["n_events"] == 72
    assert manifest["n_concurrent_nodes_min"] == 4
    assert manifest["input_sha256"] == sha256_file(path)
    assert manifest["output_sha256"] == sha256_file(dest)
    assert json.loads(dest.with_suffix(".manifest.json").read_text()) == manifest
    with dest.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 72
    assert len({(x["trial_id"], x["event_index"]) for x in rows}) == 72
    for row in rows:
        if int(row["event_index"]) == 0:
            assert float(row["baseline_mean"]) == pytest.approx(float(row["raw_ppm"]))
        assert float(row["reference_ppm"]) >= 0
        assert not any("label" in k or "reference" in k for k in manifest["features"])


def test_feature_export_rejects_single_channel(fixture_dataset, tmp_path):
    ds, path = fixture_dataset
    # Preserve labels but remove all non-primary nodes: this is not a 4-node run.
    from dataclasses import replace
    single = Dataset([replace(e, readings=(e.primary,)) for e in ds.events], ds.source, ds.manifest)
    with pytest.raises(ValueError, match="concurrent sensor readings"):
        export_features(single, tmp_path / "should_not_exist.csv", path)
    assert not (tmp_path / "should_not_exist.csv").exists()


def test_substitution_exports_initial_votes_final_class(fixture_dataset):
    ds, _ = fixture_dataset
    trace_rows = []
    preds = in_crew_fitted_predictions(ds, "gbm", trace_records=trace_rows)
    assert len(preds) == len(trace_rows) == 72
    for pred, row in zip(preds, trace_rows):
        assert row["initial_classification"] in ("NORMAL", "ANOMALY")
        assert row["final_classification"] in ("NORMAL", "ANOMALY")
        assert pred.predicted == int(row["final_classification"] == "ANOMALY")
        assert row["decision_vote"] == int(row["initial_classification"] == "ANOMALY")
        assert row["quorum_achieved"] >= row["quorum_required"]
        assert row["confidence_source"] in ("decision_agent", "decision_agent_initial_class")
        assert row["model_confidence"] == row["initial_confidence"]
        assert row["crew_support"] == row["quorum_achieved"] / row["voter_count"]
        assert row["action_released"] in (True, False)
