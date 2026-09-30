# Phase 2: contextual feature export and event-level substitution audit

This is a **new reference implementation**, not an unmodified replay of the May 2025 acquisition or the archived manuscript substitution experiment. All new outputs carry provenance; synthetic test data must never be presented as historical experimental evidence.

## Inputs required

A CSV containing one row per timestamped event with `trial_id,event_index,timestamp,label,reference_ppm` and the four concurrent `node_*_ppm,node_*_var` channels. The original acquisition stream is necessary to recompute fusion and cross-node dispersion. The reconciled V14 workbook preserves event-level analyses but is not a substitute for all original concurrent streams. The loader refuses to run a four-node experiment with a primary-channel-only export.

## Feature export

```bash
python -m experiments.export_contextual_features \
  --data /path/to/verified_four_node_d1.csv \
  --out results/phase2/d1_contextual_features.csv
```

The output records identifiers, reference labels for scoring **outside** the model input, and exactly eight agent-visible features in the order in `CONTEXTUAL_FEATURE_NAMES`. The causal baseline uses at most six strictly preceding readings within the trial; the first event uses its current raw measurement as the initialization convention. A separately implemented fitted Decision-Agent vector is checked against each exported feature row. Input and output SHA-256 are recorded in a `.manifest.json` alongside the CSV. Derived event IDs are not historical acquisition Event IDs.

## Full fitted substitution run

```bash
python -m experiments.run_decision_agent_substitution \
  --data /path/to/verified_four_node_d1.csv \
  --out results/phase2/substitution \
  --fitted-only
```

Standalone and in-crew fitted classifiers use the same held-out trials and feature representation. Each fitted reasoner produces:

- `predictions_{name}_{standalone,in_crew}.jsonl`;
- `{name}_crew_event_traces.csv` with initial model class and confidence, Sensor/Aggregator/Decision ballots, final class, agreement and confidence provenance, aligned action/severity, governance and store acknowledgments, and action release;
- `{name}_paired_events.csv` with a row per matched standalone/crew event;
- the canonical contextual feature CSV and SHA manifest.

The fitted runner scores the **final crew class**, never an unquorate result as NORMAL. If the crew fails to classify an event, the current full-coverage benchmark runner stops instead of silently dropping it or presenting complete-case F1 as all-events F1; this requires a separately specified incomplete-event estimand before quantitative publication. Offline `InMemoryChainClient` receipts are simulated acknowledgments, not measurements of public testnet persistence or latency. The built-in refit uses historical pooled calibration weights unless `--fold-local-calibration` is selected for a sensitivity analysis; pooled weights permit indirect held-out information, as disclosed in the manuscript.

To assemble the **full five-reasoner Holm family**, provide the original frozen benchmark prediction JSONL files for both Static and Gemma pairs using `--main-results`. The available historic workbook summaries alone do not reconstruct their individual ballots. Their `vote_trace_available` is false in the paired export. Without the full family, `p_holm` is deliberately omitted.

## Verification scope

New tests validate feature order and causal initialization, SHA manifest consistency, rejection of single-channel inputs, and preservation of initial and final fitted classification. A synthetic 3-trial/24-event fixture was run end-to-end for export and three fitted pairs. Its F1 values are **only test-fixture output**; they do not confirm any manuscript value. No historical D1 outcome, original four-channel stream, LLM inference, D2 deployment, or physical latency was regenerated in Phase 2.

## Outstanding design reconciliation

Phase 8 supersedes the Phase 1 confidence convention: the Decision-Agent score remains attached to its initial class, while `crew_support` separately records the fraction of matching votes. Neither is calibrated confidence in a crew-flipped final class. The existing R2 floor still reads the original model score, including on a flip; that policy meaning requires explicit manuscript review before adopting new action-release results.
