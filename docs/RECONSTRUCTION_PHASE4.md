# Phase 4 — degraded-condition reference execution

**Status:** new implementation and synthetic smoke validation. This is not a
reproduction of the archived 10-by-200 degraded-condition study. Its historical
condition-specific N1–N4 readings, prompts, retrieved records and responses are
not available in the surviving master workbook. The workbook and historical
Table/Figure remain unchanged.

## What the executable experiment does

- Requires a complete **long-form four-node input**, with
  `event_id,trial,node_id,timestamp,raw_ppm,reference_ppm,trigger_node`
  and a sidecar `<input>.manifest.json` with `source` equal to
  `physical_per_node` or `synthetic_fixture` and `sha256` of the source CSV.
  A provenance *declaration* is not independent proof of physical acquisition.
  Primary-channel workbook exports are rejected, as are missing nodes,
  duplicate event/node rows, inconsistent references or unaligned timestamps.
- Uses the NDIR channel **only** as frozen scoring ground truth, never in
  agent-visible prompts or fitted features. The 1,000-ppm reference-label
  threshold and the raw-MQ-4 screening threshold refer to different channels.
- Produces fresh condition-specific per-node streams for clean, mild drift,
  strong drift, noise and N4 dropout. Drift is a linear gain/offset ramp across
  trial progress. The offset/noise scale is each node's **within-trial** clean
  standard deviation. Mild: a=.10, b=.5; strong: a=.35, b=2.0. Noise is
  Gaussian 0.60 sigma plus signed 3.6 sigma impulses with probability .03
  per node measurement; concentrations are clipped at zero. N4 is absent from
  the last two-thirds of the trial; only surviving sensors enter fusion.
- Reuses the same perturbed CSV for all configurations in a condition/trial.
  Conditions have independent deterministic seeds derived from (condition,trial).
  `harness_manifest.csv` records each stream's digest.
- Uses pooled clean residual variances for fixed fusion weights. Clean
  training trials exclude the held-out trial when fitting Random Forest and
  Gradient Boosting. Fitted parameters are **not** refit to degraded inputs.
  The pooled weight calibration is intentionally distinguished from fold-local
  model fitting; it contains the held-out-trial dependency discussed in the paper.
- Evaluates exactly six configurations: Static Threshold, contextual Random
  Forest, contextual Gradient Boosting, Single Agent, ADAM_GBM and ADAM_LLM.
  All receive condition-specific sensor inputs. Both local-model arms make a
  new call for each event/condition; original clean predictions are not reused.
- Resets memory for each (trial, condition, system), starts from empty pretrial
  state and accumulates only prior resolved events. The offline store uses
  **concentration-nearest** retrieval; it does not reproduce Weaviate's
  embedding similarity or the historical shared service. Every retrieval
  records the exact selected objects and IDs, not just a count.
- Runs the actual Phase-1 crew-classification and governance path for ADAM
  variants. An unresolved vote is not coerced to NORMAL; F1 is not presented as
  full-coverage F1 when classifications are unresolved. The three-node
  dropout-path outlier check uses z=1.25, below its mathematical ceiling
  sqrt(2); the four-node threshold remains z=1.5. This is a documented
  **reference code choice**, not proof of the 2025 historical setting.

## Outputs and their interpretation

`perturbations/degraded_streams.csv` and `harness_manifest.csv`: raw and
perturbed physical-node input, unchanged NDIR references, randomized seeds
and condition-trial digests.

`model_call_traces.jsonl`: complete system and user prompts, retrieval inputs,
raw responses, initial DecisionObjects, provenance hashes, and the exact model
call ID. In `fixture` mode these are **test-double outputs, not LLM inference**.
For `ollama`, a fresh local call is attempted for every event and condition.
The raw response and resulting fallback/repair state are retained separately.

`reference_event_traces.jsonl`: input features, class reference (scoring
metadata only), initial and final class, all class votes, agreement status,
validation and action status, retrieval hashes and trace hashes.

`reference_trial_scores.csv` and `reference_summary.csv`: trial-level metrics,
full input denominators, classified-subset coverage and separately labeled
full-coverage trial mean F1 only where every event has a resolved class.
These generated scores must never replace the original degraded-study table
without a separately accepted new experiment and a full manuscript revision.

`reference_run_manifest.json`: SHA-256 for input, output, perturbations and
trial score files, provenance declaration, reasoner type, condition names and
reason that offline memory is not equivalent to the physical deployment.

## Run instructions

Run tests on a machine with Python/scikit-learn/pandas. This phase's offline
fixture run does not need Weaviate or Ollama. Real inference needs the local
Ollama server and `gemma3:1b`, plus the original or genuinely new four-node
acquisition CSV and verifiable hardware and inference environment records.

```bash
python -m pytest tests/test_degraded_reference.py -q
python -m experiments.degraded_harness --selftest
python -m experiments.run_degraded_reference \
  --input path/to/per_node_readings.csv \
  --outdir results/new_degraded_ollama \
  --reasoner ollama
```

A declared synthetic source **requires explicit fixture acknowledgement**:

```bash
python -m experiments.run_degraded_reference \
  --input path/to/synthetic_per_node.csv \
  --outdir results/phase4_fixture \
  --reasoner fixture --allow-fixture
```

The full manuscript workload comprises 10 x 200 labeled events x five
conditions x two LLM configurations, so its full new evaluation requires
substantial repeated local inference time; the smoke test is not that run.
Before any new scientific report, record actual Ollama build/model digest,
Python/package versions, complete input files, Weaviate settings and full
prompt/retrieval outputs in the run manifest. The standalone vs in-crew
LLM prompts and semantic-memory mechanisms differ by design.

## Unresolved release gates

1. The historical concurrent N1–N4 streams and their original LLM call traces
   cannot be reconstructed from trial-level summaries. Preserve them as
   historic measured summaries, not rerun-derived outputs.
2. Offline retrieval remains **not equivalent to Weaviate embeddings**. For
   a new field/paper experiment, implement and log the actual shared Weaviate
   service, its version, module image digest and initial memory snapshot.
3. The reference model retains the explicitly reconstructed three-node outlier
   threshold 1.25 (four-node threshold 1.5). With three equally weighted nodes,
   an internally standardized outlier cannot reach 1.5 because its upper bound
   is sqrt(2). This choice is tested but is not a recovered historical setting.
   Phase 8 keeps the original model score and records the vote fraction as
   `crew_support`; the R2 policy meaning on a class flip still needs manuscript
   review. A vote fraction is not a calibrated class probability.
4. No empirical comparative degradation results were claimed from the
   synthetic fixture or deposited in place of workbook values.
