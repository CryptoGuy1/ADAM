# ADAM Phase 4 — degradation pipeline implementation report

**Date:** 25 September 2026. **Base:** `ADAM_reconciled_phase3.zip`.
**Scope:** New condition-specific degradation experiment runner and audit traces. This is a reference reconstruction, **not a rerun of historical D1/2025 experiments**. No remote GitHub or Zenodo mutation; V14 workbook and manuscript numerical tables/figures unchanged.

## Implemented

- `experiments/degraded_harness.py` now derives drift/noise scale from each clean trial's own per-node standard deviation, applies manuscript gain/offset ramps, .60-sigma Gaussian noise, separate signed 3.6-sigma impulses (3% Bernoulli event probability), and removes N4 after one-third of the trial. It retains NDIR reference and class labels and produces condition/trial seed and stream digests.
- New `experiments/run_degraded_reference.py` requires a complete, simultaneous long-format N1–N4 input CSV and a SHA-256 source manifest. Primary-channel workbook exports and missing/unaligned streams are rejected. An explicitly declared synthetic fixture needs `--allow-fixture`; user-declared source type alone is not independent physical provenance.
- Six systems receive the same condition-specific readings: Static Threshold, contextual RF and GBM, Single Agent, ADAM_GBM and ADAM_LLM. Fitted models train only on other clean trials in leave-one-trial-out evaluation, with fixed pooled clean fusion variances. Their fitted parameters are not retrained on perturbations.
- Both local-model configurations have a fresh call per event/condition. The full model-call record retains exact system/user prompts, retrieved record objects and IDs, raw output, seven-field decision, input/prompt/response digests, and fallback/repair indicators. Memory starts empty separately for each condition/trial/system and excludes same-time or future events.
- Crew variants run the Phase1 classification-vote and policy/persistence path. Event records show initial and final classification, per-role votes, failure status, action release and confidence-source. `UNRESOLVED` is not counted as `NORMAL`; output distinguishes coverage, resolved-subset F1 and full-coverage trial-mean F1.
- Fixed a newly discovered reference-code failure: with only 3 sensor readings, `outlier_z=1.5` exceeds the mathematical maximum sqrt(2) and the preexisting aggregator raised an exception. Reference implementation uses 1.25 for exactly 3 nodes and retains 1.5 for 4. This choice is **not verified historical behavior** and must be declared if new results are published.
- The reference implementation's `InMemoryStore` uses concentration-nearest search, not Weaviate embedding similarity. The deployment's semantic-memory behavior is **not** reproduced by these outputs.

## Validation

- Six new tests cover strict provenance, four-node completeness, time alignment, within-trial drift and sigma, dropout, unchanged NDIR reference, freshly paired condition calls, prompt/response hashes and causal retrieval. Complete regression suite: **132 passed**. Python compilation: passed.
- Full synthetic smoke execution used **72 simulated events across three trials** x **five conditions** x **six systems**, producing **2,160 event records** and **720 deterministic TEST-DOUBLE call records**. These are not Gemma calls, physical tests or historical F1 estimates; do not cite their scores.
- The test fixture and complete output are packaged separately with their source and output manifests for inspection.

## Still required for scientific use

1. Actual original or newly acquired synchronized, four-node raw per-event input plus physical provenance; V14 D1 workbook is not sufficient for historical four-node replay.
2. Local Ollama runtime and verified model digest to perform real fresh Gemma calls. Full 10 x 200 x 5 x 2 local-model evaluations require substantial run time. The current smoke was a deterministic fixture only.
3. Actual Weaviate retrieval backend and archived initial memory state/version/image digest for a faithful deployed-service re-evaluation. In-memory nearest-neighbor output must not substitute for this evidence.
4. Author approval of three-node outlier threshold, post-majority confidence semantics, and treatment of unresolved events for scientific summary metrics. Do not replace manuscript Table/Figure on the basis of fixture results.

**Next:** audit security Figure 10 and numerical security provenance; audit historical CPU/RSS/service traffic conventions. Then reconcile an accepted new experiment with the manuscript, finally freeze GitHub and Zenodo.
