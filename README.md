# ADAM — Agentic Decentralized Autonomous Machines

> **Reference reconstruction (September 2026):** The crew-voting and failure-closed
> protocol in `adam/agents/roles.py`, `adam/crew.py`, and `adam/schemas.py`
> implements the revised manuscript specification. It is not presented as a
> byte-identical reconstruction of lost historical deployment code. The
> canonical analytical source is the reconciled V14 workbook; historical
> measurements remain distinct from reconstructed/reference executions.

Reference implementation and analysis code for:

> **Agentic Decentralized Autonomous Machines (ADAM): Model-Flexible Edge Intelligence and Governance for DePIN Applications**
> Benjamin C. Nweke, Gholamreza Ramezan, and Soheil Saraji (2026)

ADAM is an event-driven edge architecture that separates physical sensing,
Decision-Agent reasoning, crew agreement, governance validation, and audit
logging. The evaluated prototype uses four Raspberry Pi 5 methane-sensing nodes,
local Gemma 3 1B inference through Ollama, shared semantic memory through
Weaviate, and permissioned governance through the Fides Innova testnet.

This repository is organized so a reviewer can trace:

**manuscript claim → named experiment → implementation → saved output → metric calculation → table/figure**.

## What is reproduced here

Three evidence categories must be kept distinct.

1. **Historical reported deployment evidence.** The deposited workbook contains
   the 459-event deployment records, stage latencies, resource measurements,
   security records, and the original benchmark prediction records used for many
   reported quantities. The historical 446/459 value is an **end-to-end
   completion rate**, not an independent measurement of dual-store persistence.

2. **Workbook-backed recomputation.** `scripts/verify_manuscript_numbers.py`
   recomputes the quantities the workbook supports, including the 11-system D1
   benchmark, the Decision-Agent substitution study, the degraded-condition
   trial summaries, deployment/resource/security quantities, and all three
   multiplicity families. The numbered data-driven figures use the workbook, except Figure 7,
   which uses stage-model outputs calibrated on the workbook hardware records. Deployment semantics are derived from the
   frozen benchmark predictions rather than by rerunning the language model.

3. **New/reference reruns.** Re-executing fusion-dependent experiments from raw
   sensor inputs requires the original concurrent N1–N4 D1 acquisition stream.
   The workbook preserves fused/dispersion features and deposited predictions or
   trial results, but not the complete simultaneous four-node raw stream needed
   to re-estimate fusion or recreate degraded perturbations from first principles.

The code does not reconstruct missing historical sensor streams or present newly
implemented reference behavior as proof of historical behavior.

## Headline manuscript results

The current manuscript reports:

- ADAM_LLM benchmark-mode F1: **0.896**.
- Static Threshold F1: **0.790**.
- Random Forest (raw) F1: **0.841**.
- Random Forest (fused) F1: **0.928**.
- Gradient Boosting (fused) F1: **0.931**.
- Decision-Agent substitution: ADAM_GBM reaches **0.950** mean F1.
- Revised deployment semantics: **F1 0.830**, **FAR 0.066**.
- Live deployment: **446/459 (97.2%)** events completed within the 30-s budget.
- Completed-event median decision latency: approximately **19.0 s**.
- Local reasoning accounts for approximately **81.5%** of mean completed-event
  latency.

Reported values are treated as evidence to be verified, not numerical targets to
reverse-engineer in code.

## Requirements

- Python 3.11 recommended
- Node.js 18+ for Solidity/Hardhat tooling
- Docker for the reference Weaviate service
- Ollama with `gemma3:1b` for local-LLM runs

Create an isolated environment before installing dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install --no-package-lock
```

## Fast integrity check

```bash
python -m adam.config
python -m pytest tests/ -q
python scripts/verify_manuscript_numbers.py data/ADAM_Dataset_Master_v14_reconciled.xlsx
python scripts/export_revised_statistics.py data/ADAM_Dataset_Master_v14_reconciled.xlsx
```

For the complete workbook-backed recomputation (checks, Appendix A, and current
data-driven figures):

```bash
make recompute
```

This is intentionally not described as physical re-execution. A current reference
rerun from raw sensor inputs uses `make reference-rerun` and requires the missing
concurrent four-node D1 stream plus the relevant runtime services.

The parity tests cover the active quorum rule, conflict rule, causal baseline,
semantic-memory temporal filtering, Decision-Agent feature shape, deterministic
deployment semantics, degraded-stream invariants, optional fold-local fusion
calibration, deadline fail-closed behavior, and Python/Solidity governance-policy
surfaces.

## D1 benchmark design

### Benchmark mode

The main comparison evaluates every labeled D1 event through the configured
system so the systems classify the same 2,000 events. The current benchmark
family contains one reference plus ten comparators:

- `adam_llm`
- `static_threshold`
- `random_forest_raw`
- `random_forest_fused`
- `gradient_boosting_fused`
- `cloud_only`
- `single_agent`
- `adam_no_aggregator`
- `adam_no_llm`
- `adam_no_blockchain`
- `adam_no_weaviate`

The ten paired comparisons against ADAM_LLM form the benchmark Holm family.

### Deployment semantics

The revised manuscript does **not** run Gemma a second time under a gate.
Deployment-semantics predictions are a deterministic transform of the frozen
benchmark predictions:

```text
raw MQ-4 >= 1000 ppm  -> preserve benchmark ADAM_LLM prediction exactly
raw MQ-4 <  1000 ppm  -> NORMAL; no crew/model invocation
```

This isolates the effect of screening from language-model stochasticity.

```bash
python scripts/derive_deployment_semantics.py \
  --data data/artifacts/d1_primary_channel.csv \
  --benchmark results/trials/predictions_adam_llm.jsonl \
  --out results/trials/predictions_adam_deployment_semantics.jsonl
```

Using the deposited frozen benchmark predictions, this derivation reproduces the
current manuscript operating point of approximately precision 0.904, recall
0.767, F1 0.830, and FAR 0.066.

## Fusion calibration

The reported D1 benchmark uses the fixed inverse-variance sensor weights preserved
in the deposited event records. Those weights were estimated once from the
labeled calibration data and therefore allow a held-out trial to contribute
indirectly to the pooled weight estimate; the manuscript discloses this as a
limitation.

`data/calibration.py` implements an optional fold-local sensitivity analysis in
which each held-out trial receives weights estimated only from the other nine
trials. A complete numerical re-fusion under those fold-local weights requires
the original concurrent N1–N4 acquisition stream, which is not preserved in the
workbook. The reference runners expose this behavior through
`--fold-local-calibration`; it is not presented as the historical reported run.

## Decision-Agent substitution study

The substitution study keeps the ADAM crew runtime fixed while changing the
reasoner assigned to the Decision Agent.

Standalone / in-crew pairs are:

- Static Threshold / ADAM_Static
- Gemma 3 1B / ADAM_LLM
- Logistic Regression / ADAM_LogReg
- Random Forest / ADAM_RF
- Gradient Boosting / ADAM_GBM

The fitted standalone contextual models and the corresponding in-crew Decision Agents use the same eight features:

1. raw concentration
2. normalized threshold distance
3. threshold indicator
4. fused concentration
5. cross-node dispersion
6. causal six-reading baseline mean
7. fused / baseline ratio
8. fused − baseline difference

Holding the representation fixed means the fitted standalone/in-crew comparison
does not obtain its crew gain by silently changing the classifier input features.
The surrounding crew adds coordination, retrieval, voting, governance, and trace
handling while the substituted fitted reasoner consumes the same contextual
feature vector.

```bash
python -m experiments.run_decision_agent_substitution \
  --data data/d1_four_node.csv \
  --main-results results/trials \
  --out results/decision_agent_substitution
```

The runner executes substituted fitted backends through `ADAMNode.handle_event`;
there is no special shortcut decision path.

## Degraded-condition study

`experiments/degraded_harness.py` applies perturbations to the raw per-node MQ-4
stream **before fusion and feature construction**. The NDIR reference and labels
are unchanged.

Conditions are:

- clean control
- mild drift
- strong drift
- measurement noise plus sparse impulses
- N4 dropout beginning approximately one third into the trial

Seeds are derived from `(condition, trial)`, never from the system being tested.
Each emitted condition–trial stream is SHA-256 hashed. Under dropout, N4 is
excluded; no stale N4 value is substituted, and fusion renormalizes over the
surviving nodes.

```bash
python experiments/degraded_harness.py --selftest
python experiments/degraded_harness.py \
  --input path/to/per_node_d1.csv \
  --outdir results/degraded_inputs
```

The workbook preserves the reported per-trial degraded-condition outcomes and
supports recomputation of their tables, statistics, and figure. The original
per-node D1 acquisition file is still required to recreate the perturbed sensor
streams and re-run the study from raw measurements.

## Semantic memory

The shared memory service keeps two lifecycles separate:

- `CrewEvent`: ephemeral active coordination state, cleared when the crew
  dissolves.
- `EventTrace`: resolved historical records available for semantic retrieval.

Retrieval for event time `t` is causal: only records with `timestamp < t` are
eligible. The reference benchmark does not manufacture semantic-memory records
from training-fold labels.

## Governance and conflict resolution

Crew quorum is computed over voting agents:

```text
gamma_crew = floor(n_voters / 2) + 1
```

The Coordinator tallies and does not vote. In the deployed full crew there are
three voters, so two matching class votes are required.
The optional reference `ConsensusValidator.sol` uses the same semantics on-chain:
its ballots are explicit NORMAL/ANOMALY class votes, quorum is computed over the
eligible voting set rather than all crew roles, and a two-voter split remains
unresolved. `GovernanceRules.validateDecision` receives the number of voters and
the support count for the selected final class as a defense-in-depth check.

The active conflict resolver is:

```text
higher severity wins;
if severity is equal, the newer recommendation wins.
```

There is no active lambda-weighted conflict sweep.

The Python `LocalValidator` and Solidity `GovernanceRules` expose the same
policy surface: recognized severity, confidence floor, permitted action,
critical-concentration passive-action rejection, CRITICAL human-review rule,
and explicit degraded-mode recording. Offline/reference experiments use
`LocalValidator`; a real Fides-backed run can pass the same
`FidesInnovaClient` instance as both validator and ledger client so
`GovernanceRules.validateDecision()` is evaluated through a read-only contract
call before `DecisionLogger.logDecision()` commits the approved trace.

## Deployment and persistence terminology

For the historical D2 deployment, the workbook supports **end-to-end completion**
over all 459 events. It does not preserve separate per-event acknowledgments for
both audit stores, so 446/459 must not be relabeled as measured dual-store
persistence reliability.

New/reference traces contain explicit `persisted_chain` and
`persisted_weaviate` fields. Those fields may be used to measure commit success
for a new run, but they do not retroactively change the interpretation of the
historical deployment.

## Environment and provenance

The reported deployment and reference-rerun environments are intentionally
separated:

- historical reported Weaviate: **1.21**
- current reference Weaviate: **1.30.2**

See `ENVIRONMENT.md`.

Before a new/reference experiment, write a provenance manifest:

```bash
python scripts/run_manifest.py \
  --dataset data/d1_four_node.csv \
  --out results/<run>/run_manifest.json
```

The manifest records Git state, Python/packages, platform, dataset SHA-256,
Ollama metadata when available, inference parameters, Docker version, Weaviate
versions, chain ID, contract configuration presence, and Solidity source hashes.

## Data

The current public data record is:

**Zenodo DOI: 10.5281/zenodo.21892655**

The canonical analytical workbook is `data/ADAM_Dataset_Master_v14_reconciled.xlsx`. Its workbook export is
explicitly **primary-channel only**:

```bash
make export-primary-data
```

That export is suitable for gate/label analyses such as deterministic deployment
semantics. The master workbook itself also preserves the reported contextual
fitted-model predictions/trial metrics, substitution results, and degraded-study
trial summaries, so those reported analyses can be checked without reconstructing
raw sensor streams. However, the primary-channel export is **not** sufficient to
re-execute cross-node fusion, fitted contextual systems from raw N1–N4 readings,
or degraded perturbations. Those raw reruns require the original concurrent
four-node D1 stream.

## Repository map

```text
adam/
  config.py                   constants and manuscript parity checks
  manuscript.py               workbook-backed reference calculations
  schemas.py                  SensorReading, DecisionObject, CrewEvent, EventTrace
  mechanisms.py               trigger, fusion, quorum helpers, conflict rule
  crew.py                     event lifecycle / ADAMNode.handle_event
  agents/roles.py             Sensor, Aggregator, Decision, Coordinator
  llm/prompt.py               runtime prompt + Appendix A generator
  llm/client.py               Ollama inference, repair retry, fallback
  memory/store.py             ephemeral coordination + causal semantic memory
  governance/chain.py         policy validator and chain clients

data/
  loader.py                   D1 loader, workbook primary-channel exporter, simulator
  calibration.py              optional fold-local fusion-calibration sensitivity

analysis/metrics.py           trial-level metrics and comparison utilities
analysis/revised_statistics.py active 10+5+20 exact-Wilcoxon/Holm families

baselines/systems.py          raw and fused baselines, cloud and single-agent comparators
ablations/systems.py          ADAM_LLM and architectural ablations

experiments/
  run_trials.py               11-system D1 benchmark
  run_decision_agent_substitution.py
  decision_agent_backends.py
  degraded_harness.py
  run_deployment.py           reference replay / scaling harness
  run_security.py             active security stress-test harness
  reproduce_security.py       workbook-backed security recomputation
  phase5_provenance.py        source-linked security/resource extraction
  phase5_claim_checks.py      manuscript claim checks for retained source records

scripts/
  derive_deployment_semantics.py
  verify_manuscript_numbers.py
  export_revised_statistics.py
  phase5_verify.py
  figure8_resources_publication.py
  figure10_security_publication.py
  run_manifest.py
  deploy.js
  verify_chain.py
  figure3_confusion_matrices.py
  figure4_operating_point.py
  figure_swap_study.py
  figure_degraded_conditions.py
  figure5_coordination.py
  figure6_resources.py
  figure7_scalability.py
  figure8_security.py

contracts/
  GovernanceRules.sol
  CrewRegistry.sol
  ConsensusValidator.sol
  DecisionLogger.sol

tests/
  test_manuscript_parity.py
```

## Manuscript ↔ code traceability

| Manuscript element | Reference implementation |
|---|---|
| Raw screening rule | `adam.mechanisms.trigger` |
| Inverse-variance fusion | `adam.mechanisms.fuse_readings` |
| Reported fixed fusion weights / fold-local sensitivity | event records; `data.calibration.calibrate_fold` |
| Decision-Agent reasoning | `adam.agents.roles.DecisionAgent.reason` |
| Crew quorum | `adam.config.quorum`, `contracts/GovernanceRules.sol` |
| Severity/recency conflict rule | `adam.mechanisms.resolve_conflict` |
| Full event workflow | `adam.crew.ADAMNode.handle_event` |
| Main D1 benchmark | `experiments.run_trials` |
| Deployment-semantics derivation | `scripts/derive_deployment_semantics.py` |
| Decision-Agent substitution | `experiments.run_decision_agent_substitution` |
| Degraded-input construction | `experiments/degraded_harness.py` |
| Deployment/scaling reference replay | `experiments.run_deployment` |
| Statistical metrics | `analysis.metrics` |
| Active 10+5+20 Wilcoxon/Holm families | `analysis.revised_statistics`, `scripts/export_revised_statistics.py` |
| V14 security/resource source audit | `scripts/phase5_verify.py` |
| Source-linked resource/security figures | `scripts/figure8_resources_publication.py`, `scripts/figure10_security_publication.py` |
| Appendix A | `python -m adam.llm.prompt --latex` |
| Provenance manifest | `scripts/run_manifest.py` |

## Scope and limitations

- Four physical Raspberry Pi 5 nodes in one laboratory.
- The four MQ-4 sensors and co-located NDIR reference were exposed in the same
  laboratory fume-hood test area; this is not a spatially distributed field deployment.
- One gas species and a restricted experimental concentration range.
- The approximately 19-s workflow is an early screening/accountability system,
  not a certified emergency shutdown or ignition-protection mechanism.
- Hardware node-count measurements end at N=4; larger node counts are model
  predictions.
- The targeted security tests are small stress tests, not claims of general
  Byzantine or adversarial robustness.
- Individual ballots are not cryptographically signed in the current prototype.
- The ledger establishes what was proposed, validated, and recorded; it does not
  prove that the physical sensor measurement was correct.

## License

Code: MIT (`LICENSE`).
Deposited data: CC BY 4.0 (`LICENSE-DATA`).


## Current Figure 7: hardware-calibrated six-stage scale-out model

The revised Figure 7 uses the stage model implemented in
`experiments/scalability_stage_model.py`. Calibration uses the 72 measured
Raspberry Pi records in `data/scalability_hardware_v14.csv`; its adjacent
manifest identifies the source workbook and input checksum. The physical
measurements remain unchanged.

```bash
make scalability-figure7
make verify-manuscript
```

`make scalability-figure7` generates the strict leave-one-level-out validation,
90 software replicates, five level summaries, and a manifest under
`data/scalability_stage_model/`. It then writes the current three-panel Figure 7,
its numerical summary, and the scalability table under `figures/`.
`make deposit-figures` uses this same target for Figure 7.

Model settings are seed 42, 18 replicates per level, a fixed four-event batch,
eight logical sensor streams, four reasoning workers, and 30,000 database
vectors. Non-negative coordination-stage slopes are fitted to resampled physical
levels 1–4. Each replicate shares a full four-node stage vector across logical
node levels; reasoning, retrieval, and blockchain stages retain that vector's
measured service values. Strict validation excludes each held-out node level
from coefficient and anchor estimation.

The current result claims are stored in `data/manuscript_result_claims.json`,
separately from the model code. `scripts/verify_manuscript_numbers.py` recomputes
the stage model and checks the rounded manuscript values against those claims.
The plotter checks input/output hashes and stage additivity before plotting.
Outputs above four nodes are conditional software estimates, not additional
physical device measurements.

The older `scripts/figure7_scalability.py` reproduces the workbook's previous
software series. It is retained for archive comparison and is no longer used
by the current Figure 7 target. The original workbook and its earlier fit
parameters are preserved; prior software claims are separately identified in
the claims JSON. The new outputs must accompany the revised manuscript in the
code/data release.

## Phase 4: new degraded-condition execution (not historical reproduction)

See [`docs/RECONSTRUCTION_PHASE4.md`](docs/RECONSTRUCTION_PHASE4.md) for the
four-node provenance gate, fresh per-condition LLM inference, causal retrieval
traces and full/partial-coverage scoring. `fixture` mode is a deterministic
software test double and **must not** be cited as the historical study.


## Phase 5 — security and resource provenance gate (September 2026)

`make phase5-audit` (or specify `PHASE5_MASTER=/path/to/approved.xlsx`)
checks the workbook fingerprint before exporting the four security-event sheets and
resource-log sheet read-only. Outputs include source hashes, row-level input exports,
recomputed denominators and metrics, numeric manuscript-claim checks, and candidate
Figure 10/8 visualizations. This is **workbook-backed recomputation only**, not a
rerun of the historical physical experiments or independent verification of all
network egress or per-store persistence. The bundled earlier workbook has a
different SHA-256 and is intentionally rejected by this V14-targeted command.

`scripts/phase5_verify.py --input-dir <csv_export> --source-type synthetic_fixture
--out <output> --plot` is a diagnostic mode. Fixture plots carry an explicit
non-publication label. Historical CSV mode requires the exact V14 workbook hash
and a matching manifest produced by the read-only exporter. The old
`scripts/figure8_security.py` and `scripts/figure6_resources.py` are retained as
legacy scripts but are no longer invoked by `make deposit-figures`; their old
numbering/captions and component-memory graphics are not current publication
figures. For Figure 8, the Phase 5 candidate shows measured state means without
pretending that the component budget is process-by-process measurement.

The new source audit does not infer an independent failed-fallback denominator
from `Fallback_Triggered`, which is itself an outcome field. It separately
reports crew continuation *among* fallback-triggered events. It also distinguishes
instrumented external inference bytes from total network traffic, and sampled
window-peak CPU from time-weighted utilization.


## Phase 6 — V14 evidence reconciliation and release-candidate gate

Phase 6 closes the workbook-backed security/resource gate and replaces reliance
on stale statistical summary sheets with direct recomputation from retained
trial-level records. The reconciled V14 workbook contains a historical
`04_D1_Statistical_Tests` sheet from an earlier comparison family and no
`18_Revised_Statistical_Tests` sheet. The active manuscript families are therefore
computed by `analysis/revised_statistics.py` and exported with:

```bash
make stats
```

The resulting families contain 10 main benchmark comparisons, 5 Decision-Agent
substitution comparisons, and 20 degraded-condition comparisons. Exact
Wilcoxon tests (after zero-pair removal), 10,000 paired bootstrap resamples, and
within-family Holm adjustments reproduce the current manuscript values after
rounding. The historical workbook sheet is left unchanged for provenance.

The local release-candidate evidence gate is:

```bash
make release-audit
```

It runs the current test suite, verifies workbook-backed manuscript quantities,
exports all 35 revised comparisons, executes the V14 security/resource source
audit, and regenerates the source-linked Figure 8 and Figure 10 candidates. A
passing local gate does not publish GitHub or Zenodo and does not imply that
missing historical raw streams, ballots, per-store acknowledgments, or runtime
images have been reconstructed. See `docs/RECONSTRUCTION_PHASE6.md`.
