# ADAM: Agentic Decentralized Autonomous Machines

Reference implementation for:

> Nweke, B.C., Ramezan, G., Saraji, S. **Agentic Decentralized Autonomous Machines (ADAM): An Agentic AI Framework for Decentralized Physical Infrastructure Networks.** 2026.

A crew-based multi-agent framework for methane monitoring on a four-node
Raspberry Pi 5 testbed, combining on-device LLM reasoning (Gemma 3 1B via
Ollama), semantic memory (Weaviate), and blockchain governance (Fides Innova
PoA).

## Requirements

Python 3.9+, Node 18+ for the contracts, Docker for Weaviate.

```bash
pip install -r requirements.txt
npm install
```

## Quick start

Runs offline: no hardware, API keys, or network access.

```bash
python -m adam.config      # verify constants against the deposited dataset
python -m pytest tests/ -q # 63 tests
make offline               # trials, conflict sweep, security harnesses
```

`python -m adam.config` compares every constant in `adam/config.py` against
`data/ADAM_Dataset_Master.xlsx` and exits non-zero on any mismatch. Set
`ADAM_DATASET` if the workbook lives elsewhere; without it the structural checks
still run.

## Full reproduction

```bash
# services
docker compose up -d
ollama serve &
ollama pull gemma3:1b

# contracts
npx hardhat run scripts/deploy.js --network fides
# export the ADAM_ADDR_* values the script prints

make reproduce
```

## The two D1 runs

D1 is scored under two evaluation modes, both deposited, and the distinction
matters for reading Table 5:

* **Full pipeline** (`--eval-mode full_pipeline`). Every labeled event is
  replayed through the complete crew workflow regardless of the trigger, so all
  nine systems classify the same 2,000 events under identical conditions. This
  is the benchmark behind Table 5 (ADAM F1 = 0.896) and corresponds to
  `06A_Event_Predictions` in the workbook.
* **Gated** (`--eval-mode gated`). Deployment semantics: a reading below the
  1,000 ppm screening threshold never forms a crew and is classified normal on
  a millisecond fast path; only triggered readings receive aggregation and
  reasoning. This is the deployed operating point (F1 = 0.814) and corresponds
  to `D1_RawTrigger_Log`.

The gap between the two is the cost of the screening gate: below the threshold
nothing reaches the reasoner, so the gate caps recall while keeping sustained
resource use within budget. The live deployment runner (`run_deployment.py`)
is gated unconditionally.

```bash
make trials        # full-pipeline benchmark      -> Table 5
make trials-gated  # gated operating point        -> D1_RawTrigger_Summary
```

## Repository layout

```
adam/
  config.py               operational constants and the parity check
  manuscript.py           reference values recomputed from the dataset
  schemas.py              DecisionObject, CrewEvent, EventTrace
  mechanisms.py           Equations 1, 2, 4, 5
  crew.py                 Algorithm 1
  telemetry.py            per-stage timing, CPU accounting, egress ledger
  agents/roles.py         Sensor, Aggregator, Decision, Coordinator
  llm/prompt.py           prompt template; emits Appendix A
  llm/client.py           Ollama client, format repair, deterministic fallback
  memory/store.py         Weaviate CrewEvent and EventTrace classes
  governance/chain.py     policy validation and PoA client

baselines/systems.py      Static Threshold, Random Forest, Cloud-Only, Single-Agent
ablations/systems.py      full ADAM plus No-Aggregator, No-LLM, No-Blockchain, No-Weaviate
analysis/metrics.py       confusion matrices, Wilcoxon, Holm correction
data/                     loader, workbook exporter, label-integrity guard, simulator
contracts/                GovernanceRules, CrewRegistry, DecisionLogger, ConsensusValidator
experiments/
  reproduce_security.py   recomputes Section 4.5 from the deposited records
  run_trials.py           scores every system over D1 (both evaluation modes)
  run_deployment.py       D2 replay and the node-count scaling harness
  run_security.py         simulated attack harness (exercises the defences;
                          does not reproduce the published numbers)
  run_conflict_sweep.py   Section 4.6
scripts/
  make_manuscript_figures.py  Figures 3-8 from the dataset
  conflict_sensitivity.py     seeded sweep for Figure 9
  verify_chain.py             checks deployment events against the ledger
tests/                    parity suite
```

## Paper to code

| Paper | Code |
|---|---|
| Eq. 1, trigger at 1,000 ppm | `mechanisms.trigger` |
| Eq. 2, inverse-variance fusion | `mechanisms.fuse_readings` |
| Eq. 3, local reasoning | `agents.roles.DecisionAgent.reason` |
| Eq. 4, quorum floor(n/2)+1 over voters | `config.quorum`, `GovernanceRules.requiredQuorum` |
| Eq. 5, conflict resolution | `mechanisms.resolve_conflict` |
| Eq. 6, decision latency | `schemas.StageLatencies` |
| Algorithm 1 | `crew.ADAMNode.handle_event` |
| Table 5 | `analysis/metrics.build_table5` |
| Table 7 | `manuscript.node_scaling_fit`, `simulator_validation` |
| Table 8 | `config.quorum`, `tolerated_faults` |
| Figures 3-8 | `scripts/make_manuscript_figures.py` |
| Figure 9 | `scripts/conflict_sensitivity.py` |
| Table 9, Figure 8 | `experiments/reproduce_security.py` |
| Appendix A | `python -m adam.llm.prompt --latex` |

## Datasets

`data/ADAM_Dataset_Master.xlsx` holds both datasets:

* **D1**, 10 trials of 200 labeled events (2,000 total; 900 anomaly, 1,100
  normal). Labels derive from the co-located NDIR reference analyzer, never
  from the MQ-4 readings under evaluation. Both D1 runs are recorded: the
  full-pipeline predictions in `06A_Event_Predictions` and the trigger-gated
  run in `D1_RawTrigger_Log`.
* **D2**, 459 deployment coordination events with six per-stage latencies each,
  446 of which completed end to end within the 30-second deadline.

Also deposited at <https://doi.org/10.21227/hyqx-bn32>.

```
sha256  9a9e1e77eb87e9a45743e95e662d3bc8226ab19d572e2cfc0d5715ef4fd9af28
```

Two things to know when working with D1.

**Both systems receive the same raw input.** The Static Threshold baseline is
`Raw_Instantaneous_PPM` compared against 1,000 ppm (F1 = 0.790, FAR = 0.165),
and the same raw sample is what gates ADAM's crew formation. `Start_PPM` and
`End_PPM` describe the exposure profile of the event and are not detector input
channels. Per-node error variances for Equation 2 come from residuals of the
raw readings against the reference; on this testbed the four variances are
closely matched, so the fusion weights are near uniform (ratio about 1.09).

**`data/validate.py` rejects any D1 whose labels are recoverable from the
screening rule**, and has no override. On the deposit, agreement between the
raw threshold rule and the labels is 0.813, which is what leaves detection
headroom above the fixed rule.

## Scalability design

The scalability study varies participating node count N while holding load
fixed at the reference configuration (4 concurrent events, 8 sensor streams,
4 logical workers, 30,000 vectors), so latency changes attribute to node
count. N = 1-4 are physical Raspberry Pi 5 runs; N = 6, 8, 12, 16 come from a
Python scale-out model validated against the matched N = 1-4 hardware runs
(decision-latency MAPE 2.4%, bias below 0.1%). `08_Scalability_Log` flags every
row with its `Run_Mode`, and `09_Fitted_Models` holds the fitted relations
T(N) = T0 + alpha * N^beta for both domains.

## Notes on scope

* Four physical nodes, one gas species, one site, 72 hours.
* Node counts above four are model-based scale-out, not hardware, and are
  flagged per row in `08_Scalability_Log`.
* The poisoning trial is underpowered at 7 to 8 events per level; no effect was
  detectable (Fisher exact, p = 0.47).
* Equation 5 never fired during the deployment. Section 4.6 characterises it
  through the seeded sweep in `scripts/conflict_sensitivity.py`.
* Quorum is computed over the agents that cast a ballot, not the agents in the
  crew. The Coordinator tallies and does not vote, so a four-agent crew
  supplies three ballots and the deployed threshold is quorum(3) = 2: any two
  of the three role-specific checks must agree, no single voter can approve an
  action alone, and one unavailable voter cannot block the other two. Two
  colluding voters can supply quorum, which is the integrity bound Table 8
  states. `config.DEPLOYED_VOTER_COUNT` records the voter count.
* Table 8's bounds assume votes are attributable to distinct registered agents.
* The ledger records what was decided and on what evidence, not whether the
  measurement was true.
* `mechanisms.fuse_readings` rejects an `outlier_z` at or above sqrt(n-1), which
  is 1.732 for four nodes. Above that ceiling no reading can be flagged.
* No per-decision dollar cost is reported for the Cloud-Only comparator: the
  deposit contains no token or billing records. The measured comparison is
  external egress (zero for ADAM in all 12 windows; about 117 KB and 19 API
  calls per 30-minute window for Cloud-Only).

## License

MIT for code (`LICENSE`), CC BY 4.0 for the datasets (`LICENSE-DATA`). Citation
metadata in `CITATION.cff`.
