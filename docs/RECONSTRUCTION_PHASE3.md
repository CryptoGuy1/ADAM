# Phase 3 — fixed-load six-stage scalability reference model

This is a **new reference-model execution**, separate from the earlier recorded
scale-out simulation in the historical workbook. Only the `Run_Mode=HARDWARE`
rows of the V14 `08_Scalability_Log` enter this model's calibration and
strict held-out validation. Nothing is written to the historical workbook,
Figure 7 source, or Zenodo.

## Physical input and traceability

`data/scalability_hardware_v14.csv` contains 72 recorded Pi 5 hardware runs:
18 at each N=1,2,3,4. Its sidecar `.manifest.json` records the source V14
workbook's filename, SHA-256, sheet, extraction filter, data SHA-256 and
counts. The local candidate V14 workbook is bundled under
`data/ADAM_Dataset_Master_v14_reconciled.xlsx` and retains its original hash.
It is not asserted to be the current Zenodo deposit. The six recorded stages add exactly to each recorded total (within
floating-point roundoff). If the V14 workbook changes, regenerate and verify
this extraction; do not silently reuse a stale CSV and provenance claim.

The artifact-tool workbook reader successfully inspected sample cells but
closed its backend when asked to materialize the full table. The source was
therefore **read-only** extracted directly from cached OOXML cells of sheet 12
using Python's standard-library ZIP/XML reader. The known sheet's header and
72-row hardware filter were checked against the artifact-tool preview.

## Generative assumptions

The reference workload is a fixed simultaneous four-event batch, eight
logical sensor streams, four reasoning workers, and one shared 30,000-vector
store. More logical nodes **do not** create more physical Pis, more
inference workers, more concurrent events, or more blockchain requests. The
model does not introduce WAN propagation, partitions, packet loss, variable
network capacity, growing database size, or additional ledger queueing.

For each of 18 generated replicates, 18 full six-stage physical vectors
are drawn *with replacement* from each available physical node level. Three
coordination-stage increments—cross-node exchange, network transfer, result
merge—are estimated by nonnegative least-squares slopes on the resampled
node-level means. One complete N=4 stage vector is drawn from the N=4
bootstrap sample. That **same** vector and fitted slopes are reused at
N=4,6,8,12,16 within the replicate. The query, reasoning, and blockchain
stages retain the drawn N=4 values. Every emitted total is the sum of its six
nonnegative components. Each row is explicitly labeled
`REFERENCE_MODEL_ESTIMATE` with its physical anchor run ID and seed.

These generated values are *not* the archived `PYTHON_SIMULATION` records.
The power-law and linear fits summarize the five generated level means and
never generate stage latencies; the generative model is linear in additional
logical nodes by construction. A perfect fitted R-squared against those five
means is consequently **tautological**, not independent model validation.

## Strict validation

Each physical N in 1--4 is withheld once. The slope, baseline and empirical
vectors come exclusively from the other three physical levels. The nearest
available training level provides the stage baseline. A single physical node
has zero cross-node exchange by definition, independently of recorded held-out
measurements. The resulting prediction is compared with the withheld physical
level mean. The output reports both total-latency MAPE/signed bias and
per-stage absolute errors: aggregate accuracy alone can conceal compensating
stage errors. This evaluation tests interpolation / short-range extrapolation
*within* N=1..4, not performance at 6..16 actual physical nodes.

The historical workbook's **2.373% matched-level MAPE** is a different
comparison, performed on its archived simulator records. It is never used as
a coefficient, a pass threshold, or a new strict-validation result.

## Files and reproduction

```bash
python -m experiments.scalability_stage_model \
  --hardware data/scalability_hardware_v14.csv \
  --out results/phase3
python -m pytest tests/test_scalability_stage_model.py -q
python scripts/plot_reference_stage_scalability.py \
  --hardware data/scalability_hardware_v14.csv \
  --level-means results/phase3/stage_scaleout_level_means.csv \
  --out results/phase3
```

The initial deterministic run is recorded under `results/phase3/`, with a
SHA-256 manifest for all CSV outputs. Preview charts deliberately do not
replace archived Figure 7. Workbook values, plots, manuscript figures and
historical statistics are not silently modified.

## Manuscript reconciliation before submission

If this reference simulator becomes the reported method, replace the archived
2.37% matched-level result with a *distinctly labeled* strict validation
result, and replace the historical extrapolation rows/curve with the new
model outputs. Keep the original hardware records unchanged. Explicitly state
the positive fixed-load assumption and that no physical N>4 experiments were
performed. Do not present this as validation of sixteen physical nodes.
