# Phase 5: Security and resource provenance verification

**Status:** code and synthetic workflow tests complete; V14 historical workbook import and full claim check must be executed successfully before using publication figures. No original field experiments have been rerun.

## Approved analytical source

- Original V14 reconciled workbook: `data/ADAM_Dataset_Master_v14_reconciled.xlsx`.
- Expected SHA-256: `aa21f5c0272d6fcee27ef27b0a46d2af0fee09f82f1b36bd402b4084fc73629d`.
- The separately bundled `data/ADAM_Dataset_Master.xlsx` is an older predecessor and **must not** silently substitute for V14.
- V14 includes analytical results and preserved event records, **not** all original per-event crew votes, synchronized raw N1--N4 streams, complete endpoint traffic, or independent per-store ACKs.

## Execution

```sh
python -m pytest tests/ -q
make phase5-audit
# or:
python scripts/phase5_verify.py \
  --workbook data/ADAM_Dataset_Master_v14_reconciled.xlsx \
  --out results/phase5 --check-manuscript --plot
```

The script performs a read-only five-sheet export, records original workbook and CSV SHA-256s in `source_manifest.json`, recalculates security and resource outcomes, writes `phase5_metrics.json`, and compares calculations with manuscript claims in `phase5_manuscript_claim_checks.json`. A mismatch exits nonzero and must be resolved against the source, **not** by editing the result to match the manuscript. Production figures are candidate Figure 10 and Figure 8 replacements and require final textual/caption reconciliation before publication.

CSV re-analysis is also supported, but an export claimed as historical must carry the matching exporter manifest and approved workbook digest. Synthetic input must be explicitly labeled `--source-type synthetic_fixture`, and its plots have a visible non-manuscript warning.

## What the verification means

- Injection detection is counted independently from correctness of final methane classification.
- Poisoning levels have *different event sets*; no paired treatment effect or general attack immunity can be inferred from their displayed scores.
- The number of entries where `Fallback_Triggered=Yes` is not an independent count of every induced-failure attempt. The runner reports continuity **among** fallback-triggered events instead of the tautological fraction n/n.
- Model-failure timings are only for the documented early-failure condition; late deadline-exhausting failures remain untested.
- Egress measurement means *externally addressed inference calls and bytes at instrumented call sites*, not complete external network traffic.
- Resource CPU means **mean of sampled 60-second window maxima** in each state, not average utilization across the deployment. Per-node memory means measured `RAM_MB`, whereas historical stacked component allocations were budget reconciliation, not per-process measurement. Bandwidth means accounted LLM, Weaviate, and blockchain bytes; it excludes uninstrumented packet/network overhead.
- 909 recorded windows across four nodes correspond to ~15.15 aggregate sampled hours / ~3.79 sampled hours per node; the timestamp span and coverage are computed separately.

## Outputs not yet authorized as manuscript evidence

Fixture-derived figures and metrics, original old `figure8_security.py` and `figure6_resources.py` output, and any rejected workbook version. The repository Makefile no longer calls those two legacy figure scripts through the `deposit-figures` target. The manuscript still references its historical figures and has not been silently changed.
