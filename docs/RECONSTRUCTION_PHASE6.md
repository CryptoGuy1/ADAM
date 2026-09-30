# ADAM Phase 6 — V14 Evidence Reconciliation and Release-Candidate Audit

**Date:** 28 September 2026
**Status:** Complete locally. Remote GitHub and Zenodo remain unchanged.

## Purpose

Phase 6 closes the historical-evidence gate left open at the end of Phase 5. The goal was not to rerun the May 2025 physical experiment, but to verify every manuscript quantity that the reconciled V14 workbook can actually support, correct the active statistical-family implementation, and generate the resource/security publication figures directly from recorded source data.

## Canonical evidence source

- Workbook: `data/ADAM_Dataset_Master_v14_reconciled.xlsx`
- SHA-256: `aa21f5c0272d6fcee27ef27b0a46d2af0fee09f82f1b36bd402b4084fc73629d`
- Older repository workbook SHA-256: `d079ead317a67865251438b3cafc7eceee834cb46ce1b4c90f69060bde4bb6dd`

The older workbook remains provenance history and is not allowed to silently replace V14 in the release-candidate verification path.

## 1. Phase 5 source gate closed

The V14 workbook was successfully imported and the security/resource source sheets were verified. The Phase 5 checker now runs directly against V14 and reports:

- security/resource manuscript claim checks: **28/28 passed**;
- sensor-injection detection: **27/30**;
- fallback-triggered events: **19**, all **19** continued through the crew pipeline in the retained records;
- resource windows: **909** = 256 idle + 434 monitoring + 219 crew-active;
- mean sampled non-inference window-peak CPU: **21.807%**;
- fallback latency mean/median/P95: **55.721 / 54.600 / 81.640 ms**;
- Cloud-Only external inference traffic: **117.404 KB per 30-min window** mean;
- ADAM_LLM external inference bytes: **0** in the 12 instrumented inference windows.

These are recomputations from retained analytical records. They do not recover missing synchronized N1–N4 raw streams, historical per-agent ballots, independent per-store acknowledgments, packet-level network capture, or the historical runtime image.

## 2. Statistical-family inconsistency repaired

The reconciled V14 workbook contains a historical `04_D1_Statistical_Tests` sheet corresponding to an earlier comparison family and does **not** contain an `18_Revised_Statistical_Tests` sheet. The previous verifier therefore could not serve as the authoritative implementation of the revised manuscript's 10 + 5 + 20 comparison families.

Phase 6 adds `analysis/revised_statistics.py` and `scripts/export_revised_statistics.py`. The active test families are now recomputed directly from the preserved per-trial outcomes using:

- exact two-sided Wilcoxon signed-rank tests after removing zero-difference pairs;
- percentile bootstrap 95% confidence intervals from 10,000 paired-trial resamples;
- Holm adjustment within each manuscript-defined family.

Verified family sizes:

- Main benchmark: **10 comparisons**;
- Decision-Agent substitution: **5 comparisons**;
- Degraded conditions: **20 comparisons**.

Key exact values reproduce the manuscript after rounding:

- main significant comparisons: `p_Holm = 0.01953125` → **0.020**;
- Cloud-Only and ADAM-No-Blockchain: `p_Holm = 0.09765625` → **0.098**;
- substitution Static and Gemma: `p_Holm = 0.009765625` → **0.010**;
- substitution LR/RF/GBM: `p_Holm = 0.4921875` → **0.492**;
- degraded-condition family: `p_Holm = 0.0390625` → **0.039** for the retained contrasts.

The 35 active comparisons are exported to `results/revised_statistical_tests.csv` and `.json`. The stale historical workbook statistics sheet is not rewritten or presented as the active revised family.

## 3. Workbook-backed manuscript verification

`scripts/verify_manuscript_numbers.py` now targets V14 by default and recomputes the active statistical families instead of relying on the stale workbook statistics sheet. The complete source-backed numerical verification finished with:

**ALL CHECKS PASSED**

This includes benchmark metrics, screening-gate analysis, deployment latency/completion, resource measurements, archived scalability values, security records, substitution results, degraded-condition summaries, and the 10 + 5 + 20 statistical families.

## 4. Publication Figure 8 and Figure 10

Two source-linked publication generators are included:

- `scripts/figure8_resources_publication.py`
- `scripts/figure10_security_publication.py`

Figure 8 uses all 909 retained resource windows. Panel (a) shows sampled window-peak CPU over the actual ~58.45-hour timestamp span. Panel (b) keeps the measured per-node state totals separate from the reconciled component estimates. A timestamp-decoding issue found during visual QA was corrected: Excel timestamps may be decoded by pandas as datetimes and are now converted to elapsed hours correctly. Component colors/legend are consistent across operating states.

Figure 10 is generated from the retained 19 fallback latencies and 20 external-inference windows. The zero local value is explicitly external inference traffic, not total network egress.

Both PNG and PDF versions were regenerated and visually inspected.

## 5. Repository changes

Phase 6 adds or revises:

- `analysis/revised_statistics.py`
- `scripts/export_revised_statistics.py`
- `scripts/verify_manuscript_numbers.py`
- `scripts/figure8_resources_publication.py`
- `scripts/figure10_security_publication.py`
- `adam/manuscript.py`
- `tests/test_revised_statistics.py`
- V14 defaults in data-driven figure scripts
- `Makefile` release-audit targets
- `README.md` and `CITATION.cff`
- `docs/RECONSTRUCTION_PHASE6.md`

The user-confirmed apparatus wording is retained as a shared laboratory **fume-hood test area**, not a separate sealed exposure chamber.

## 6. Local release-candidate gate

`make release-audit` completed successfully after the final figure correction:

- configuration/manuscript invariant check: **passed**;
- automated test suite: **145 passed**;
- workbook-backed manuscript numerical audit: **passed**;
- revised statistics export: **35 comparisons** with family sizes 10/5/20;
- Phase 5 historical security/resource audit: **28/28 claims passed**;
- Figure 8 source generation: **passed**;
- Figure 10 source generation: **passed**.

This gate verifies the local release candidate. It does not upload, publish, or modify GitHub or Zenodo.

## 7. Evidence boundaries that remain

The release candidate can recompute the historical results that are preserved in V14, but it cannot recreate from first principles evidence that is no longer preserved. In particular:

- complete original synchronized four-node D1 acquisition streams are unavailable;
- historical event-level substitution ballots are unavailable;
- historical degraded-condition prompts/retrieval/model-call traces are unavailable;
- D2 does not independently record acknowledgments from every required audit store or a complete Event-ID-to-transaction mapping;
- scale-out above four physical nodes remains model-based;
- reference reconstruction behavior must not be described as byte-identical historical execution.

These limitations are deliberate provenance boundaries, not test failures.

## Recommended next step

Treat the Phase 6 ZIP as the code release candidate. Before a public GitHub/Zenodo release, compare it against the user's **latest locally edited manuscript source**, replace manuscript Figure 8 and Figure 10 with the source-linked versions if accepted, reconcile the scalability text if the new Phase 3 model is to be adopted, then freeze commit/tag/dataset hashes and update the Data Availability statement with those final identifiers.
