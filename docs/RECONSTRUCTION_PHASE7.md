# ADAM Phase 7 — Final Protocol/Contract Consistency Hardening

**Date:** 28 September 2026
**Status:** Complete locally. Remote GitHub and Zenodo remain unchanged.

## Purpose

Phase 7 performs the last code-level consistency pass after the V14 evidence reconciliation. The focus is not new experimentation. It is to ensure that the reconstructed Python workflow, Solidity reference contracts, documentation, and verification language all implement the same **classification-voting** protocol now stated in the manuscript.

## 1. Classification-voting semantics aligned across Python and Solidity

The reconstructed Python workflow already used explicit `NORMAL` / `ANOMALY` ballots and treated a two-voter split as unresolved. The optional on-chain `ConsensusValidator.sol` still used legacy `approve/reject` terminology and counted approvals against crew size. That was inconsistent with the manuscript and with the corrected Python implementation.

Phase 7 rewrites the reference Solidity voting layer so that:

- every ballot is an explicit class vote (`ANOMALY` when `true`, `NORMAL` when `false`);
- the contract tracks `anomalyVotes` and `normalVotes` separately;
- either class must independently reach strict-majority support;
- a two-voter split remains `CLASS_UNRESOLVED`;
- quorum is calculated over the **eligible voting set**, not the full crew size;
- the eligible set is checked as a duplicate-free subset of the registered crew;
- the eligible set is restricted to the Sensor, Aggregator, and Decision roles; the Coordinator cannot be counted as a voter;
- `GovernanceRules.validateDecision` now receives `voterCount` and `classSupport` rather than the legacy `crewSize` / `approvals` names.

This preserves the Coordinator as a non-voting tallying role and makes the Solidity reference semantics match the Python protocol.

## 2. Legacy manuscript-number references removed from active code comments

The revised manuscript has changed equation/table numbering during the rewrite. Several source comments still referred to old equation or table numbers such as “Equation (4)”, “Table 6”, “Table 7”, and “Table 8”. Those references could become wrong again if the manuscript is renumbered.

Phase 7 replaces those active-code references with stable semantic language, e.g.:

- “strict-majority classification quorum”;
- “deployment latency decomposition”;
- “archived node-scaling fit”;
- “deterministic conflict resolution”.

Legacy plotting scripts retained only for provenance are explicitly marked as legacy and are not part of the active publication figure path.

## 3. Apparatus wording kept consistent

The repository documentation now uses the user-confirmed wording:

> the four MQ-4 sensors and co-located NDIR reference were exposed in the same laboratory fume-hood test area

It does not introduce a distinct sealed exposure chamber.

## 4. New contract-semantic regression tests

Added `tests/test_contract_classification_voting.py` to guard the exact protocol properties that previously drifted:

- both event classes are counted independently;
- unresolved class state exists;
- quorum uses `eligibleVoters.length` rather than total crew size;
- `GovernanceRules` receives final-class support;
- the Coordinator is recognized as a crew role but excluded from the eligible voting set.

These are source-level tests so the verification suite does not require a Solidity compiler in every Python-only environment.

## 5. Final local verification

The complete local release-candidate gate now reports:

- automated tests: **149 passed**;
- workbook-backed manuscript numerical audit: **ALL CHECKS PASSED**;
- V14 security/resource manuscript checks: **28/28 passed**;
- revised statistical families: **35 comparisons** = 10 main + 5 substitution + 20 degraded;
- source-linked Figure 8 generation: **passed**;
- source-linked Figure 10 generation: **passed**;
- release-candidate evidence gate: **passed**.

The successful gate is a local code/evidence consistency check. It does not imply a rerun of the historical May 2025 physical experiment.

## 6. Remaining evidence boundaries

The following are still deliberately not reconstructed or fabricated:

- original synchronized raw N1–N4 D1 acquisition streams;
- historical event-level substitution ballots;
- historical degraded-condition prompt/retrieval/model-call traces;
- independent per-store acknowledgments for D2;
- complete Event-ID-to-transaction mapping;
- physical scalability above four Raspberry Pi nodes;
- byte-identical historical software/runtime image.

These are manuscript limitations, not code failures.

## 7. Release status

The Phase 7 ZIP is now the strongest local code release candidate produced in this reconstruction sequence. Before public release, the remaining tasks are editorial/release tasks rather than implementation fixes:

1. obtain the user's latest post-audit manuscript source and compare it against the final code semantics;
2. decide whether the historical scalability presentation or the separately reconstructed Phase 3 stage model will appear in the paper;
3. replace Figure 8 and Figure 10 in the manuscript with the source-linked versions if accepted;
4. freeze the final repository commit/tag and dataset hashes;
5. update GitHub and Zenodo only after those identifiers are frozen;
6. update the Data Availability statement with the final release identifiers.
