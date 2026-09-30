# Phase 8 local reference implementation and release audit

This package continues from the locally reconciled Phase 7 ZIP. The historical
V14 workbook, GitHub, Zenodo, and manuscript numerical results were not edited.

## Confidence and class support

`DecisionObject.confidence` is retained from the initial Decision-Agent output
even when a crew majority selects the other class. `EventTrace.model_confidence`
copies that initial score. `EventTrace.crew_support` is the matching-voter count
divided by the eligible voting-role count. It is absent for unresolved votes.
`confidence_source=decision_agent_initial_class` marks a class flip. The
reconciled event trace, substitution export, degraded-reference output, and
Weaviate record expose the separate values. Existing Weaviate collections gain
the two new numeric properties when `ensure_schema()` runs.

The legacy `confidence` field in prediction and memory records, and the
`confidenceScaled` contract argument, remain the initial Decision-Agent score.
In a flipped event they do not measure confidence in the final class. The R2
governance floor still evaluates that initial score. This is a policy convention
that must be stated or revised before new action-release results are reported;
the current code does not establish calibration of either score.

## Dropout and scalability decisions

The reference Aggregator keeps a 1.5 standardized outlier threshold with four
active nodes and 1.25 with three. A three-node internally standardized outlier
cannot reach 1.5 under equal weighting (the upper limit is sqrt(2)); the 1.25
setting is a documented reconstruction, not a recovered historical parameter.
The added test checks both settings and an actual three-node detection.

The historical matched-level scalability presentation (2.373% MAPE, N=16
20.94 s) remains separate from the strict leave-one-physical-level-out
reference stage model (1.333% MAPE, N=16 20.890 s). No physical N>4 result is
inferred from the latter. Selecting a replacement manuscript presentation
requires an explicit Methods, figure, and result revision together.

## Verification and evidence limits

The Phase 8 release audit passed: 150 tests; manuscript workbook numerical
checks; 35 active statistical comparisons in 10+5+20 families; 28/28 source
security/resource checks; and the source-linked Figures 8 and 10 generators.
The bundled V14 workbook has SHA-256
`aa21f5c0272d6fcee27ef27b0a46d2af0fee09f82f1b36bd402b4084fc73629d`.
This verifies the local reference code and retained records, not historical
byte-identical execution. Complete synchronized N1–N4 D1 streams, historical
substitution ballots, degraded prompt/retrieval traces, individual D2 store
acknowledgments, and physical N>4 measurements remain unavailable.
