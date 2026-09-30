# Phase 12 semantic-memory record consistency

The EventTrace checksum is recomputed at ingestion and retrieval from the
tagged event identifier, trigger node/concentration, fused concentration,
classification, and final action. The Weaviate query now returns each field
needed for that check. A record changed after tagging is rejected.

The checksum is not a signature or access-control mechanism. The direct-write
poisoning scenario computes a matching checksum for its fabricated records,
then places them before the replayed test events in event time. A small
fixture run verifies the scenario can execute with the revised validation.

The manuscript and retained results were not edited. No reported result
value is encoded in this validation path.

The prior handoff's confidence separation and three-node outlier setting are
implemented in Phase 8 and covered by tests. Historical scalability analysis
and the strict holdout software model remain separate. The local release
cleanup and eventual remote repository update remain subsequent work.
