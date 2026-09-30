# Phase 15 conflict order and trace

Registered concurrent recommendations for the same location within the event
window are ranked by severity, then timestamp, before the current crew votes.
The selected decision proceeds through the ordinary class-majority and
governance checks. A policy rejection records a withheld action and cannot
release the competing recommendation.

The trace retains the current event's local reasoner output, the selected
pre-vote decision, and the selected source event identifier. These fields are
also stored in EventTrace memory records, with schema additions for existing
Weaviate collections. No result value or measured latency is used to choose
a recommendation.

The manuscript source was not edited. Manuscript wording for the vote formula,
confidence field, Aggregator vote, conflict scope, and holdout validation is
provided separately for the author to review and paste.
