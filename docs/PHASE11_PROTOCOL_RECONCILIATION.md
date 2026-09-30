# Crew ballot and confidence semantics

The formed crew has an eligible voting set of Sensor, Aggregator, and Decision
roles. The Coordinator counts ballots and does not vote. A role that fails to
return a valid ballot contributes neither an ANOMALY nor a NORMAL vote. The
quorum denominator remains the number of eligible voting roles in the formed
crew. Two matching votes resolve a three-voter crew; one NORMAL and one
ANOMALY vote with a missing third ballot leave the class unresolved.

The Decision Agent's emitted confidence is retained after voting, including
when the crew changes the class. The event trace stores it as
`model_confidence` and stores the fraction of eligible roles supporting the
final class as `crew_support`. Governance R2 evaluates the initial
Decision-Agent confidence field against its configured floor. The vote
fraction supplies the quorum count; it does not replace that confidence field.
The on-chain call receives the initial confidence and final-class support as
separate arguments.

`tests/test_classification_voting.py` covers a missing ballot, a crew class
flip with a score below the R2 floor, and the separate on-chain arguments.
