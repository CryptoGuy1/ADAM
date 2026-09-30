# ADAM reference reconstruction — Phase 1 (classification voting)

This reconstruction does not retroactively establish the exact implementation
used in the historical deployment. It implements the final revised manuscript
protocol and is subject to fresh execution and data reconciliation.

- The Sensor votes on the local trigger or >=2x causal baseline; the Aggregator
  votes on the fused measurement crossing the 1,000-ppm operating point; the
  Decision Agent votes its initial class. Ground-truth NDIR values are not inputs.
- The Coordinator does not vote. Its quorum is floor(formed voter count/2)+1.
  Missing or failed votes do not silently become NORMAL or shrink the denominator.
- Both NORMAL and ANOMALY can win. Two-voter disagreement yields UNRESOLVED;
  action is withheld before policy validation and ledger execution.
- The trace distinguishes initial model output, individual class votes,
  final class, vote errors, and whether an action was released.
- When voting changes the class, severity/action follow the deterministic
  policy bands (NONE/continue monitoring; LOW/raise alert;
  HIGH/dispatch inspection; CRITICAL/escalate). The final decision's
  `confidence` remains the original Decision-Agent score for its initial class;
  it does not become confidence in the selected class. The trace records that
  score as `model_confidence` and the final-class vote fraction as `crew_support`.
  `confidence_source` explicitly marks flips as `decision_agent_initial_class`.
- The on-chain `approvals` argument is an inherited ABI field and now carries
  the number of votes supporting the final class. A NORMAL majority must
  pass the same quorum check as an ANOMALY majority.
- Early-return `classification_quorum` traces are observable to callers but
  are not claimed as successfully validated or persisted blockchain actions.
- Historical workbook and old figure data are untouched.

**Manuscript reconciliation required before submission:** Explicitly describe
that confidence on a crew flip still refers to the initial class, and specify the Sensor/
Aggregator independent classification rules; regenerate substitution outcomes
from a genuine complete per-node dataset before claiming parity.
