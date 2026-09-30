# Result claims kept separate from executable methods

`data/manuscript_result_claims.json` holds the numerical results and cohort
counts used to check the manuscript against the preserved workbook. The
verification scripts compute observed values from the supplied records and
compare them with this external file. The recorded claim values were moved
without recalibration or alteration.

The same claim file supplies the strict cohort checks for archived security
and resource exports and the result expectations in manuscript-facing tests.
Operational thresholds, hardware settings, mathematical rules, and small
synthetic unit-test inputs remain in code where they define behavior.

The manuscript was not edited. The saved claim file is an audit input, not a
source of predictions or a control parameter for the event-processing system.
