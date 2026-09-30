# Code, equipment, and reported results

The operational code should generate outputs from input data and configured
experimental conditions. Reported numerical outcomes belong in result records,
figures, and separate manuscript audit checks, not as target coefficients or
substitutes for measurements in the runtime.

This revision removes embedded scalability fit coefficients, archived
validation values, and measured deployment summary values from runtime
configuration and runner output. The deployment replay now processes all
triggering input records by default unless `--events` is specified. The
software-only in-process diagnostic is explicitly distinguished from physical
Raspberry Pi runs and the fixed-load stage model. Manuscript verification
scripts still compare workbook-derived numbers to reported claims; they never
feed the targets into the experiment.

The bundled workbook contains rows marked `HARDWARE` for N=1..4 with 18 records
per level and rows marked `PYTHON_SIMULATION` for the larger modeled counts.
The stage-model script consumes the hardware rows and computes software
estimates. The manuscript specifies four Raspberry Pi 5 nodes for the measured
domain and software participants above four nodes. The figure script reads the
workbook rows; the stage-model script generates a separate output set. Neither
execution path uses manuscript result values as calculation inputs.
