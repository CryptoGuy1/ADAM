# Current Figure 7 integration

The manuscript now uses the six-stage scale-out model and strict held-out-node-level validation. `make scalability-figure7` regenerates its software records and publication figure. `make deposit-figures` calls this target for Figure 7. The manuscript verification script checks this model's current result claims, while the former workbook software claims and fit parameters remain under `archived_scalability` in the claims JSON.

The model implementation and measured hardware input are unchanged. Added files provide the publication plotter and matching software outputs. Figure files are generated locally into the ignored `figures/` directory; the model-output CSVs and manifest are versioned under `data/scalability_stage_model/`.

The recalculated software estimates, updated figure, table, and Methods/Results text must be released together. The existing Zenodo workbook still contains the earlier software series; publish the current software outputs alongside it in a new version or clearly identified supplement. No physical measurement above four nodes is introduced by this update.
