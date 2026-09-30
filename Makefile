# ADAM verification, workbook recomputation, and reference-rerun targets.
#
# Important evidence distinction:
# - `make recompute` verifies the deposited workbook and regenerates the current
#   data-driven manuscript figures. It does NOT re-execute the historical
#   physical experiment or re-fuse D1 from raw N1-N4 streams.
# - `make reference-rerun` requires the original concurrent four-node D1 event
#   stream plus the runtime services needed by the selected systems.

PY ?= python3
DEPOSIT ?= data/ADAM_Dataset_Master_v14_reconciled.xlsx
PHASE5_MASTER ?= data/ADAM_Dataset_Master_v14_reconciled.xlsx
DATA ?= data/d1_four_node.csv
PRIMARY_DATA ?= data/artifacts/d1_primary_channel.csv
FIXTURE := data/artifacts/d1_simulated.csv
RESULTS ?= results
DEPOSIT_FIGURES ?= figures

.PHONY: help install test verify verify-manuscript fixture export-primary-data check-data \
        appendix appendix-file offline recompute deposit-figures diagnostics \
        trials substitution degraded-inputs deployment-replay scalability-reference \
        security-reference reference-rerun contracts clean reproduce phase5-audit stats publication-figures release-audit

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-22s %s\n", $$1, $$2}'

install:  ## install Python and Node dependencies (creates local npm metadata if absent)
	$(PY) -m pip install -r requirements.txt
	npm install --no-package-lock

verify:  ## check code constants against manuscript-derived invariants
	$(PY) -m adam.config

verify-manuscript:  ## verify all workbook-backed reported quantities and test families
	$(PY) scripts/verify_manuscript_numbers.py $(DEPOSIT)

test: verify  ## run manuscript-parity and runtime regression tests
	$(PY) -m pytest tests/ -q

fixture:  ## generate synthetic D1 fixture (NOT paper evidence)
	$(PY) -m data.loader --simulate --out $(FIXTURE)

export-primary-data:  ## export workbook primary MQ-4 channel only (NOT fusion-capable)
	$(PY) -m data.loader --export $(DEPOSIT) --out $(PRIMARY_DATA)

check-data:  ## require a concurrent multi-node event file for fusion-dependent reruns
	$(PY) -m data.loader --check $(DATA)

appendix:  ## print Appendix A LaTeX from the live Decision-Agent prompt
	$(PY) -m adam.llm.prompt --latex

appendix-file:  ## regenerate repository appendix_a.tex from live code
	$(PY) -m adam.llm.prompt --latex > appendix_a.tex

# ---------------------------------------------------------------------------
# Deposited-evidence path: no hardware, Ollama, RPC, or raw N1-N4 stream needed
# ---------------------------------------------------------------------------

deposit-figures:  ## regenerate current data-driven manuscript figures from workbook
	mkdir -p $(DEPOSIT_FIGURES)
	$(PY) scripts/figure3_confusion_matrices.py $(DEPOSIT) $(DEPOSIT_FIGURES)
	$(PY) scripts/figure4_operating_point.py $(DEPOSIT) $(DEPOSIT_FIGURES)
	$(PY) scripts/figure_swap_study.py $(DEPOSIT) $(DEPOSIT_FIGURES)
	$(PY) scripts/figure_degraded_conditions.py $(DEPOSIT) $(DEPOSIT_FIGURES)
	$(PY) scripts/figure5_coordination.py $(DEPOSIT) $(DEPOSIT_FIGURES)
	@echo "Resource Figure 8 requires the Phase 5 source audit; legacy figure6 is not a publication figure."
	$(PY) scripts/figure7_scalability.py $(DEPOSIT) $(DEPOSIT_FIGURES)
	@echo "Security Figure 10 requires the Phase 5 source audit; legacy figure8 is not a publication figure."

phase5-audit:  ## verify approved V14 security/resource records and generate source-linked figure candidates
	$(PY) scripts/phase5_verify.py --workbook $(PHASE5_MASTER) --out $(RESULTS)/phase5 --check-manuscript --plot

recompute: test verify-manuscript appendix-file deposit-figures  ## recompute workbook-backed checks/figures (not physical re-execution)
	@echo "Workbook-backed recomputation complete. This does not re-execute the historical physical deployment or raw four-node fusion."

stats:  ## recompute the 10+5+20 trial-level comparison families from V14
	$(PY) scripts/export_revised_statistics.py $(DEPOSIT) --out $(RESULTS)/revised_statistical_tests.csv

publication-figures: phase5-audit  ## regenerate source-linked resource/security publication figures
	$(PY) scripts/figure8_resources_publication.py $(DEPOSIT) $(RESULTS)/publication_figures
	$(PY) scripts/figure10_security_publication.py $(DEPOSIT) $(RESULTS)/publication_figures

release-audit: test verify-manuscript stats phase5-audit publication-figures  ## local release-candidate evidence gate
	@echo "Release-candidate evidence gate passed locally. This does not publish GitHub or Zenodo."

# Backward-compatible convenience target. Deliberately maps to recomputation,
# not to a claim of full experimental reproduction.
reproduce: recompute  ## backward-compatible alias for workbook-backed recomputation

# ---------------------------------------------------------------------------
# New/reference execution path: requires original concurrent N1-N4 D1 stream
# ---------------------------------------------------------------------------

offline: test fixture  ## exercise reference code on synthetic data; not manuscript reproduction
	$(PY) -m experiments.run_trials --data $(FIXTURE) --no-llm --skip cloud_only --out $(RESULTS)/trials_fixture
	$(PY) -m experiments.run_security --data $(FIXTURE) --no-llm --out $(RESULTS)/security_fixture

trials: check-data  ## run the 11-system reference benchmark on concurrent four-node D1
	$(PY) -m experiments.run_trials --data $(DATA) --eval-mode full_pipeline --out $(RESULTS)/trials

substitution: check-data  ## run Decision-Agent substitution on concurrent four-node D1
	$(PY) -m experiments.run_decision_agent_substitution --data $(DATA) --out $(RESULTS)/decision_agent_substitution

degraded-inputs: check-data  ## generate deterministic degraded per-node streams from raw D1
	$(PY) experiments/degraded_harness.py --input $(DATA) --outdir $(RESULTS)/degraded_inputs

deployment-replay: check-data  ## reference code-path replay; NOT the May 2025 physical deployment
	$(PY) -m experiments.run_deployment --data $(DATA) --out $(RESULTS)/deployment_replay

scalability-reference: check-data  ## reference scaling harness; >4 nodes remain model-based
	$(PY) -m experiments.run_deployment --data $(DATA) --scalability --out $(RESULTS)/scalability_reference

security-reference: check-data  ## run current security stress-test harness
	$(PY) -m experiments.run_security --data $(DATA) --out $(RESULTS)/security_reference

diagnostics:  ## render diagnostic plots from new/reference result JSON/CSV artifacts
	$(PY) -m analysis.make_figures --results $(RESULTS) --out $(RESULTS)/diagnostic_figures

reference-rerun: test trials substitution deployment-replay scalability-reference security-reference diagnostics  ## current code rerun; requires raw N1-N4 + services
	@echo "Reference rerun complete. Results are new/reference outputs, not a reconstruction of historical hardware evidence."

contracts:  ## compile and test governance contracts
	npx hardhat compile
	npx hardhat test

clean:
	rm -rf $(RESULTS) $(DEPOSIT_FIGURES) blockchain/artifacts blockchain/cache
	find . -name __pycache__ -type d -exec rm -rf {} +
