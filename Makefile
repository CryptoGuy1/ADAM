# ADAM reproduction targets.
#
# `make reproduce` is the full path from a clean checkout to every table and
# figure in the manuscript. It requires the deposited datasets, Ollama with
# gemma3:1b, and Weaviate.
#
# `make offline` runs everything that does not need hardware or API keys, and
# is what CI runs.

PY ?= python3
DEPOSIT ?= data/ADAM_Dataset_Master.xlsx
DATA ?= data/artifacts/d1_deposit.csv
FIXTURE := data/artifacts/d1_simulated.csv
RESULTS ?= results

.PHONY: help install test verify verify-manuscript fixture export-data offline reproduce trials \
        trials-gated deployment scalability security conflict figures contracts \
        clean appendix

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## install Python and contract dependencies
	$(PY) -m pip install -r requirements.txt
	npm install

verify:  ## check constants reproduce the manuscript's derived figures
	$(PY) -m adam.config

test: verify  ## run the parity test suite
	$(PY) -m pytest tests/ -v

fixture:  ## generate a synthetic D1 (NOT a paper reproduction)
	$(PY) -m data.loader --simulate --out $(FIXTURE)

export-data:  ## export D1 from the deposited workbook to CSV
	$(PY) -m data.loader --export $(DEPOSIT) --out $(DATA)

check-data:  ## run integrity checks on the real D1 deposit
	$(PY) -m data.loader --check $(DATA)

appendix:  ## emit Appendix A LaTeX from the live prompt template
	$(PY) -m adam.llm.prompt --latex

offline: test fixture  ## everything that needs no hardware or API keys
	$(PY) -m experiments.run_trials --data $(FIXTURE) --no-llm --skip cloud_only --out $(RESULTS)/trials
	$(PY) -m experiments.run_conflict_sweep --out $(RESULTS)/conflict
	$(PY) -m experiments.run_security --data $(FIXTURE) --no-llm --out $(RESULTS)/security

trials: export-data  ## Table 5 benchmark run (full pipeline), requires Ollama
	$(PY) -m experiments.run_trials --data $(DATA) --eval-mode full_pipeline --out $(RESULTS)/trials

trials-gated: export-data  ## deployed operating point over D1, requires Ollama
	$(PY) -m experiments.run_trials --data $(DATA) --eval-mode gated --out $(RESULTS)/trials_gated

deployment:  ## Figure 5 and Table 6, requires Ollama
	$(PY) -m experiments.run_deployment --data $(DATA) --out $(RESULTS)/deployment

scalability:  ## Table 7 and Figure 8, requires Ollama
	$(PY) -m experiments.run_deployment --data $(DATA) --scalability --out $(RESULTS)/deployment

security:  ## Section 4.5, requires Ollama for the poisoning scenario
	$(PY) -m experiments.run_security --data $(DATA) --out $(RESULTS)/security

conflict:  ## Section 4.6 and Figure 9
	$(PY) -m experiments.run_conflict_sweep --out $(RESULTS)/conflict

figures:  ## render all figures from results/
	$(PY) -m analysis.make_figures --results $(RESULTS) --out $(RESULTS)/figures

contracts:  ## compile and check the governance contracts
	npx hardhat compile
	npx hardhat test

reproduce: test trials deployment scalability security conflict figures  ## full reproduction
	@echo "Reproduction complete. Results in $(RESULTS)/"

clean:
	rm -rf $(RESULTS) blockchain/artifacts blockchain/cache
	find . -name __pycache__ -type d -exec rm -rf {} +
