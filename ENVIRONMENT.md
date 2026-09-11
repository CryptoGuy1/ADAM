# ADAM environment record

This repository distinguishes the **historical reported deployment** from the
**current reference rerun environment**. A reference rerun is not evidence that
the historical deployment used the newer software version.

| Component | Historical reported deployment | Current reference rerun |
|---|---|---|
| Edge nodes | 4 × Raspberry Pi 5, 8 GB | Same hardware target |
| Python | 3.11 | See `run_manifest.json` |
| Decision model | Gemma 3 1B, Q4_K_M, Ollama | `gemma3:1b`; exact local digest recorded by manifest when available |
| Temperature | 0.1 | 0.1 |
| Max generated tokens | 256 | 256 |
| Ollama seed | Not supplied | Not supplied unless an experiment explicitly overrides it |
| Weaviate | **1.21** | **1.30.2** |
| Vectorizer | `text2vec-transformers` | `text2vec-transformers` |
| Blockchain | Fides Innova permissioned PoA testnet | Chain ID/configuration recorded by manifest |
| Dataset DOI | `10.5281/zenodo.21892655` | Same public record; revised result artifacts must be deposited separately if not present there |

## Reference-run provenance

Before a new benchmark, substitution, degraded-condition, deployment replay, or
scalability run, create a manifest, for example:

```bash
python scripts/run_manifest.py \
  --dataset data/path/to/d1.csv \
  --experiment-config path/to/config.json \
  --out results/<run>/run_manifest.json
```

The manifest records the Git commit and dirty state, Python/platform details,
installed packages, Ollama version/model metadata when available, model
inference settings, Docker version, dataset SHA-256, Weaviate historical and
reference versions, blockchain chain ID, contract-address presence, and hashes
of the Solidity sources. Service URLs are sanitized to scheme/host/port; secrets
and URL query/path data are not written to the manifest.
