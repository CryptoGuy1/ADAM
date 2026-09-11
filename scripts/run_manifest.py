#!/usr/bin/env python3
"""Write a provenance manifest for a new/reference ADAM experiment run.

The manifest is intentionally descriptive rather than self-certifying. It
records the software, model, dataset, service, contract, and experiment state
needed to interpret a rerun without claiming that those values describe the
historical May 2025 deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

# Allow direct execution as ``python scripts/run_manifest.py`` from a clean
# checkout without requiring an editable package install.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adam.config import (
    CHAIN_ID,
    CHAIN_RPC_URL,
    CONTRACT_ADDRESSES,
    LLM_FORMAT_REPAIR_RETRIES,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    WEAVIATE_HOST,
)

HISTORICAL_WEAVIATE_VERSION = "1.21"
REFERENCE_WEAVIATE_VERSION = "1.30.2"


def _run(cmd: list[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return p.stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_endpoint(value: str) -> str:
    """Record only scheme/host/port; never credentials, query, or path data."""
    try:
        u = urlsplit(value)
        host = u.hostname or ""
        if not host:
            return "configured" if value else ""
        port = f":{u.port}" if u.port else ""
        scheme = f"{u.scheme}://" if u.scheme else ""
        return f"{scheme}{host}{port}"
    except Exception:
        return "configured" if value else ""


def _git(repo: Path) -> Dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"], repo)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)
    status = _run(["git", "status", "--porcelain"], repo)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
    }


def _ollama() -> Dict[str, Any]:
    version = _run(["ollama", "--version"])
    listing = _run(["ollama", "list"])
    model_line = None
    if listing:
        for line in listing.splitlines():
            if line.strip().startswith(OLLAMA_MODEL):
                model_line = re.sub(r"\s+", " ", line.strip())
                break
    show = _run(["ollama", "show", OLLAMA_MODEL])
    digest = None
    quantization = None
    if show:
        md = re.search(r"(?im)^\s*digest\s+([a-f0-9:]+)\s*$", show)
        mq = re.search(r"(?im)^\s*quantization\s+(.+?)\s*$", show)
        digest = md.group(1) if md else None
        quantization = mq.group(1).strip() if mq else None
    return {
        "host": safe_endpoint(OLLAMA_HOST),
        "version": version,
        "model": OLLAMA_MODEL,
        "model_list_entry": model_line,
        "model_digest": digest,
        "quantization": quantization,
        "inference": {
            "temperature": LLM_TEMPERATURE,
            "num_predict": LLM_MAX_TOKENS,
            "format_repair_retries": LLM_FORMAT_REPAIR_RETRIES,
            "seed": None,
        },
    }


def _contract_hashes(repo: Path) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in ("GovernanceRules", "CrewRegistry", "ConsensusValidator", "DecisionLogger"):
        path = repo / "contracts" / f"{name}.sol"
        out[name] = sha256_file(path) if path.exists() else None
    return out


def build_manifest(
    repo: Path,
    dataset: Optional[Path] = None,
    experiment_config: Optional[Path] = None,
) -> Dict[str, Any]:
    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"])
    docker_version = _run(["docker", "--version"])
    experiment: Any = None
    if experiment_config:
        raw = experiment_config.read_text()
        try:
            experiment = json.loads(raw)
        except json.JSONDecodeError:
            experiment = {"path": str(experiment_config), "sha256": sha256_file(experiment_config)}

    dataset_info = None
    if dataset:
        dataset_info = {
            "path": str(dataset),
            "sha256": sha256_file(dataset),
            "bytes": dataset.stat().st_size,
        }

    return {
        "manifest_schema": "adam.run_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "new/reference run; not a reconstruction of historical deployment provenance",
        "git": _git(repo),
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "node": platform.node(),
            "pip_freeze": pip_freeze.splitlines() if pip_freeze else [],
        },
        "ollama": _ollama(),
        "weaviate": {
            "host": safe_endpoint(WEAVIATE_HOST),
            "historical_reported_version": HISTORICAL_WEAVIATE_VERSION,
            "reference_rerun_version": REFERENCE_WEAVIATE_VERSION,
        },
        "docker": {"version": docker_version},
        "blockchain": {
            "chain_id": CHAIN_ID,
            "rpc": safe_endpoint(CHAIN_RPC_URL),
            "contract_addresses_configured": {
                k: bool(v) for k, v in CONTRACT_ADDRESSES.items()
            },
            "contract_source_sha256": _contract_hashes(repo),
        },
        "dataset": dataset_info,
        "experiment": experiment,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--dataset")
    ap.add_argument("--experiment-config")
    ap.add_argument("--out", default="run_manifest.json")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    manifest = build_manifest(
        repo,
        Path(args.dataset).resolve() if args.dataset else None,
        Path(args.experiment_config).resolve() if args.experiment_config else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
