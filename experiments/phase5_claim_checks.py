"""Compare computed security/resource metrics with external manuscript claims."""

from __future__ import annotations

import json
from pathlib import Path

CLAIMS_PATH = Path(__file__).resolve().parents[1] / "data" / "manuscript_result_claims.json"


def check_claims(payload: dict) -> list[dict]:
    sec = payload["security"]
    res = payload["resources"]
    injury = sec["injection"]
    failure = sec["model_failure"]
    pois = sec["poisoning"]
    eg = sec["external_inference_traffic"]

    observed = {
        "injection.detected": injury["detected"],
        "injection.F1": injury["f1"],
        "injection.FAR": injury["false_alarm_rate"],
        "poisoning.clean_correct": pois["by_level"]["0"]["correct"],
        "poisoning.5_correct": pois["by_level"]["5"]["correct"],
        "poisoning.10_correct": pois["by_level"]["10"]["correct"],
        "poisoning.20_correct": pois["by_level"]["20"]["correct"],
        "poisoning.retrieval_affected": pois["retrieval_affected"],
        "fallback.triggered": failure["fallback_triggered_n"],
        "fallback.continued": failure["continued_given_fallback"],
        "fallback.mean_ms": failure["fallback_latency_ms"]["mean"],
        "fallback.median_ms": failure["fallback_latency_ms"]["median"],
        "fallback.p95_ms": failure["fallback_latency_ms"]["p95_linear"],
        "fallback.F1": failure["fallback_only"]["f1"],
        "egress.local_mean_KB": eg["ADAM_LLM"]["mean_kb_per_window"],
        "egress.cloud_mean_KB": eg["Cloud-Only"]["mean_kb_per_window"],
        "egress.cloud_calls_mean": eg["Cloud-Only"]["mean_api_calls_per_window"],
        "resource.windows": res["n_recorded_windows"],
        "resource.noninfer_cpu": res["mean_non_inference_window_peak_cpu_pct"],
    }
    for state in ("idle", "monitoring", "crew_active"):
        row = res["by_state"][state]
        observed[f"resource.{state}.CPU"] = row["mean_window_peak_cpu_pct"]
        observed[f"resource.{state}.memory"] = row["mean_node_memory_mb"]
        observed[f"resource.{state}.accounted_KB_per_60s"] = row["mean_accounted_kb_per_60s_per_node"]

    with CLAIMS_PATH.open(encoding="utf-8") as claims_file:
        expected = json.load(claims_file)["phase5"]
    if set(observed) != set(expected):
        raise ValueError("Phase 5 claim keys do not match computed metric keys")

    return [
        {
            "metric": name,
            "computed": observed[name],
            "manuscript": spec["expected"],
            "tolerance": spec["tolerance"],
            "pass": observed[name] is not None
            and abs(observed[name] - spec["expected"]) <= spec["tolerance"] + 1e-12,
        }
        for name, spec in expected.items()
    ]
