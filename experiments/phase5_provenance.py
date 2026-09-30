"""Read-only, source-linked Section 4.5 / resource verification.

Historical records, reconstructed experiments, and synthetic fixtures must not be
mixed. This module consumes explicit CSV exports, never manufactures measurements,
and makes no claim about uninstrumented network traffic or historical per-store ACKs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

SOURCE_SHEETS = {
    "injection": "10_Security_Injection",
    "failure": "11_Security_LLM_Failure",
    "poisoning": "12_Security_Poisoning",
    "egress": "13_Security_Data_Leakage",
    "resources": "07_D2_Resource_Log",
}
STATES = ("idle", "monitoring", "crew_active")
CLAIMS_PATH = Path(__file__).resolve().parents[1] / "data" / "manuscript_result_claims.json"


def _cohort_claims() -> dict:
    with CLAIMS_PATH.open(encoding="utf-8") as claims_file:
        return json.load(claims_file)["cohorts"]


class EvidenceError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path) -> List[dict]:
    if not path.is_file():
        raise EvidenceError(f"Missing source CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise EvidenceError(f"Missing headers: {path}")
        records = list(reader)
    if not records or any(all(not str(v or "").strip() for v in r.values()) for r in records):
        raise EvidenceError(f"Empty data or blank rows: {path}")
    return records


def required(rows: Sequence[Mapping[str, Any]], *fields: str) -> None:
    if not rows:
        raise EvidenceError("No event records")
    missing = set(fields) - set(rows[0])
    if missing:
        raise EvidenceError(f"Missing required fields: {sorted(missing)}")


def num(value: Any, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{name} not numeric: {value!r}") from exc
    if not math.isfinite(v):
        raise EvidenceError(f"{name} not finite: {value!r}")
    return v


def yes(v: Any) -> bool:
    s = str(v or "").strip().lower()
    if s not in ("yes", "no"):
        raise EvidenceError(f"Expected Yes/No, got {v!r}")
    return s == "yes"


def classification(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s not in ("normal", "anomaly"):
        raise EvidenceError(f"Invalid or unresolved class {v!r}; do not coerce to normal")
    return s


def f1_and_far(rows: Sequence[Mapping[str, Any]], pred: str, truth: str) -> dict:
    required(rows, pred, truth)
    tp = fp = tn = fn = 0
    for r in rows:
        p, t = classification(r[pred]), classification(r[truth])
        if p == "anomaly" and t == "anomaly": tp += 1
        elif p == "anomaly": fp += 1
        elif t == "normal": tn += 1
        else: fn += 1
    denom = 2 * tp + fp + fn
    return {
        "confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "f1": 2 * tp / denom if denom else 0.0,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else None,
        "correct": tp + tn,
        "n": len(rows),
    }


def quantile_linear(data: Sequence[float], frac: float) -> float:
    xs = sorted(data)
    if not xs:
        raise EvidenceError("Cannot compute percentile of an empty cohort")
    pos = (len(xs)-1) * frac
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return xs[lo] + (pos - lo) * (xs[hi] - xs[lo])


def verify_security(injection: Sequence[dict], failure: Sequence[dict],
                    poisoning: Sequence[dict], egress: Sequence[dict], *,
                    strict_historical_counts: bool = True) -> dict:
    required(injection, "Attack_Type", "ADAM_Detected_Attack", "ADAM_Prediction", "Ground_Truth")
    patterns: dict = defaultdict(lambda: {"n": 0, "detected": 0, "correct": 0})
    for r in injection:
        p = str(r["Attack_Type"]).strip()
        if not p:
            raise EvidenceError("Missing attack pattern")
        group = patterns[p]
        group["n"] += 1
        group["detected"] += int(yes(r["ADAM_Detected_Attack"]))
        group["correct"] += int(classification(r["ADAM_Prediction"]) == classification(r["Ground_Truth"]))
    injection_metrics = f1_and_far(injection, "ADAM_Prediction", "Ground_Truth")
    injection_metrics.update({"detected": sum(p["detected"] for p in patterns.values()),
                              "by_pattern": dict(sorted(patterns.items()))})
    injection_metrics["attack_detection_rate"] = injection_metrics["detected"] / len(injection)

    required(poisoning, "Num_Poisoned_Entries", "ADAM_Prediction", "Ground_Truth", "Retrieval_Affected")
    groups: dict = defaultdict(list)
    affected = 0
    for r in poisoning:
        level = num(r["Num_Poisoned_Entries"], "Num_Poisoned_Entries")
        if level < 0 or level != int(level):
            raise EvidenceError("Poisoned-entry count must be a nonnegative integer")
        groups[int(level)].append(r)
        affected += int(yes(r["Retrieval_Affected"]))
    pois = {str(n): f1_and_far(v, "ADAM_Prediction", "Ground_Truth") for n,v in sorted(groups.items())}
    poisoning_metrics = {"n": len(poisoning), "by_level": pois, "retrieval_affected": affected,
                         "comparison_type": "unpaired, nonidentical event groups; descriptive only"}

    required(failure, "Fallback_Triggered", "Crew_Continued", "Fallback_Latency_ms",
             "Prediction", "Ground_Truth")
    triggered = [r for r in failure if yes(r["Fallback_Triggered"])]
    if not triggered:
        raise EvidenceError("No fallback-triggered events")
    latency = [num(r["Fallback_Latency_ms"], "Fallback_Latency_ms") for r in triggered]
    if any(x < 0 for x in latency):
        raise EvidenceError("Negative fallback activation latency")
    completed = sum(yes(r["Crew_Continued"]) for r in failure)
    completed_fb = sum(yes(r["Crew_Continued"]) for r in triggered)
    failure_metrics = {
        "n": len(failure), "fallback_triggered_n": len(triggered),
        "fallback_activated_given_triggered": len(triggered),
        "continued_given_fallback": completed_fb,
        "crew_continued_all": completed,
        "fallback_latency_ms": {"mean": mean(latency), "median": median(latency),
                                "p95_linear": quantile_linear(latency, .95), "values_sorted": sorted(latency)},
        "episode": f1_and_far(failure, "Prediction", "Ground_Truth"),
        "fallback_only": f1_and_far(triggered, "Prediction", "Ground_Truth"),
        "limitation": "Fallback_Triggered is an outcome flag, not an independent record of the attempted intervention count or late-failure recovery.",
    }
    if "Degraded_Mode" in failure[0]:
        flags = [str(r["Degraded_Mode"]).strip().lower() for r in triggered]
        if not all(s in ("true", "yes", "1") for s in flags):
            raise EvidenceError("Fallback-triggered event without degraded-mode marker")

    required(egress, "System", "Total_Bytes_External", "External_API_Calls")
    by_system: dict = defaultdict(list)
    aliases = {"ADAM_LLM": "ADAM_LLM", "ADAM": "ADAM_LLM", "ADAM (local)": "ADAM_LLM",
               "Cloud-Only": "Cloud-Only", "Cloud_Only": "Cloud-Only"}
    for r in egress:
        system = str(r["System"]).strip()
        if system not in aliases:
            raise EvidenceError(f"Unknown egress system {system!r}; explicit categorization required")
        by_system[aliases[system]].append(r)
    egress_metrics = {}
    for system,rows in sorted(by_system.items()):
        byte_values = [num(r["Total_Bytes_External"], "external bytes") for r in rows]
        call_values = [num(r["External_API_Calls"], "external calls") for r in rows]
        if any(x < 0 for x in byte_values+call_values):
            raise EvidenceError("Negative traffic/call count")
        if any(x != int(x) for x in byte_values+call_values):
            raise EvidenceError("Noninteger traffic/call count")
        egress_metrics[system] = {"n_windows": len(rows), "kb_per_window": [b/1024 for b in byte_values],
                                  "mean_kb_per_window": mean(byte_values)/1024,
                                  "min_kb_per_window": min(byte_values)/1024,
                                  "max_kb_per_window": max(byte_values)/1024,
                                  "mean_api_calls_per_window": mean(call_values),
                                  "nonzero_windows": sum(b > 0 for b in byte_values)}
    if strict_historical_counts:
        cohorts = _cohort_claims()
        if len(injection) != cohorts["injection_total"] or {k: v["n"] for k,v in patterns.items()} != cohorts["attack_patterns"]:
            raise EvidenceError("Historical injection cohort size/pattern allocation mismatch")
        if len(poisoning) != cohorts["poisoning_total"] or {str(k): len(v) for k,v in groups.items()} != cohorts["poisoning_levels"]:
            raise EvidenceError("Historical poisoning cohort size/levels mismatch")
        if len(failure) != cohorts["failure_total"] or len(triggered) != cohorts["fallback_triggered"]:
            raise EvidenceError("Historical model-failure cohort mismatch")
        if {k:v["n_windows"] for k,v in egress_metrics.items()} != cohorts["egress_windows"]:
            raise EvidenceError("Historical external-inference cohort mismatch")
    return {"injection": injection_metrics, "poisoning": poisoning_metrics,
            "model_failure": failure_metrics, "external_inference_traffic": egress_metrics,
            "scope": "Recorded test events and instrumented inference call sites; not general attack tolerance or all egress"}


def verify_resources(rows: Sequence[dict], *, strict_historical_counts: bool = True) -> dict:
    required(rows, "Timestamp", "State", "CPU_Peak_%", "RAM_MB", "LLM_Bytes", "Weaviate_Bytes", "Blockchain_Bytes")
    grouped: dict = defaultdict(list)
    for r in rows:
        state = str(r["State"]).strip()
        if state not in STATES:
            raise EvidenceError(f"Unknown resource state {state!r}")
        cpu, ram = num(r["CPU_Peak_%"], "CPU_Peak_%"), num(r["RAM_MB"], "RAM_MB")
        if not 0 <= cpu <= 100 or ram < 0:
            raise EvidenceError("Out-of-range CPU/RAM measurement")
        components = {name: num(r[name], name) for name in ("LLM_Bytes", "Weaviate_Bytes", "Blockchain_Bytes")}
        if any(v < 0 for v in components.values()):
            raise EvidenceError("Negative service byte counter")
        if "Total_Bandwidth_Bytes" in r and str(r["Total_Bandwidth_Bytes"]).strip():
            total = num(r["Total_Bandwidth_Bytes"], "Total_Bandwidth_Bytes")
            if abs(total - sum(components.values())) > 0.5:
                raise EvidenceError("Total service bytes disagree with component counters")
        grouped[state].append(r)
    # The source may store a real timestamp string or an Excel date serial;
    # take differences, not the absolute epoch. Never infer 10-Hz sampling from
    # 60-second summaries.
    stamps = []
    stamp_kind = None
    for r in rows:
        raw = str(r["Timestamp"]).strip()
        try:
            value = float(raw)
            kind = "excel_serial"
        except ValueError:
            try:
                value = datetime.fromisoformat(raw).timestamp()
                kind = "iso_timestamp"
            except ValueError as exc:
                raise EvidenceError(f"Unparseable resource timestamp {raw!r}") from exc
        if stamp_kind is not None and kind != stamp_kind:
            raise EvidenceError("Mixed timestamp representations in resource records")
        stamp_kind = kind
        stamps.append(value)
    span_hours = (max(stamps)-min(stamps)) * (24 if stamp_kind=="excel_serial" else 1/3600)
    if span_hours <= 0:
        raise EvidenceError("Resource logging span must be positive")
    if strict_historical_counts and {k:len(v) for k,v in grouped.items()} != _cohort_claims()["resource_states"]:
        raise EvidenceError("Historical resource state counts mismatch")
    out: dict = {}
    for state,group in grouped.items():
        out[state] = {
            "n_windows":len(group),
            "mean_window_peak_cpu_pct":mean(num(r["CPU_Peak_%"], "CPU") for r in group),
            "mean_node_memory_mb":mean(num(r["RAM_MB"], "RAM") for r in group),
            "mean_accounted_kb_per_60s_per_node":mean(sum(num(r[k],k) for k in ("LLM_Bytes","Weaviate_Bytes","Blockchain_Bytes"))/1024 for r in group),
            "components_kb_per_60s": {k:mean(num(r[k],k)/1024 for r in group) for k in ("LLM_Bytes","Weaviate_Bytes","Blockchain_Bytes")},
        }
    noninf=[r for r in rows if r["State"] != "crew_active"]
    return {"n_recorded_windows":len(rows), "by_state":out,
            "mean_non_inference_window_peak_cpu_pct":mean(num(r["CPU_Peak_%"],"CPU") for r in noninf),
            "aggregate_sampled_hours_all_nodes":len(rows)/60,
            "aggregate_sampled_hours_per_node_if_four_nodes":len(rows)/60/4,
            "source_timestamp_span_h":span_hours,
            "approximate_four_node_coverage_fraction":len(rows)/(60*4*span_hours),
            "interpretation": "Samples are 60-s window peaks and accounted service bytes; not sustained time-weighted CPU, process-level RAM, or wire-level network traffic."}


def run(input_dir: Path, out: Path, *, source_type: str, workbook_sha256: Optional[str]=None,
        strict_historical_counts: bool = True) -> dict:
    if source_type not in ("historical_workbook_export", "synthetic_fixture", "new_reference_run"):
        raise EvidenceError("Explicit source_type is required")
    if source_type == "historical_workbook_export" and not workbook_sha256:
        raise EvidenceError("A historical export requires the original workbook SHA-256")
    src = {key: input_dir / (sheet + ".csv") for key,sheet in SOURCE_SHEETS.items()}
    if source_type=="historical_workbook_export":
        manifest_file=input_dir/"source_manifest.json"
        if not manifest_file.is_file():
            raise EvidenceError("Historical CSVs require a workbook-export manifest")
        manifest=json.loads(manifest_file.read_text())
        if manifest.get("workbook_sha256")!=workbook_sha256:
            raise EvidenceError("Historical CSV export workbook hash mismatch")
        for key,path in src.items():
            if manifest.get("csv_sha256",{}).get(key)!=sha256_file(path):
                raise EvidenceError(f"Export manifest digest mismatch: {key}")
    rows = {key: read_csv(path) for key,path in src.items()}
    checks = verify_security(rows["injection"],rows["failure"],rows["poisoning"], rows["egress"],
                             strict_historical_counts=strict_historical_counts)
    resource = verify_resources(rows["resources"], strict_historical_counts=strict_historical_counts)
    output = {"provenance": {"source_type":source_type,"workbook_sha256":workbook_sha256,
                              "csv_sha256":{k:sha256_file(v) for k,v in src.items()},
                              "record_class":"historical recorded evidence" if source_type=="historical_workbook_export" else source_type,
                              "independent_hardware_replay":False},
              "security":checks, "resources":resource}
    out.mkdir(parents=True,exist_ok=True)
    (out/"phase5_metrics.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    return output
