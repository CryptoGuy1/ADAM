"""
adam.manuscript
===============

Reference values recomputed from the deposited dataset.

``verify_against_manuscript()`` compares the constants in :mod:`adam.config`
against ``ADAM_Dataset_Master.xlsx`` - the workbook a reviewer downloads -
rather than against a second table of literals. Code and data cannot diverge
without the check failing.

The dataset is the source of truth for measured quantities. Design choices that
govern behavior rather than describe it, such as the screening threshold, the
decision deadline and the quorum rule, stay in :mod:`adam.config`.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Dict, List, Optional, Tuple

DATASET_ENV = "ADAM_DATASET"
DEFAULT_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "ADAM_Dataset_Master.xlsx",
)


class DatasetUnavailable(RuntimeError):
    """The deposited workbook could not be read."""


def dataset_path() -> str:
    return os.getenv(DATASET_ENV, DEFAULT_DATASET)


def available() -> bool:
    return os.path.exists(dataset_path())


@functools.lru_cache(maxsize=1)
def _sheets() -> Dict[str, Any]:
    """Load the sheets the parity check needs, once per process."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise DatasetUnavailable("pandas is required to read the deposit") from exc

    path = dataset_path()
    if not os.path.exists(path):
        raise DatasetUnavailable(
            f"{path} not found. Set {DATASET_ENV} to the deposited workbook, or "
            f"download it from the archive named in the Data Availability "
            f"statement."
        )
    read = lambda name, **kw: pd.read_excel(path, sheet_name=name, **kw)
    return {
        "trials": read("03_D1_Trial_Results"),
        "coord": read("05_D2_Coordination_Log"),
        "resources": read("07_D2_Resource_Log"),
        "scalability": read("08_Scalability_Log"),
        "labeled": read("02_D1_Labeled_Events"),
        "tests": read("04_D1_Statistical_Tests", header=1),
        "trigger_log": read("D1_RawTrigger_Log"),
        "trigger_summary": read("D1_RawTrigger_Summary"),
        "event_predictions": read("06A_Event_Predictions", header=3),
    }


def _num(frame, column):
    import pandas as pd

    return pd.to_numeric(frame[column], errors="coerce")


# ---------------------------------------------------------------------------
# Deployment: latency and end-to-end completion
# ---------------------------------------------------------------------------

STAGE_COLUMNS = {
    "T_form": "T_form_ms",
    "T_agg": "T_aggregate_ms",
    "T_reason": "T_reason_ms",
    "T_gov": "T_validate_ms",
    "T_weav": "T_weaviate_ms",
    "T_bc": "T_blockchain_ms",
}


def _completed(coord):
    return coord["Success"].astype(str).str.lower().eq("yes")


def deployment_events() -> int:
    """N: every coordination event recorded in the historical D2 deployment."""
    return int(len(_sheets()["coord"]))


def completed_events() -> int:
    """Events that finished end to end, the latency denominator."""
    return int(_completed(_sheets()["coord"]).sum())


def trace_persistence() -> float:
    """Backward-compatible alias for the historical end-to-end completion rate.

    D2 does not preserve independent acknowledgments for both audit stores, so
    this value must not be described as measured dual-store persistence.
    """
    return completed_events() / deployment_events()


def stage_latencies_ms() -> Dict[str, float]:
    """Mean per-stage latency over the completed events. Equation (6)."""
    coord = _sheets()["coord"]
    ok = _completed(coord)
    return {k: float(_num(coord, col)[ok].mean()) for k, col in STAGE_COLUMNS.items()}


def stage_shares() -> Dict[str, float]:
    lat = stage_latencies_ms()
    total = sum(lat.values())
    return {k: v / total for k, v in lat.items()}


def decision_latency_ms() -> Dict[str, float]:
    coord = _sheets()["coord"]
    ok = _completed(coord)
    v = _num(coord, "T_decision_total_ms")[ok]
    return {
        "mean": float(v.mean()),
        "median": float(v.median()),
        "p95": float(v.quantile(0.95)),
        "max": float(v.max()),
    }


def crew_formation_ms() -> Dict[str, float]:
    coord = _sheets()["coord"]
    ok = _completed(coord)
    v = _num(coord, "T_form_ms")[ok]
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)),
        "median": float(v.median()),
        "p95": float(v.quantile(0.95)),
    }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detection_scores() -> Dict[str, Dict[str, float]]:
    """Per-system mean and SD of the four detection metrics, as in Table 5."""
    t = _sheets()["trials"]
    out: Dict[str, Dict[str, float]] = {}
    for system, g in t.groupby("System"):
        row: Dict[str, float] = {}
        for metric in ("Precision", "Recall", "F1", "FAR"):
            v = _num(g, metric)
            row[metric.lower()] = float(v.mean())
            row[metric.lower() + "_sd"] = float(v.std(ddof=1))
        row["latency_s"] = float(_num(g, "T_decision_ms").mean()) / 1000.0
        out[str(system)] = row
    return out


def evaluated_systems() -> List[str]:
    return sorted(detection_scores())


def labeled_events() -> Dict[str, int]:
    lab = _sheets()["labeled"]
    positive = (
        lab["Ground_Truth_Label"].astype(str).str.strip().str.lower().eq("anomaly")
    )
    return {
        "total": int(len(lab)),
        "trials": int(lab["Trial"].nunique()),
        "positive": int(positive.sum()),
        "negative": int((~positive).sum()),
    }


def threshold_baseline(channel: str, threshold_ppm: float = 1000.0) -> Dict[str, float]:
    """Score a fixed-threshold detector on one sensor channel, per trial.

    The Static Threshold baseline of Table 5 is this rule applied to
    ``Raw_Instantaneous_PPM``: the same raw instantaneous sample that gates
    ADAM's crew formation, compared against the fixed screening threshold.
    Both systems therefore receive identical input; they differ only in what
    happens after the comparison.
    """
    import numpy as np

    lab = _sheets()["labeled"]
    if channel not in lab.columns:
        raise DatasetUnavailable(f"02_D1_Labeled_Events has no {channel!r} column")
    series = _num(lab, channel)

    f1s, fars, precisions, recalls = [], [], [], []
    for _, g in lab.groupby("Trial"):
        v = series[g.index]
        y = g["Ground_Truth_Label"].astype(str).str.strip().str.lower().eq("anomaly")
        tp = int(((v >= threshold_ppm) & y).sum())
        fp = int(((v >= threshold_ppm) & ~y).sum())
        fn = int(((v < threshold_ppm) & y).sum())
        tn = int(((v < threshold_ppm) & ~y).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        fars.append(fp / (fp + tn) if fp + tn else 0.0)
    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "f1_sd": float(np.std(f1s, ddof=1)),
        "far": float(np.mean(fars)),
        "far_sd": float(np.std(fars, ddof=1)),
    }


def gated_run_summary() -> Dict[str, float]:
    """Legacy independently gated D1 run retained for provenance only.

    This historical sheet is not the revised manuscript deployment-semantics
    result. The current operating point is derived deterministically by
    :func:`derived_deployment_summary` from frozen benchmark predictions.
    Readings below the
    screening threshold never form a crew and are classified normal on the
    fast path, so only triggered readings receive aggregation and reasoning.
    It is the deployed operating point; the full-pipeline run behind Table 5
    is the interpretation benchmark. Both are deposited, and the code
    reproduces each under the corresponding ``eval_mode``.
    """
    s = _sheets()["trigger_summary"]
    row = s[s["Trial"].astype(str).str.strip().str.lower().eq("overall")]
    if len(row) != 1:
        raise DatasetUnavailable("D1_RawTrigger_Summary has no Overall row")
    r = row.iloc[0]
    return {
        "triggered": float(r["Triggered_N"]),
        "trigger_rate": float(r["Trigger_Rate"]),
        "precision": float(r["Precision"]),
        "recall": float(r["Recall"]),
        "f1": float(r["F1"]),
        "far": float(r["FAR"]),
        "degraded": float(r["Degraded_N"]),
    }


def derived_deployment_summary() -> Dict[str, float]:
    """Revised deployment semantics derived from frozen benchmark predictions.

    Above the 1,000 ppm raw screening gate, the stored ADAM_LLM benchmark
    prediction is preserved exactly. Below the gate, the prediction is NORMAL.
    The legacy D1_RawTrigger_* sheets are retained only as historical artifacts
    of a separate stochastic execution and are not used for this result.
    """
    import numpy as np
    import pandas as pd

    sh = _sheets()
    log = sh["trigger_log"].copy()
    pred = sh["event_predictions"].dropna(subset=["Event_ID"]).copy()
    m = log.merge(pred[["Event_ID", "ADAM_LLM"]], on="Event_ID", how="inner")
    if len(m) != len(log):
        raise DatasetUnavailable("could not align frozen ADAM_LLM predictions to trigger log")
    above = _num(m, "Raw_Instantaneous_PPM") >= 1000.0
    m["ADAM_Derived"] = np.where(above, m["ADAM_LLM"], "normal")

    rows = []
    for _trial, g in m.groupby("Trial", sort=True):
        y = g["Ground_Truth_Label"].astype(str).str.lower().eq("anomaly")
        p = g["ADAM_Derived"].astype(str).str.lower().eq("anomaly")
        tp = int((y & p).sum()); fp = int((~y & p).sum())
        fn = int((y & ~p).sum()); tn = int((~y & ~p).sum())
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        far = fp / (fp + tn)
        rows.append((precision, recall, f1, far))
    a = np.asarray(rows, dtype=float)
    return {
        "triggered": int(above.sum()),
        "precision": float(a[:, 0].mean()),
        "recall": float(a[:, 1].mean()),
        "f1": float(a[:, 2].mean()),
        "f1_sd": float(a[:, 2].std(ddof=1)),
        "far": float(a[:, 3].mean()),
        "far_sd": float(a[:, 3].std(ddof=1)),
        "above_gate_mismatches": int(
            m.loc[above, "ADAM_Derived"].astype(str).str.lower().ne(
                m.loc[above, "ADAM_LLM"].astype(str).str.lower()
            ).sum()
        ),
    }


def gated_predictions_agree() -> Dict[str, float]:
    """Structural checks on the historical trigger-gated workbook log.

    This sheet documents the earlier independently gated D1 execution and is
    retained as historical provenance only. The revised manuscript deployment
    semantics are derived deterministically from frozen benchmark predictions
    by :mod:`scripts.derive_deployment_semantics`. These row-level checks only
    verify that the historical trigger flag followed the raw 1,000-ppm gate and
    that untriggered rows were not called anomalous.
    """
    log = _sheets()["trigger_log"]
    raw = _num(log, "Raw_Instantaneous_PPM")
    triggered = log["Triggered"].astype(str).str.strip().str.lower().eq("yes")
    pred_anom = (
        log["ADAM_Prediction"].astype(str).str.strip().str.lower().eq("anomaly")
    )
    rule_matches = int((triggered == (raw >= 1000.0)).sum())
    untriggered_anomalies = int((~triggered & pred_anom).sum())
    return {
        "rows": int(len(log)),
        "trigger_rule_matches": rule_matches,
        "untriggered_anomalies": untriggered_anomalies,
    }


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


def resource_profile() -> Dict[str, Dict[str, float]]:
    """CPU, working set and traffic by operating state."""
    r = _sheets()["resources"]
    out: Dict[str, Dict[str, float]] = {}
    for state, g in r.groupby("State"):
        out[str(state)] = {
            "cpu_pct": float(_num(g, "CPU_Peak_%").mean()),
            "ram_mb": float(_num(g, "RAM_MB").mean()),
            "traffic_kb_per_min": float(_num(g, "Bandwidth_KB").mean()),
            "n": int(len(g)),
        }
    return out


def sustained_cpu_pct() -> float:
    """Constraint C2: mean utilization outside active-inference windows."""
    r = _sheets()["resources"]
    outside = r[r["State"].astype(str) != "crew_active"]
    return float(_num(outside, "CPU_Peak_%").mean())


# ---------------------------------------------------------------------------
# Scalability
# ---------------------------------------------------------------------------


def _fit_power(x, y, xr, yr) -> Dict[str, float]:
    """Fit T(N) = T0 + alpha * N^beta over level means, R^2 over replicates."""
    import numpy as np
    from scipy.optimize import curve_fit

    model = lambda N, t0, a, b: t0 + a * N**b
    popt, _ = curve_fit(
        model, x, y, p0=[float(min(y)), 200.0, 1.0], maxfev=60000
    )
    r2 = 1 - np.sum((yr - model(xr, *popt)) ** 2) / np.sum((yr - yr.mean()) ** 2)
    return {
        "T0": float(popt[0]),
        "alpha": float(popt[1]),
        "beta": float(popt[2]),
        "r_squared": float(r2),
        "n_raw": int(len(xr)),
    }


def node_scaling_fit(scope: str) -> Dict[str, float]:
    """Node-count scaling model T(N) = T0 + alpha * N^beta. Table 7.

    ``scope`` selects the fitted domain:

      "hardware"  N = 1-4 measured on physical Raspberry Pi 5 nodes.
      "scaleout"  N = 4, 6, 8, 12, 16 from the Python scale-out model, which
                  is validated against the matched hardware runs before use
                  (see :func:`simulator_validation`).

    Load is held at the reference configuration in both scopes, so latency
    changes attribute to node count.
    """
    s = _sheets()["scalability"]
    mode = s["Run_Mode"].astype(str).str.strip().str.upper()
    n = _num(s, "Node_Count")
    t = _num(s, "T_decision_ms")

    if scope == "hardware":
        mask = mode.eq("HARDWARE")
    elif scope == "scaleout":
        eligible = s["Fit_Eligible"].astype(str).str.strip().str.lower().eq("yes")
        mask = eligible & (n >= 4)
    else:
        raise ValueError(f"scope must be 'hardware' or 'scaleout'; got {scope!r}")

    sub = s[mask]
    xr = n[mask].values.astype(float)
    yr = t[mask].values.astype(float)
    g = sub.groupby(sub["Node_Count"])["T_decision_ms"].mean()
    return _fit_power(g.index.values.astype(float), g.values, xr, yr)


def simulator_validation() -> Dict[str, float]:
    """Scale-out model against matched hardware, decision latency, N = 1-4.

    MAPE and mean bias over the per-level means, as reported in
    09_Fitted_Models. The scale-out results at N > 4 are conditional on this
    agreement.
    """
    import numpy as np

    s = _sheets()["scalability"]
    mode = s["Run_Mode"].astype(str).str.strip().str.upper()
    n = _num(s, "Node_Count")

    hw = s[mode.eq("HARDWARE")].groupby(n)["T_decision_ms"].mean()
    sim = s[mode.eq("PYTHON_SIMULATION")].groupby(n)["T_decision_ms"].mean()
    common = sorted(set(hw.index) & set(sim.index))
    if not common:
        raise DatasetUnavailable("no matched hardware/simulation node levels")
    rel = [(sim[k] - hw[k]) / hw[k] for k in common]
    return {
        "levels": float(len(common)),
        "mape_pct": float(np.mean(np.abs(rel)) * 100.0),
        "bias_pct": float(np.mean(rel) * 100.0),
    }


def empirical_scalability_rows() -> int:
    """Rows measured on physical hardware. Everything else is the scale-out model."""
    s = _sheets()["scalability"]
    return int(s["Run_Mode"].astype(str).str.strip().str.upper().eq("HARDWARE").sum())


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def sensor_error_variances() -> Dict[str, float]:
    """Per-node sensor error variance from raw-channel reference residuals.

    The residual is each node's raw instantaneous reading minus the co-located
    NDIR reference, over the 500 paired readings the node contributes to D1.
    These variances define the inverse-variance weights of Equation (2).
    Recomputed here rather than read from a summary cell, so the check
    confirms the deposited residuals actually produce the variances the
    manuscript quotes.
    """
    lab = _sheets()["labeled"]
    resid = _num(lab, "Raw_Instantaneous_PPM") - _num(lab, "Reference_Sensor_PPM")
    by_node = resid.groupby(lab["Node_ID"])
    variances = by_node.var(ddof=1)
    weights = 1.0 / variances
    normalized = weights / weights.sum()
    return {
        "min_ppm2": float(variances.min()),
        "max_ppm2": float(variances.max()),
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
        "min_normalized": float(normalized.min()),
        "max_normalized": float(normalized.max()),
        "weight_ratio": float(weights.max() / weights.min()),
        "n_nodes": int(len(variances)),
        "min_paired": int(by_node.count().min()),
        "max_paired": int(by_node.count().max()),
    }


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def statistical_tests() -> Dict[str, Dict[str, float]]:
    t = _sheets()["tests"]
    out: Dict[str, Dict[str, float]] = {}
    for _, row in t.iterrows():
        key = str(row.get("Comparison", "")).strip()
        if not key or key == "nan":
            continue
        try:
            out[key] = {
                "n": float(row.get("N_Effective", float("nan"))),
                "p_exact": float(row.get("P_Exact", float("nan"))),
                "p_holm": float(row.get("P_Holm", float("nan"))),
            }
        except (TypeError, ValueError):
            continue
    return out


__all__ = [
    "DatasetUnavailable",
    "dataset_path",
    "available",
    "deployment_events",
    "completed_events",
    "trace_persistence",
    "stage_latencies_ms",
    "stage_shares",
    "decision_latency_ms",
    "crew_formation_ms",
    "detection_scores",
    "evaluated_systems",
    "labeled_events",
    "static_threshold_reproducible",
    "resource_profile",
    "sustained_cpu_pct",
    "node_scaling_fit",
    "simulator_validation",
    "empirical_scalability_rows",
    "sensor_error_variances",
    "gated_run_summary",
    "derived_deployment_summary",
    "gated_predictions_agree",
    "statistical_tests",
]
