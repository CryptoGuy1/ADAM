#!/usr/bin/env python3
"""Verify every quantitative claim in the article's Results section against the
deposited workbook and hardware-calibrated stage model. Exits non-zero on any mismatch.

Usage:  python scripts/verify_manuscript_numbers.py [workbook.xlsx]

Sections covered: 4.1 detection and operating point, 4.2 coordination and
failures, 4.3 resources, 4.4 node scaling, 4.5 security. Values quoted in the
article at coarser precision are checked at that precision.
"""

import json
import math
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

CLAIMS_PATH = ROOT / "data" / "manuscript_result_claims.json"
with CLAIMS_PATH.open(encoding="utf-8") as claims_file:
    CLAIMS = json.load(claims_file)

import numpy as np
import pandas as pd
from adam.config import DECISION_DEADLINE_S, THRESHOLD_PPM

WB = sys.argv[1] if len(sys.argv) > 1 else "data/ADAM_Dataset_Master_v14_reconciled.xlsx"
xl = pd.ExcelFile(WB)

FAILURES = []


def chk(label, got, want, tol=0.0005):
    ok = abs(float(got) - float(want)) <= tol
    if not ok:
        FAILURES.append(f"{label}: article {want}, dataset {got}")
    print(f"{'PASS' if ok else 'FAIL'}  {label:52s} {want:<10} {float(got):.4f}")


# ---------------------------------------------------------------- 4.1
tr = pd.read_excel(xl, "03_D1_Trial_Results")
pred = pd.read_excel(xl, "06A_Event_Predictions", header=3).dropna(subset=["Event_ID"])
trig = pd.read_excel(xl, "D1_RawTrigger_Log")
from analysis.revised_statistics import main_benchmark_family, substitution_family, degraded_family

main_stats = {r.comparison: r for r in main_benchmark_family(tr)}

S = lambda s, c: (tr[tr.System == s][c].mean(), tr[tr.System == s][c].std(ddof=1))
TABLE4 = CLAIMS["tables"]["TABLE4"]
for sysname, (p_, r_, f_, fa_) in TABLE4.items():
    for col, want in zip(("Precision", "Recall", "F1", "FAR"), (p_, r_, f_, fa_)):
        chk(f"4.1 {sysname} {col}", S(sysname, col)[0], want)
chk("4.1 ADAM F1 sd", S("ADAM_LLM", "F1")[1], CLAIMS["checks"]['4.1 ADAM F1 sd'])
chk("4.1 margin over Static (pts)", S("ADAM_LLM", "F1")[0] - S("Static_Threshold", "F1")[0], CLAIMS["checks"]['4.1 margin over Static (pts)'], 0.001)
chk("4.1 SingleAgent gap (pts)", S("ADAM_LLM", "F1")[0] - S("SingleAgent", "F1")[0], CLAIMS["checks"]['4.1 SingleAgent gap (pts)'], 0.001)

lab = pd.read_excel(xl, "02_D1_Labeled_Events")
v = (lab["Raw_Instantaneous_PPM"] - lab["Reference_Sensor_PPM"]).groupby(lab["Node_ID"]).var(ddof=1)
w = (1 / v) / (1 / v).sum()
chk("4.1 variance min ppm2", v.min(), CLAIMS["checks"]['4.1 variance min ppm2'], 0.5)
chk("4.1 variance max ppm2", v.max(), CLAIMS["checks"]['4.1 variance max ppm2'], 0.5)
chk("4.1 norm weight min", w.min(), CLAIMS["checks"]['4.1 norm weight min'], 0.001)
chk("4.1 norm weight max", w.max(), CLAIMS["checks"]['4.1 norm weight max'], 0.001)

WILCOXON = CLAIMS["tables"]["WILCOXON"]
for key, (pe, ph) in WILCOXON.items():
    row = main_stats[key]
    chk(f"4.1 {key} p_exact", row.p_exact, pe)
    if ph is not None:
        chk(f"4.1 {key} p_Holm", row.p_holm, ph)
f1p = tr.pivot(index="Trial", columns="System", values="F1")
dc = f1p["Cloud_Only"] - f1p["ADAM_LLM"]
chk("4.1 Cloud median diff", dc.median(), CLAIMS["checks"]['4.1 Cloud median diff'])
chk("4.1 Cloud trials favoring cloud", (dc > 0).sum(), CLAIMS["checks"]['4.1 Cloud trials favoring cloud'], 0)
db = f1p["ADAM_LLM"] - f1p["ADAM_NoBlockchain"]
nz = db[db != 0]
chk("4.1 NoBlockchain effective n", len(nz), CLAIMS["checks"]['4.1 NoBlockchain effective n'], 0)
chk("4.1 NoBlockchain favoring ADAM", (nz > 0).sum(), CLAIMS["checks"]['4.1 NoBlockchain favoring ADAM'], 0)
chk("4.1 NoBlockchain median", nz.median(), CLAIMS["checks"]['4.1 NoBlockchain median'])

m = trig.merge(pred[["Event_ID", "ADAM_LLM", "Static_Threshold"]], on="Event_ID")
above = m["Raw_Instantaneous_PPM"] >= THRESHOLD_PPM
anom = m["Ground_Truth_Label"] == "anomaly"
chk("4.1 anomalies above gate", (anom & above).sum(), CLAIMS["checks"]['4.1 anomalies above gate'], 0)
chk("4.1 anomalies below gate", (anom & ~above).sum(), CLAIMS["checks"]['4.1 anomalies below gate'], 0)
chk("4.1 triggered events", above.sum(), CLAIMS["checks"]['4.1 triggered events'], 0)
m["ADAM_Derived"] = np.where(above, m["ADAM_LLM"], "normal")

for col, ra, rb, ro in CLAIMS["series"]["recall_by_system"]:
    hit = (m[col] == "anomaly") & anom
    chk(f"4.1 {col} recall above", hit[above].sum() / (anom & above).sum(), ra)
    chk(f"4.1 {col} recall below", hit[~above].sum() / (anom & ~above).sum(), rb)
    chk(f"4.1 {col} recall overall", hit.sum() / anom.sum(), ro)
chk("4.1 triggered FP static", ((m["Static_Threshold"] == "anomaly") & ~anom & above).sum(), CLAIMS["checks"]['4.1 triggered FP static'], 0)
chk("4.1 triggered FP derived deployment", ((m["ADAM_Derived"] == "anomaly") & ~anom & above).sum(), CLAIMS["checks"]['4.1 triggered FP derived deployment'], 0)

# Deployment semantics are a deterministic transform of frozen benchmark
# predictions: exact identity above the gate, NORMAL below it.
gate_ident = m.loc[above, "ADAM_Derived"].str.lower().ne(m.loc[above, "ADAM_LLM"].str.lower())
chk("4.1 above-gate prediction identity (mismatches)", gate_ident.sum(), CLAIMS["checks"]['4.1 above-gate prediction identity (mismatches)'], 0)
chk("4.1 below-gate all normal (exceptions)",
    m.loc[~above, "ADAM_Derived"].str.lower().ne("normal").sum(), CLAIMS["checks"]['4.1 below-gate all normal (exceptions)'], 0)

def _trial_metrics(g):
    yy = g["Ground_Truth_Label"].str.lower().eq("anomaly")
    pp = g["ADAM_Derived"].str.lower().eq("anomaly")
    tp = int((yy & pp).sum()); fp = int((~yy & pp).sum())
    fn = int((yy & ~pp).sum()); tn = int((~yy & ~pp).sum())
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    far = fp / (fp + tn)
    return pd.Series({"Precision": precision, "Recall": recall, "F1": f1, "FAR": far})

g10 = m.groupby("Trial", sort=True).apply(_trial_metrics)
for col, mean_, sd_ in CLAIMS["series"]["deployment_metrics"]:
    chk(f"4.1 deployment {col} mean", g10[col].mean(), mean_)
    chk(f"4.1 deployment {col} sd", g10[col].std(ddof=1), sd_)

latp = tr.pivot(index="Trial", columns="System", values="T_decision_ms")
for sysname, want in CLAIMS["series"]["latency_by_system"]:
    chk(f"4.1 latency {sysname} (s)", latp[sysname].mean() / 1e3, want, 0.05)
d = latp["ADAM_LLM"] - latp["ADAM_NoBlockchain"]
chk("4.1 blockchain delta (ms)", d.mean(), CLAIMS["checks"]['4.1 blockchain delta (ms)'], 1); chk("4.1 blockchain trials", (d > 0).sum(), CLAIMS["checks"]['4.1 blockchain trials'], 0)
d = latp["ADAM_LLM"] - latp["ADAM_NoWeaviate"]
chk("4.1 weaviate delta (ms)", d.mean(), CLAIMS["checks"]['4.1 weaviate delta (ms)'], 1); chk("4.1 weaviate trials", (d > 0).sum(), CLAIMS["checks"]['4.1 weaviate trials'], 0)
d = latp["ADAM_NoAgg"] - latp["ADAM_LLM"]
chk("4.1 no-agg delta (ms)", d.mean(), CLAIMS["checks"]['4.1 no-agg delta (ms)'], 1); chk("4.1 no-agg trials", (d > 0).sum(), CLAIMS["checks"]['4.1 no-agg trials'], 0)

# ---------------------------------------------------------------- 4.2
co = pd.read_excel(xl, "05_D2_Coordination_Log")
done = co[co["Success"].astype(str).str.lower() == "yes"]
fail = co[co["Success"].astype(str).str.lower() != "yes"]
chk("4.2 events", len(co), CLAIMS["checks"]['4.2 events'], 0)
chk("4.2 completed", len(done), CLAIMS["checks"]['4.2 completed'], 0)
chk("4.2 failures", len(fail), CLAIMS["checks"]['4.2 failures'], 0)
tf = done["T_form_ms"]
chk("4.2 formation mean", tf.mean(), CLAIMS["checks"]['4.2 formation mean'], 1)
chk("4.2 formation sd", tf.std(ddof=1), CLAIMS["checks"]['4.2 formation sd'], 1)
chk("4.2 formation median", tf.median(), CLAIMS["checks"]['4.2 formation median'], 1)
chk("4.2 formation P95", np.percentile(tf, 95), CLAIMS["checks"]['4.2 formation P95'], 1)
fday = done.groupby("Day")["T_form_ms"].mean()
chk("4.2 per-day formation span (ms)", fday.max() - fday.min(), CLAIMS["checks"]['4.2 per-day formation span (ms)'], 1.5)
lday = done.groupby("Day")["T_decision_total_ms"].mean()
chk("4.2 per-day latency span (ms)", lday.max() - lday.min(), CLAIMS["checks"]['4.2 per-day latency span (ms)'], 15)

# Trigger publication: t_trigger -> first role acknowledgement (Sections 3.1.2, 4.2).
pub = (done["t_sensor_join"] - done["t_trigger"]).dt.total_seconds() * 1e3
chk("4.2 publication mean (ms)", pub.mean(), CLAIMS["checks"]['4.2 publication mean (ms)'], 1)
chk("4.2 publication sd (ms)", pub.std(ddof=1), CLAIMS["checks"]['4.2 publication sd (ms)'], 1)
chk("4.2 publication median (ms)", pub.median(), CLAIMS["checks"]['4.2 publication median (ms)'], 1)
chk("4.2 publication P95 (ms)", np.percentile(pub, 95), CLAIMS["checks"]['4.2 publication P95 (ms)'], 2)
chk("4.2 publication share of T_form (%)", pub.mean() / tf.mean() * 100, CLAIMS["checks"]['4.2 publication share of T_form (%)'], 0.1)
chk("4.2 publication share of budget (%)",
    pub.mean() / done["T_decision_total_ms"].mean() * 100, CLAIMS["checks"]['4.2 publication share of budget (%)'], 0.1)

# Deployment window: the run spans three calendar days, not 72 h (Section 3.4.3).
span_h = (co["Timestamp"].max() - co["Timestamp"].min()).total_seconds() / 3600
chk("3.4.3 coordination span (h)", span_h, CLAIMS["checks"]['3.4.3 coordination span (h)'], 0.1)
chk("4.3 event rate (per h)", len(co) / span_h, CLAIMS["checks"]['4.3 event rate (per h)'], 0.05)
td = done["T_decision_total_ms"]
chk("4.2 decision mean (s)", td.mean() / 1e3, CLAIMS["checks"]['4.2 decision mean (s)'], 0.01)
chk("4.2 decision median (s)", td.median() / 1e3, CLAIMS["checks"]['4.2 decision median (s)'], 0.01)
chk("4.2 decision P95 (s)", np.percentile(td, 95) / 1e3, CLAIMS["checks"]['4.2 decision P95 (s)'], 0.01)
chk("4.2 decision max (s)", td.max() / 1e3, CLAIMS["checks"]['4.2 decision max (s)'], 0.01)
STAGES42 = CLAIMS["tables"]["STAGES42"]
tot = sum(done[c].mean() for c in STAGES42)
for c, (ms_, pct) in STAGES42.items():
    chk(f"4.2 {c} mean", done[c].mean(), ms_, 1)
    chk(f"4.2 {c} share %", 100 * done[c].mean() / tot, pct, 0.06)
chk("4.2 gov+bc share %", 100 * (done["T_validate_ms"] + done["T_blockchain_ms"]).mean() / td.mean(), CLAIMS["checks"]['4.2 gov+bc share %'], 0.05)
chk("4.2 end-to-end completion %", 100 * len(done) / len(co), CLAIMS["checks"]['4.2 end-to-end completion %'], 0.05)
codes = fail["Notes"].str.extract(r"^(FAIL_[A-Z_]+)")[0].value_counts()
for code, n in CLAIMS["series"]["failure_codes"]:
    chk(f"4.2 {code}", codes.get(code, 0), n, 0)
chk("4.2 all failures censored at 30 s", (fail["T_decision_total_ms"] == DECISION_DEADLINE_S * 1000).sum(), CLAIMS["checks"]['4.2 all failures censored at 30 s'], 0)

# ---------------------------------------------------------------- 4.3
r = pd.read_excel(xl, "07_D2_Resource_Log")
STATES = CLAIMS["tables"]["STATES"]
for st, (cpu, ram, kb, n) in STATES.items():
    g = r[r["State"] == st]
    chk(f"4.3 {st} n", len(g), n, 0)
    chk(f"4.3 {st} CPU", g["CPU_Peak_%"].mean(), cpu, 0.05)
    chk(f"4.3 {st} RAM", g["RAM_MB"].mean(), ram, 1)
    chk(f"4.3 {st} KB/min", (g["Total_Bandwidth_Bytes"] / 1024).mean(), kb, 0.05)
out = r[r["State"] != "crew_active"]
chk("4.3 sustained CPU %", out["CPU_Peak_%"].mean(), CLAIMS["checks"]['4.3 sustained CPU %'], 0.05)
chk("4.3 max CPU %", r["CPU_Peak_%"].max(), CLAIMS["checks"]['4.3 max CPU %'], 0.05)
ca = r[r["State"] == "crew_active"]
chk("4.3 LLM KB/min", (ca["LLM_Bytes"] / 1024).mean(), CLAIMS["checks"]['4.3 LLM KB/min'], 0.05)
chk("4.3 Weaviate KB/min", (ca["Weaviate_Bytes"] / 1024).mean(), CLAIMS["checks"]['4.3 Weaviate KB/min'], 0.05)
chk("4.3 Blockchain KB/min", (ca["Blockchain_Bytes"] / 1024).mean(), CLAIMS["checks"]['4.3 Blockchain KB/min'], 0.05)
chk("4.3 Full-NoLLM RAM (MB)", S("ADAM_LLM", "RAM_MB")[0] - S("ADAM_NoLLM", "RAM_MB")[0], CLAIMS["checks"]['4.3 Full-NoLLM RAM (MB)'], 1)
chk("4.3 share of 8 GB %", 100 * ca["RAM_MB"].mean() / 8192, CLAIMS["checks"]['4.3 share of 8 GB %'], 0.05)
TABLE6 = CLAIMS["tables"]["TABLE6"]
for sysname, (cpu, ram, bw) in TABLE6.items():
    chk(f"4.3 {sysname} CPU", S(sysname, "CPU_WindowPeak_Mean_%")[0], cpu, 0.05)
    chk(f"4.3 {sysname} RAM", S(sysname, "RAM_MB")[0], ram, 1)
    chk(f"4.3 {sysname} BW", S(sysname, "Bandwidth_KB_per_60s_4nodes")[0], bw, 0.05)

# ---------------------------------------------------------------- 4.4: current stage model
from experiments.scalability_stage_model import (
    STAGES, load_hardware, simulate_scaleout, strict_leave_one_level_out, summarize,
)
sc = pd.read_excel(xl, "08_Scalability_Log")
hw = sc[sc["Run_Mode"] == "HARDWARE"]
hardware_path = ROOT / "data" / "scalability_hardware_v14.csv"
hardware_csv = pd.read_csv(hardware_path)
if set(hw.Run_ID) != set(hardware_csv.Run_ID):
    raise ValueError("Workbook and scale-model hardware Run_ID sets differ")
workbook_hardware = hw.set_index("Run_ID").loc[hardware_csv.Run_ID]
for field in (*STAGES, "T_decision_ms", "Node_Count"):
    if not np.allclose(workbook_hardware[field].to_numpy(), hardware_csv[field].to_numpy(), rtol=0, atol=1e-8):
        raise ValueError("Workbook and scale-model hardware differ: " + field)
hm = hw.groupby("Node_Count")["T_decision_ms"].mean()
chk("4.4 HW N=1 (s)", hm[1] / 1e3, CLAIMS["checks"]['4.4 HW N=1 (s)'], 0.005)
chk("4.4 HW N=4 (s)", hm[4] / 1e3, CLAIMS["checks"]['4.4 HW N=4 (s)'], 0.005)
chk("4.4 HW growth %", 100 * (hm[4] / hm[1] - 1), CLAIMS["checks"]['4.4 HW growth %'], 0.05)
current = CLAIMS["scalability_stage_model"]
grouped = load_hardware(hardware_path)
model_runs, _ = simulate_scaleout(grouped, seed=current["seed"], replicates=current["replicates_per_level"])
level_means = {row["node_count"]: row for row in summarize(model_runs)}
for node, values in current["levels_mean_sd_s"].items():
    row = level_means[int(node)]
    chk(f"4.4 stage-model N={node} mean (s)", row["mean_ms"] / 1000, values[0], 0.005)
    chk(f"4.4 stage-model N={node} SD (s)", row["std_ms"] / 1000, values[1], 0.005)
    chk(f"4.4 stage-model N={node} replicates", row["n"], current["replicates_per_level"], 0)
validation_rows, validation = strict_leave_one_level_out(grouped)
for name in ("mape_pct", "signed_mean_percentage_error_pct"):
    chk("4.4 strict LOLO " + name, validation[name], current["validation"][name], 0.005)
chk("4.4 strict LOLO MAE (ms)", validation["mae_ms"], current["validation"]["mae_ms"], 0.5)
worst = max(validation_rows, key=lambda row: row["abs_error_pct"])
chk("4.4 strict LOLO largest absolute error %", worst["abs_error_pct"], current["validation"]["largest_absolute_error_pct"], 0.005)
chk("4.4 strict LOLO largest error node", worst["held_out_N"], current["validation"]["largest_absolute_error_node"], 0)
chk("4.4 stage-model growth %", 100 * (level_means[16]["mean_ms"] / level_means[4]["mean_ms"] - 1), current["growth_pct"], 0.05)
gm = hw.groupby("Node_Count")[list(STAGES)].mean()
coord = [c for c in gm.columns if c != "T_reason_ms"]
chk("4.4 coordination growth (ms)", gm.loc[4, coord].sum() - gm.loc[1, coord].sum(), CLAIMS["checks"]['4.4 coordination growth (ms)'], 1)
chk("4.4 reasoning growth (ms)", gm.loc[4, "T_reason_ms"] - gm.loc[1, "T_reason_ms"], CLAIMS["checks"]['4.4 reasoning growth (ms)'], 1)
chk("4.4 stage sum = total N=1", gm.loc[1].sum(), hm[1], 1.5)
chk("4.4 stage sum = total N=4", gm.loc[4].sum(), hm[4], 1.5)

# ---------------------------------------------------------------- 4.5
inj = pd.read_excel(xl, "10_Security_Injection", header=2)
llm = pd.read_excel(xl, "11_Security_LLM_Failure", header=2)
poi = pd.read_excel(xl, "12_Security_Poisoning", header=2)
egr = pd.read_excel(xl, "13_Security_Data_Leakage", header=2)
det = inj["ADAM_Detected_Attack"].astype(str).str.lower().eq("yes")
chk("4.5 injection detected", det.sum(), CLAIMS["checks"]['4.5 injection detected'], 0)
BY_TYPE = CLAIMS["tables"]["BY_TYPE"]
for atk, (d_, n_) in BY_TYPE.items():
    g = inj[inj["Attack_Type"] == atk]
    chk(f"4.5 {atk} n", len(g), n_, 0)
    chk(f"4.5 {atk} detected", g["ADAM_Detected_Attack"].astype(str).str.lower().eq("yes").sum(), d_, 0)
tp = ((inj["ADAM_Prediction"] == "anomaly") & (inj["Ground_Truth"] == "anomaly")).sum()
fp = ((inj["ADAM_Prediction"] == "anomaly") & (inj["Ground_Truth"] == "normal")).sum()
fn = ((inj["ADAM_Prediction"] == "normal") & (inj["Ground_Truth"] == "anomaly")).sum()
tn = ((inj["ADAM_Prediction"] == "normal") & (inj["Ground_Truth"] == "normal")).sum()
chk("4.5 attack F1", 2 * tp / (2 * tp + fp + fn), CLAIMS["checks"]['4.5 attack F1'])
chk("4.5 attack FAR", fp / (fp + tn), CLAIMS["checks"]['4.5 attack FAR'])
lvl = poi.groupby("Num_Poisoned_Entries").apply(lambda g: ((g["ADAM_Prediction"] == g["Ground_Truth"]).sum(), len(g)))
for level, (ok_, n_) in CLAIMS["series"]["poisoning_groups"]:
    chk(f"4.5 poisoning L{level} correct", lvl.loc[level][0], ok_, 0)
    chk(f"4.5 poisoning L{level} n", lvl.loc[level][1], n_, 0)
chk("4.5 retrieval affected", poi["Retrieval_Affected"].astype(str).str.lower().eq("yes").sum(), CLAIMS["checks"]['4.5 retrieval affected'], 0)
fb = llm[llm["Fallback_Triggered"].astype(str).str.lower() == "yes"]
lat = fb["Fallback_Latency_ms"].astype(float)
chk("4.5 fallback n", len(fb), CLAIMS["checks"]['4.5 fallback n'], 0)
chk("4.5 fallback mean", lat.mean(), CLAIMS["checks"]['4.5 fallback mean'], 0.05)
chk("4.5 fallback median", lat.median(), CLAIMS["checks"]['4.5 fallback median'], 0.05)
chk("4.5 fallback P95", np.percentile(lat, 95), CLAIMS["checks"]['4.5 fallback P95'], 0.05)
chk("4.5 crews completed", llm["Crew_Continued"].astype(str).str.lower().eq("yes").sum(), CLAIMS["checks"]['4.5 crews completed'], 0)
tpf = ((fb["Prediction"] == "anomaly") & (fb["Ground_Truth"] == "anomaly")).sum()
fpf = ((fb["Prediction"] == "anomaly") & (fb["Ground_Truth"] == "normal")).sum()
fnf = ((fb["Prediction"] == "normal") & (fb["Ground_Truth"] == "anomaly")).sum()
chk("4.5 fallback-only F1", 2 * tpf / (2 * tpf + fpf + fnf), CLAIMS["checks"]['4.5 fallback-only F1'])
adam_e = egr[egr["System"].astype(str).str.contains("ADAM", case=False)]
cloud_e = egr[~egr["System"].astype(str).str.contains("ADAM", case=False)]
chk("4.5 ADAM windows", len(adam_e), CLAIMS["checks"]['4.5 ADAM windows'], 0)
chk("4.5 ADAM egress zero", (adam_e["Total_Bytes_External"] == 0).all(), CLAIMS["checks"]['4.5 ADAM egress zero'], 0)
chk("4.5 cloud windows", len(cloud_e), CLAIMS["checks"]['4.5 cloud windows'], 0)
chk("4.5 cloud KB/window", (cloud_e["Total_Bytes_External"] / 1024).mean(), CLAIMS["checks"]['4.5 cloud KB/window'], 0.05)
chk("4.5 cloud calls/window", cloud_e["External_API_Calls"].mean(), CLAIMS["checks"]['4.5 cloud calls/window'], 0.05)

# ---------------------------------------------------------------- substitution + degraded families
swap = pd.read_excel(xl, "17_Swap_Study_Summary", header=1)
for agent, alone, crew in CLAIMS["series"]["substitution_results"]:
    row = swap[swap["Decision_Agent"] == agent].iloc[0]
    chk(f"swap {agent} standalone F1", row["Standalone_F1"], alone, 0.0006)
    chk(f"swap {agent} in-crew F1", row["InCrew_F1"], crew, 0.0006)

deg = pd.read_excel(xl, "14_Degraded_Conditions", header=1)
for cond, want_llm, want_gbm in CLAIMS["series"]["degraded_results"]:
    llm_mean = deg[(deg.Condition == cond) & (deg.System == "ADAM_LLM")]["F1_Score"].mean()
    gbm_mean = deg[(deg.Condition == cond) & (deg.System == "ADAM_GBM")]["F1_Score"].mean()
    chk(f"degraded {cond} ADAM_LLM F1", llm_mean, want_llm, 0.0006)
    chk(f"degraded {cond} ADAM_GBM F1", gbm_mean, want_gbm, 0.0006)

swap_trials = pd.read_excel(xl, "15_Swap_Study_Trials", header=1)
sub_stats = substitution_family(tr, swap_trials)
deg_stats = degraded_family(deg)
chk("stat family main size", len(main_stats), CLAIMS["checks"]['stat family main size'], 0)
chk("stat family substitution size", len(sub_stats), CLAIMS["checks"]['stat family substitution size'], 0)
chk("stat family degraded size", len(deg_stats), CLAIMS["checks"]['stat family degraded size'], 0)

sub_by_name = {r.comparison: r for r in sub_stats}
chk("swap Static p_Holm", sub_by_name["Static threshold"].p_holm, CLAIMS["checks"]['swap Static p_Holm'], 1e-12)
chk("swap LLM p_Holm", sub_by_name["LLM (Gemma 3 1B)"].p_holm, CLAIMS["checks"]['swap LLM p_Holm'], 1e-12)
for label in ("Logistic regression", "Random forest", "Gradient boosting"):
    chk(f"swap {label} p_Holm", sub_by_name[label].p_holm, CLAIMS["checks"]["swap fitted p_Holm"], 1e-12)
for row in deg_stats:
    chk(f"degraded Holm {row.comparison}", row.p_holm, CLAIMS["checks"]["degraded Holm common"], 1e-12)

print("Workbook-backed checks passed, including the 11-system benchmark, ")
print("Decision-Agent substitution study, and degraded-condition family.")
print("NOTE: the workbook preserves fused/dispersion/result records but not the ")
print("complete concurrent N1-N4 raw stream needed to re-execute sensor fusion ")
print("under alternative calibration weights or regenerate per-node perturbations.")

# ---------------------------------------------------------------- summary
if FAILURES:
    print(f"\n{len(FAILURES)} MISMATCH(ES):")
    for f_ in FAILURES:
        print("  " + f_)
    sys.exit(1)
print("\nALL CHECKS PASSED")
