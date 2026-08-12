#!/usr/bin/env python3
"""Verify every quantitative claim in the article's Results section against the
deposited workbook. Exits non-zero on any mismatch.

Usage:  python scripts/verify_manuscript_numbers.py [workbook.xlsx]

Sections covered: 4.1 detection and operating point, 4.2 coordination and
failures, 4.3 resources, 4.4 node scaling, 4.5 security. Values quoted in the
article at coarser precision are checked at that precision.
"""

import math
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

WB = sys.argv[1] if len(sys.argv) > 1 else "data/ADAM_Dataset_Master.xlsx"
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
gs = pd.read_excel(xl, "D1_RawTrigger_Summary")
tests = pd.read_excel(xl, "04_D1_Statistical_Tests", header=1).set_index("Comparison")

S = lambda s, c: (tr[tr.System == s][c].mean(), tr[tr.System == s][c].std(ddof=1))
TABLE4 = {  # system: (P, R, F1, FAR)
    "ADAM_Full": (0.901, 0.891, 0.896, 0.080),
    "Static_Threshold": (0.795, 0.786, 0.790, 0.165),
    "Random_Forest": (0.948, 0.730, 0.825, 0.033),
    "Cloud_Only": (0.904, 0.899, 0.901, 0.078),
    "SingleAgent": (0.881, 0.830, 0.855, 0.092),
    "ADAM_NoAgg": (0.889, 0.851, 0.869, 0.087),
    "ADAM_NoLLM": (0.864, 0.817, 0.840, 0.105),
    "ADAM_NoBlockchain": (0.902, 0.877, 0.889, 0.078),
    "ADAM_NoWeaviate": (0.883, 0.853, 0.868, 0.093),
}
for sysname, (p_, r_, f_, fa_) in TABLE4.items():
    for col, want in zip(("Precision", "Recall", "F1", "FAR"), (p_, r_, f_, fa_)):
        chk(f"4.1 {sysname} {col}", S(sysname, col)[0], want)
chk("4.1 ADAM F1 sd", S("ADAM_Full", "F1")[1], 0.009)
chk("4.1 margin over Static (pts)", S("ADAM_Full", "F1")[0] - S("Static_Threshold", "F1")[0], 0.106, 0.001)
chk("4.1 SingleAgent gap (pts)", S("ADAM_Full", "F1")[0] - S("SingleAgent", "F1")[0], 0.041, 0.001)

lab = pd.read_excel(xl, "02_D1_Labeled_Events")
v = (lab["Raw_Instantaneous_PPM"] - lab["Reference_Sensor_PPM"]).groupby(lab["Node_ID"]).var(ddof=1)
w = (1 / v) / (1 / v).sum()
chk("4.1 variance min ppm2", v.min(), 6073.2, 0.5)
chk("4.1 variance max ppm2", v.max(), 6609.6, 0.5)
chk("4.1 norm weight min", w.min(), 0.239, 0.001)
chk("4.1 norm weight max", w.max(), 0.260, 0.001)

WILCOXON = {"ADAM_vs_Static": (0.002, 0.016), "ADAM_vs_RF": (0.002, 0.016),
            "ADAM_vs_SingleAgent": (0.002, 0.016), "ADAM_vs_NoLLM": (0.002, 0.016),
            "ADAM_vs_NoAgg": (0.004, 0.016), "ADAM_vs_NoWeaviate": (0.004, 0.016),
            "ADAM_vs_Cloud": (0.049, 0.098), "ADAM_vs_NoBlockchain": (0.098, None)}
for key, (pe, ph) in WILCOXON.items():
    chk(f"4.1 {key} p_exact", tests.loc[key, "P_Exact"], pe)
    if ph is not None:
        chk(f"4.1 {key} p_Holm", tests.loc[key, "P_Holm"], ph)
f1p = tr.pivot(index="Trial", columns="System", values="F1")
dc = f1p["Cloud_Only"] - f1p["ADAM_Full"]
chk("4.1 Cloud median diff", dc.median(), 0.007)
chk("4.1 Cloud trials favoring cloud", (dc > 0).sum(), 8, 0)
db = f1p["ADAM_Full"] - f1p["ADAM_NoBlockchain"]
nz = db[db != 0]
chk("4.1 NoBlockchain effective n", len(nz), 9, 0)
chk("4.1 NoBlockchain favoring ADAM", (nz > 0).sum(), 8, 0)
chk("4.1 NoBlockchain median", nz.median(), 0.009)

m = trig.merge(pred[["Event_ID", "ADAM_Full", "Static_Threshold"]], on="Event_ID")
above = m["Raw_Instantaneous_PPM"] >= 1000
anom = m["Ground_Truth_Label"] == "anomaly"
chk("4.1 anomalies above gate", (anom & above).sum(), 707, 0)
chk("4.1 anomalies below gate", (anom & ~above).sum(), 193, 0)
chk("4.1 triggered events", above.sum(), 889, 0)
for col, ra, rb, ro in [("Static_Threshold", 1.000, 0.000, 0.786),
                        ("ADAM_Prediction", 0.976, 0.000, 0.767),
                        ("ADAM_Full", 0.976, 0.580, 0.891)]:
    hit = (m[col] == "anomaly") & anom
    chk(f"4.1 {col} recall above", hit[above].sum() / (anom & above).sum(), ra)
    chk(f"4.1 {col} recall below", hit[~above].sum() / (anom & ~above).sum(), rb)
    chk(f"4.1 {col} recall overall", hit.sum() / anom.sum(), ro)
chk("4.1 triggered FP static", ((m["Static_Threshold"] == "anomaly") & ~anom & above).sum(), 182, 0)
chk("4.1 triggered FP gated", ((m["ADAM_Prediction"] == "anomaly") & ~anom & above).sum(), 105, 0)
g10 = gs[gs["Trial"].astype(str) != "Overall"]
for col, mean_, sd_ in [("Precision", 0.869, 0.028), ("Recall", 0.767, 0.029),
                        ("F1", 0.814, 0.021), ("FAR", 0.095, 0.023)]:
    chk(f"4.1 deployment {col} mean", g10[col].mean(), mean_)
    chk(f"4.1 deployment {col} sd", g10[col].std(ddof=1), sd_)

latp = tr.pivot(index="Trial", columns="System", values="T_decision_ms")
for sysname, want in [("Static_Threshold", 0.3), ("Random_Forest", 0.4),
                      ("ADAM_NoLLM", 1.3), ("ADAM_Full", 18.8), ("Cloud_Only", 12.9)]:
    chk(f"4.1 latency {sysname} (s)", latp[sysname].mean() / 1e3, want, 0.05)
d = latp["ADAM_Full"] - latp["ADAM_NoBlockchain"]
chk("4.1 blockchain delta (ms)", d.mean(), 246, 1); chk("4.1 blockchain trials", (d > 0).sum(), 7, 0)
d = latp["ADAM_Full"] - latp["ADAM_NoWeaviate"]
chk("4.1 weaviate delta (ms)", d.mean(), 83, 1); chk("4.1 weaviate trials", (d > 0).sum(), 6, 0)
d = latp["ADAM_NoAgg"] - latp["ADAM_Full"]
chk("4.1 no-agg delta (ms)", d.mean(), 451, 1); chk("4.1 no-agg trials", (d > 0).sum(), 9, 0)

# ---------------------------------------------------------------- 4.2
co = pd.read_excel(xl, "05_D2_Coordination_Log")
done = co[co["Success"].astype(str).str.lower() == "yes"]
fail = co[co["Success"].astype(str).str.lower() != "yes"]
chk("4.2 events", len(co), 459, 0)
chk("4.2 completed", len(done), 446, 0)
chk("4.2 failures", len(fail), 13, 0)
tf = done["T_form_ms"]
chk("4.2 formation mean", tf.mean(), 2076, 1)
chk("4.2 formation sd", tf.std(ddof=1), 182, 1)
chk("4.2 formation median", tf.median(), 2078, 1)
chk("4.2 formation P95", np.percentile(tf, 95), 2373, 1)
fday = done.groupby("Day")["T_form_ms"].mean()
chk("4.2 per-day formation span (ms)", fday.max() - fday.min(), 9, 1.5)
lday = done.groupby("Day")["T_decision_total_ms"].mean()
chk("4.2 per-day latency span (ms)", lday.max() - lday.min(), 80, 15)
td = done["T_decision_total_ms"]
chk("4.2 decision mean (s)", td.mean() / 1e3, 18.99, 0.01)
chk("4.2 decision median (s)", td.median() / 1e3, 19.00, 0.01)
chk("4.2 decision P95 (s)", np.percentile(td, 95) / 1e3, 20.39, 0.01)
chk("4.2 decision max (s)", td.max() / 1e3, 21.38, 0.01)
STAGES42 = {"T_reason_ms": (15474, 81.5), "T_form_ms": (2076, 10.9),
            "T_aggregate_ms": (437, 2.3), "T_validate_ms": (308, 1.6),
            "T_weaviate_ms": (256, 1.3), "T_blockchain_ms": (439, 2.3)}
tot = sum(done[c].mean() for c in STAGES42)
for c, (ms_, pct) in STAGES42.items():
    chk(f"4.2 {c} mean", done[c].mean(), ms_, 1)
    chk(f"4.2 {c} share %", 100 * done[c].mean() / tot, pct, 0.06)
chk("4.2 gov+bc share %", 100 * (done["T_validate_ms"] + done["T_blockchain_ms"]).mean() / td.mean(), 3.9, 0.05)
chk("4.2 persistence %", 100 * len(done) / len(co), 97.2, 0.05)
codes = fail["Notes"].str.extract(r"^(FAIL_[A-Z_]+)")[0].value_counts()
for code, n in [("FAIL_LLM_DEADLINE", 3), ("FAIL_BLOCKCHAIN_COMMIT", 3),
                ("FAIL_VALIDATION_REPAIR", 3), ("FAIL_CREW_QUORUM", 2),
                ("FAIL_WEAVIATE_TIMEOUT", 2)]:
    chk(f"4.2 {code}", codes.get(code, 0), n, 0)
chk("4.2 all failures censored at 30 s", (fail["T_decision_total_ms"] == 30000).sum(), 13, 0)

# ---------------------------------------------------------------- 4.3
r = pd.read_excel(xl, "07_D2_Resource_Log")
STATES = {"crew_active": (94.7, 3890, 41.0, 219), "monitoring": (28.0, 2349, 12.7, 434),
          "idle": (11.3, 1530, 3.0, 256)}
for st, (cpu, ram, kb, n) in STATES.items():
    g = r[r["State"] == st]
    chk(f"4.3 {st} n", len(g), n, 0)
    chk(f"4.3 {st} CPU", g["CPU_Peak_%"].mean(), cpu, 0.05)
    chk(f"4.3 {st} RAM", g["RAM_MB"].mean(), ram, 1)
    chk(f"4.3 {st} KB/min", (g["Total_Bandwidth_Bytes"] / 1024).mean(), kb, 0.05)
out = r[r["State"] != "crew_active"]
chk("4.3 sustained CPU %", out["CPU_Peak_%"].mean(), 21.8, 0.05)
chk("4.3 max CPU %", r["CPU_Peak_%"].max(), 97.7, 0.05)
ca = r[r["State"] == "crew_active"]
chk("4.3 LLM KB/min", (ca["LLM_Bytes"] / 1024).mean(), 21.1, 0.05)
chk("4.3 Weaviate KB/min", (ca["Weaviate_Bytes"] / 1024).mean(), 13.8, 0.05)
chk("4.3 Blockchain KB/min", (ca["Blockchain_Bytes"] / 1024).mean(), 6.1, 0.05)
chk("4.3 Full-NoLLM RAM (MB)", S("ADAM_Full", "RAM_MB")[0] - S("ADAM_NoLLM", "RAM_MB")[0], 1540, 1)
chk("4.3 share of 8 GB %", 100 * 3890 / 8192, 47.5, 0.05)
TABLE6 = {"ADAM_Full": (94.7, 3890, 163.9), "Static_Threshold": (14.5, 766, 26.5),
          "Random_Forest": (18.0, 800, 34.0), "Cloud_Only": (22.3, 1063, 228.0),
          "SingleAgent": (91.3, 3650, 133.0), "ADAM_NoAgg": (92.5, 3780, 205.0),
          "ADAM_NoLLM": (32.8, 2350, 87.3), "ADAM_NoBlockchain": (92.4, 3710, 140.0),
          "ADAM_NoWeaviate": (85.2, 3400, 112.0)}
for sysname, (cpu, ram, bw) in TABLE6.items():
    chk(f"4.3 {sysname} CPU", S(sysname, "CPU_WindowPeak_Mean_%")[0], cpu, 0.05)
    chk(f"4.3 {sysname} RAM", S(sysname, "RAM_MB")[0], ram, 1)
    chk(f"4.3 {sysname} BW", S(sysname, "Bandwidth_KB_per_60s_4nodes")[0], bw, 0.05)

# ---------------------------------------------------------------- 4.4
sc = pd.read_excel(xl, "08_Scalability_Log")
hw = sc[sc["Run_Mode"] == "HARDWARE"]
sim = sc[sc["Run_Mode"] == "PYTHON_SIMULATION"]
so = sc[(sc["Fit_Eligible"].astype(str).str.lower() == "yes") & (sc["Node_Count"] >= 4)]
hm = hw.groupby("Node_Count")["T_decision_ms"].mean()
sm = so.groupby("Node_Count")["T_decision_ms"].mean()
im = sim.groupby("Node_Count")["T_decision_ms"].mean()
chk("4.4 HW N=1 (s)", hm[1] / 1e3, 17.64, 0.005)
chk("4.4 HW N=4 (s)", hm[4] / 1e3, 18.82, 0.005)
chk("4.4 HW growth %", 100 * (hm[4] / hm[1] - 1), 6.7, 0.05)
chk("4.4 SO N=4 (s)", sm[4] / 1e3, 18.55, 0.005)
chk("4.4 SO N=16 (s)", sm[16] / 1e3, 20.94, 0.005)
chk("4.4 SO growth %", 100 * (sm[16] / sm[4] - 1), 12.9, 0.05)
chk("4.4 overall growth %", 100 * (sm[16] / hm[1] - 1), 18.7, 0.05)
chk("4.4 margin at N=16 (s)", 30 - sm[16] / 1e3, 9.1, 0.05)
rel = [(im[k] - hm[k]) / hm[k] * 100 for k in [1, 2, 3, 4]]
chk("4.4 MAPE %", np.mean(np.abs(rel)), 2.37, 0.01)
chk("4.4 bias %", np.mean(rel), -0.02, 0.01)
chk("4.4 worst level dev % (N=2)", rel[1], 4.0, 0.05)
from scipy.optimize import curve_fit
mdl = lambda N, t0, a, b: t0 + a * N ** b
for name, df, want in [("HW", hw, (17380, 198.8, 1.462)), ("SO", so, (14262, 2950.6, 0.288))]:
    g = df.groupby("Node_Count")["T_decision_ms"].mean()
    pw, _ = curve_fit(mdl, g.index.astype(float), g.values, p0=[15000, 500, 1.0], maxfev=60000)
    for pname, got, want_ in zip(("T0", "alpha", "beta"), pw, want):
        chk(f"4.4 {name} fit {pname}", got, want_, max(abs(want_) * 0.005, 0.005))
gm = hw.groupby("Node_Count")[["T_reason_ms", "T_cross_node_ms", "T_network_ms",
                               "T_merge_ms", "T_query_ms", "T_blockchain_ms"]].mean()
coord = [c for c in gm.columns if c != "T_reason_ms"]
chk("4.4 coordination growth (ms)", gm.loc[4, coord].sum() - gm.loc[1, coord].sum(), 594, 1)
chk("4.4 reasoning growth (ms)", gm.loc[4, "T_reason_ms"] - gm.loc[1, "T_reason_ms"], 584, 1)
chk("4.4 stage sum = total N=1", gm.loc[1].sum(), hm[1], 1.5)
chk("4.4 stage sum = total N=4", gm.loc[4].sum(), hm[4], 1.5)

# ---------------------------------------------------------------- 4.5
inj = pd.read_excel(xl, "10_Security_Injection", header=2)
llm = pd.read_excel(xl, "11_Security_LLM_Failure", header=2)
poi = pd.read_excel(xl, "12_Security_Poisoning", header=2)
egr = pd.read_excel(xl, "13_Security_Data_Leakage", header=2)
det = inj["ADAM_Detected_Attack"].astype(str).str.lower().eq("yes")
chk("4.5 injection detected", det.sum(), 27, 0)
BY_TYPE = {"replay": (7, 7), "spike_inject": (10, 11), "zero_inject": (8, 9),
           "constant_offset": (2, 3)}
for atk, (d_, n_) in BY_TYPE.items():
    g = inj[inj["Attack_Type"] == atk]
    chk(f"4.5 {atk} n", len(g), n_, 0)
    chk(f"4.5 {atk} detected", g["ADAM_Detected_Attack"].astype(str).str.lower().eq("yes").sum(), d_, 0)
tp = ((inj["ADAM_Prediction"] == "anomaly") & (inj["Ground_Truth"] == "anomaly")).sum()
fp = ((inj["ADAM_Prediction"] == "anomaly") & (inj["Ground_Truth"] == "normal")).sum()
fn = ((inj["ADAM_Prediction"] == "normal") & (inj["Ground_Truth"] == "anomaly")).sum()
tn = ((inj["ADAM_Prediction"] == "normal") & (inj["Ground_Truth"] == "normal")).sum()
chk("4.5 attack F1", 2 * tp / (2 * tp + fp + fn), 0.769)
chk("4.5 attack FAR", fp / (fp + tn), 0.176)
lvl = poi.groupby("Num_Poisoned_Entries").apply(lambda g: ((g["ADAM_Prediction"] == g["Ground_Truth"]).sum(), len(g)))
for level, (ok_, n_) in [(0, (8, 8)), (5, (7, 8)), (10, (6, 7)), (20, (6, 7))]:
    chk(f"4.5 poisoning L{level} correct", lvl.loc[level][0], ok_, 0)
    chk(f"4.5 poisoning L{level} n", lvl.loc[level][1], n_, 0)
chk("4.5 retrieval affected", poi["Retrieval_Affected"].astype(str).str.lower().eq("yes").sum(), 3, 0)
fb = llm[llm["Fallback_Triggered"].astype(str).str.lower() == "yes"]
lat = fb["Fallback_Latency_ms"].astype(float)
chk("4.5 fallback n", len(fb), 19, 0)
chk("4.5 fallback mean", lat.mean(), 55.7, 0.05)
chk("4.5 fallback median", lat.median(), 54.6, 0.05)
chk("4.5 fallback P95", np.percentile(lat, 95), 81.6, 0.05)
chk("4.5 crews completed", llm["Crew_Continued"].astype(str).str.lower().eq("yes").sum(), 30, 0)
tpf = ((fb["Prediction"] == "anomaly") & (fb["Ground_Truth"] == "anomaly")).sum()
fpf = ((fb["Prediction"] == "anomaly") & (fb["Ground_Truth"] == "normal")).sum()
fnf = ((fb["Prediction"] == "normal") & (fb["Ground_Truth"] == "anomaly")).sum()
chk("4.5 fallback-only F1", 2 * tpf / (2 * tpf + fpf + fnf), 0.842)
adam_e = egr[egr["System"].astype(str).str.contains("ADAM", case=False)]
cloud_e = egr[~egr["System"].astype(str).str.contains("ADAM", case=False)]
chk("4.5 ADAM windows", len(adam_e), 12, 0)
chk("4.5 ADAM egress zero", (adam_e["Total_Bytes_External"] == 0).all(), 1, 0)
chk("4.5 cloud windows", len(cloud_e), 8, 0)
chk("4.5 cloud KB/window", (cloud_e["Total_Bytes_External"] / 1024).mean(), 117.4, 0.05)
chk("4.5 cloud calls/window", cloud_e["External_API_Calls"].mean(), 19.1, 0.05)

# ---------------------------------------------------------------- summary
print(f"\n{'ALL ' + str(len(FAILURES) == 0 and 'CHECKS PASSED' or '')}"
      if not FAILURES else "")
if FAILURES:
    print(f"\n{len(FAILURES)} MISMATCH(ES):")
    for f_ in FAILURES:
        print("  " + f_)
    sys.exit(1)
print("Every quantitative Results claim reproduces from the deposited workbook.")
