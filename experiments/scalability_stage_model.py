"""Hardware-calibrated, fixed-load six-stage software scale-out model.

This script produces model estimates from the supplied measured hardware rows.
It does not perform Raspberry Pi acquisition. No result target is used in the
calculation, and this script cannot establish whether other experiments were
rerun outside this repository.

Usage:
  python -m experiments.scalability_stage_model \
      --hardware data/scalability_hardware_v14.csv --out results/phase3

Only recorded HARDWARE rows enter calibration; an explicit strict held-out
level is excluded from every parameter and anchor estimated for that fold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

STAGES: Tuple[str, ...] = (
    "T_cross_node_ms", "T_network_ms", "T_merge_ms",
    "T_query_ms", "T_reason_ms", "T_blockchain_ms",
)
NODE_DEPENDENT = STAGES[:3]
SHARED_OR_FIXED = STAGES[3:]
PHYSICAL_LEVELS = (1, 2, 3, 4)
SCALEOUT_LEVELS = (4, 6, 8, 12, 16)
REPLICATES = 18
SEED = 42
WORKLOAD = {"concurrent_events": 4, "logical_sensor_streams": 8,
            "logical_workers": 4, "database_vectors": 30000,
            "blockchain_requests_per_event": 1,
            "arrival_process": "one simultaneous four-event batch",
            "reasoning_workers_added_for_logical_nodes": 0}
FIELDS = (
    "Run_ID", "Day", "Replicate", "Node_Count", "Physical_Pi_Count",
    "Simulated_Node_Count", "Run_Mode", "Concurrent_Events", "Num_Sensors",
    "Database_Size_Vectors", *STAGES, "T_decision_ms",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs)


def load_hardware(path: Path) -> Dict[int, List[dict]]:
    """Reject missing, synthetic, inconsistent, incomplete, or non-additive input."""
    if not path.is_file():
        raise FileNotFoundError(path)
    grouped: Dict[int, List[dict]] = {n: [] for n in PHYSICAL_LEVELS}
    ids = set()
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not set(FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError(f"missing columns {sorted(set(FIELDS) - set(reader.fieldnames or ()))}")
        for row in reader:
            run_id = row["Run_ID"]
            if not run_id or run_id in ids:
                raise ValueError(f"missing or duplicate Run_ID {run_id!r}")
            ids.add(run_id)
            if row["Run_Mode"] != "HARDWARE":
                raise ValueError(f"not a physical record: {run_id} ({row['Run_Mode']})")
            node = int(row["Node_Count"])
            if node not in grouped:
                raise ValueError(f"unexpected physical node count {node}")
            for key, expected in (("Physical_Pi_Count", node),
                                  ("Simulated_Node_Count", 0),
                                  ("Concurrent_Events", 4),
                                  ("Num_Sensors", 8),
                                  ("Database_Size_Vectors", 30000)):
                if int(row[key]) != expected:
                    raise ValueError(f"{run_id}: {key}={row[key]}, expected {expected}")
            stages = {}
            for stage in STAGES:
                value = float(row[stage])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"{run_id}: invalid {stage}={value}")
                stages[stage] = value
            reported = float(row["T_decision_ms"])
            if not math.isfinite(reported) or abs(reported - sum(stages.values())) > 0.11:
                raise ValueError(f"{run_id}: T_decision_ms does not equal six-stage sum")
            grouped[node].append({"run_id": run_id, "node": node, "stages": stages,
                                  "reported_total": reported,
                                  "day": int(row["Day"]), "replicate": int(row["Replicate"])})
    if {n: len(v) for n, v in grouped.items()} != {n: REPLICATES for n in PHYSICAL_LEVELS}:
        raise ValueError(f"expected 18 physical runs per level; got { {n:len(v) for n,v in grouped.items()} }")
    for node, records in grouped.items():
        if len({(r["day"], r["replicate"]) for r in records}) != REPLICATES:
            raise ValueError(f"duplicate day/replicate within N={node}")
    return grouped


def stage_means(grouped: Mapping[int, Sequence[dict]]) -> Dict[int, Dict[str, float]]:
    return {n: {s: _mean([r["stages"][s] for r in records]) for s in STAGES}
            for n, records in grouped.items()}


def fit_slopes(means: Mapping[int, Mapping[str, float]]) -> Dict[str, float]:
    """Non-negative OLS slope through node-level stage means."""
    levels = sorted(means)
    if len(levels) < 2:
        raise ValueError("need >=2 training node levels")
    nbar = _mean(levels)
    den = sum((n - nbar)**2 for n in levels)
    return {s: max(0., sum((n-nbar)*(means[n][s] -
                                _mean([means[k][s] for k in levels]))
                         for n in levels)/den) for s in NODE_DEPENDENT}


def predict_stages(target: int, means: Mapping[int, Mapping[str, float]],
                   slopes: Mapping[str, float], anchor: int) -> Dict[str, float]:
    if anchor not in means:
        raise ValueError("anchor must belong to training levels")
    result = {s: max(0., means[anchor][s] + slopes[s]*(target-anchor))
              if s in NODE_DEPENDENT else means[anchor][s] for s in STAGES}
    # Structural condition independent of the withheld data: a single node
    # has no peers with which to exchange cross-node information.
    if target == 1:
        result["T_cross_node_ms"] = 0.0
    return result


def strict_leave_one_level_out(grouped: Mapping[int, Sequence[dict]]) -> Tuple[List[dict], dict]:
    """Every withheld level is absent from slopes, stage base, and residuals."""
    validation = []
    for held in PHYSICAL_LEVELS:
        training = {n: grouped[n] for n in PHYSICAL_LEVELS if n != held}
        means = stage_means(training)
        slopes = fit_slopes(means)
        anchor = min(means, key=lambda n: (abs(n-held),n))
        predicted = predict_stages(held, means, slopes, anchor)
        observed = _mean([r["reported_total"] for r in grouped[held]])
        total = sum(predicted.values())
        error = 100*(total-observed)/observed
        actual_stages = {s:_mean([r["stages"][s] for r in grouped[held]]) for s in STAGES}
        absolute_stage_error = {s:abs(predicted[s]-actual_stages[s]) for s in STAGES}
        validation.append({"held_out_N":held, "train_levels": ",".join(map(str, sorted(training))),
                           "anchor_N":anchor, "hardware_mean_ms":observed,
                           "predicted_mean_ms":total, "error_pct":error,
                           "abs_error_pct":abs(error), "absolute_error_ms":abs(total-observed),
                           "sum_absolute_stage_error_ms":sum(absolute_stage_error.values()),
                           **{f"predicted_{s}":predicted[s] for s in STAGES},
                           **{f"observed_{s}":actual_stages[s] for s in STAGES}})
    stats = {"method":"strict leave-one-physical-node-level-out stage prediction",
             "level_count":len(validation),
             "mape_pct":_mean([r["abs_error_pct"] for r in validation]),
             "signed_mean_percentage_error_pct":_mean([r["error_pct"] for r in validation]),
             "mae_ms":_mean([r["absolute_error_ms"] for r in validation]),
             "mean_sum_absolute_stage_error_ms":_mean([r["sum_absolute_stage_error_ms"] for r in validation]),
             "stage_mae_ms":{
                 s:_mean([abs(r[f"predicted_{s}"]-r[f"observed_{s}"]) for r in validation])
                 for s in STAGES},
             "note":"Aggregate error may mask stage errors; compare validation methods separately."}
    return validation, stats


def simulate_scaleout(grouped: Mapping[int, Sequence[dict]], seed: int=SEED,
                      replicates: int=REPLICATES,
                      levels: Sequence[int]=SCALEOUT_LEVELS) -> Tuple[List[dict],dict]:
    """Paired bootstrap: sample all physical levels, draw a full N4 stage vector.

    Reusing one vector/one fitted coefficient set for different N within each
    replicate preserves covariance between stages and scale-out levels.
    """
    if not levels or any(n < 4 for n in levels) or len(set(levels)) != len(levels):
        raise ValueError("levels must be distinct logical node counts >=4")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    rng = random.Random(seed)
    output: List[dict] = []
    for replicate in range(1,replicates+1):
        sampled = {n: [rng.choice(grouped[n]) for _ in range(len(grouped[n]))]
                   for n in PHYSICAL_LEVELS}
        means = stage_means(sampled)
        slopes = fit_slopes(means)
        base = rng.choice(sampled[4])  # paired observed six-stage vector
        for n in levels:
            stages={s: max(0., base["stages"][s]+slopes[s]*(n-4))
                    if s in NODE_DEPENDENT else base["stages"][s] for s in STAGES}
            output.append({"run_id":f"REF-STAGE-N{n:02d}-R{replicate:02d}",
                           "run_mode":"REFERENCE_MODEL_ESTIMATE", "node_count":n,
                           "physical_pi_count":4,"simulated_node_count":n-4,
                           "replicate":replicate, "random_seed":seed,
                           "reference_hardware_run_id":base["run_id"],
                           **WORKLOAD, **stages,"T_decision_ms":sum(stages.values())})
    fit={"model":"T_j(N)=observed N4 vector + max(0,OLS stage slope)*(N-4), j=cross/network/merge; fixed other stages",
         "physical_calibration_levels":list(PHYSICAL_LEVELS),"anchor_level":4,
         "empirical_resampling":"18 physical records per level with replacement; paired full N4 stage vector per replicate",
         "simulated_node_counts":list(levels),"replicates_per_level":replicates,"random_seed":seed,
         "fixed_workload":WORKLOAD,
         "assumptions":["no additional reasoning workers or inference demand",
                        "shared database fixed at 30000 vectors",
                        "one decision-recording request per event",
                        "no additional WAN loss, queue, capacity change or network partition",
                        "outputs beyond 4 physical nodes are conditional software estimates"]}
    return output,fit


def summarize(output: Sequence[dict]) -> List[dict]:
    levels=sorted({r["node_count"] for r in output})
    summary=[]
    for n in levels:
        subset=[r for r in output if r["node_count"]==n]
        totals=[r["T_decision_ms"] for r in subset]
        summary.append({"node_count":n, "domain":"REFERENCE_MODEL_ESTIMATE",
                        "n":len(subset),"mean_ms":_mean(totals),
                        "std_ms":statistics.stdev(totals) if len(totals)>1 else 0.,
                        **{f"mean_{s}":_mean([r[s] for r in subset]) for s in STAGES}})
    return summary


def descriptive_fits(rows: Sequence[dict]) -> dict:
    """Fits summarize NEW generated level means; never generate stage records."""
    x=[float(r["node_count"]) for r in rows]
    y=[r["mean_ms"] for r in rows]
    xbar,ybar=_mean(x),_mean(y)
    slope=sum((xx-xbar)*(yy-ybar) for xx,yy in zip(x,y))/sum((xx-xbar)**2 for xx in x)
    intercept=ybar-slope*xbar
    fit={"linear":{"intercept_ms":intercept,"slope_ms_per_node":slope,
                   "r_squared_level_means":_r_squared(y,[intercept+slope*xx for xx in x])}}
    try:
        from scipy.optimize import curve_fit
        def curve(n,t0,alpha,beta):return t0+alpha*n**beta
        import numpy as np
        params,cov=curve_fit(curve,np.asarray(x),np.asarray(y),
                             p0=[min(y)*0.8,500.,0.5],bounds=([0.,0.,0.],[1e7,1e7,5.]),maxfev=50000)
        fit["power_law"]={"T0_ms":float(params[0]),"alpha_ms":float(params[1]),
                          "beta":float(params[2]),
                          "r_squared_level_means":_r_squared(y,[float(curve(xx,*params)) for xx in x]),
                          "warning":"descriptive fit to five model-generated means; no extrapolation guarantee"}
    except (ImportError,RuntimeError,ValueError,OverflowError) as ex:
        fit["power_law"]={"status":"unavailable","reason":str(ex)}
    return fit


def _r_squared(y,yp):
    den=sum((a-_mean(y))**2 for a in y)
    return 1-sum((a-b)**2 for a,b in zip(y,yp))/den if den else float('nan')


def _write_csv(path:Path,records:Sequence[dict]):
    if not records:raise ValueError('empty output')
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(records)


def run(hardware:Path,out:Path,seed:int=SEED,replicates:int=REPLICATES)->dict:
    grouped=load_hardware(hardware)
    val,metrics=strict_leave_one_level_out(grouped)
    rows,model=simulate_scaleout(grouped,seed,replicates)
    levels=summarize(rows)
    fit=descriptive_fits(levels)
    out.mkdir(parents=True,exist_ok=True)
    files={"strict_lolo":"strict_lolo_validation.csv",
           "reference_model_runs":"stage_scaleout_reference_model.csv",
           "model_level_means":"stage_scaleout_level_means.csv"}
    for key,rs in (("strict_lolo",val),("reference_model_runs",rows),("model_level_means",levels)):
        _write_csv(out/files[key],rs)
    metadata={"schema":"adam.scalability.stage_model.v2", "input_path":str(hardware),
              "input_sha256":sha256(hardware),"calibration":"only 72 physical Pi measurements, N=1..4",
              "historical_archived_simulation_used_for_calibration":False,
              "strict_lolo_validation":metrics,"reference_model":model,
              "descriptive_fits":fit,
              "outputs":{key:{"file":name,"sha256":sha256(out/name)} for key,name in files.items()},
              "provenance_warning":"New reference-model output; not May 2025 measurements or reproduction of archived simulated outputs."}
    (out/'scalability_manifest.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    return metadata


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hardware',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--replicates',type=int,default=REPLICATES)
    args=p.parse_args()
    report=run(args.hardware,args.out,args.seed,args.replicates)
    print('STRICT LOO MAPE %:',round(report['strict_lolo_validation']['mape_pct'],4))
    print('STRICT LOO BIAS %:',round(report['strict_lolo_validation']['signed_mean_percentage_error_pct'],4))
    for name,details in report['outputs'].items():print(name,details['file'],details['sha256'])
    return 0

if __name__=='__main__':raise SystemExit(main())
