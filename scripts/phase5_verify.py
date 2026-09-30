#!/usr/bin/env python3
"""Verify historical Section 4.5 and resource claims, preserving source hashes.

Run against a source workbook with --workbook or against separately exported CSVs
with --input-dir. CSV mode requires an explicit source type; never infer that a
file containing historical-looking numbers is historical evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

V14_SHA256 = "aa21f5c0272d6fcee27ef27b0a46d2af0fee09f82f1b36bd402b4084fc73629d"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from experiments.phase5_provenance import SOURCE_SHEETS, EvidenceError, run, sha256_file
from experiments.phase5_claim_checks import check_claims


def export_read_only(workbook: Path, target: Path) -> str:
    """Export exactly five underlying sheets without modifying the original XLSX.

    The project already depends on pandas for its original workbook readers.
    This export is a utility for user-side execution; it is not a new experiment.
    """
    import pandas as pd
    if not workbook.is_file():
        raise EvidenceError(f"Missing workbook {workbook}")
    target.mkdir(parents=True,exist_ok=True)
    for key,sheet in SOURCE_SHEETS.items():
        frame = pd.read_excel(workbook,sheet_name=sheet,header=0 if key=="resources" else 2)
        # Do not normalize measured content or silently drop incomplete event rows.
        if frame.empty or frame.columns.duplicated().any():
            raise EvidenceError(f"Invalid or duplicate headers in {sheet}")
        frame.to_csv(target/(sheet+".csv"),index=False,float_format="%.15g")
    digest=sha256_file(workbook)
    manifest={"workbook_sha256":digest,
              "csv_sha256":{k:sha256_file(target/(sheet+".csv")) for k,sheet in SOURCE_SHEETS.items()},
              "sheet_names":SOURCE_SHEETS,
              "meaning":"Read-only V14 workbook export; not a physical experiment or independently attested record"}
    (target/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return digest


def plot_security(metrics: dict, dest: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    a=metrics["model_failure"]["fallback_latency_ms"]
    vals=np.asarray(a["values_sorted"],dtype=float)
    cloud=metrics["external_inference_traffic"]["Cloud-Only"]
    local=metrics["external_inference_traffic"]["ADAM_LLM"]
    fig,(ax, bx)=plt.subplots(1,2,figsize=(12,4.3),layout="constrained")
    ax.step(vals, np.arange(1,len(vals)+1)/len(vals), where="post", linewidth=1.8)
    ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),"o",markersize=4)
    ax.axvline(a["median"],linestyle="--",linewidth=1)
    ax.axvline(a["p95_linear"],linestyle=":",linewidth=1)
    ax.text(.97,.06,f'Median {a["median"]:.1f} ms; P95 {a["p95_linear"]:.1f} ms',
            transform=ax.transAxes,ha="right",va="bottom",fontsize=9)
    ax.set(xlabel="Fallback activation latency (ms)",ylabel="Empirical cumulative fraction",ylim=(0,1.04),title=f'(a) Fallback-triggered events (n={len(vals)})')
    ax.grid(alpha=.2)
    by=[local["kb_per_window"],cloud["kb_per_window"]]
    for i,x in enumerate(by):
        xs=np.linspace(i-.1,i+.1,len(x))
        bx.scatter(xs,x,s=30)
        bx.hlines(np.mean(x),i-.21,i+.21,linewidth=1.6)
    bx.set(xticks=[0,1],xticklabels=["ADAM_LLM","Cloud-Only"],ylabel="External inference KB per 30-min window",
           title=f'(b) Instrumented windows ({len(by[0])} local; {len(by[1])} cloud)')
    bx.grid(alpha=.2,axis="y")
    if metrics.get("_synthetic_fixture"):
        fig.suptitle('SYNTHETIC FIXTURE — NOT MANUSCRIPT DATA',fontsize=11)
    dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/"figure10_security_evidence.png",dpi=200)
    fig.savefig(dest/"figure10_security_evidence.pdf")
    plt.close(fig)


def plot_resources(stats: dict, dest: Path, synthetic: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels=["Idle","Monitoring","Crew active"]
    states=["idle","monitoring","crew_active"]
    cpu=[stats["by_state"][s]["mean_window_peak_cpu_pct"] for s in states]
    mem=[stats["by_state"][s]["mean_node_memory_mb"] for s in states]
    for series,y_label,name in [(cpu,"Mean of sampled 60-s window-peak CPU (%)","figure8a_cpu_window_peaks"),
                                (mem,"Mean measured node-level memory (MB)","figure8b_memory_footprint")]:
        fig,ax=plt.subplots(figsize=(6.5,4.2),layout="constrained")
        ax.bar(labels,series,width=.60)
        for i,v in enumerate(series):
            ax.text(i,v+max(series)*.02,f"{v:,.1f}",ha="center")
        if series is cpu:
            ax.axhline(80,linestyle="--",linewidth=1)
            ax.text(.98,.98,"C2: 80% criterion applies to NON-INFERENCE windows",
                    ha="right",va="top",fontsize=8,transform=ax.transAxes)
        ax.set_ylim(0,max(series)*1.20)
        ax.set_ylabel(y_label)
        ax.set_title("Source-recorded resource summary" if not synthetic else "SYNTHETIC FIXTURE — NOT MANUSCRIPT")
        dest.mkdir(parents=True,exist_ok=True)
        fig.savefig(dest/(name+".png"),dpi=200)
        fig.savefig(dest/(name+".pdf"))
        plt.close(fig)


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--workbook",type=Path)
    group.add_argument("--input-dir",type=Path)
    p.add_argument("--source-type",choices=["historical_workbook_export","synthetic_fixture","new_reference_run"])
    p.add_argument("--workbook-sha256",help="required for historical CSV export mode")
    p.add_argument("--expected-sha256",default=V14_SHA256,help="approved master-workbook fingerprint; defaults to V14 reconciled")
    p.add_argument("--out",type=Path,default=Path("results/phase5"))
    p.add_argument("--no-strict-historical-counts",action="store_true")
    p.add_argument("--plot",action="store_true")
    p.add_argument("--check-manuscript",action="store_true",help="check numeric claims without changing recorded values")
    args=p.parse_args()
    if args.no_strict_historical_counts and args.source_type in (None,"historical_workbook_export") and args.workbook:
        p.error("Historical workbook mode requires the historical cohort sizes")
    if args.workbook:
        if args.source_type not in (None,"historical_workbook_export"):
            p.error("--workbook is the archived historical data source, not a synthetic fixture")
        input_dir=args.out/"source_csv"
        actual = sha256_file(args.workbook)
        if actual.lower() != args.expected_sha256.lower():
            raise EvidenceError(f"Workbook SHA-256 {actual} differs from approved source "
                                f"{args.expected_sha256}. Review the new workbook before explicitly "
                                "passing its confirmed digest through --expected-sha256.")
        digest=export_read_only(args.workbook,input_dir)
        source_type="historical_workbook_export"
    else:
        if not args.source_type:
            p.error("--source-type is required for CSV inputs")
        input_dir=args.input_dir
        digest=args.workbook_sha256
        if args.source_type=="historical_workbook_export" and (
            not digest or digest.lower()!=args.expected_sha256.lower()
        ):
            raise EvidenceError("Historical CSV input must carry the approved master-workbook SHA-256")
        source_type=args.source_type
    if args.no_strict_historical_counts and source_type=="historical_workbook_export":
        p.error("Historical verification cannot disable cohort-count checks")
    data=run(input_dir,args.out,source_type=source_type,workbook_sha256=digest,
             strict_historical_counts=not args.no_strict_historical_counts)
    print(f"Source: {source_type}; files: {len(data['provenance']['csv_sha256'])}")
    print(f"Injection: {data['security']['injection']['detected']}/{data['security']['injection']['n']} detected")
    print(f"Fallback: {data['security']['model_failure']['fallback_triggered_n']} activated; {data['security']['model_failure']['continued_given_fallback']} continued")
    print(f"Sampled resource windows: {data['resources']['n_recorded_windows']}")
    print('NOTE: This calculation does not reconstruct missing sensor/vote/ledger-level traces.')
    if args.check_manuscript:
        checks=check_claims(data)
        (args.out/"phase5_manuscript_claim_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
        failures=[r for r in checks if not r["pass"]]
        print(f"Manuscript claim checks: {len(checks)-len(failures)}/{len(checks)} passed")
        if failures:
            for row in failures:
                print(f"MISMATCH {row['metric']}: computed={row['computed']} manuscript={row['manuscript']}")
            return 2
    if args.plot:
        data['security']['_synthetic_fixture']=source_type!='historical_workbook_export'
        plot_security(data["security"],args.out)
        plot_resources(data["resources"],args.out,source_type!="historical_workbook_export")
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except EvidenceError as exc: raise SystemExit(f"EVIDENCE CHECK FAILED: {exc}")
