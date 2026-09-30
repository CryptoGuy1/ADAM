import json
from pathlib import Path

import pandas as pd

from analysis.revised_statistics import degraded_family, main_benchmark_family, substitution_family

ROOT = Path(__file__).resolve().parent.parent
WB = ROOT / "data" / "ADAM_Dataset_Master_v14_reconciled.xlsx"
CLAIMS = json.loads((ROOT / "data" / "manuscript_result_claims.json").read_text())


def _frames():
    tr = pd.read_excel(WB, sheet_name="03_D1_Trial_Results")
    swap = pd.read_excel(WB, sheet_name="15_Swap_Study_Trials", header=1)
    deg = pd.read_excel(WB, sheet_name="14_Degraded_Conditions", header=1)
    return tr, swap, deg


def test_revised_family_sizes_and_main_holm():
    tr, swap, deg = _frames()
    main = main_benchmark_family(tr)
    sub = substitution_family(tr, swap)
    degraded = degraded_family(deg)
    assert (len(main), len(sub), len(degraded)) == (10, 5, 20)
    by = {r.comparison: r for r in main}
    for key in (
        "ADAM_vs_Static", "ADAM_vs_RF_Raw", "ADAM_vs_RF_Fused",
        "ADAM_vs_GBM_Fused", "ADAM_vs_SingleAgent", "ADAM_vs_NoAgg",
        "ADAM_vs_NoLLM", "ADAM_vs_NoWeaviate",
    ):
        assert by[key].p_holm == CLAIMS["tables"]["WILCOXON"][key][1]
    assert by["ADAM_vs_Cloud"].p_holm == CLAIMS["tables"]["WILCOXON"]["ADAM_vs_Cloud"][1]
    assert by["ADAM_vs_NoBlockchain"].p_holm == CLAIMS["tables"]["WILCOXON"]["ADAM_vs_NoBlockchain"][1]


def test_substitution_and_degraded_adjustments_match_manuscript():
    tr, swap, deg = _frames()
    sub = {r.comparison: r for r in substitution_family(tr, swap)}
    assert sub["Static threshold"].p_holm == CLAIMS["checks"]["swap Static p_Holm"]
    assert sub["LLM (Gemma 3 1B)"].p_holm == CLAIMS["checks"]["swap LLM p_Holm"]
    for key in ("Logistic regression", "Random forest", "Gradient boosting"):
        assert sub[key].p_holm == CLAIMS["checks"]["swap fitted p_Holm"]
    assert all(r.p_exact == CLAIMS["checks"]["degraded exact common"] for r in degraded_family(deg))
    assert all(r.p_holm == CLAIMS["checks"]["degraded Holm common"] for r in degraded_family(deg))
