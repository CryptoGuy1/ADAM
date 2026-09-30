"""Phase 3 correctness, provenance, leakage, and reproducibility checks."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from experiments.scalability_stage_model import (
    NODE_DEPENDENT, PHYSICAL_LEVELS, REPLICATES, SCALEOUT_LEVELS, STAGES,
    load_hardware, run, sha256, simulate_scaleout, stage_means,
    strict_leave_one_level_out, summarize,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'data/scalability_hardware_v14.csv'


def test_archived_hardware_source_and_additivity():
    manifest=json.loads(INPUT.with_suffix('.manifest.json').read_text())
    assert manifest['source_sheet']=='08_Scalability_Log'
    assert manifest['rows']==72
    assert manifest['hardware_csv_sha256']==sha256(INPUT)
    candidate=ROOT/'data'/'ADAM_Dataset_Master_v14_reconciled.xlsx'
    assert candidate.is_file()
    assert manifest['source_sha256']==sha256(candidate)
    data=load_hardware(INPUT)
    assert {n:len(rows) for n,rows in data.items()}=={n:18 for n in PHYSICAL_LEVELS}
    for rows in data.values():
        for row in rows:
            assert abs(sum(row['stages'].values())-row['reported_total'])<1e-8


def test_reject_simulated_or_malformed_data(tmp_path):
    with INPUT.open(newline='') as f:
        original=list(csv.DictReader(f))
        fields=list(original[0])
    def attempt(field,value):
        records=[dict(r) for r in original]
        records[0][field]=str(value)
        path=tmp_path/f'invalid_{field}.csv'
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
        with pytest.raises(ValueError):load_hardware(path)
    attempt('Run_Mode','PYTHON_SIMULATION')
    attempt('Simulated_Node_Count',1)
    attempt('Concurrent_Events',5)
    attempt('Num_Sensors',9)
    attempt('Database_Size_Vectors',31000)
    attempt('T_decision_ms',1)
    attempt('T_reason_ms',-1)
    attempt('Run_ID',original[1]['Run_ID'])


def test_strict_heldout_level_exclusion():
    original=load_hardware(INPUT)
    baseline,_=strict_leave_one_level_out(original)
    for held in PHYSICAL_LEVELS:
        altered={n:[{**r, 'stages':dict(r['stages'])} for r in rows]
                 for n,rows in original.items()}
        for row in altered[held]:
            row['stages']['T_reason_ms']+=2000.
            row['reported_total']+=2000.
        validation,_=strict_leave_one_level_out(altered)
        b=next(x for x in baseline if x['held_out_N']==held)
        a=next(x for x in validation if x['held_out_N']==held)
        assert a['predicted_mean_ms']==pytest.approx(b['predicted_mean_ms'])
        assert a['hardware_mean_ms']==pytest.approx(b['hardware_mean_ms']+2000)
        assert str(held) not in a['train_levels'].split(',')
        assert a['anchor_N'] != held
        assert a['sum_absolute_stage_error_ms'] >= a['absolute_error_ms']-1e-8
        if held==1: assert a['predicted_T_cross_node_ms']==0.0


def test_bootstrap_uses_paired_six_stage_vectors_and_fixed_services():
    original=load_hardware(INPUT)
    output,manifest=simulate_scaleout(original)
    assert len(output)==5*REPLICATES
    assert Counter(r['node_count'] for r in output)=={n:18 for n in SCALEOUT_LEVELS}
    for replicate in range(1,REPLICATES+1):
        cell={r['node_count']:r for r in output if r['replicate']==replicate}
        anchor_id=cell[4]['reference_hardware_run_id']
        anchor=next(r for r in original[4] if r['run_id']==anchor_id)
        for n,entry in cell.items():
            assert entry['reference_hardware_run_id']==anchor_id
            assert entry['physical_pi_count']==4
            assert entry['simulated_node_count']==n-4
            assert entry['database_vectors']==30000
            assert entry['concurrent_events']==4
            assert entry['logical_sensor_streams']==8
            assert entry['logical_workers']==4
            assert entry['run_mode']=='REFERENCE_MODEL_ESTIMATE'
            assert entry['T_decision_ms']==pytest.approx(sum(entry[s] for s in STAGES))
            for s in STAGES:
                if s not in NODE_DEPENDENT:
                    assert entry[s]==anchor['stages'][s]
            for s in NODE_DEPENDENT:
                assert entry[s]>=0.
        for s in NODE_DEPENDENT:
            increments=[(cell[n][s]-cell[4][s])/(n-4) for n in (6,8,12,16)]
            assert max(increments)-min(increments)<1e-8
    assert manifest['anchor_level']==4


def test_seed_and_run_manifest_are_repeatable(tmp_path):
    data=load_hardware(INPUT)
    a,_=simulate_scaleout(data,seed=42)
    b,_=simulate_scaleout(data,seed=42)
    c,_=simulate_scaleout(data,seed=43)
    assert a==b
    assert a!=c
    info=run(INPUT,tmp_path)
    assert info['historical_archived_simulation_used_for_calibration'] is False
    assert info['strict_lolo_validation']['level_count']==4
    assert info['strict_lolo_validation']['mape_pct']>0.
    assert info['input_sha256']==sha256(INPUT)
    for ent in info['outputs'].values():
        assert sha256(tmp_path/ent['file'])==ent['sha256']
    assert len(summarize(a))==5
