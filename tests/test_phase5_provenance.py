"""Phase 5 verifies meanings, denominators, and provenance—not synthetic scores."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from experiments.phase5_provenance import (
    SOURCE_SHEETS, EvidenceError, f1_and_far, quantile_linear,
    run, verify_resources, verify_security,
)

COHORTS = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "manuscript_result_claims.json").read_text()
)["cohorts"]


def fixture():
    types=[name for name,n in COHORTS['attack_patterns'].items() for _ in range(n)]
    inj=[{'Attack_Type':p,'ADAM_Detected_Attack':'No' if i in (7,18,29) else 'Yes',
          'ADAM_Prediction':'anomaly' if i%2 else 'normal',
          'Ground_Truth':'anomaly' if i%3 else 'normal'} for i,p in enumerate(types)]
    fail=[{'Fallback_Triggered':'Yes' if i<COHORTS['fallback_triggered'] else 'No',
           'Crew_Continued':'No' if i==3 else 'Yes',
           'Fallback_Latency_ms':str(40+i) if i<COHORTS['fallback_triggered'] else '',
           'Prediction':'anomaly' if i%2 else 'normal',
           'Ground_Truth':'anomaly' if i%3 else 'normal',
           'Degraded_Mode':'True' if i<COHORTS['fallback_triggered'] else 'False'} for i in range(COHORTS['failure_total'])]
    levels=[int(level) for level,n in COHORTS['poisoning_levels'].items() for _ in range(n)]
    poi=[{'Num_Poisoned_Entries':str(k),'ADAM_Prediction':'anomaly' if i%2 else 'normal',
          'Ground_Truth':'anomaly' if i%3 else 'normal',
          'Retrieval_Affected':'Yes' if i in (12,20,29) else 'No'} for i,k in enumerate(levels)]
    eg=[{'System':'ADAM_LLM','Total_Bytes_External':'0','External_API_Calls':'0'} for _ in range(COHORTS['egress_windows']['ADAM_LLM'])]
    eg += [{'System':'Cloud-Only','Total_Bytes_External':str(112640+i*1024),
            'External_API_Calls':str(19+i%2)} for i in range(COHORTS['egress_windows']['Cloud-Only'])]
    rr=[]
    for state,n in COHORTS['resource_states'].items():
        for j in range(n):
            a,b,c=(0,5000,1000) if state=='idle' else (15000,8000,2000)
            rr.append({'Timestamp':str(j),'State':state,'CPU_Peak_%':'24.5',
                       'RAM_MB':'2500','LLM_Bytes':str(a),'Weaviate_Bytes':str(b),
                       'Blockchain_Bytes':str(c),'Total_Bandwidth_Bytes':str(a+b+c)})
    return inj,fail,poi,eg,rr


def test_security_cohorts_are_separate_and_denominators_are_actual():
    result=verify_security(*fixture()[:4])
    assert result['injection']['detected']==COHORTS['injection_total']-3
    assert result['injection']['n']==COHORTS['injection_total']
    assert result['model_failure']['fallback_triggered_n']==COHORTS['fallback_triggered']
    assert result['model_failure']['continued_given_fallback']==COHORTS['fallback_triggered']-1
    assert result['model_failure']['crew_continued_all']==COHORTS['failure_total']-1
    assert result['poisoning']['by_level']['20']['n']==COHORTS['poisoning_levels']['20']
    assert result['external_inference_traffic']['ADAM_LLM']['mean_kb_per_window']==0
    assert result['external_inference_traffic']['Cloud-Only']['n_windows']==COHORTS['egress_windows']['Cloud-Only']


def test_security_refuses_incorrect_denominator():
    inj,fb,p,e,_=fixture()
    with pytest.raises(EvidenceError,match='Historical injection'):
        verify_security(inj[:-1],fb,p,e)


def test_security_refuses_unresolved_class_or_unknown_system():
    inj,fb,p,e,_=fixture()
    inj[0]['ADAM_Prediction']='UNRESOLVED'
    with pytest.raises(EvidenceError,match='unresolved class'):
        verify_security(inj,fb,p,e)
    inj,fb,p,e,_=fixture()
    e[0]['System']='unknown'
    with pytest.raises(EvidenceError,match='Unknown egress'):
        verify_security(inj,fb,p,e)


def test_security_requires_degraded_mode_marker_on_fallback():
    inj,fb,p,e,_=fixture()
    fb[0]['Degraded_Mode']='False'
    with pytest.raises(EvidenceError,match='degraded-mode'):
        verify_security(inj,fb,p,e)


def test_resource_component_sums_and_noninference_statistic():
    r=verify_resources(fixture()[4]); assert r['n_recorded_windows']==sum(COHORTS['resource_states'].values())
    assert r['mean_non_inference_window_peak_cpu_pct']==24.5
    assert r['by_state']['crew_active']['n_windows']==COHORTS['resource_states']['crew_active']
    assert r['by_state']['crew_active']['mean_accounted_kb_per_60s_per_node']==25000/1024
    assert r['source_timestamp_span_h']>0


def test_resource_fails_on_mismatched_accounted_total():
    rr=fixture()[4]
    rr[0]['Total_Bandwidth_Bytes']='0'
    with pytest.raises(EvidenceError,match='disagree'):
        verify_resources(rr)


def test_csv_export_requires_source_declaration_and_workbook_hash(tmp_path):
    inj,fail,poison,eg,rs=fixture()
    pairs=zip(SOURCE_SHEETS.values(),[inj,fail,poison,eg,rs])
    for name,rows in pairs:
        with (tmp_path/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    with pytest.raises(EvidenceError,match='requires the original workbook SHA'):
        run(tmp_path,tmp_path/'out',source_type='historical_workbook_export')
    data=run(tmp_path,tmp_path/'out',source_type='synthetic_fixture')
    assert data['provenance']['source_type']=='synthetic_fixture'
    assert data['provenance']['independent_hardware_replay'] is False
    assert (tmp_path/'out'/'phase5_metrics.json').is_file()
    assert len(data['provenance']['csv_sha256'])==5


def test_linear_percentile_uses_all_event_values():
    assert quantile_linear([3,1,5,2,4],.95)==pytest.approx(4.8)


def test_historical_export_requires_matching_manifest(tmp_path):
    inj,fail,poison,eg,rs=fixture()
    for name,rows in zip(SOURCE_SHEETS.values(),[inj,fail,poison,eg,rs]):
        with (tmp_path/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    with pytest.raises(EvidenceError,match='workbook-export manifest'):
        run(tmp_path,tmp_path/'out',source_type='historical_workbook_export',workbook_sha256='test-hash')


def test_historical_export_rejects_changed_csv_after_manifest(tmp_path):
    from experiments.phase5_provenance import sha256_file
    inj,fail,poison,eg,rs=fixture()
    src={}
    for key,(name,rows) in enumerate(zip(SOURCE_SHEETS.values(),[inj,fail,poison,eg,rs])):
        p=tmp_path/(name+'.csv')
        with p.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        src[list(SOURCE_SHEETS)[key]]=sha256_file(p)
    (tmp_path/'source_manifest.json').write_text(json.dumps({
        'workbook_sha256':'approved-test-digest', 'csv_sha256':src
    }))
    path=tmp_path/(SOURCE_SHEETS['injection']+'.csv')
    path.write_text(path.read_text().replace('replay','changed_attack',1))
    with pytest.raises(EvidenceError,match='Export manifest digest mismatch'):
        run(tmp_path,tmp_path/'out',source_type='historical_workbook_export',workbook_sha256='approved-test-digest')


def test_manuscript_claim_check_cannot_pass_synthetic_metrics(tmp_path):
    from experiments.phase5_claim_checks import check_claims
    inj,fail,poison,eg,rs=fixture()
    payload={'security':verify_security(inj,fail,poison,eg),'resources':verify_resources(rs)}
    checks=check_claims(payload)
    assert len(checks)>20
    assert any(not row['pass'] for row in checks)
    assert any(row['metric']=='fallback.continued' and not row['pass'] for row in checks)
