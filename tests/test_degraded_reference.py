"""Phase 4 gates: no historical-data substitution or clean-response reuse."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.degraded_harness import CONDITIONS, apply_condition, condition_seed, run
from experiments.run_degraded_reference import (
    SYSTEMS, TracedBackend, condition_events, file_sha, pooled_variances,
    run_reference, validate_source,
)


def fixture_input(tmp_path: Path, trials=2, n=6):
    rows=[]
    rng=np.random.default_rng(123)
    for trial in range(1,trials+1):
        for i in range(n):
            ref=700+160*i+20*trial
            for node in range(1,5):
                ppm=ref+int(node*13)+float(rng.normal(0,30))
                rows.append(dict(trial=trial,event_id=f'T{trial}-{i}',node_id=f'N{node}',
                    trigger_node='N1',timestamp=float(i+1),raw_ppm=ppm,reference_ppm=ref))
    p=tmp_path/'four_node.csv'
    pd.DataFrame(rows).to_csv(p,index=False)
    sidecar=p.with_suffix('.manifest.json')
    sidecar.write_text(json.dumps({'source':'synthetic_fixture','sha256':file_sha(p)}))
    return p


def test_missing_or_bad_source_manifest_fails_closed(tmp_path):
    p=fixture_input(tmp_path)
    with pytest.raises(ValueError,match='explicit'):
        validate_source(p)
    m=p.with_suffix('.manifest.json')
    m.unlink()
    with pytest.raises(ValueError,match='manifest required'):
        validate_source(p,allow_fixture=True)
    m.write_text(json.dumps({'source':'physical_per_node','sha256':'bad'}))
    with pytest.raises(ValueError,match='SHA-256'):
        validate_source(p)
    m.write_text(json.dumps({'source':'deposit','sha256':file_sha(p)}))
    with pytest.raises(ValueError,match='source must'):
        validate_source(p)


def test_requires_every_node_and_unique_event_timestamps(tmp_path):
    p=fixture_input(tmp_path)
    df=pd.read_csv(p)
    df=df[df.node_id!='N4']
    df.to_csv(p,index=False)
    p.with_suffix('.manifest.json').write_text(json.dumps({'source':'synthetic_fixture','sha256':file_sha(p)}))
    with pytest.raises(ValueError,match='N1--N4'):
        validate_source(p,allow_fixture=True)
    p=fixture_input(tmp_path)
    df=pd.read_csv(p);df.loc[df.event_id=='T1-1','timestamp']=1.0
    df.to_csv(p,index=False)
    p.with_suffix('.manifest.json').write_text(json.dumps({'source':'synthetic_fixture','sha256':file_sha(p)}))
    with pytest.raises(ValueError,match='unique timestamps'):
        validate_source(p,allow_fixture=True)


def test_sigma_is_within_trial_not_pooled_and_strong_drift_formula(tmp_path):
    p=fixture_input(tmp_path)
    df,_=validate_source(p,allow_fixture=True)
    stream,_=run(df,str(tmp_path/'perturb'))
    cell=stream[(stream.condition=='strong_drift')&(stream.trial==1)&(stream.node_id=='N1')].sort_values('timestamp')
    sig=df[(df.trial==1)&(df.node_id=='N1')].raw_ppm.std(ddof=1)
    n=len(cell)
    assert n>=3
    for j,r in enumerate(cell.itertuples()):
        frac=j/(n-1)
        assert r.perturbed_ppm == pytest.approx((1+.35*frac)*r.raw_ppm+2*sig*frac)
    original=df[(df.trial==1)&(df.node_id=='N1')]
    assert np.allclose(stream[(stream.condition=='clean')&(stream.trial==1)&(stream.node_id=='N1')].raw_ppm,
                       original.raw_ppm)


def test_four_node_dropout_and_reference_are_never_fabricated(tmp_path):
    p=fixture_input(tmp_path)
    df,_=validate_source(p,allow_fixture=True)
    stream,_=run(df,str(tmp_path/'perturb'))
    var=pooled_variances(df)
    events=condition_events(stream,var,'one_node_dropout',1)
    assert len(events)==6
    assert any(len(e.readings)==3 for _,e in events)
    for _,e in events:
        assert e.label==int(e.reference_ppm>=1000)
        assert all(r.node_id in ('N1','N2','N3','N4') for r in e.readings)
    control={str(r.event_id):r.reference_ppm for r in df[df.trial==1].itertuples()}
    assert all(e.reference_ppm==control[eid] for eid,e in events)


def test_full_fixture_run_causal_pairing_fresh_condition_calls(tmp_path):
    p=fixture_input(tmp_path,trials=3,n=7)
    out=tmp_path/'run'
    manifest=run_reference(p,out,allow_fixture=True,reasoner='fixture')
    rows=[json.loads(x) for x in (out/'reference_event_traces.jsonl').read_text().splitlines()]
    calls=[json.loads(x) for x in (out/'model_call_traces.jsonl').read_text().splitlines()]
    assert len(rows)==3*7*5*6
    assert len(calls)==3*7*5*2
    assert manifest['input_source']=='synthetic_fixture'
    assert not manifest['historical_reproduction']
    assert manifest['semantic_memory']=='IN_MEMORY_CONCENTRATION_NEAREST_NOT_WEAVIATE'
    assert len({x['call_id'] for x in calls})==len(calls)
    assert {r['system'] for r in rows}==set(SYSTEMS)
    index={(r['condition'],r['trial'],r['event_id'],r['system']):r for r in rows}
    for r in rows:
        assert r['label']==int(r['reference_ppm']>=1000)
        assert all('reference_ppm' not in x for x in r['node_readings'])
        assert (r['prediction_resolved'] or r['prediction']=='UNRESOLVED')
        if r['system'] in ('adam_llm','adam_gbm'):
            assert len(r['votes'])==3
            assert r['initial_classification'] in ('ANOMALY','NORMAL')
            assert r['quorum_required']==2
            if r['prediction']=='UNRESOLVED':assert r['action_released'] is False
        if r['system']=='single_agent':assert not r['retrieved_ids']
    for c in calls:
        r=index[(c['condition'],c['trial'],c['event_id'],c['system'])]
        assert r['llm_call_id']==c['call_id']
        assert c['trace']['prompt_sha256']==hashlib.sha256(c['artifacts']['user_prompt'].encode()).hexdigest()
        assert c['trace']['response_sha256']==hashlib.sha256(c['artifacts']['raw_response'].encode()).hexdigest()
        if c['system']=='single_agent':assert c['artifacts']['retrieved_records']==[]
        assert all(float(x['timestamp'])<float(r['node_readings'][0]['timestamp']) for x in c['artifacts']['retrieved_records'])
        assert 'reference_ppm' not in c['artifacts']['user_prompt']
    # Verify drift/noise produce distinct inputs and independent calls for the same event.
    sample=index[('clean',1,'T1-6','adam_llm')]
    altered=index[('strong_drift',1,'T1-6','adam_llm')]
    assert sample['input_sha256'] != altered['input_sha256']
    assert sample['model_call']['prompt_sha256'] != altered['model_call']['prompt_sha256']
    assert manifest['output_sha256']['events']==file_sha(out/'reference_event_traces.jsonl')
    assert manifest['output_sha256']['model_calls']==file_sha(out/'model_call_traces.jsonl')


def test_empty_fixture_or_missing_ollama_never_claimed_historical(tmp_path):
    p=fixture_input(tmp_path)
    with pytest.raises(ValueError,match='test double'):
        run_reference(p,tmp_path/'no',reasoner='fixture')
