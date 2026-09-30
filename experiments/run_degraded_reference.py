#!/usr/bin/env python3
"""New, provenance-gated degradation study; NEVER a historical D1 reproduction.

Input is a complete, concurrent *long-form* four-node acquisition CSV plus a
sidecar provenance manifest. The archived workbook is not an acceptable input.
All six study configurations see condition-specific inputs. Fitted classifiers
are trained only on clean, non-held-out trials. Each LLM arm makes fresh calls
for every condition/event; no clean-condition response cache is used.

`--reasoner fixture` exercises the pipeline with a deterministic TEST double;
it is not an LLM experiment. `--reasoner ollama` performs real local inference.
The offline InMemoryStore is a concentration-nearest-neighbor approximation,
NOT the historical Weaviate embedding retrieval. Each condition/system/trial
starts with a new empty memory and processes its own records chronologically.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from adam.config import ADAMConfig, RF_PARAMS, SEED, THRESHOLD_PPM
from adam.crew import ADAMNode
from adam.governance.chain import InMemoryChainClient, LocalValidator
from adam.llm.client import InferenceResult, OllamaClient
from adam.llm.prompt import build_user_prompt, build_system_prompt
from adam.mechanisms import FusionResult, fuse_readings
from adam.memory.store import InMemoryStore
from adam.schemas import CrewEvent, DecisionObject, LabeledEvent, SensorReading
from baselines.systems import BASELINE_WINDOW, fused_context_matrix
from experiments.decision_agent_backends import decision_feature_vector, decision_from_probability, make_backend
from experiments.degraded_harness import CONDITIONS, run as apply_degradation, validate_input

SCHEMA = "adam.degraded_reference.v1"
SYSTEMS = ("static_threshold", "random_forest_fused", "gradient_boosting_fused",
           "single_agent", "adam_gbm", "adam_llm")
NODES = ("N1", "N2", "N3", "N4")


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False, allow_nan=False).encode('utf-8'))


def validate_source(path: Path, *, allow_fixture: bool = False) -> tuple[pd.DataFrame, dict]:
    """Require explicit source declaration and identical bytes; reject archived D1 export."""
    sidecar = path.with_suffix('.manifest.json')
    if not sidecar.is_file():
        raise ValueError(f'provenance manifest required: {sidecar}')
    meta = json.loads(sidecar.read_text())
    source = meta.get('source')
    if source not in ('physical_per_node', 'synthetic_fixture'):
        raise ValueError('source must be physical_per_node or synthetic_fixture; primary-channel D1 is not sufficient')
    if source == 'synthetic_fixture' and not allow_fixture:
        raise ValueError('synthetic fixture requires explicit --allow-fixture')
    if meta.get('sha256') != file_sha(path):
        raise ValueError('input SHA-256 does not match provenance manifest')
    df = pd.read_csv(path, dtype={'event_id': str, 'node_id': str, 'trigger_node': str})
    validate_input(df)
    # Each event is one simultaneous 4-node observation, not four events.
    if 'trigger_node' not in df:
        raise ValueError('trigger_node is required to preserve the primary channel')
    if set(df.node_id) != set(NODES):
        raise ValueError('complete source must contain exactly N1--N4')
    event_times = df.groupby(['trial','event_id']).timestamp.nunique(dropna=False)
    node_sets = df.groupby(['trial','event_id']).node_id.agg(lambda x: set(x))
    if not (event_times == 1).all() or not all(x == set(NODES) for x in node_sets):
        raise ValueError('all input events require one shared timestamp and all four node measurements')
    if not np.isfinite(df[['raw_ppm','reference_ppm','timestamp']].astype(float).values).all():
        raise ValueError('nonfinite acquisition input')
    per_trial = df.groupby(['trial','node_id']).raw_ppm.agg(['size','std'])
    if (per_trial['size'] < 3).any() or (per_trial['std'] <= 0).any():
        raise ValueError('at least 3 varying observations are required per node per trial')
    # Numeric time cannot repeat within a trial across distinct events.
    event_idx = df[['trial','event_id','timestamp']].drop_duplicates()
    if event_idx.duplicated(['trial','timestamp']).any():
        raise ValueError('events within a trial need unique timestamps for causal retrieval')
    return df, meta


def pooled_variances(df: pd.DataFrame) -> dict[str, float]:
    out = (df.assign(residual=df.raw_ppm-df.reference_ppm)
           .groupby('node_id').residual.var(ddof=1).to_dict())
    if set(out) != set(NODES) or any(not math.isfinite(v) or v <= 0 for v in out.values()):
        raise ValueError('positive residual variance required for each clean node')
    return {str(k):float(v) for k,v in out.items()}


def condition_events(df: pd.DataFrame, variances: dict[str,float],
                     condition: str, trial: int) -> list[tuple[str,LabeledEvent]]:
    cell = df[(df.condition == condition) & (df.trial == trial)].copy()
    if cell.empty:
        raise ValueError(f'missing condition/trial: {condition}/{trial}')
    events = []
    for index, (eid, frame) in enumerate(cell.groupby('event_id', sort=False)):
        trigger_node = str(frame.trigger_node.iloc[0])
        frame = frame.assign(_trigger_first=frame.node_id.ne(trigger_node)).sort_values(['_trigger_first','node_id'])
        readings = tuple(SensorReading(str(r.node_id), float(r.timestamp),
                                       float(r.perturbed_ppm), error_variance=variances[str(r.node_id)])
                         for r in frame.itertuples(index=False) if bool(r.node_online))
        if trigger_node not in {r.node_id for r in readings}:
            raise ValueError('trigger node unavailable; no invented substitution allowed')
        expected = set(NODES)
        if not set(r.node_id for r in readings).issubset(expected) or len(readings) < 3:
            raise ValueError('unexpected missing nodes in degraded stream')
        ref = float(frame.reference_ppm.iloc[0])
        events.append((str(eid), LabeledEvent(int(trial), index, float(frame.timestamp.iloc[0]),
                              readings, int(ref>=THRESHOLD_PPM), ref)))
    events.sort(key=lambda pair: (pair[1].timestamp, pair[0]))
    # Reindex chronological order so the temporal baseline cannot view future events.
    from dataclasses import replace
    return [(eid, replace(event,event_index=i)) for i,(eid,event) in enumerate(events)]


class RecordingMemory(InMemoryStore):
    def __init__(self):
        super().__init__()
        self.last_retrieved: list[dict] = []
    def retrieve(self, fused_ppm, k=5, cutoff_timestamp=None):
        records = super().retrieve(fused_ppm,k,cutoff_timestamp)
        self.last_retrieved = [dict(x) for x in records]
        return records


class TracedBackend:
    """Wrap a fresh local model call and save exact prompt/response digests."""
    name='ADAM_Tracing_Decision'
    def __init__(self, mode: str, fitted=None, client=None):
        self.mode, self.fitted = mode, fitted
        self.client = client if client is not None else (OllamaClient() if mode=='ollama' else None)
        self.last: dict = {}
        self.last_payload: dict = {}
        self.calls = 0

    def reason(self, *, event, fusion, node_readings, baseline_window,
               history, deadline_s, threshold_ppm):
        self.calls += 1
        prompt = build_user_prompt(trigger_ppm=event.trigger_ppm, trigger_node=event.trigger_node,
                   fused_ppm=fusion.fused_ppm, node_readings=list(node_readings),
                   baseline_window=list(baseline_window), history=list(history),
                   dispersion_ppm=fusion.dispersion_ppm, outlier_nodes=list(fusion.outliers))
        evidence = {'event_id':event.event_id, 'trigger_ppm':event.trigger_ppm,
                    'fused_ppm':fusion.fused_ppm,'dispersion_ppm':fusion.dispersion_ppm,
                    'nodes':list(node_readings),'prior_baseline':list(baseline_window),
                    'retrieved_ids':[str(x['event_id']) for x in history]}
        self.last={'call_sequence':self.calls,'evidence_sha256':canonical_sha(evidence),
                   'system_prompt_sha256':canonical_sha(build_system_prompt(threshold_ppm)),
                   'prompt_sha256':digest_bytes(prompt.encode('utf-8')),
                   'retrieved_ids':evidence['retrieved_ids'],
                   'retrieval_sha256':canonical_sha(list(history)),
                   'retrieval_backend':'in_memory_concentration_nearest',
                   'model_kind':self.mode}
        if self.mode == 'ollama':
            result=self.client.decide(user_prompt=prompt,fused_ppm=fusion.fused_ppm,
                       deadline_s=deadline_s,threshold_ppm=threshold_ppm)
        elif self.mode == 'fixture':
            # An executable test double, not a language model or scientific result.
            baseline = float(np.mean(baseline_window)) if baseline_window else event.trigger_ppm
            p = .75 if fusion.fused_ppm > max(threshold_ppm,1.30*baseline) else .25
            decision=decision_from_probability(probability_anomaly=p,
                         fused_ppm=fusion.fused_ppm,dispersion_ppm=fusion.dispersion_ppm,
                         variant='NON_LLM_TEST_DOUBLE',threshold_ppm=threshold_ppm)
            raw=json.dumps(decision.to_dict(), sort_keys=True)
            result=InferenceResult(decision=decision,latency_ms=0.,raw_output=raw)
        else:
            raise ValueError('expected fixture or ollama reasoner')
        self.last_payload = {
            'system_prompt':build_system_prompt(threshold_ppm),
            'user_prompt':prompt,
            'raw_response':result.raw_output,
            'retrieved_records':list(history),
            'decision_object':result.decision.to_dict(),
            'evidence':evidence,
            'model_name':getattr(self.client,'model', None) if self.mode == 'ollama' else 'NON_LLM_TEST_DOUBLE',
            'model_temperature':getattr(self.client,'temperature', None) if self.mode == 'ollama' else None,
            'model_max_tokens':getattr(self.client,'max_tokens', None) if self.mode == 'ollama' else None,
        }
        self.last.update({'response_sha256':digest_bytes(result.raw_output.encode('utf-8')),
                   'decision_sha256':canonical_sha(result.decision.to_dict()),
                   'repair_attempted':bool(result.repair_attempted),
                   'fallback':bool(result.fell_back)})
        return result


def make_models(train: list[LabeledEvent]):
    # Confirm label alignment against the exact returned feature ordering.
    ordered, x=fused_context_matrix(train)
    y=[e.label for e in ordered]
    models={'random_forest_fused':RandomForestClassifier(**RF_PARAMS),
            'gradient_boosting_fused':GradientBoostingClassifier(random_state=SEED)}
    for model in models.values():
        model.fit(np.asarray(x,dtype=float),np.asarray(y,dtype=int))
    gbm_backend=make_backend('adam_gbm').fit(train)
    return models,gbm_backend


def run_reference(input_path:Path,outdir:Path,*,allow_fixture=False, reasoner='fixture',
                  max_events_per_cell:Optional[int]=None) -> dict:
    if reasoner not in ('fixture','ollama'):raise ValueError('reasoner must be fixture or ollama')
    if reasoner=='fixture' and not allow_fixture:
        raise ValueError('test double must be explicitly enabled with --allow-fixture')
    clean,source=validate_source(input_path,allow_fixture=allow_fixture)
    if reasoner=='ollama' and not OllamaClient().health():
        raise RuntimeError('local Ollama runtime/model unavailable; refusing to substitute fake predictions')
    outdir.mkdir(parents=True,exist_ok=True)
    perturbdir=outdir/'perturbations'
    perturbed,perturb_manifest=apply_degradation(clean, str(perturbdir))
    variances=pooled_variances(clean)
    trials=sorted(clean.trial.unique())
    clean_by_trial={int(t):[ev for _,ev in condition_events(perturbed,variances,'clean',int(t))] for t in trials}
    if len(trials)<2:raise ValueError('leave-one-trial-out requires at least two trials')
    records=[]
    llm_calls=[]
    for held_out in trials:
        train=[e for t in trials if t!=held_out for e in clean_by_trial[int(t)]]
        fitted,gbm_backend=make_models(train)
        for condition in CONDITIONS:
            events=condition_events(perturbed,variances,condition,int(held_out))
            if max_events_per_cell is not None:events=events[:max_events_per_cell]
            for system in SYSTEMS:
                memory=RecordingMemory() # separate and empty for every system/condition/trial
                backend=TracedBackend(reasoner) if system in ('single_agent','adam_llm') else None
                nodes={}  # One local role stack for each actual triggering node.
                if system in ('adam_llm','adam_gbm'):
                    for trigger_node in NODES:
                        nodes[trigger_node]=ADAMNode(trigger_node,config=ADAMConfig(eval_mode='full_pipeline'),
                            memory=memory,chain=InMemoryChainClient(),validator=LocalValidator(),
                            decision_backend=(backend if system=='adam_llm' else gbm_backend))
                prior=[]
                for original_id,event in events:
                    baseline=prior[-BASELINE_WINDOW:]
                    feature=decision_feature_vector(raw_ppm=event.primary.methane_ppm,
                          fused_ppm=fuse_readings(event.readings).fused_ppm,
                          dispersion_ppm=fuse_readings(event.readings).dispersion_ppm,
                          baseline_window=baseline)
                    inputs={'condition':condition,'trial':int(held_out),'event_id':original_id,
                            'node_readings':[r.redacted() for r in event.readings],
                            'feature_values':feature.tolist(), 'baseline':baseline}
                    prediction=None; trace=None; inference=None
                    if system=='static_threshold':
                        prediction='ANOMALY' if event.primary.methane_ppm>=THRESHOLD_PPM else 'NORMAL'
                    elif system in fitted:
                        p=float(fitted[system].predict_proba(feature.reshape(1,-1))[0,1])
                        prediction='ANOMALY' if p>=.5 else 'NORMAL'
                    elif system=='single_agent':
                        local=event.primary
                        fusion=FusionResult(local.methane_ppm,{local.node_id:1.0},(local.node_id,),0.0)
                        ev=CrewEvent(f'{condition}-{held_out}-{original_id}-single',local.node_id,
                                     local.methane_ppm,event.timestamp)
                        inference=backend.reason(event=ev,fusion=fusion,
                            node_readings=[local.redacted()],baseline_window=baseline,history=[],
                            deadline_s=30.,threshold_ppm=THRESHOLD_PPM)
                        prediction=inference.decision.classification
                    else:
                        # New, fresh event identity. Current reading cannot enter baseline.
                        node=nodes[event.primary.node_id]
                        node.sensor.observe(event.primary)
                        ev=node.sensor.publish_trigger(event.primary)
                        ev.event_id=f'{condition}-{held_out}-{original_id}-{system}'
                        trace=node.handle_event(ev,list(event.readings),sample_resources=False,
                                                baseline_window=baseline)
                        prediction=trace.final_classification
                    retrieved=memory.last_retrieved if trace is not None else []
                    if trace is not None and system=='adam_llm':
                        if backend.last.get('retrieved_ids') != [str(r['event_id']) for r in retrieved]:
                            raise AssertionError('retrieval log differs from inference input')
                        if any(float(r['timestamp'])>=event.timestamp for r in retrieved):
                            raise AssertionError('future/current event leaked into retrieval')
                    record={**inputs, 'system':system,'label':event.label,
                        'reference_ppm':event.reference_ppm, 'raw_ppm':event.primary.methane_ppm,
                        'fused_ppm':fuse_readings(event.readings).fused_ppm,
                        'dispersion_ppm':fuse_readings(event.readings).dispersion_ppm,
                        'prediction':prediction, 'prediction_resolved':prediction in ('NORMAL','ANOMALY'),
                        'input_sha256':canonical_sha(inputs),
                        'retrieved_ids':[r.get('event_id') for r in retrieved],
                        'retrieval_sha256':canonical_sha(retrieved),
                        'model_call':(backend.last.copy() if backend and system in ('single_agent','adam_llm') else None),
                        'initial_classification':trace.initial_classification if trace else prediction,
                        'votes':trace.classification_votes if trace else {},
                        'final_classification':trace.final_classification if trace else prediction,
                        'quorum_required':trace.quorum_required if trace else None,
                        'quorum_achieved':trace.quorum_achieved if trace else None,
                        'confidence_source':trace.confidence_source if trace else None,
                        'model_confidence':trace.model_confidence if trace else (inference.decision.confidence if inference else None),
                        'crew_support':trace.crew_support if trace else None,
                        'action_released':trace.executed if trace else None,
                        'failure_stage':trace.failure_stage if trace else None,
                        'degraded_mode':trace.degraded_mode if trace else (inference.decision.degraded_mode if inference else False),
                        'trace_sha256':canonical_sha(trace.to_dict()) if trace else None,
                    }
                    if system in ('single_agent','adam_llm'):
                        call_id=f'{condition}:{held_out}:{original_id}:{system}'
                        record['llm_call_id']=call_id
                        llm_calls.append({'call_id':call_id,'condition':condition,'trial':int(held_out),
                            'event_id':original_id,'system':system,'model_kind':reasoner,
                            'trace':backend.last.copy(),'artifacts':backend.last_payload.copy()})
                    if system=='adam_gbm':
                        record['model_call']={'model_kind':'fitted_gbm_clean_training',
                                               'feature_sha256':canonical_sha(feature.tolist()),
                                               'initial_decision_sha256':canonical_sha(trace.initial_decision.to_dict()) if trace and trace.initial_decision else None}
                    if system in ('single_agent','adam_llm') and backend.calls != len(prior)+1:
                        raise AssertionError('fresh condition-specific reasoner call missing or cached')
                    records.append(record)
                    prior.append(float(event.primary.methane_ppm))
    out=outdir/'reference_event_traces.jsonl'
    with out.open('w') as f:
        for r in records:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    model_file=outdir/'model_call_traces.jsonl'
    with model_file.open('w') as f:
        for x in llm_calls:f.write(json.dumps(x,sort_keys=True,allow_nan=False)+'\n')
    # Report trial-level scores and class-coverage separately: abstentions are
    # never silently converted to NORMAL or excluded from a full-coverage claim.
    table=pd.DataFrame([{'condition':r['condition'],'trial':r['trial'],
           'system':r['system'],'label':r['label'],'prediction':r['prediction']}
           for r in records])
    def score_cell(group):
        n=len(group); good=group[group.prediction.isin(['NORMAL','ANOMALY'])]
        y=good.label.to_numpy(dtype=int)
        pred=(good.prediction=='ANOMALY').to_numpy(dtype=int)
        tp=int(((y==1)&(pred==1)).sum())
        fp=int(((y==0)&(pred==1)).sum())
        fn=int(((y==1)&(pred==0)).sum())
        tn=int(((y==0)&(pred==0)).sum())
        f1=(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn)>0 else None
        return {'n_events':n,'n_resolved':len(good),'n_unresolved':n-len(good),
                'coverage':len(good)/n,'tp':tp,'fp':fp,'tn':tn,'fn':fn,
                'f1_resolved_subset':f1,
                'f1_full_coverage':f1 if len(good)==n else None}
    trial_scores=[]
    for (cond,trial,system),group in table.groupby(['condition','trial','system']):
        trial_scores.append({'condition':cond,'trial':int(trial),'system':system,
                             **score_cell(group),'not_historical':True})
    trial_file=outdir/'reference_trial_scores.csv'
    with trial_file.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(trial_scores[0]));w.writeheader();w.writerows(trial_scores)
    summaries=[]
    for (cond,system),group in table.groupby(['condition','system']):
        cells=[r for r in trial_scores if r['condition']==cond and r['system']==system]
        all_complete=all(x['n_unresolved']==0 for x in cells)
        per_trial=[x['f1_full_coverage'] for x in cells]
        valid=[x for x in per_trial if x is not None]
        summaries.append({'condition':cond,'system':system,**score_cell(group),
            'n_trials':len(cells),
            'mean_trial_f1_full_coverage':float(np.mean(valid)) if all_complete and len(valid)==len(cells) else None,
            'sd_trial_f1_full_coverage':float(np.std(valid,ddof=1)) if all_complete and len(valid)>1 else None,
            'not_historical':True})
    summary_file=outdir/'reference_summary.csv'
    with summary_file.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    manifest={'schema':SCHEMA,'study_type':'NEW_REFERENCE_RECONSTRUCTION',
           'historical_reproduction':False,'input_source':source['source'],
           'input_sha256':file_sha(input_path),'source_manifest_sha256':file_sha(input_path.with_suffix('.manifest.json')),
           'reconstructed_model_type':reasoner,'semantic_memory':'IN_MEMORY_CONCENTRATION_NEAREST_NOT_WEAVIATE',
           'confidence_note':'model_confidence refers to initial Decision-Agent class; crew_support is vote fraction, not calibrated class probability',
           'clean_train_only':True,'physical_four_node_input_required':True,
           'n_trials':len(trials),'conditions':list(CONDITIONS),'systems':list(SYSTEMS),
           'max_events_per_cell':max_events_per_cell,
           'partial_execution':max_events_per_cell is not None,
           'historical_study_trial_size_expected':200,
           'event_records':len(records),'llm_or_fixture_call_records':len(llm_calls),
           'fitted_crew_decision_records':sum(r['system']=='adam_gbm' for r in records),
           'unresolved_events':sum(not r['prediction_resolved'] for r in records),
           'output_sha256':{'events':file_sha(out),'model_calls':file_sha(model_file),'trial_scores':file_sha(trial_file),'summary':file_sha(summary_file),
                            'perturbations':file_sha(perturbdir/'degraded_streams.csv'),
                            'perturbation_manifest':file_sha(perturbdir/'harness_manifest.csv')},
           'declared_input_provenance_is_not_external_verification':True,
           'not_equivalent_to_original':'historical four-node streams/prompt/retrieval records unavailable',
           'not_equivalent_to_deployed_semantic_memory':'offline fixture uses concentration proximity, not Weaviate embeddings'}
    (outdir/'reference_run_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--reasoner',choices=('fixture','ollama'),required=True)
    ap.add_argument('--allow-fixture',action='store_true')
    ap.add_argument('--max-events-per-cell',type=int)
    args=ap.parse_args()
    manifest=run_reference(args.input,args.outdir,allow_fixture=args.allow_fixture,
                            reasoner=args.reasoner,max_events_per_cell=args.max_events_per_cell)
    print(json.dumps({k:manifest[k] for k in ('study_type','input_source','event_records',
                                           'llm_or_fixture_call_records','unresolved_events')},indent=2))

if __name__=='__main__':main()
