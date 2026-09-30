#!/usr/bin/env python3
"""Figure 7 and manuscript statistics from measured inputs and stage-model outputs.

No target result values are supplied. Predictions and validation are read from
the stage-model output files; measured values are read from the hardware CSV.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAGES = [('T_reason_ms', 'Reasoning', '#0072B2'),
          ('T_cross_node_ms', 'Cross-node exchange', '#E69F00'),
          ('T_blockchain_ms', 'Blockchain commit', '#009E73'),
          ('T_query_ms', 'Memory retrieval', '#CC79A7'),
          ('T_network_ms', 'Network transfer', '#D55E00'),
          ('T_merge_ms', 'Result merge', '#777777')]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(hardware, results, out):
    hw = pd.read_csv(hardware)
    runs = pd.read_csv(results / 'stage_scaleout_reference_model.csv')
    validation = pd.read_csv(results / 'strict_lolo_validation.csv')
    metadata = json.loads((results / 'scalability_manifest.json').read_text())
    if set(hw.Run_Mode) != {'HARDWARE'}:
        raise ValueError('Hardware input must contain only measured rows')
    if digest(hardware) != metadata['input_sha256']:
        raise ValueError('Model outputs were calibrated on a different input')
    for key, item in metadata['outputs'].items():
        if digest(results / item['file']) != item['sha256']:
            raise ValueError('Changed model output: ' + key)
    hw_g = hw.groupby('Node_Count').T_decision_ms.agg(['mean','std','count'])
    model_g = runs.groupby('node_count').T_decision_ms.agg(['mean','std','count'])
    if not np.allclose(runs.T_decision_ms, runs[[s for s,_,_ in STAGES]].sum(axis=1)):
        raise ValueError('Generated total differs from stage sum')
    recorded = pd.read_csv(results / 'stage_scaleout_level_means.csv').set_index('node_count')
    if not np.allclose(model_g['mean'], recorded.mean_ms):
        raise ValueError('Model summary differs from generated runs')
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'serif', 'font.serif':['DejaVu Serif'],
        'font.size':12, 'axes.labelsize':13, 'axes.titlesize':13,
        'legend.fontsize':10, 'axes.linewidth':1, 'pdf.fonttype':42,
        'ps.fonttype':42, 'savefig.bbox':'tight', 'savefig.pad_inches':0.08})
    fig, axes = plt.subplots(1,3,figsize=(18,5.6))
    rng = np.random.default_rng(42)  # horizontal visual jitter only
    a,b,c=axes
    for n, group in hw.groupby('Node_Count'):
        a.scatter(n+rng.uniform(-0.09,0.09,len(group)),group.T_decision_ms/1000,
                  s=19,color='#0072B2',alpha=.27,zorder=2)
    a.errorbar(hw_g.index,hw_g['mean']/1000,yerr=hw_g['std']/1000,
        fmt='o',capsize=4,color='#0072B2',markersize=8,label='Measured mean ± SD')
    a.scatter(validation.held_out_N+.14,validation.predicted_mean_ms/1000,
        marker='D',s=55,facecolors='white',edgecolors='#D55E00',linewidths=1.5,
        label='Held-out level prediction',zorder=4)
    a.set(xlabel='Physical Raspberry Pi nodes, N',ylabel='Decision latency (s)',
          xticks=[1,2,3,4],ylim=(16.2,20.2),title='Hardware and held-out predictions')
    a.legend(loc='upper left',frameon=False)
    metrics=metadata['strict_lolo_validation']
    a.text(.96,.055,f"LOLO MAPE: {metrics['mape_pct']:.2f}%\n"
           f"Mean signed error: {metrics['signed_mean_percentage_error_pct']:+.2f}%",
           transform=a.transAxes,ha='right',va='bottom',fontsize=11)
    for n, group in runs.groupby('node_count'):
        b.scatter(n+rng.uniform(-.16,.16,len(group)),group.T_decision_ms/1000,
                  s=19,color='#56B4E9',alpha=.35,zorder=2)
    b.errorbar(model_g.index,model_g['mean']/1000,yerr=model_g['std']/1000,
        fmt='s-',capsize=4,color='#0072B2',markersize=7,linewidth=1.5,
        label='Software estimate: mean ± SD')
    growth=100*(model_g.loc[16,'mean']/model_g.loc[4,'mean']-1)
    b.set(xlabel='Participating logical nodes, N',ylabel='Decision latency (s)',
          xticks=list(model_g.index),ylim=(17.5,22.2),title='Hardware-calibrated scale-out')
    b.legend(loc='upper left',frameon=False)
    b.text(.96,.055,f'+{growth:.1f}% from N=4 to N=16\n18 replicates per level',
           transform=b.transAxes,ha='right',va='bottom',fontsize=11)
    stage_g=hw.groupby('Node_Count')[[s for s,_,_ in STAGES]].mean()
    for col,label,color in STAGES:
        values=stage_g[col].where(stage_g[col]>0,np.nan)
        c.plot(stage_g.index,values,'o-',color=color,markersize=5,linewidth=1.5,label=label)
    c.set(xlabel='Physical Raspberry Pi nodes, N',ylabel='Mean stage latency (ms)',
          xticks=[1,2,3,4],yscale='log',ylim=(25,80000),title='Measured stage decomposition')
    c.legend(loc='center right',frameon=False,fontsize=9.5)
    c.text(.02,.48,'Cross-node exchange is zero at N=1;',transform=c.transAxes,fontsize=9.5)
    c.text(.02,.43,'its line begins at N=2.',transform=c.transAxes,fontsize=9.5)
    for index,ax in enumerate(axes):
        ax.grid(linestyle=':',alpha=.35,which='both');ax.set_axisbelow(True)
        ax.text(-.12,1.05,f'({chr(97+index)})',transform=ax.transAxes,
                fontsize=18,fontweight='bold')
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out/'figure7_scalability.pdf')
    fig.savefig(out/'figure7_scalability.png',dpi=600)
    fig.savefig(out/'figure7_preview.png',dpi=170)
    plt.close(fig)
    summary={'validation':metrics,'software_growth_pct':growth,
             'hardware':hw_g.reset_index().to_dict(orient='records'),
             'software':model_g.reset_index().to_dict(orient='records'),
             'source_hardware_sha256':digest(hardware),
             'model_runs_sha256':digest(results/'stage_scaleout_reference_model.csv')}
    (out/'figure7_statistics.json').write_text(json.dumps(summary,indent=2)+'\n')
    rows=[]
    for name,frame in [('Hardware',hw_g),('Scale-out',model_g)]:
        basis='Measured (Raspberry~Pi~5)' if name=='Hardware' else 'Model prediction'
        for i,(n,r) in enumerate(frame.iterrows()):
            prefix=(rf'\multirow{{{len(frame)}}}{{*}}{{{name}}} & '
                    rf'\multirow{{{len(frame)}}}{{*}}{{{basis}}}') if i==0 else ' & '
            rows.append(prefix+f" & {n} & {r['mean']/1000:.2f} $\\pm$ {r['std']/1000:.2f} & {int(r['count'])} \\\\")
        if name=='Hardware':rows.append(r'\midrule')
    table=r'''\begin{table}[H]
\centering
\caption{Decision latency by node count under fixed load.}
\label{tab:scalability}
\renewcommand{\arraystretch}{1.2}
\footnotesize
\begin{threeparttable}
\begin{tabular}{@{}lcccc@{}}
\toprule
Domain & Basis & $N$ & Level mean (s) & $n$/level \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\begin{tablenotes}[flushleft]
\footnotesize
\item Values are mean $\pm$ standard deviation over 18 measurements or model replicates per node count. Hardware values are measured on Raspberry~Pi~5 nodes. Scale-out values are generated by the six-stage model calibrated on those hardware measurements, with random seed 42. Configurations above $N=4$ are software estimates under the fixed workload and service assumptions of Section~\ref{sec:scalability_method}. Predicted means do not establish worst-case latency or deadline compliance for larger physical deployments.
\end{tablenotes}
\end{threeparttable}
\end{table}
'''
    (out/'table_scalability.tex').write_text(table)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hardware',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.hardware,a.results,a.out),indent=2))

if __name__=='__main__':main()
