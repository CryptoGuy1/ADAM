#!/usr/bin/env python3
"""Plot Phase 3 *reference-model* outputs without altering archived manuscript figures.

Input: archived 72 physical-run CSV and new stage-model output. The figure
explicitly separates physically measured N=1..4 from conditional N=4..16.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def _csv(path):
    with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def render(hardware:Path,model_summary:Path,out:Path)->tuple[Path,Path]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    physical=defaultdict(list)
    for row in _csv(hardware):
        if row['Run_Mode']!='HARDWARE':raise ValueError('input contains non-hardware rows')
        physical[int(row['Node_Count'])].append(float(row['T_decision_ms'])/1000)
    predicted=sorted(_csv(model_summary),key=lambda r:int(r['node_count']))
    out.mkdir(parents=True,exist_ok=True)
    x=sorted(physical)
    fig,ax=plt.subplots(figsize=(8.5,5.0))
    ax.errorbar(x,[statistics.fmean(physical[n]) for n in x],
                yerr=[statistics.stdev(physical[n]) for n in x],
                fmt='o-',capsize=4,label='Measured: physical Pi nodes (18 runs/level)')
    ax.errorbar([int(r['node_count']) for r in predicted],
                [float(r['mean_ms'])/1000 for r in predicted],
                yerr=[float(r['std_ms'])/1000 for r in predicted],
                fmt='s--',capsize=4,label='Reference-model estimates (18 replicates/level)')
    ax.axvline(4,linestyle=':',linewidth=1)
    ax.set(xlabel='Participating nodes, N',ylabel='Decision latency (s)',
           title='Fixed-load scaling: measured versus conditional estimates')
    ax.set_xticks([1,2,3,4,6,8,12,16]);ax.grid(alpha=.3);ax.legend(loc='best',fontsize=8)
    fig.tight_layout()
    first=out/'reference_stage_scalability_preview.png'
    fig.savefig(first,dpi=220);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.5,5.0))
    base=next(r for r in predicted if int(r['node_count'])==4)
    for key,label in [('T_cross_node_ms','Cross-node exchange'),
                      ('T_network_ms','Network transfer'),('T_merge_ms','Result merge')]:
        ax.plot([int(r['node_count']) for r in predicted],
                [(float(r['mean_'+key])-float(base['mean_'+key]))/1000 for r in predicted],
                marker='o',label=label)
    ax.set(xlabel='Participating logical nodes, N',
           ylabel='Incremental modeled latency relative to N=4 (s)',
           title='Coordination-stage growth under fixed offered load')
    ax.set_xticks([4,6,8,12,16]);ax.grid(alpha=.3);ax.legend()
    fig.tight_layout()
    second=out/'reference_stage_growth_preview.png'
    fig.savefig(second,dpi=220);plt.close(fig)
    return first,second


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hardware',type=Path,required=True)
    p.add_argument('--level-means',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(*render(a.hardware,a.level_means,a.out),sep='\n')

if __name__=='__main__':main()
