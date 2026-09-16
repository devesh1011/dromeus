"""Build M2 report figures from retained run evidence, without altering it."""
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / 'charts'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','axes.titlecolor':'#285477',
                     'axes.labelcolor':'#263444','figure.facecolor':'white','savefig.facecolor':'white'})
COLORS = ['#3274A5','#D58A35','#4F9380']
SEEDS = [17,29,41]
SCALES = [4,8,16]
sources = {}
def read(p):
    sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches='tight')
    plt.close(fig)

runs = {}
fig, axes = plt.subplots(3,2,figsize=(6.5,7.1),sharex=True)
for row,w in enumerate(SCALES):
    for seed,color in zip(SEEDS,COLORS):
        base = ROOT / f'benchmark-results/{w}-nodes/seed-{seed}'
        report = read(base/'reports/acceptance-report.json')
        rec = read(base/'reports/reconciliation.json')
        div = read(base/'reports/divergence-report.json')
        metrics = defaultdict(list)
        count = 0
        for p in sorted((base/'logs').glob('node-*/dromeus.jsonl')):
            digest=hashlib.sha256()
            with p.open('rb') as f:
                for line in f:
                    digest.update(line)
                    value=json.loads(line)
                    if value.get('event')=='round_metrics':
                        metrics[value['round_id']].append(value)
                        count += 1
            sources[str(p.relative_to(ROOT))]=digest.hexdigest()
        assert count==w*500, (w,seed,count)
        assert set(metrics)==set(range(500)) and all(len(v)==w for v in metrics.values())
        evaluation=[r for r in sorted(metrics) if all('evaluation_accuracy' in v for v in metrics[r])]
        acc=[np.mean([v['evaluation_accuracy'] for v in metrics[r]])*100 for r in evaluation]
        loss=[np.mean([v['local_loss'] for v in metrics[r]]) for r in sorted(metrics)]
        axes[row,0].plot(np.array(evaluation)+1,acc,color=color,label=f'Seed {seed}',lw=1.5)
        axes[row,1].plot(range(1,501),loss,color=color,label=f'Seed {seed}',lw=1.2,alpha=.85)
        timings={k:float(np.mean([r[k] for r in rec['rounds']])) for k in
                 ['local_compute_seconds','peer_wait_seconds','transfer_seconds','mixing_seconds','evaluation_seconds']}
        ex=report['execution']
        start=min(datetime.fromisoformat(t) for t in ex['node_start_timestamps'])
        end=max(datetime.fromisoformat(t) for t in ex['node_complete_timestamps'])
        runs[f'{w}-{seed}']={'accuracy':report['accuracy'],'timings':timings,
                            'lifecycle_seconds':(end-start).total_seconds(),
                            'residual':report['residual'],'divergence':div,
                            'communication':report['communication'],
                            'curves':{'evaluation_outer_rounds':[r+1 for r in evaluation],
                                      'mean_accuracy_percent':[float(v) for v in acc],
                                      'outer_rounds':list(range(1,501)),
                                      'mean_local_batch_loss':[float(v) for v in loss]}}
    axes[row,0].set_title(f'{w} workers: test accuracy')
    axes[row,1].set_title(f'{w} workers: local training loss')
    axes[row,0].set_ylabel('Accuracy (%)'); axes[row,0].set_ylim(10,100)
    axes[row,1].set_ylabel('Cross-entropy'); axes[row,1].set_ylim(bottom=0)
    for ax in axes[row]: ax.grid(alpha=.16); ax.set_xlim(0,500)
axes[0,0].legend(fontsize=8,loc='lower right')
axes[-1,0].set_xlabel('Completed outer rounds'); axes[-1,1].set_xlabel('Completed outer rounds')
fig.tight_layout(h_pad=1.5)
save(fig,'learning.png')

fig,axes=plt.subplots(1,3,figsize=(6.5,2.8),sharey=True)
for ax,w in zip(axes,SCALES):
    for seed,color in zip(SEEDS,COLORS):
        points=runs[f'{w}-{seed}']['divergence']['points']
        ax.plot([p['completed_outer_steps'] for p in points],[p['weight_std_l2'] for p in points],color=color,label=f'Seed {seed}',lw=1.5)
    ax.set_title(f'{w} workers'); ax.set_xlabel('Outer rounds'); ax.grid(alpha=.16)
axes[0].set_ylabel('Weight standard deviation (L2)')
axes[0].legend(fontsize=7)
fig.tight_layout(); save(fig,'divergence.png')

identity_report=read(ROOT/'identity-ablation/4-nodes/reports/acceptance-report.json')
identity_accuracy=identity_report['accuracy']['mean']*100
compressed_accuracy=runs['4-17']['accuracy']['mean']*100
identity_wire=runs['4-17']['communication']['identity_complete_wire_bytes']/1e6
compressed_wire=runs['4-17']['communication']['nominal_complete_wire_bytes']/1e6
fig,axes=plt.subplots(1,2,figsize=(6.5,3.0))
axes[0].bar(['Identity','Compressed'],[identity_wire,compressed_wire],color=[COLORS[0],COLORS[2]])
axes[0].set_ylabel('Nominal complete-wire MB / update')
axes[0].set_title('Communication baseline')
for i,v in enumerate([identity_wire,compressed_wire]): axes[0].text(i,v+2,f'{v:.2f}',ha='center')
axes[0].set_ylim(0,105)
axes[1].bar(['Identity','Compressed'],[identity_accuracy,compressed_accuracy],color=[COLORS[0],COLORS[2]])
axes[1].set_ylim(0,100); axes[1].set_ylabel('Final mean test accuracy (%)'); axes[1].set_title('W4 seed 17 ablation')
for i,v in enumerate([identity_accuracy,compressed_accuracy]): axes[1].text(i,v+1,f'{v:.2f}%',ha='center',fontsize=8)
fig.tight_layout(); save(fig,'compression.png')

fig,ax=plt.subplots(figsize=(6.5,3.3))
labels=[]; data=[]
for w in SCALES:
    for seed in SEEDS:
        labels.append(f'W{w}\ns{seed}')
        data.append(list(runs[f'{w}-{seed}']['timings'].values()))
bottom=np.zeros(9)
for col,(name,color) in enumerate(zip(['Local compute','Peer wait','Transfer','Outer update','Evaluation'],['#3274A5','#A9BDCC','#4F9380','#D58A35','#856AA2'])):
    vals=np.array(data)[:,col]
    ax.bar(range(9),vals,bottom=bottom,label=name,color=color); bottom+=vals
ax.set_xticks(range(9),labels); ax.set_ylabel('Mean seconds / node-round'); ax.set_title('AXL recorded round-time components')
ax.legend(fontsize=7,ncol=3,loc='upper left'); ax.set_ylim(0,max(bottom)*1.3)
fig.tight_layout(); save(fig,'timing.png')

summary={'runs':runs,'identity_ablation':{'mean_accuracy_percent':identity_accuracy,'source':'identity-ablation/4-nodes/reports/acceptance-report.json'},'sources_sha256':sources,'notes':{'learning':'Worker means; evaluation at retained cadence; loss is unsmoothed final local batch loss per round.', 'timing':'Means of recorded per-node timing fields; not a synchronized global wall-clock measurement.'}}
(OUT.parent/'chart-data.json').write_text(json.dumps(summary,indent=2)+'\n')
for w in SCALES:
    print('W',w,'mean accuracy',np.mean([runs[f'{w}-{s}']['accuracy']['mean'] for s in SEEDS])*100,
          'lifecycle minutes',[round(runs[f'{w}-{s}']['lifecycle_seconds']/60,2) for s in SEEDS],
          'mean round seconds',[round(sum(runs[f'{w}-{s}']['timings'].values()),2) for s in SEEDS],
          'residual maxima',max(runs[f'{w}-{s}']['residual']['maximum_l2_norm'] for s in SEEDS),
          'ratio maxima',max(runs[f'{w}-{s}']['residual']['maximum_to_signal_ratio'] for s in SEEDS))
