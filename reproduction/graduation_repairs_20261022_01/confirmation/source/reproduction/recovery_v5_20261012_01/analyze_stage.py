"""Five-version paired evaluation, fixed training seeds averaged within each start."""
import argparse,csv,json
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from setup_study import ROOT,RUN,write
VERSIONS=['v2','v4','timing_only','side_only','v5']
def main(stage):
    folder=RUN/stage;out=RUN/f'analysis_{stage}';out.mkdir(exist_ok=False)
    jobs=json.loads((folder/'jobs.json').read_text());records={};jobrows=[];episodes=[]
    for j in jobs:
        result=json.loads((folder/j['id']/'result.json').read_text());assert result['status']=='complete'
        rows=result['records'];assert [r['demo'] for r in rows]==j['demos']
        key=(j['version'],j['method'],j['seed'],j['noise']);assert key not in records;records[key]=rows
        if stage.startswith('dev'):
            for v in ['v2','v4']:
                base=json.loads((ROOT/j['baseline_result_'+v]).read_text())['records']
                assert [r['demo'] for r in base]==j['demos'];records[(v,j['method'],j['seed'],j['noise'])]=base
    for (v,m,s,n),rows in records.items():
        jobrows.append({'version':v,'method':m,'seed':s,'noise':n,'successes':sum(r['success'] for r in rows),'episodes':len(rows),
            'intervened_episodes':sum(r['attempts']>0 for r in rows),'mean_steps_capped':float(np.mean([r['steps'] for r in rows])),
            'events':dict(Counter(e['reason'] for r in rows for e in r['events']))})
        for r in rows:episodes.append({'version':v,'method':m,'seed':s,'noise':n,'demo':r['demo'],'success':r['success'],'steps':r['steps'],'attempts':r['attempts']})
    def outcomes(v,m,s,n):
        rows=records[(v,m,s,n)];assert [r['demo'] for r in rows]==[r['demo'] for r in records[('v4',m,s,n)]]
        return np.array([r['success'] for r in rows],float)
    pairs=[('v4','v2')]+[(v,ref) for v in VERSIONS[2:] for ref in ['v2','v4']]+[('v5','timing_only'),('v5','side_only')]
    comparisons=[]
    for v,ref in pairs:
        for noise in [0.,.05]:
            for method,seed in [('bc_k2',None),('full',17),('full',23),('full',41)]:
                a=outcomes(v,method,seed,noise);b=outcomes(ref,method,seed,noise);ids=[r['demo'] for r in records[(v,method,seed,noise)]]
                comparisons.append({'version':v,'reference':ref,'method':method,'seed':seed,'noise':noise,'successes':int(a.sum()),'baseline_successes':int(b.sum()),'episodes':len(a),
                    'rescues':int(((a==1)&(b==0)).sum()),'regressions':int(((a==0)&(b==1)).sum()),
                    'rescued_demos':[d for d,x,y in zip(ids,a,b) if x and not y],'regressed_demos':[d for d,x,y in zip(ids,a,b) if y and not x]})
    rng=np.random.default_rng(20261016);stats={}
    for noise,label in [(0.,'clean'),(.05,'noise005')]:
        n=len(outcomes('v4','bc_k2',None,noise));idx=rng.integers(0,n,(20000,n));values={}
        for v in VERSIONS:
            values[v+'_bc']=outcomes(v,'bc_k2',None,noise)
            values[v+'_full']=np.stack([outcomes(v,'full',s,noise) for s in [17,23,41]]).mean(0)
        diffs={}
        for v,ref in pairs:
            for m in ['bc','full']:diffs[v+'_'+m+'_minus_'+ref]=values[v+'_'+m]-values[ref+'_'+m]
        for m in ['bc','full']:diffs[m+'_interaction_TS_minus_T_minus_S_plus_v4']=values['v5_'+m]-values['timing_only_'+m]-values['side_only_'+m]+values['v4_'+m]
        for v in VERSIONS:diffs[v+'_full_minus_bc']=values[v+'_full']-values[v+'_bc']
        stats[label]={k:{'estimate':float(v.mean()),'ci95':np.quantile(v[idx].mean(1),[.025,.975]).tolist()} for k,v in {**values,**diffs}.items()}
    # Preserve the protocol's fixed execution mixture, but resample starts,
    # never the 8 correlated executions as if they were independent people.
    mixed={v:np.stack([outcomes(v,m,s,noise) for noise in [0.,.05] for m,s in [('bc_k2',None),('full',17),('full',23),('full',41)]]).mean(0) for v in VERSIONS}
    idx=rng.integers(0,n,(20000,n))
    mixed.update({v+'_minus_'+ref:mixed[v]-mixed[ref] for v,ref in pairs})
    stats['overall_execution_mix']={k:{'estimate':float(a.mean()),'ci95':np.quantile(a[idx].mean(1),[.025,.975]).tolist()} for k,a in mixed.items()}
    totals={}
    for v in VERSIONS:
        jr=[r for r in jobrows if r['version']==v];t={'successes':sum(r['successes'] for r in jr),'episodes':sum(r['episodes'] for r in jr)}
        for ref in ['v2','v4']:
            cr=[r for r in comparisons if r['version']==v and r['reference']==ref]
            t['rescues_vs_'+ref]=sum(r['rescues'] for r in cr);t['regressions_vs_'+ref]=sum(r['regressions'] for r in cr)
            t['distinct_rescued_starts_vs_'+ref]=sorted({d for r in cr for d in r['rescued_demos']})
        totals[v]=t
    summary={'stage':stage,'unique_starts':n,'new_executed_episodes':sum(len(j['demos']) for j in jobs),'totals':totals,'jobs':jobrows,
        'comparisons':comparisons,'statistics':stats,'inference':'Paired-start percentile bootstrap 20000 draws, seed20261016. Full averages 3 fixed seeds within start; uncertainty does not cover arbitrary retraining. Repeated conditions are not independent starts. Overall mixture equally weights the 8 prescribed runs per start. Intervals are pointwise, not multiplicity-adjusted. Rare discordance can make CI degenerate; zero-width CI is not proof of equivalence.'}
    write(out/'summary.json',summary)
    for name,rows in [('episodes.csv',episodes),('comparisons.csv',comparisons)]:
        with (out/name).open('x',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.9),sharey=True)
    labels=['v2','v4','Timing','Side clearance','v5 combined'];colors=['#a4abb4','#65717e','#4677b8','#b37b39','#15816c']
    for ax,label in zip(axes,['clean','noise005']):
        for i,(v,color,title) in enumerate(zip(VERSIONS,colors,labels)):
            vals=[100*stats[label][v+'_'+m]['estimate'] for m in ['bc','full']]
            bars=ax.bar(np.arange(2)+(i-2)*.16,vals,.15,label=title,color=color);ax.bar_label(bars,fmt='%.1f',fontsize=7,padding=3)
        ax.set_xticks([0,1],['BC k=2','Full (3-seed mean)']);ax.set_ylim(0,113);ax.set_title('Clean' if label=='clean' else 'Action noise 0.05');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Task success (%)');axes[1].legend(loc='lower right');fig.suptitle(f'{stage} | {n} paired starts per condition');fig.tight_layout()
    fig.savefig(out/'success_comparison.png',dpi=300);fig.savefig(out/'success_comparison.svg');fig.savefig(out/'success_comparison.pdf');plt.close(fig)
    print(json.dumps(totals,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();main(a.stage)
