"""Descriptive summary for the separate, predeclared 16-start stress test."""
import csv,json,sys
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from setup_study import ROOT,RUN,write
sys.path.insert(0,str(ROOT))
from variants import VERSIONS

def main():
    stage=RUN/'geometry_confirmation'
    assert json.loads((stage/'completed.json').read_text())['episodes']==80
    assert json.loads((RUN/'validation_geometry_confirmation.json').read_text())['passed']
    design=json.loads((RUN/'geometry_confirmation_design.json').read_text())
    groups=dict(zip(design['state_names'],design['groups']))
    jobs=json.loads((stage/'jobs.json').read_text());records={};rows=[]
    for j in jobs:
        r=json.loads((stage/j['id']/'result.json').read_text())['records']
        assert [x['demo'] for x in r]==design['state_names']
        records[j['version']]={x['demo']:x for x in r}
        for x in r:
            rows.append({'version':j['version'],'group':groups[x['demo']],'demo':x['demo'],
                         'success':x['success'],'steps':x['steps'],'attempts':x['attempts']})
    kinds=list(dict.fromkeys(design['groups']));totals={};pairs=[]
    for v in VERSIONS:
        totals[v]={g:{'successes':sum(x['success'] for x in rows if x['version']==v and x['group']==g),'episodes':4} for g in kinds}
        totals[v]['all']={'successes':sum(x['success'] for x in records[v].values()),'episodes':16}
    for v,ref in [('v4','v2'),('timing_only','v4'),('side_only','v4'),('v5','v4'),('v5','v2'),('v5','timing_only'),('v5','side_only')]:
        pairs.append({'version':v,'reference':ref,
            'rescued_demos':[d for d in groups if records[v][d]['success'] and not records[ref][d]['success']],
            'regressed_demos':[d for d in groups if records[ref][d]['success'] and not records[v][d]['success']]})
    out=RUN/'analysis_geometry_confirmation';out.mkdir(exist_ok=False)
    write(out/'summary.json',{'stage':'geometry_confirmation','unique_starts':16,'new_executed_episodes':80,
        'totals':totals,'comparisons':pairs,'episodes':rows,
        'scope':'Descriptive stress test, four starts per hand-specified geometry stratum. Stationary-open proposal and forced recovery at step 80; not learned-policy distribution, not pooled with native confirmation.'})
    with (out/'episodes.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,4.8));colors=['#a4abb4','#65717e','#4677b8','#b37b39','#15816c']
    for i,(v,color) in enumerate(zip(VERSIONS,colors)):
        vals=[totals[v][g]['successes'] for g in kinds]
        b=ax.bar(np.arange(4)+(i-2)*.16,vals,.15,color=color,label=v)
        ax.bar_label(b,labels=[f'{n}/4' for n in vals],fontsize=8,padding=3)
    ax.set_xticks(range(4),['Far alignment','Destination corner','Sideways source','Sideways destination'])
    ax.set_ylim(0,5.2);ax.set_yticks(range(5));ax.set_ylabel('Successful starts (of 4)')
    ax.set_title('Independent geometry stress test | 16 paired starts | 400-step cap')
    ax.legend(ncol=5,loc='upper center',frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(out/f'geometry_comparison.{ext}',dpi=300)
    plt.close(fig);print(json.dumps({'totals':totals,'comparisons':pairs},indent=2))

if __name__=='__main__':main()
