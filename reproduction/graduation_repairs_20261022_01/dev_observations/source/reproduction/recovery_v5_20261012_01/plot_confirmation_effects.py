"""Plot the registered paired-start estimates and pointwise intervals."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from setup_study import RUN
def main():
    s=json.loads((RUN/'analysis_confirmation/summary.json').read_text())['statistics'];rows=[]
    for condition,method,label in [('clean','bc','BC, clean'),('noise005','bc','BC, noise'),('clean','full','Full, clean'),('noise005','full','Full, noise')]:
        rows.append((label,s[condition]['v5_'+method+'_minus_v4']))
    rows.append(('Prespecified mixture',s['overall_execution_mix']['v5_minus_v4']))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),gridspec_kw={'width_ratios':[1.2,1]})
    for ax,values,title in [(axes[0],rows,'v5 minus v4'),(axes[1],[(n,s['overall_execution_mix'][v+'_minus_v4']) for n,v in [('Timing','timing_only'),('Side clearance','side_only'),('Combined v5','v5')]],'Ablations minus v4 | prescribed mixture')]:
        for y,(label,r) in enumerate(values):
            value=100*r['estimate'];lo,hi=np.array(r['ci95'])*100
            ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt='o',color='#15816c',capsize=4,ms=7)
            ax.annotate(f'{value:+.2f} [{lo:+.2f}, {hi:+.2f}]',(value,y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
        ax.axvline(0,color='#777',linestyle='--',linewidth=1);ax.set_yticks(range(len(values)),[v[0] for v in values]);ax.invert_yaxis();ax.margins(y=.24,x=.30)
        ax.set_xlabel('Success difference (percentage points)');ax.set_title(title,fontsize=12);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Independent confirmation | 60 paired starts | 95% pointwise bootstrap intervals',fontsize=12)
    fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(RUN/'analysis_confirmation'/f'paired_effects.{ext}',dpi=300)
    plt.close(fig)
if __name__=='__main__':main()
