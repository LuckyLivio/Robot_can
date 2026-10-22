"""Compact, print-readable figures from frozen result CSVs; never rerun analysis."""
import csv
import hashlib
import json
from pathlib import Path
import sys

DEPS=Path.home()/"AppData/Local/Temp/robot_can_design_20261022_01/doc_plot_deps"
sys.path.insert(0,str(DEPS))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=HERE.parent.parent/'reproduction/graduation_repairs_20261022_01'


def generate():
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10.5,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none',
                         'savefig.facecolor':'white','figure.facecolor':'white'})
    cells_path=RUN/'analysis/confirmation/cells.csv'
    effects_path=RUN/'analysis/confirmation/paired_effects.csv'
    with cells_path.open(encoding='utf-8') as f:cells=list(csv.DictReader(f))
    with effects_path.open(encoding='utf-8') as f:effects=list(csv.DictReader(f))
    plotted=[]
    fig,ax=plt.subplots(figsize=(6.2,3.6))
    labels=['BC\n无恢复','BC\nv5恢复','Full\n无恢复','Full\nv5恢复']
    groups=[('bc','none'),('bc','v5'),('full','none'),('full','v5')]
    for shift,noise,color,label in [(-.19,0.,'#315e85','干净'),(.19,.05,'#bd773e','噪声0.05')]:
        values=[]
        for x,(method,controller) in enumerate(groups):
            rows=[r for r in cells if r['study_arm']=='recovery_factorial' and r['method']==method and r['controller']==controller and float(r['noise'])==noise]
            a=sum(int(r['successes']) for r in rows);b=sum(int(r['executions']) for r in rows)
            percent=100*a/b;values.append(percent)
            ax.text(x+shift,percent+1.2,f'{a}/{b}',ha='center',va='bottom',fontsize=8.5)
            plotted.append({'figure':'recovery_factorial_success','method':method,'controller':controller,'noise':noise,'successes':a,'executions':b})
        ax.bar(np.arange(4)+shift,values,.36,color=color,label=label,zorder=3)
    ax.set_xticks(np.arange(4),labels);ax.set_ylim(0,112);ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel('成功率  %')
    ax.grid(axis='y',color='#e5e8eb',zorder=0);ax.legend(ncol=2,loc='lower left',frameon=False)
    fig.subplots_adjust(left=.12,right=.98,top=.96,bottom=.19)
    save(fig,'recovery_factorial_success')

    def effect(contrast,comparison,method):
        rows=[r for r in effects if r['contrast']==contrast and r['comparison']==comparison and r['method']==method and r['metric']=='success' and r['noise']=='equal_noise_joint']
        if len(rows)!=1:raise ValueError((contrast,comparison,method,len(rows)))
        return rows[0]
    primary=[('BC 恢复作用',effect('recovery_effect','v5 - none','bc')),
             ('Full 恢复作用',effect('recovery_effect','v5 - none','full')),
             ('v5下Full相对BC',effect('world_model_effect','Full - BC with v5','v5')),
             ('恢复交互效应',effect('recovery_interaction','(Full v5-none) - (BC v5-none)','full_minus_bc'))]
    effect_plot(primary,'recovery_factorial_effects',(-7,17),3.5)
    candidate=[('BC 候选减v5',effect('recovery_effect','queue_budget - v5','bc')),
               ('Full 候选减v5',effect('recovery_effect','queue_budget - v5','full'))]
    effect_plot(candidate,'queue_budget_candidate_effects',(-3.8,2),2.9)
    for figure,rows in [('recovery_factorial_effects',primary),('queue_budget_candidate_effects',candidate)]:
        for label,row in rows:plotted.append({'figure':figure,'label':label,**row})
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    receipt={'inputs_sha256':{str(p.relative_to(HERE.parent.parent)):sha(p) for p in (cells_path,effects_path)},
             'matplotlib':matplotlib.__version__,'numpy':np.__version__,'size_inches':'6.2 wide; text 8.5-10.5 pt',
             'scope':'Redrawn from frozen CSV only; no simulations, statistical recomputation or new samples.',
             'plotted_rows':plotted}
    (HERE/'research/document_figures.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def save(fig,name):
    for suffix in ('png','svg'):fig.savefig(HERE/'assets'/f'{name}.{suffix}',dpi=300)
    plt.close(fig)


def effect_plot(rows,name,limits,height):
    fig,ax=plt.subplots(figsize=(6.2,height))
    for index,(label,row) in enumerate(reversed(rows)):
        a,lo,hi=(float(row[key]) for key in ('effect','ci95_low','ci95_high'))
        ax.errorbar(a,index,xerr=[[a-lo],[hi-a]],fmt='o',capsize=4,color='#315e85',markersize=6)
        ax.text(limits[1],index+.24,f'{a:+.2f}  [{lo:.2f}, {hi:.2f}]',ha='right',fontsize=9)
    ax.axvline(0,color='#888888',ls='--',lw=1)
    ax.set_yticks(range(len(rows)),[label for label,_ in reversed(rows)])
    ax.set_xlim(*limits);ax.set_ylim(-.6,len(rows)-.35);ax.set_xlabel('成功率差  百分点  及95%配对区间')
    ax.grid(axis='x',color='#eceeef');fig.subplots_adjust(left=.30,right=.97,top=.95,bottom=.23)
    save(fig,name)


if __name__=='__main__':generate()
