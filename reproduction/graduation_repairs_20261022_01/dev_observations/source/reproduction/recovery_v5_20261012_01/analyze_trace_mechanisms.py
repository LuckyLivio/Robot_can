"""Post-hoc observation/command diagnostics; no intervention or tuning."""
import argparse,json,sys
from collections import Counter
import numpy as np
from setup_study import ROOT,RUN,write
sys.path.insert(0,str(ROOT))
from variants import controller_class

def main(stage):
    jobs=json.loads((RUN/stage/'jobs.json').read_text());summary={};episodes=[]
    for j in jobs:
        cls,cfg=controller_class(j['version']);last_name=None;acc=None;controller=None
        for line in (RUN/stage/j['id']/'trace.jsonl').open():
            r=json.loads(line);name=r['demo'];x=np.asarray(r['state_before'],np.float32)
            if name!=last_name:
                if acc is not None:episodes.append(acc)
                controller=cls([.1975,.4025,.8],.8,x,j['enabled'],cfg(**j.get('config',{})))
                previous=np.zeros(7,np.float32)
                acc={'job':j['id'],'version':j['version'],'method':j['method'],'seed':j['seed'],'noise':j['noise'],'demo':name,
                    'side_plans':[],'budget_pressure_steps':0,'accelerated_eligible_steps':0,'phases':Counter()}
                last_name=name;old_plan=None
            if j.get('force_recovery_at')==r['step']:controller.start_attempt('diagnostic_forced_start',x)
            _,diag=controller.choose(x,np.asarray(r['original_proposal'],np.float32),previous)
            assert diag==r['recovery']
            previous=np.asarray(r['executed_action'],np.float32)
            if hasattr(controller,'side_plan'):
                plan=controller.side_plan
                if plan!=old_plan and plan is not None:
                    acc['side_plans'].append({'step':r['step'],**plan});old_plan=dict(plan)
                pressure=controller.time_pressure(x) if controller.cfg.time_budget else False
                acc['budget_pressure_steps']+=bool(pressure)
                # Count eligibility only, not a claim that norm clipping changed.
                acc['accelerated_eligible_steps']+=bool(pressure and diag['mode'] in ['RETRACT','ALIGN','LIFT','CARRY'] and (diag['mode'] in ['RETRACT','ALIGN'] or controller.stable))
            acc['phases'][diag['mode']]+=1
            acc['steps']=r['step'];acc['success']=r['success_after'];acc['final_mode']=r['recovery'].get('next_mode',r['recovery']['mode'])
            acc['final_observation']=r['state_after'];acc['events']=controller.events.copy()
        if acc is not None:episodes.append(acc)
        print(j['id'],flush=True)
    for v in dict.fromkeys(j['version'] for j in jobs):
        rows=[r for r in episodes if r['version']==v]
        summary[v]={'episodes':len(rows),'with_feasible_side_plan':sum(any(p['feasible'] for p in r['side_plans']) for r in rows),
            'with_infeasible_side_plan':sum(any(not p['feasible'] for p in r['side_plans']) for r in rows),
            'with_budget_pressure':sum(r['budget_pressure_steps']>0 for r in rows),
            'with_early_stall':sum(any(e['reason']=='early_stall_preserve_budget' for e in r['events']) for r in rows),
            'with_early_descent_abort':sum(any(e['reason']=='descent_stalled_preserve_budget' for e in r['events']) for r in rows),
            'failure_final_modes':dict(Counter(r['final_mode'] for r in rows if not r['success']))}
    write(RUN/f'mechanisms_{stage}.json',{'post_hoc':True,'summary':summary,'episodes':episodes,
        'limitation':'Mechanism activation is reconstructed on logged observations, not a causal effect estimate; stage labels are not collision diagnoses. Accelerated eligibility is not a measured speed change.'})
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();main(a.stage)
