"""Descriptive paired failures from completed, frozen results; no tuning."""
import json
from collections import Counter
from setup_study import RUN,write
VERSIONS=['v2','v4','timing_only','side_only','v5']
def main():
    diag=json.loads((RUN/'mechanisms_confirmation.json').read_text())['episodes']
    records={v:{} for v in VERSIONS}
    for r in diag:records[r['version']][(r['method'],r['seed'],r['noise'],r['demo'])]=r
    keys=list(records['v5']);assert len(keys)==480
    for v in VERSIONS:assert set(records[v])==set(keys)
    def brief(r):
        x=r['final_observation'];reasons=[e['reason'] for e in r['events']]
        if x[18]<.76:kind='object_below_recovery_floor'
        elif r['final_mode'] in ['CARRY','PLACE','RELEASE','RETREAT','LIFT']:kind='budget_ended_during_'+r['final_mode'].lower()
        elif r['final_mode']=='EXHAUSTED' and 'observed_grasp_loss' in reasons:kind='attempts_exhausted_after_observed_loss'
        elif r['final_mode']=='EXHAUSTED':kind='attempts_exhausted_before_completion'
        else:kind='unfinished_other'
        return {k:r[k] for k in ['job','version','method','seed','noise','demo','success','steps','final_mode','phases','side_plans','events']}|{
            'observation_based_category':kind,'final_object_position':x[16:19],'final_object_quaternion':x[19:23]}
    failures=[brief(records['v5'][k]) for k in keys if not records['v5'][k]['success']]
    common=[brief(records['v5'][k]) for k in keys if all(not records[v][k]['success'] for v in VERSIONS)]
    pairs={}
    for ref in VERSIONS[:-1]:
        pair={}
        for label in ['rescues','regressions']:
            selected=[k for k in keys if records['v5'][k]['success']!=records[ref][k]['success'] and records['v5'][k]['success']==(label=='rescues')]
            pair[label]=[{'candidate':brief(records['v5'][k]),'reference':brief(records[ref][k])} for k in selected]
        pair['joint_successes']=sum(records['v5'][k]['success'] and records[ref][k]['success'] for k in keys)
        deltas=[records['v5'][k]['steps']-records[ref][k]['steps'] for k in keys if records['v5'][k]['success'] and records[ref][k]['success']]
        pair['mean_step_difference_among_joint_successes']=sum(deltas)/len(deltas) if deltas else None
        pairs[ref]=pair
    outcome={'post_hoc':True,'v5_failures':failures,'v5_failure_executions':len(failures),
        'v5_failure_distinct_starts':sorted({r['demo'] for r in failures}),'v5_failure_categories':dict(Counter(r['observation_based_category'] for r in failures)),
        'common_failed_executions_all_five':len(common),'common_failed_distinct_starts':sorted({r['demo'] for r in common}),
        'common_failed':common,'paired_changes':pairs,
        'limitation':'Categories describe observations/events, not proof of physical cause. Repeated training seeds and noise conditions share starts. Stress contacts are measured separately in stress_failure_physics.json.'}
    write(RUN/'posthoc_failure_review.json',outcome)
    print(json.dumps({k:v for k,v in outcome.items() if k not in ['v5_failures','common_failed','paired_changes']},indent=2))
    print(json.dumps({v:{'rescues':len(p['rescues']),'regressions':len(p['regressions']),'joint_successes':p['joint_successes'],'step_difference':p['mean_step_difference_among_joint_successes']} for v,p in pairs.items()},indent=2))
if __name__=='__main__':main()
