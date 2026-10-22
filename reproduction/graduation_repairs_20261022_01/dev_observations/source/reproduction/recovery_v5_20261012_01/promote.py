import argparse,json,datetime
from setup_study import ROOT,RUN,write,digest
def main(stage):
    s=json.loads((RUN/f'analysis_{stage}/summary.json').read_text());plan=json.loads((RUN/'study_plan.json').read_text());g=plan['dev_gate']
    assert json.loads((RUN/f'validation_{stage}.json').read_text())['passed'];candidates={}
    for v in ['timing_only','side_only','v5']:
        t=s['totals'][v];rows=[r for r in s['jobs'] if r['version']==v]
        count=lambda m,n:sum(r['successes'] for r in rows if r['method']==m and r['noise']==n)
        checks={'total':t['successes']>=g['min_success'],'regressions_v4':t['regressions_vs_v4']<=g['max_regressions_vs_v4'],
            'regressions_v2':t['regressions_vs_v2']<=g['max_regressions_vs_v2'],
            'distinct_rescued_starts':len(t['distinct_rescued_starts_vs_v4'])>=g['min_distinct_rescued_starts_vs_v4'],
            'bc_clean':count('bc_k2',0.)>=g['min_bc_clean'],'bc_noise':count('bc_k2',.05)>=g['min_bc_noise'],
            'full_clean':count('full',0.)>=g['min_full_clean'],'full_noise':count('full',.05)>=g['min_full_noise']}
        candidates[v]={'passed':all(checks.values()),'checks':checks,**t}
    passing=[v for v,c in candidates.items() if c['passed']]
    selected=min(passing,key=lambda v:(-candidates[v]['successes'],candidates[v]['regressions_vs_v4'],2 if v=='v5' else 1,0 if v=='timing_only' else 1)) if passing else None
    receipt={'passed':bool(passing),'selected_variant':selected,'stage':stage,'candidates':candidates,'study_plan_sha256':digest(RUN/'study_plan.json')}
    write(RUN/f'promotion_{stage}.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
    if not passing:return
    write(RUN/'promotion.json',receipt);sources=json.loads((RUN/stage/'started.json').read_text())['sha256']
    for name,sha in sources.items():assert digest(ROOT/name)==sha,name
    geometry_sources=json.loads((RUN/'gripper_geometry.json').read_text())['source_sha256']
    for name,sha in geometry_sources.items():assert digest(ROOT/name)==sha,name
    sources.update(geometry_sources)
    for p in [*RUN.glob('*.py'),RUN/'gripper_geometry.json',RUN/'study_plan.json',RUN/'DEVELOPMENT_NOTES.md']:sources[p.relative_to(ROOT).as_posix()]=digest(p)
    c=plan['confirmation_if_pass']
    write(RUN/'freeze_before_states.json',{'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'selected_variant':selected,
        'initialization_seeds':c['native_seeds'],'state_names':[f'demo_{i}' for i in c['demo_ids']],'noise_seeds':[70000+i for i in c['demo_ids']],
        'source_sha256':sources,'promotion_sha256':digest(RUN/'promotion.json'),'selection':'All registered starts retained; all five versions frozen before new state generation; no tuning after confirmation.'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();main(a.stage)
