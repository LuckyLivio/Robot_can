"""Final completeness / integrity audit, without rerunning any policy."""
import datetime,json
from pathlib import Path
from setup_study import ROOT,RUN,digest,write

def main():
    old=json.loads((RUN/'before_manifest.json').read_text())
    for name,sha in old.items():assert (ROOT/name).is_file() and digest(ROOT/name)==sha,name
    frozen=json.loads((RUN/'confirmation_lock.json').read_text())
    for name,sha in frozen['sha256'].items():assert digest(ROOT/name)==sha,name
    expected={'pilot_a':132,'pilot_b':132,'dev_a':720,'confirmation':2400,'geometry_confirmation':80}
    audited=[]
    for stage,count in expected.items():
        folder=RUN/stage;complete=json.loads((folder/('complete.json' if stage.startswith('pilot_') else 'completed.json')).read_text())
        assert complete.get('complete',complete.get('inputs_unchanged',False))
        if stage.startswith('pilot_'):
            for name,sha in complete['code_sha256'].items():assert digest(folder/'source'/name)==sha,(stage,name)
        jobs=json.loads((folder/'jobs.json').read_text());assert sum(len(j['demos']) for j in jobs)==count
        ids=[j['id'] for j in jobs];assert len(set(ids))==len(ids)
        found=[];steps=0
        for j in jobs:
            p=folder/j['id']/'result.json';r=json.loads(p.read_text());assert r['status']=='complete';assert r['job']==j
            assert [s['demo'] for s in r['records']]==j['demos']
            assert all(1<=s['steps']<=400 and s['attempts']<=2 and (s['success'] or s['steps']==400) for s in r['records'])
            steps+=sum(s['steps'] for s in r['records']);found.append(p.parent.name)
        assert set(found)==set(ids)
        # Source snapshots are part of the original experiment evidence.
        if stage!='pilot_a':
            for name in ['recovery_controller.py','clearance.py','gripper_geometry.json','variants.py','evaluate_recovery.py']:
                assert digest(folder/'source'/name)==digest(RUN/name),(stage,name)
        audited.append({'stage':stage,'new_episodes':count,'controls':steps,'jobs':len(jobs)})
    a=json.loads((RUN/'validation_pilot_a.json').read_text());b=json.loads((RUN/'pilot_acceptance.json').read_text())
    receipts=[json.loads((RUN/f'validation_{s}.json').read_text()) for s in ['dev_a','confirmation','geometry_confirmation']]
    assert a['passed'] and b['passed'] and all(r['passed'] for r in receipts)
    assert a['control_steps']==audited[0]['controls'] and b['verified_control_steps']==audited[1]['controls']
    for r,s in zip(receipts,audited[2:]):
        assert r['episodes']==s['new_episodes'] and r['controls_checked']==s['controls']
        for name,sha in r['outputs_sha256'].items():assert digest(RUN/name)==sha,name
    for stage in ['dev_a','confirmation','geometry_confirmation']:
        summary=json.loads((RUN/f'analysis_{stage}'/'summary.json').read_text())
        assert summary['new_executed_episodes']==expected[stage]
    fresh=json.loads((RUN/'confirmation_state_validation.json').read_text());assert fresh['passed'] and fresh['no_old_overlap']
    assert len(fresh['native_starts'])==60 and len(fresh['geometry_starts'])==16
    required=['ASSESSMENT.md','README.md','study_plan.json','DEVELOPMENT_NOTES.md','promotion_dev_a.json','freeze_before_states.json',
              'confirmation_state_validation.json','initialization_validation_amendment.json','serialized_model_diagnostic.json',
              'mechanism_physics.json','mechanism_tests_b.json','diagnosis_confirmation.json','diagnosis_geometry_confirmation.json',
              'mechanisms_confirmation.json','mechanisms_geometry_confirmation.json','posthoc_failure_review.json','stress_failure_physics.json','native_change_physics.json']
    for name in required:assert (RUN/name).is_file(),name
    files=[p for p in RUN.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_validation.json']
    physical=len(b['physical_replays'])+sum(len(r['physical_replays']) for r in receipts)+len(json.loads((RUN/'mechanism_physics.json').read_text())['results'])+len(json.loads((RUN/'stress_failure_physics.json').read_text())['results'])
    physical+=len(json.loads((RUN/'native_change_physics.json').read_text())['results'])
    payload={'passed':True,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_files_unchanged':len(old),
        'frozen_files_unchanged':len(frozen['sha256']),'stages':audited,'new_executed_episodes':sum(x['new_episodes'] for x in audited),
        'controls_checked':sum(x['controls'] for x in audited),'physical_replays':physical,'old_dev_baseline_runs_reused':480,
        'new_independent_native_starts':60,'new_independent_stress_starts':16,'all_episodes_at_most_400_steps':True,
        'paper_written':False,'output_files':len(files),'outputs_sha256':{p.relative_to(RUN).as_posix():digest(p) for p in files}}
    write(RUN/'delivery_validation.json',payload)
    print(json.dumps({k:v for k,v in payload.items() if k!='outputs_sha256'},indent=2))
if __name__=='__main__':main()
