"""Verify frozen inputs, replay every controller/FIFO/noise step, and sample MuJoCo replay."""
import argparse,json,sys,datetime
from collections import deque,Counter
import numpy as np
from setup_study import ROOT,RUN,digest,write
sys.path.insert(0,str(ROOT))
from recovery_controller import RecoveryController,Config
from reproduction.recovery_20261012_01.recovery_controller import RecoveryController as V1Controller,Config as V1Config


from reproduction.recovery_v2_20261012_01.recovery_controller import RecoveryController as V2Controller,Config as V2Config


from variants import controller_class,VERSIONS


def audit(directory):
    result=json.loads((directory/'result.json').read_text());assert result['status']=='complete';job=result['job']
    rows={r['demo']:r for r in result['records']};seen=[];current=None;count=0;replays=[]
    cls,cfg=controller_class(job['version'])
    def finish(name,steps,last,controller,interventions,stable,modes):
        s=rows[name];assert s['steps']==steps and s['success']==last['success_after']
        assert s['success'] or steps==400
        assert s['attempts']==controller.attempts and s['events']==controller.events
        assert s['verified_regrasps']==controller.verified_regrasps and s['failed_attempts']==controller.failed_attempts
        assert s['intervention_steps']==interventions and s['stable_hold_steps']==stable and s['phases']==dict(modes)
        if s['success']:assert not earlier_success
    for line in (directory/'trace.jsonl').open():
        row=json.loads(line);name=row['demo'];x=np.asarray(row['state_before'],np.float32)
        if name!=current:
            if current is not None:finish(current,steps,last,controller,interventions,stable,modes)
            assert name not in seen;seen.append(name);current=name
            controller=cls([.1975,.4025,.8],.8,x,job['enabled'],cfg(**job.get('config',{})))
            pending=deque([np.zeros(7,np.float32) for _ in range(2)],maxlen=3)
            rng=np.random.default_rng(70000+int(name.split('_')[1]));previous=np.zeros(7,np.float32);before=x.copy()
            steps=0;interventions=0;stable=0;modes=Counter();earlier_success=False
            fault_left=0;fault_step=None
        else:earlier_success=earlier_success or last['success_after']
        steps+=1;count+=1;assert row['step']==steps<=400
        np.testing.assert_array_equal(before,x);np.testing.assert_array_equal(row['queue_before'],list(pending))
        if job.get('force_recovery_at')==steps:controller.start_attempt('diagnostic_forced_start',x)
        command,diag=controller.choose(x,np.asarray(row['original_proposal'],np.float32),previous)
        assert diag==row['recovery'],(job['id'],name,steps,'controller replay')
        if job.get('fault')=='open_after_verified_lift' and fault_step is None and diag['stable_hold'] and .89<x[18]<.96 and np.linalg.norm(x[16:18]-controller.target[:2])>.30:
            fault_step=steps;fault_left=12
        assert row['fault_injected']==(fault_left>0)
        if fault_left>0:command[6]=-1.;fault_left-=1
        np.testing.assert_allclose(command,row['submitted_action'],rtol=0,atol=1e-7)
        assert abs(command).max()<=1 and np.isfinite(command).all() and controller.attempts<=2
        if not job.get('fault_policy'):
            np.testing.assert_array_equal(row['original_proposal'],row['base_chunk'][row['original_selected']])
        if job['method']=='full':
            scores=np.array(row['original_scores']);raw=int(scores.argmax());chosen=2 if raw!=2 and scores[raw]-scores[2]<.02 else raw
            assert chosen==row['original_selected']
        else:assert row['original_selected']==2
        pending.append(np.asarray(row['submitted_action'],np.float32));action=pending.popleft()
        np.testing.assert_array_equal(action,row['executed_action_before_noise'])
        if job['noise']:action[:3]+=rng.normal(0,job['noise'],3)
        previous=np.clip(action,-1,1);np.testing.assert_array_equal(previous,row['executed_action'])
        before=np.asarray(row['state_after'],np.float32);assert np.isfinite(before).all()
        interventions+=diag['intervened'];stable+=diag['stable_hold'];modes[diag['mode']]+=1;last=row
    finish(current,steps,last,controller,interventions,stable,modes)
    assert seen==job['demos'] and len(seen)==len(rows)
    return {'job':job['id'],'episodes':len(rows),'controls':count}


def physics_samples(stage,jobs):
    import h5py
    from common import DATA,make_env,restore_demo,pack_obs
    selected=[]
    for version in VERSIONS:
        for noise in [0.,.05]:
            match=next((j for j in jobs if j['version']==version and j['noise']==noise and j['method']=='full'),None)
            if match is None:continue
            rows=json.loads((RUN/stage/match['id']/'result.json').read_text())['records']
            # Outcome-independent ordering within explicit diagnostic strata.
            row=next((r for r in rows if r['attempts'] and r['success']),next((r for r in rows if not r['success']),rows[0]))
            selected.append((match,row['demo']))
    if not selected:selected=[(jobs[0],jobs[0]['demos'][0]),(jobs[-1],jobs[-1]['demos'][0])]
    env=make_env(False);results=[]
    try:
        for job,name in selected:
            path=ROOT/job['dataset'] if job.get('dataset') else DATA
            with h5py.File(path) as f:obs=restore_demo(env,f['data'][name])
            n=0
            for line in (RUN/stage/job['id']/'trace.jsonl').open():
                r=json.loads(line)
                if r['demo']!=name:continue
                np.testing.assert_array_equal(pack_obs(obs),r['state_before'])
                obs,*_=env.step(np.asarray(r['executed_action'],np.float32));np.testing.assert_array_equal(pack_obs(obs),r['state_after'])
                assert bool(env._check_success())==r['success_after'];n+=1
            results.append({'job':job['id'],'demo':name,'steps':n,'max_state_error':0.})
    finally:env.close()
    return results


def main(stage,receipt):
    folder=RUN/stage;assert json.loads((folder/'completed.json').read_text())['inputs_unchanged']
    started=json.loads((folder/'started.json').read_text())
    for path,sha in started['sha256'].items():assert digest(ROOT/path)==sha,path
    if stage in ['confirmation','geometry_confirmation']:
        for path,sha in json.loads((RUN/'confirmation_lock.json').read_text())['sha256'].items():assert digest(ROOT/path)==sha,path
    before=json.loads((RUN/'before_manifest.json').read_text())
    for path,sha in before.items():assert (ROOT/path).is_file() and digest(ROOT/path)==sha,path
    jobs=json.loads((folder/'jobs.json').read_text());audits=[]
    for job in jobs:
        audits.append(audit(folder/job['id']));print(json.dumps(audits[-1]),flush=True)
    physics=physics_samples(stage,jobs)
    result={'passed':True,'stage':stage,'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'old_files_unchanged':len(before),'frozen_inputs_unchanged':True,'episodes':sum(a['episodes'] for a in audits),
          'controls_checked':sum(a['controls'] for a in audits),'audits':audits,'physical_replays':physics,
          'outputs_sha256':{p.relative_to(RUN).as_posix():digest(p) for p in folder.rglob('*') if p.is_file()}}
    write(RUN/(receipt or f'validation_{stage}.json'),result);print(json.dumps({k:v for k,v in result.items() if k not in ['audits','outputs_sha256']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--receipt');a=p.parse_args();main(a.stage,a.receipt)
