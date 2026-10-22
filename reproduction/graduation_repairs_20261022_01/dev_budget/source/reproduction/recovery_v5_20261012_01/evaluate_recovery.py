"""Run the original BC/Full policy plus an optional shared recovery supervisor."""
import argparse,json,sys,time
from collections import deque,Counter
from pathlib import Path
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'reproduction/extensions_20261012_01'))
import h5py,numpy as np,torch
from common import DATA,make_env,restore_demo,pack_obs,load_policy
from reproduction.extensions_20261012_01.evaluate import load_model,predict,sequences,select
from recovery_controller import RecoveryController,Config
from reproduction.recovery_20261012_01.recovery_controller import RecoveryController as V1Controller,Config as V1Config


from reproduction.recovery_v2_20261012_01.recovery_controller import RecoveryController as V2Controller,Config as V2Config


from variants import controller_class


def verify_control(env):
    arm=env.robots[0].part_controllers['right']
    assert arm.input_type=='delta' and arm.input_ref_frame=='world'
    assert np.array_equal(arm.output_max,[.05,.05,.05,.5,.5,.5])
    assert np.array_equal(arm.output_min,[-.05,-.05,-.05,-.5,-.5,-.5])
    assert env.control_freq==20


def run(job,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name in ['result.json','trace.jsonl']:
        if (out/name).exists():raise FileExistsError(out/name)
    torch.set_num_threads(2)
    policy=load_policy(ROOT/'policy/bc_chunk_dev_v4/policy_15000.pt')
    model=load_model(job['seed']) if job['method']=='full' else None
    cls,cfg_cls=controller_class(job['version'])
    cfg=cfg_cls(**job.get('config',{}));env=make_env(False);rows=[]
    payload={'status':'running','job':job,'records':rows,'input':'PRIVILEGED_CURRENT_STATE_AND_FIXED_TASK_GOAL',
             'supervisor_note':'Observation history only. Reward/contact/success are not used in choose(). No learned weight changes.'}
    dataset=ROOT/job['dataset'] if job.get('dataset') else DATA
    with (out/'result.json').open('x',encoding='utf-8') as result,(out/'trace.jsonl').open('x',encoding='utf-8') as trace,h5py.File(dataset) as data:
        def save(status):
            payload['status']=status;result.seek(0);json.dump(payload,result,indent=2,allow_nan=False);result.truncate();result.flush()
        save('running')
        try:
            for name in job['demos']:
                obs=restore_demo(env,data['data'][name]);verify_control(env)
                x0=pack_obs(obs);target=env.target_bin_placements[env.object_to_id['can']].copy()
                supervisor=cls(target,env.bin2_pos[2],x0,job.get('enabled',True),cfg)
                pending=deque([np.zeros(7,np.float32) for _ in range(2)],maxlen=3)
                rng=np.random.default_rng(70000+int(name.split('_')[1]));previous=np.zeros(7,np.float32)
                peak=0.;ever_contact=False;success=False;interventions=0;stable_steps=0;decisions=[];normal_choices=[];modes=Counter()
                fault_left=0;fault_step=None;start=time.perf_counter()
                for step in range(job.get('steps',400)):
                    x=pack_obs(obs);base=policy.best(torch.from_numpy(x[None]))[0].numpy();tic=time.perf_counter()
                    if model is None:proposal=base[2].copy();selected=2;scores=None
                    else:
                        seq=sequences(list(pending),base[:4],8);_,_,scores=predict(model,x,seq)
                        _,selected=select(scores,[0,1,2,3]);proposal=base[selected].copy()
                    if job.get('fault_policy')=='stationary_open':proposal=np.array([0,0,0,0,0,0,-1],np.float32)
                    if job.get('fault_policy')=='stationary_after_lift' and (fault_step is not None or supervisor.ever_stable):
                        if fault_step is None:fault_step=step+1
                        proposal=np.array([0,0,0,0,0,0,1],np.float32)
                    if job.get('force_recovery_at')==step+1:
                        supervisor.start_attempt('diagnostic_forced_start',x)
                    action,diagnostic=supervisor.choose(x,proposal,previous)
                    if job.get('fault')=='open_after_verified_lift' and fault_step is None and diagnostic['stable_hold'] and .89<x[18]<.96 and np.linalg.norm(x[16:18]-target[:2])>.30:
                        fault_step=step+1;fault_left=12
                    injected=fault_left>0
                    if injected:action[6]=-1.;fault_left-=1
                    decisions.append(time.perf_counter()-tic);normal_choices.append(selected)
                    row={'demo':name,'step':step+1,'state_before':x.tolist(),'base_chunk':base.tolist(),
                         'queue_before':np.asarray(pending).tolist(),'original_proposal':proposal.tolist(),
                         'original_selected':selected,'original_scores':None if scores is None else scores.tolist(),
                         'submitted_action':action.tolist(),'recovery':diagnostic,'fault_injected':injected}
                    interventions+=diagnostic['intervened'];stable_steps+=diagnostic['stable_hold'];modes[diagnostic['mode']]+=1
                    pending.append(action.copy());executed=pending.popleft();row['executed_action_before_noise']=executed.copy().tolist()
                    if job['noise']:executed[:3]+=rng.normal(0,job['noise'],3)
                    applied=np.clip(executed,-1,1);previous=applied.copy()
                    obs,*_=env.step(applied);stages=env.staged_rewards();success=bool(env._check_success())
                    peak=max(peak,float(max(stages)));ever_contact=bool(ever_contact or stages[1]>.3)
                    row.update(state_after=pack_obs(obs).tolist(),executed_action=applied.tolist(),stages_after=np.asarray(stages).tolist(),
                               success_after=success,decision_ms=decisions[-1]*1000)
                    trace.write(json.dumps(row,allow_nan=False)+'\n')
                    if success:break
                summary={'demo':name,'success':success,'steps':step+1,'peak_stage_reward':peak,'ever_grasped':ever_contact,
                         'contact_note':'ever_grasped is the legacy bilateral-contact criterion, not stable holding',
                         'stable_hold_observed':supervisor.ever_stable,'stable_hold_steps':int(stable_steps),
                         'intervention_steps':int(interventions),'intervention_fraction':interventions/(step+1),
                         'attempts':supervisor.attempts,'verified_regrasps':supervisor.verified_regrasps,
                         'failed_attempts':supervisor.failed_attempts,'phases':dict(modes),'events':supervisor.events,
                         'fault_step':fault_step,'mean_decision_ms':float(np.mean(decisions)*1000),
                         'original_selection_counts':dict(Counter(normal_choices)),'elapsed_s':time.perf_counter()-start}
                rows.append(summary);trace.flush();save('running');print(json.dumps(summary),flush=True)
            save('complete')
        finally:env.close()
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-file',required=True);p.add_argument('--job-id');p.add_argument('--output-dir')
    args=p.parse_args();job=json.loads(Path(args.job_file).read_text(encoding='utf-8'))
    if args.job_id:job=next(j for j in job if j['id']==args.job_id)
    run(job,args.output_dir or RUN/'evaluations'/job['id'])
