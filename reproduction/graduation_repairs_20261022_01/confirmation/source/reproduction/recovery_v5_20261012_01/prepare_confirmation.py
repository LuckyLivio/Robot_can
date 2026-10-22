"""Prospective held-out native starts and a separate geometry challenge, after freeze."""
import datetime,importlib.metadata,json,sys,hashlib
from dataclasses import asdict
import numpy as np,h5py
from scipy.spatial.transform import Rotation
from setup_study import ROOT,RUN,digest,write
sys.path.insert(0,str(ROOT))
from common import DATA,make_env,restore_demo,pack_obs
from reproduction.confirmation_20261012_01.prepare_states import native_reset
from recovery_controller import Config
from variants import config,VERSIONS


def main():
    design=json.loads((RUN/'freeze_before_states.json').read_text());assert json.loads((RUN/'promotion.json').read_text())['passed']
    for name,sha in design['source_sha256'].items():assert digest(ROOT/name)==sha,name
    challenge={'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'initialization_seeds':list(range(1026101401,1026101417)),'pose_rng_seed':1126101401,
        'state_names':[f'demo_{11000000+i}' for i in range(16)],
        'groups':['source_far_alignment']*4+['destination_corner']*4+['sideways_source']*4+['sideways_destination']*4,
        'evaluation':'All five frozen recovery controllers, fixed BC, clean execution. Force the same recovery start at step80 with stationary/open base proposal; mechanism test only, excluded from primary task-success totals.',
        'selection':'All 16 predetermined starts retained. Initial pose changes and settling happen before evaluation. No teleporting during control.'}
    write(RUN/'geometry_confirmation_design.json',challenge)
    with h5py.File(DATA) as f:
        previous=[d['states'][0] for d in f['data'].values()];meta=f['data'].attrs['env_args']
    old_files=sorted(p for p in (ROOT/'reproduction').rglob('*starts*.hdf5') if RUN not in p.parents)
    for path in old_files:
        with h5py.File(path) as f:
            previous.extend(d['states'][0] for d in f['data'].values())
    old=np.stack(previous);env=make_env(False);env2=make_env(False);description=[]
    try:
        with h5py.File(RUN/'confirmation_starts.hdf5','x') as f:
            group=f.create_group('data');group.attrs['env_args']=meta;hashes=set()
            for seed,name,noise_seed in zip(design['initialization_seeds'],design['state_names'],design['noise_seeds']):
                state,obs=native_reset(env,seed);state_hash=hashlib.sha256(state.tobytes()).hexdigest()
                assert state_hash not in hashes;hashes.add(state_hash)
                assert np.isfinite(state).all() and np.isfinite(obs).all() and not env._check_success()
                distance=float(np.max(abs(old-state),axis=1).min());assert distance>1e-8
                d=group.create_group(name);d.attrs['model_file']=env.sim.model.get_xml();d.attrs['initialization_seed']=seed
                d.create_dataset('states',data=state[None]);d.create_dataset('initial_observation',data=obs)
                noise=np.random.default_rng(noise_seed).normal(0,.05,(400,3));d.create_dataset('execution_noise',data=noise)
                description.append({'name':name,'seed':seed,'noise_seed':noise_seed,'state_sha256':state_hash,
                    'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),'nearest_old_state_max_abs_distance':distance})
        regeneration=[]
        with h5py.File(RUN/'confirmation_starts.hdf5') as f:
            for item in reversed(description):
                state,obs=native_reset(env2,item['seed']);d=f['data'][item['name']]
                np.testing.assert_array_equal(state,d['states'][0]);np.testing.assert_array_equal(obs,d['initial_observation'][:])
                restored=pack_obs(restore_demo(env,d));error=float(np.max(abs(restored-obs)));assert error<1e-5
                regeneration.append({'name':item['name'],'native_regeneration_exact':True,'serialized_xml_observation_error':error})
        # Additional geometry starts use different native seeds and independently drawn poses.
        rng=np.random.default_rng(challenge['pose_rng_seed']);geometry=[]
        with h5py.File(RUN/'geometry_starts.hdf5','x') as f:
            group=f.create_group('data');group.attrs['env_args']=meta
            for i,(seed,name,kind) in enumerate(zip(challenge['initialization_seeds'],challenge['state_names'],challenge['groups'])):
                native_reset(env2,seed)
                if kind=='sideways_source':position=[rng.uniform(-.01,.18),rng.uniform(-.34,-.15),.89];pitch=90.;yaw=rng.uniform(-180,180)
                elif kind=='source_far_alignment':position=[rng.uniform(.20,.25),rng.uniform(-.43,-.35),.88];pitch=0.;yaw=rng.uniform(-180,180)
                elif kind=='destination_corner':position=[rng.uniform(.147,.158),rng.uniform(.076,.087),.88];pitch=0.;yaw=rng.uniform(-180,180)
                else:position=[rng.uniform(-.015,.02),rng.uniform(.14,.18),.89];pitch=90.;yaw=rng.uniform(-180,180)
                quat=(Rotation.from_euler('z',yaw,degrees=True)*Rotation.from_euler('y',pitch,degrees=True)).as_quat()
                obj=next(o for o in env2.objects if o.name.lower()=='can');joint=obj.joints[0]
                env2.sim.data.set_joint_qpos(joint,np.r_[position,quat[3],quat[:3]]);env2.sim.data.set_joint_qvel(joint,np.zeros(6));env2.sim.forward()
                for _ in range(40):obs,*_=env2.step(np.array([0,0,0,0,0,0,-1],np.float32))
                state=env2.sim.get_state().flatten().copy();state[0]=0.
                d=group.create_group(name);d.attrs['model_file']=env2.sim.model.get_xml();d.attrs['initialization_seed']=seed;d.attrs['kind']=kind
                initial_obs=pack_obs(obs);state_hash=hashlib.sha256(state.tobytes()).hexdigest()
                distance=float(np.max(abs(old-state),axis=1).min());assert distance>1e-8
                assert np.isfinite(state).all() and not env2._check_success()
                d.create_dataset('states',data=state[None]);d.create_dataset('initial_observation',data=initial_obs)
                geometry.append({'name':name,'kind':kind,'initialization_seed':seed,'position':initial_obs[16:19].tolist(),'quaternion':initial_obs[19:23].tolist(),
                    'sampled_position':position,'sampled_quaternion_xyzw':quat.tolist(),'state_sha256':state_hash,'nearest_old_state_max_abs_distance':distance})
        assert len({g['state_sha256'] for g in geometry})==16
        assert not ({g['state_sha256'] for g in geometry}&{g['state_sha256'] for g in description})
        with h5py.File(RUN/'geometry_starts.hdf5') as f:
            for item in reversed(geometry):
                native_reset(env,item['initialization_seed']);q=np.array(item['sampled_quaternion_xyzw'])
                obj=next(o for o in env.objects if o.name.lower()=='can');joint=obj.joints[0]
                env.sim.data.set_joint_qpos(joint,np.r_[item['sampled_position'],q[3],q[:3]])
                env.sim.data.set_joint_qvel(joint,np.zeros(6));env.sim.forward()
                for _ in range(40):obs,*_=env.step(np.array([0,0,0,0,0,0,-1],np.float32))
                state=env.sim.get_state().flatten().copy();state[0]=0.;d=f['data'][item['name']]
                np.testing.assert_array_equal(state,d['states'][0]);np.testing.assert_array_equal(pack_obs(obs),d['initial_observation'][:])
                restored=pack_obs(restore_demo(env2,d));error=float(np.max(abs(restored-d['initial_observation'][:])));assert error<1e-5
                item['regeneration_exact']=True;item['serialized_xml_observation_error']=error
    finally:env.close();env2.close()
    write(RUN/'confirmation_state_validation.json',{'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'old_initial_states_compared':len(old),'old_start_files':[p.relative_to(ROOT).as_posix() for p in old_files],'native_starts':description,'regeneration':regeneration,'geometry_starts':geometry,
        'no_old_overlap':True,'no_policy_used_or_outcome_filtering':True})
    jobs=[]
    # Version order is fixed before launch. All use exactly the same starts and noise.
    for version in VERSIONS:
        for noise in [0.,.05]:
            for method,seed in [('bc_k2',None),('full',17),('full',23),('full',41)]:
                tag=f'{version}_{method}'+(f'_s{seed}' if seed else '')+('_clean' if not noise else '_noise005')
                jobs.append({'id':tag,'version':version,'method':method,'seed':seed,'noise':noise,'enabled':version!='disabled',
                    'dataset':(RUN/'confirmation_starts.hdf5').relative_to(ROOT).as_posix(),'demos':design['state_names'],
                    'steps':400,'config':config(version)})
    write(RUN/'confirmation_jobs.json',jobs)
    gjobs=[{'id':f'{v}_geometry','version':v,'method':'bc_k2','seed':None,'noise':0.,'enabled':True,'steps':400,
            'dataset':(RUN/'geometry_starts.hdf5').relative_to(ROOT).as_posix(),'demos':challenge['state_names'],
            'fault_policy':'stationary_open','force_recovery_at':80,'config':config(v)} for v in VERSIONS]
    write(RUN/'geometry_jobs.json',gjobs)
    hashes=dict(design['source_sha256'])
    for p in [RUN/'confirmation_starts.hdf5',RUN/'geometry_starts.hdf5',RUN/'confirmation_jobs.json',RUN/'geometry_jobs.json',
              RUN/'geometry_confirmation_design.json',RUN/'prepare_confirmation.py',RUN/'freeze_before_states.json',RUN/'analyze_stage.py']:
        hashes[p.relative_to(ROOT).as_posix()]=digest(p)
    write(RUN/'confirmation_lock.json',{'locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashes,
           'native_episodes':2400,'additional_geometry_episodes':80,'packages':{n:importlib.metadata.version(n) for n in ['numpy','torch','robosuite','mujoco','scipy','h5py']}})
    print(json.dumps({'starts_verified':60,'old_overlap':0,'primary_episodes':2400,'additional_geometry_episodes':80}),flush=True)

if __name__=='__main__':main()
