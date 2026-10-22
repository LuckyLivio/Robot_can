"""Finish preflight with separate native and serialized-XML environments.

The frozen first generator reused a serialized-XML environment for native
geometry regeneration. This validator repairs that separation only. It never
edits any generated state, seed, pose, controller, threshold, or model weight.
"""
import datetime,hashlib,importlib.metadata,json,sys
import h5py,numpy as np
from scipy.spatial.transform import Rotation
from setup_study import ROOT,RUN,write,digest
sys.path.insert(0,str(ROOT))
from common import DATA,make_env,restore_demo,pack_obs
from reproduction.confirmation_20261012_01.prepare_states import native_reset
from variants import config,VERSIONS

def poses(design):
    rng=np.random.default_rng(design['pose_rng_seed']);out=[]
    for kind in design['groups']:
        if kind=='sideways_source':position=[rng.uniform(-.01,.18),rng.uniform(-.34,-.15),.89];pitch=90.;yaw=rng.uniform(-180,180)
        elif kind=='source_far_alignment':position=[rng.uniform(.20,.25),rng.uniform(-.43,-.35),.88];pitch=0.;yaw=rng.uniform(-180,180)
        elif kind=='destination_corner':position=[rng.uniform(.147,.158),rng.uniform(.076,.087),.88];pitch=0.;yaw=rng.uniform(-180,180)
        else:position=[rng.uniform(-.015,.02),rng.uniform(.14,.18),.89];pitch=90.;yaw=rng.uniform(-180,180)
        q=(Rotation.from_euler('z',yaw,degrees=True)*Rotation.from_euler('y',pitch,degrees=True)).as_quat();out.append((position,q))
    return out

def settled(env,seed,position,q):
    native_reset(env,seed);obj=next(o for o in env.objects if o.name.lower()=='can');joint=obj.joints[0]
    env.sim.data.set_joint_qpos(joint,np.r_[position,q[3],q[:3]]);env.sim.data.set_joint_qvel(joint,np.zeros(6));env.sim.forward()
    for _ in range(40):obs,*_=env.step(np.array([0,0,0,0,0,0,-1],np.float32))
    state=env.sim.get_state().flatten().copy();state[0]=0.
    return state,pack_obs(obs)

def main():
    frozen=json.loads((RUN/'freeze_before_states.json').read_text());plan=json.loads((RUN/'study_plan.json').read_text())
    for name,sha in frozen['source_sha256'].items():assert digest(ROOT/name)==sha,name
    input_paths=[RUN/'confirmation_starts.hdf5',RUN/'geometry_starts.hdf5',RUN/'geometry_confirmation_design.json']
    original_hashes={p.relative_to(ROOT).as_posix():digest(p) for p in input_paths}
    design=json.loads((RUN/'geometry_confirmation_design.json').read_text());samples=poses(design)
    # These two native environments are never reset from serialized XML.
    native=make_env(False);native2=make_env(False);serialized=make_env(False);primary=[];geometry=[];regeneration=[]
    old_files=sorted(p for p in (ROOT/'reproduction').rglob('*starts*.hdf5') if RUN not in p.parents)
    with h5py.File(DATA) as f:old=[d['states'][0] for d in f['data'].values()]
    for path in old_files:
        with h5py.File(path) as f:old.extend(d['states'][0] for d in f['data'].values())
    old=np.stack(old)
    try:
        with h5py.File(RUN/'confirmation_starts.hdf5') as f:
            for seed,name,noise_seed in reversed(list(zip(frozen['initialization_seeds'],frozen['state_names'],frozen['noise_seeds']))):
                state,obs=native_reset(native,seed);d=f['data'][name]
                np.testing.assert_array_equal(state,d['states'][0]);np.testing.assert_array_equal(obs,d['initial_observation'][:])
                noise=np.random.default_rng(noise_seed).normal(0,.05,(400,3));np.testing.assert_array_equal(noise,d['execution_noise'][:])
                distance=float(np.max(abs(old-state),axis=1).min());assert distance>1e-8 and not native._check_success()
                restored=pack_obs(restore_demo(serialized,d));error=float(np.max(abs(restored-obs)));assert error<1e-5
                primary.append({'name':name,'seed':seed,'noise_seed':noise_seed,'state_sha256':hashlib.sha256(state.tobytes()).hexdigest(),
                    'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),'nearest_old_state_max_abs_distance':distance})
                regeneration.append({'name':name,'native_regeneration_exact':True,'serialized_xml_observation_error':error})
        with h5py.File(RUN/'geometry_starts.hdf5') as f:
            for i in reversed(range(16)):
                name=design['state_names'][i];seed=design['initialization_seeds'][i];position,q=samples[i];d=f['data'][name]
                state,obs=settled(native2,seed,position,q)
                # Never accept only an approximate state match for generation.
                np.testing.assert_array_equal(state,d['states'][0]);np.testing.assert_array_equal(obs,d['initial_observation'][:])
                distance=float(np.max(abs(old-state),axis=1).min());assert distance>1e-8 and not native2._check_success()
                # env.step's last sensor values precede the final integration's
                # forward kinematics. Compare restored state observations only
                # after the same forward/observable refresh used by restore_demo.
                native2.sim.forward();native2._obs_cache.clear();native2._get_observations(force_update=True)
                canonical=pack_obs(native2._get_observations(force_update=True))
                restored=pack_obs(restore_demo(serialized,d))
                np.testing.assert_array_equal(serialized.sim.get_state().flatten(),state)
                raw_error=float(np.max(abs(restored-obs)))
                error=float(np.max(abs(restored-canonical)));assert error<1e-5
                geometry.append({'name':name,'kind':design['groups'][i],'seed':seed,'sampled_position':position,'sampled_quaternion_xyzw':q.tolist(),
                    'position':obs[16:19].tolist(),'quaternion':obs[19:23].tolist(),'state_sha256':hashlib.sha256(state.tobytes()).hexdigest(),
                    'nearest_old_state_max_abs_distance':distance,'regeneration_exact':True,'serialized_state_exact':True,
                    'raw_step_observation_error':raw_error,'canonical_native_observation':canonical.tolist(),
                    'canonical_serialized_observation':restored.tolist(),'serialized_xml_observation_error':error})
            # Reproduce the old mixed-model failure on the first previously
            # failing geometry case, then compare native static model arrays.
            i=15;position,q=samples[i];seed=design['initialization_seeds'][i];d=f['data'][design['state_names'][i]]
            mixed_state,mixed_obs=settled(serialized,seed,position,q)
            mixed_error=float(np.max(abs(mixed_state-d['states'][0])))
            model_differences={name:float(np.max(abs(np.asarray(getattr(native2.sim.model,name))-np.asarray(getattr(serialized.sim.model,name))))) for name in ['body_pos','body_quat','body_mass','body_inertia','geom_pos','geom_quat','geom_size']}
            native_state,native_obs=settled(native2,seed,position,q);np.testing.assert_array_equal(native_state,d['states'][0])
    finally:native.close();native2.close();serialized.close()
    assert len({i['state_sha256'] for i in primary+geometry})==76
    for name,sha in original_hashes.items():assert digest(ROOT/name)==sha,name
    amendment={'registered_before_any_policy_evaluation_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reason':'Two validation initialization errors: native regeneration reused a serialized XML model; env.step observations were compared with restored-state observations without first refreshing forward kinematics. Separate native models and identical sim.forward plus double observable refresh resolve both.',
        'repair_scope':'Validation environment separation and observation refresh only; all 76 existing generated states and every frozen source remain byte-identical. All policies still use the unchanged restore_demo and identical saved state. No policy outputs inspected; no filtering or seed changes.',
        'original_generator_failure_max_state_error':.80964663,'reproduced_mixed_model_error':mixed_error,'native_vs_serialized_model_max_differences':model_differences,
        'max_raw_geometry_observation_error':max(x['raw_step_observation_error'] for x in geometry),
        'max_canonical_geometry_observation_error':max(x['serialized_xml_observation_error'] for x in geometry),
        'unchanged_generated_files':original_hashes,'validator_sha256':digest(RUN/'complete_state_validation.py')}
    write(RUN/'initialization_validation_amendment.json',amendment)
    write(RUN/'confirmation_state_validation.json',{'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'old_initial_states_compared':len(old),
        'old_start_files':[p.relative_to(ROOT).as_posix() for p in old_files],'native_starts':list(reversed(primary)),'regeneration':regeneration,'geometry_starts':list(reversed(geometry)),
        'no_old_overlap':True,'no_policy_used_or_outcome_filtering':True,'validation_amendment':'initialization_validation_amendment.json'})
    jobs=[]
    for v in VERSIONS:
        for noise in [0.,.05]:
            for method,seed in [('bc_k2',None),('full',17),('full',23),('full',41)]:
                tag=f'{v}_{method}'+(f'_s{seed}' if seed else '')+('_clean' if not noise else '_noise005')
                jobs.append({'id':tag,'version':v,'method':method,'seed':seed,'noise':noise,'enabled':True,'dataset':(RUN/'confirmation_starts.hdf5').relative_to(ROOT).as_posix(),
                    'demos':frozen['state_names'],'steps':400,'config':config(v)})
    write(RUN/'confirmation_jobs.json',jobs)
    gjobs=[{'id':f'{v}_geometry','version':v,'method':'bc_k2','seed':None,'noise':0.,'enabled':True,'steps':400,
        'dataset':(RUN/'geometry_starts.hdf5').relative_to(ROOT).as_posix(),'demos':design['state_names'],'fault_policy':'stationary_open','force_recovery_at':80,'config':config(v)} for v in VERSIONS]
    write(RUN/'geometry_jobs.json',gjobs)
    hashes=dict(frozen['source_sha256'])
    for p in [*input_paths,RUN/'confirmation_jobs.json',RUN/'geometry_jobs.json',RUN/'complete_state_validation.py',RUN/'initialization_validation_amendment.json',
        RUN/'freeze_before_states.json',RUN/'confirmation_state_validation.json']:hashes[p.relative_to(ROOT).as_posix()]=digest(p)
    write(RUN/'confirmation_lock.json',{'locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashes,'native_episodes':2400,'additional_geometry_episodes':80,
        'packages':{n:importlib.metadata.version(n) for n in ['numpy','torch','robosuite','mujoco','scipy','h5py']}})
    print(json.dumps({'passed':True,'native_starts':60,'geometry_starts':16,'existing_state_files_unchanged':True,'old_states_compared':len(old),'primary_episodes':2400,'additional_geometry_episodes':80,'amendment':amendment}),flush=True)
if __name__=='__main__':main()
