"""Post-hoc physical replay of four remaining v5 stress failures and one control."""
import json,sys
from collections import Counter
import h5py,numpy as np
from scipy.spatial.transform import Rotation
from setup_study import ROOT,RUN,write
sys.path.insert(0,str(ROOT))
from common import make_env,restore_demo,pack_obs
from variants import controller_class

def main():
    jobs=json.loads((RUN/'geometry_confirmation/jobs.json').read_text());env=make_env(False);out=[]
    cases=[('v5','demo_11000006'),('v5','demo_11000012'),('v5','demo_11000013'),('v5','demo_11000014'),('v4','demo_11000013')]
    try:
        for version,name in cases:
            j=next(j for j in jobs if j['version']==version);folder=RUN/'geometry_confirmation'/j['id']
            with h5py.File(ROOT/j['dataset']) as f:obs=restore_demo(env,f['data'][name])
            cls,cfg=controller_class(version);controller=cls([.1975,.4025,.8],.8,pack_obs(obs),config=cfg(**j['config']))
            previous=np.zeros(7,np.float32);contacts=Counter();phase_contacts=Counter();descents=[];phases=Counter();controls=0
            for line in (folder/'trace.jsonl').open():
                r=json.loads(line)
                if r['demo']!=name:continue
                x=pack_obs(obs);np.testing.assert_array_equal(x,r['state_before'])
                if r['step']==80:controller.start_attempt('diagnostic_forced_start',x)
                action,diag=controller.choose(x,np.asarray(r['original_proposal'],np.float32),previous)
                assert diag==r['recovery'];np.testing.assert_array_equal(action,r['submitted_action'])
                phase=diag['mode'];phases[phase]+=1
                if phase=='DESCEND':
                    angle=float(np.linalg.norm(Rotation.from_matrix(controller.goal_rotation@Rotation.from_quat(x[3:7]).as_matrix().T).as_rotvec()))
                    descents.append({'step':r['step'],'attempt':controller.attempts,'position_error_m':float(np.linalg.norm(controller.grasp_target(x)-x[:3])),
                        'angle_error_rad':angle,'observed_gap_m':float(abs(x[7]-x[8])),'planned_gap_m':controller.approach_gap})
                previous=np.asarray(r['executed_action'],np.float32);obs,*_=env.step(previous)
                np.testing.assert_array_equal(pack_obs(obs),r['state_after']);assert bool(env._check_success())==r['success_after'];controls+=1
                pairs=set()
                for c in env.sim.data.contact[:env.sim.data.ncon]:
                    bodies=[str(env.sim.model.body_id2name(int(env.sim.model.geom_bodyid[g]))) for g in [c.geom1,c.geom2]]
                    if any(b.startswith('bin') for b in bodies) and any('gripper' in b or 'robot0' in b for b in bodies):pairs.add(' / '.join(sorted(bodies)))
                contacts.update(pairs);phase_contacts.update(f'{phase}: {p}' for p in pairs)
            out.append({'version':version,'demo':name,'exact_physics_replay':True,'controls':controls,'success':bool(env._check_success()),'phases':dict(phases),
                'wall_robot_contact_boundaries':dict(contacts),'contacts_by_phase':dict(phase_contacts),'descent_observations':descents,'events':controller.events})
            print(json.dumps({k:v for k,v in out[-1].items() if k not in ['events','descent_observations']}) ,flush=True)
    finally:env.close()
    write(RUN/'stress_failure_physics.json',{'post_hoc':True,'selection':'Four v5 failures in the 16-start stress test, plus v4 for the sole initially feasible new side-plan failure. No tuning.',
        'measurement':'Physical contacts are sampled at 20 Hz control boundaries, not every physics substep. Diagnostics only, never inputs to control.','results':out})
if __name__=='__main__':main()
