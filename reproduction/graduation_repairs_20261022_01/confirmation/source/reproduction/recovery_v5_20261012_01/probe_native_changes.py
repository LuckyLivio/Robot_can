"""Post-hoc physical replay of one timing rescue, one side rescue and the v2 regression."""
import json,sys
from collections import Counter
import h5py,numpy as np
from setup_study import ROOT,RUN,write
sys.path.insert(0,str(ROOT))
from common import make_env,restore_demo,pack_obs
from variants import controller_class
def main():
    cases=[('v4_bc_k2_clean','demo_10000029'),('timing_only_bc_k2_clean','demo_10000029'),
        ('v4_full_s41_clean','demo_10000012'),('side_only_full_s41_clean','demo_10000012'),
        ('v2_full_s17_noise005','demo_10000034'),('v5_full_s17_noise005','demo_10000034')]
    env=make_env(False);results=[]
    try:
        for name,demo in cases:
            folder=RUN/'confirmation'/name;payload=json.loads((folder/'result.json').read_text());j=payload['job']
            with h5py.File(ROOT/j['dataset']) as f:obs=restore_demo(env,f['data'][demo])
            cls,cfg=controller_class(j['version']);c=cls([.1975,.4025,.8],.8,pack_obs(obs),config=cfg(**j['config']))
            previous=np.zeros(7,np.float32);contacts=Counter();phases=Counter();controls=0;plans=[];lastplan=None
            for line in (folder/'trace.jsonl').open():
                r=json.loads(line)
                if r['demo']!=demo:continue
                x=pack_obs(obs);np.testing.assert_array_equal(x,r['state_before'])
                action,d=c.choose(x,np.asarray(r['original_proposal'],np.float32),previous)
                np.testing.assert_array_equal(action,r['submitted_action']);assert d==r['recovery']
                plan=getattr(c,'side_plan',None)
                if plan is not None and plan!=lastplan:plans.append({'step':r['step'],**plan});lastplan=dict(plan)
                previous=np.asarray(r['executed_action'],np.float32);obs,*_=env.step(previous)
                np.testing.assert_array_equal(pack_obs(obs),r['state_after']);assert bool(env._check_success())==r['success_after']
                controls+=1;phases[d['mode']]+=1;pairs=set()
                for con in env.sim.data.contact[:env.sim.data.ncon]:
                    bodies=[str(env.sim.model.body_id2name(int(env.sim.model.geom_bodyid[g]))) for g in [con.geom1,con.geom2]]
                    if any(b.startswith('bin') for b in bodies) and any('gripper' in b or 'robot0' in b for b in bodies):pairs.add(d['mode']+': '+' / '.join(sorted(bodies)))
                contacts.update(pairs)
            record=next(r for r in payload['records'] if r['demo']==demo)
            results.append({'job':name,'demo':demo,'exact_physics_replay':True,'steps':controls,'success':record['success'],
                'phases':dict(phases),'contacts_by_phase':dict(contacts),'side_plans':plans,'events':record['events']})
            print(json.dumps({k:v for k,v in results[-1].items() if k!='events'}),flush=True)
    finally:env.close()
    write(RUN/'native_change_physics.json',{'post_hoc':True,'selection':'One representative execution of each distinct new v4-relative rescue start, plus the sole v2-relative regression. This selection is diagnostic, not an extra test set.',
        'measurement':'Contacts sampled at 20 Hz control boundaries, not every MuJoCo substep. Never fed to control.','results':results})
if __name__=='__main__':main()
