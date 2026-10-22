"""Exact replay of known timing and side-wall cases; contacts are diagnostic only."""
import json,sys
from collections import Counter
import h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from setup_study import ROOT,RUN,write
sys.path.insert(0,str(ROOT))
from common import DATA,make_env,restore_demo,pack_obs
from variants import controller_class
def main():
    cases=[('v4_known_alignment','demo_73'),('timing_only_known_alignment','demo_73'),
        ('v4_known_budget','demo_124'),('timing_only_known_budget','demo_124'),
        ('v4_side_wall','demo_6200000'),('side_only_side_wall','demo_6200000'),
        ('v4_side_wall','demo_6200001'),('side_only_side_wall','demo_6200001')]
    env=make_env(False);results=[]
    try:
        for name,demo in cases:
            folder=RUN/'pilot_b'/name;payload=json.loads((folder/'result.json').read_text());j=payload['job']
            with h5py.File(ROOT/j['dataset'] if j.get('dataset') else DATA) as f:obs=restore_demo(env,f['data'][demo])
            cls,cfg=controller_class(j['version']);c=cls([.1975,.4025,.8],.8,pack_obs(obs),config=cfg(**j.get('config',{})))
            previous=np.zeros(7,np.float32);contacts=Counter();descent=0;controls=0;plans=[];last_plan=None;phases=[]
            for line in (folder/'trace.jsonl').open():
                r=json.loads(line)
                if r['demo']!=demo:continue
                x=pack_obs(obs);np.testing.assert_array_equal(x,r['state_before'])
                if j.get('force_recovery_at')==r['step']:c.start_attempt('diagnostic_forced_start',x)
                a,d=c.choose(x,np.array(r['original_proposal'],np.float32),previous)
                np.testing.assert_array_equal(a,r['submitted_action']);assert d==r['recovery']
                p=getattr(c,'side_plan',None)
                if p is not None and p!=last_plan:plans.append({'step':r['step'],**p});last_plan=p
                phases.append(d['mode']);previous=np.array(r['executed_action'],np.float32);obs,*_=env.step(previous)
                np.testing.assert_array_equal(pack_obs(obs),r['state_after']);assert bool(env._check_success())==r['success_after'];controls+=1
                if d['mode']=='DESCEND':
                    descent+=1;pairs=set()
                    for contact in env.sim.data.contact[:env.sim.data.ncon]:
                        bodies=[env.sim.model.body_id2name(int(env.sim.model.geom_bodyid[g])) for g in [contact.geom1,contact.geom2]]
                        if any(str(b).startswith('bin') for b in bodies) and any('gripper' in str(b) for b in bodies):pairs.add(' / '.join(sorted(str(b) for b in bodies)))
                    contacts.update(pairs)
            record=next(r for r in payload['records'] if r['demo']==demo)
            results.append({'job':name,'demo':demo,'success':record['success'],'steps':controls,'physics_replay_exact':True,'descent_steps':descent,
                'wall_gripper_contact_boundaries':dict(contacts),'side_plans':plans,'events':record['events'],'phases':phases})
    finally:env.close()
    write(RUN/'mechanism_physics.json',{'known_diagnostic_only':True,'measurement':'Contacts sampled at 20 Hz control boundaries, not each physics substep. Diagnostics never feed control.','results':results})
    mapping={'NORMAL':0,'RETRACT':1,'ALIGN':1,'DESCEND':2,'CLOSE':2,'TEST_LIFT':2,'LIFT':3,'CARRY':3,'PLACE':4,'RELEASE':4,'RETREAT':4,'EXHAUSTED':5}
    colors=['#afb7c0','#5e84bf','#b48145','#278f82','#8e6eae','#c2504b'];labels=['Base policy','Approach','Grasp / verify','Transport','Place / release','Exhausted']
    fig,ax=plt.subplots(figsize=(10,4.4))
    for i,r in enumerate(results[:4]):
        values=[mapping[p] for p in r['phases']];start=0
        for k in range(1,len(values)+1):
            if k==len(values) or values[k]!=values[start]:ax.barh(i,k-start,left=start,color=colors[values[start]],height=.55);start=k
        ax.text(r['steps']+3,i,'success' if r['success'] else 'unfinished',va='center',fontsize=9)
    ax.set_yticks(range(4),['v4 / case 73','Timing / case 73','v4 / case 124','Timing / case 124']);ax.invert_yaxis()
    ax.axvline(400,color='#333',ls='--');ax.set_xlim(0,465);ax.set_xlabel('Control steps (unchanged 400-step limit)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,labels)],loc='upper center',bbox_to_anchor=(.5,1.25),ncol=3,fontsize=8)
    ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(RUN/'timing_mechanism.png',dpi=300);fig.savefig(RUN/'timing_mechanism.svg');fig.savefig(RUN/'timing_mechanism.pdf');plt.close(fig)
    print(json.dumps([{k:v for k,v in r.items() if k not in ['events','phases']} for r in results],indent=2),flush=True)
if __name__=='__main__':main()
