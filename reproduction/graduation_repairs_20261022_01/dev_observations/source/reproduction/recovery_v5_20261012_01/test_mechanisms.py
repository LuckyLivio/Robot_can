import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from setup_study import ROOT,RUN,write,digest
sys.path.insert(0,str(ROOT))
from recovery_controller import RecoveryController,Config
from reproduction.recovery_v4_20261012_01.recovery_controller import RecoveryController as V4
from clearance import side_cell,clearance_margin
def main(receipt):
    path=ROOT/'reproduction/recovery_v4_20261012_01/dev_a/v4_bc_k2_clean/trace.jsonl'
    current=None;steps=0;episodes=0
    for line in path.open():
        r=json.loads(line);x=np.array(r['state_before'],np.float32)
        if r['demo']!=current:
            current=r['demo'];episodes+=1;previous=np.zeros(7,np.float32)
            old=V4([.1975,.4025,.8],.8,x);new=RecoveryController([.1975,.4025,.8],.8,x,config=Config(time_budget=False,side_clearance=False))
        proposal=np.array(r['original_proposal'],np.float32)
        a,da=old.choose(x,proposal,previous);b,db=new.choose(x,proposal,previous)
        np.testing.assert_array_equal(a,b);assert da==db and old.events==new.events
        previous=np.array(r['executed_action'],np.float32);steps+=1
    geometry=[]
    for name,pos,axis,valid in [('free',[.1,-.25,.845],[1,0,0],True),('side_cell',[.02,.38,.845],[1,0,0],True),
        ('divider',[.10,.38,.845],[1,0,0],False),('wall_top',[.28,.308,.94],[1,0,0],False),
        ('outside',[.296,.308,.845],[0,1,0],False),('overlap_endcap',[.13,.14,.845],[1,0,0],False)]:
        cell=side_cell(np.array(pos),np.array(axis),.8);assert (cell is not None)==valid,name
        geometry.append({'case':name,'valid':valid})
    down=np.diag([1.,-1.,-1.]);cell=(-.09,.29,-.49,-.01)
    assert clearance_margin(np.array([.1,-.25,.855]),down,cell,.8,.08)>0
    assert clearance_margin(np.array([.285,-.25,.855]),down,cell,.8,.08)<0
    # Record planning on already-known fixtures before any new physical rollout.
    plans=[]
    known=ROOT/'reproduction/recovery_v3_20261012_01/pilot_e/v2_side_wall/trace.jsonl'
    for line in known.open():
        r=json.loads(line)
        if r['step']!=80:continue
        x=np.array(r['state_before'],np.float32);c=RecoveryController([.1975,.4025,.8],.8,x);c.plan_grasp(x)
        plans.append({'demo':r['demo'],'plan':c.side_plan,'object':x[16:19].tolist(),'axis':c.grasp_axis.tolist()})
    write(RUN/receipt,{'passed':True,'fallback_episodes':episodes,'fallback_control_steps_exact':steps,'geometry_cases':geometry,'known_side_plans':plans,
        'source_sha256':{n:digest(RUN/n) for n in ['recovery_controller.py','clearance.py','gripper_geometry.json']}})
    print(json.dumps({'passed':True,'fallback_steps':steps,'known_side_plans':plans},indent=2))
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'mechanism_tests_a.json')
