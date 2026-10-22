"""Small-test gate precedes the full paired development batch."""
import argparse,json,sys
from setup_study import ROOT,RUN,write,digest
sys.path.insert(0,str(ROOT))
from variants import config
from verify_stage import audit,physics_samples
def main(revision):
    pilot=RUN/revision;assert json.loads((pilot/'complete.json').read_text())['complete']
    for name in ['recovery_controller.py','evaluate_recovery.py','clearance.py','gripper_geometry.json','variants.py']:
        assert (pilot/'source'/name).read_bytes()==(RUN/name).read_bytes(),name
    checks={};mechanisms={}
    for version in ['v4','timing_only','side_only','v5']:
        for suffix in ['train','loss','held_stall','old_geometry','full_noise','regression_6000049','regression_6000000','regression_6000005']:
            rows=json.loads((pilot/f'{version}_{suffix}/result.json').read_text())['records'];assert all(r['success'] for r in rows),(version,suffix)
            checks[version+'_'+suffix]=len(rows)
        natural=json.loads((pilot/f'{version}_train/result.json').read_text())['records']
        original=json.loads((ROOT/'reproduction/recovery_20261012_01/pilot_v3/bc_original/result.json').read_text())['records']
        assert all(not o['success'] or (r['success'] and r['attempts']==0 and r['steps']==o['steps']) for o,r in zip(original,natural))
        geometry=json.loads((pilot/f'{version}_new_geometry/result.json').read_text())['records']
        assert all(r['success'] for r in geometry[1:]),(version,'lost known corner/source success')
    for v in ['timing_only','v5']:
        successes=[]
        for suffix in ['known_alignment','known_budget']:
            r=json.loads((pilot/f'{v}_{suffix}/result.json').read_text())['records'][0]
            if r['success']:successes.append({'case':suffix,'steps':r['steps']})
        assert successes,(v,'no budget rescue');mechanisms[v+'_timing']=successes
    for v in ['side_only','v5']:
        rows=json.loads((pilot/f'{v}_side_wall/result.json').read_text())['records'];successful=[r['demo'] for r in rows if r['success']]
        assert successful,(v,'no side-wall rescue');mechanisms[v+'_side']=successful
    jobs=json.loads((pilot/'jobs.json').read_text());audits=[audit(pilot/j['id']) for j in jobs];physics=physics_samples(revision,jobs)
    write(RUN/'pilot_acceptance.json',{'passed':True,'revision':revision,'checks':checks,'mechanisms':mechanisms,
        'verified_episodes':sum(a['episodes'] for a in audits),'verified_control_steps':sum(a['controls'] for a in audits),'physical_replays':physics,
        'source_sha256':{n:digest(RUN/n) for n in ['recovery_controller.py','evaluate_recovery.py','clearance.py','gripper_geometry.json','variants.py']}})
    jobs=[]
    for v in ['timing_only','side_only','v5']:
        for j in json.loads((ROOT/'reproduction/recovery_v2_20261012_01/dev_jobs.json').read_text()):
            jobs.append({**j,'id':v+'_'+j['id'],'version':v,'config':config(v),
                'baseline_result_v2':f"reproduction/recovery_v2_20261012_01/dev_a/{j['id']}/result.json",
                'baseline_result_v4':f"reproduction/recovery_v4_20261012_01/dev_a/v4_{j['id']}/result.json"})
    write(RUN/'dev_jobs.json',jobs);print(json.dumps({'pilot_accepted':True,'episodes_verified':sum(a['episodes'] for a in audits),'dev_episodes':720,'mechanisms':mechanisms}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',default='pilot_b');a=p.parse_args();main(a.revision)
