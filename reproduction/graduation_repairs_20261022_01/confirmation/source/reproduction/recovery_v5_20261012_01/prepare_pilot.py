import json,sys
from setup_study import ROOT,RUN,write
sys.path.insert(0,str(ROOT))
from variants import config
def main():
    old=json.loads((ROOT/'reproduction/recovery_v4_20261012_01/pilot_jobs_b.json').read_text())
    templates=[j for j in old if j['version']=='v4']
    templates.append({'id':'v4_side_wall','method':'bc_k2','seed':None,'noise':0.,'enabled':True,'steps':400,
        'dataset':'reproduction/recovery_v3_20261012_01/wall_pilot_starts.hdf5','demos':['demo_6200000','demo_6200001','demo_6200002'],
        'fault_policy':'stationary_open','force_recovery_at':80})
    templates.append({'id':'v4_known_budget','method':'bc_k2','seed':None,'noise':.05,'enabled':True,'demos':['demo_124']})
    for seed,demos in [(17,['demo_97']),(23,['demo_128','demo_188']),(41,['demo_97'])]:
        templates.append({'id':f'v4_known_side_s{seed}','method':'full','seed':seed,'noise':0.,'enabled':True,'demos':demos})
    jobs=[]
    for v in ['v4','timing_only','side_only','v5']:
        for j in templates:jobs.append({**j,'id':v+j['id'][2:],'version':v,'config':config(v)})
    write(RUN/'pilot_jobs_a.json',jobs)
    print(json.dumps({'jobs':len(jobs),'episodes':sum(len(j['demos']) for j in jobs)}))
if __name__=='__main__':main()
