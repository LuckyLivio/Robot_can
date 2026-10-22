"""Run an immutable batch, snapshot inputs, and never overwrite an earlier stage."""
import argparse,concurrent.futures,datetime,json,subprocess,sys
from setup_study import ROOT,RUN,digest,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--jobs',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    out=RUN/a.stage;assert out.parent.resolve()==RUN.resolve();out.mkdir(exist_ok=False)
    source=out/'source';source.mkdir()
    for name in ['recovery_controller.py','evaluate_recovery.py','run_stage.py','clearance.py','gripper_geometry.json','variants.py']:(source/name).write_bytes((RUN/name).read_bytes())
    jobs=json.loads((RUN/a.jobs).read_text());write(out/'jobs.json',jobs)
    files=[ROOT/'common.py',ROOT/'world_model.py',ROOT/'data/splits.json',ROOT/'data/can_ph_lowdim_v15.hdf5',
           ROOT/'policy/bc_chunk_dev_v4/policy_15000.pt',ROOT/'reproduction/recovery_20261012_01/recovery_controller.py',ROOT/'reproduction/recovery_v2_20261012_01/recovery_controller.py',
           ROOT/'reproduction/extensions_20261012_01/evaluate.py',RUN/a.jobs]
    files.extend(RUN/n for n in ['recovery_controller.py','evaluate_recovery.py','run_stage.py','clearance.py','gripper_geometry.json','variants.py'])
    files.append(ROOT/'reproduction/recovery_v4_20261012_01/recovery_controller.py')
    files.extend(ROOT/f'models/multiseed_20261012_01_full_s{s}/model.pt' for s in [17,23,41])
    files.extend(ROOT/j['dataset'] for j in jobs if j.get('dataset'))
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in set(files)}
    if a.stage in ['confirmation','geometry_confirmation']:
        lock=json.loads((RUN/'confirmation_lock.json').read_text())
        for name,sha in lock['sha256'].items():assert digest(ROOT/name)==sha,name
    write(out/'started.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashes,
                             'episodes':sum(len(j['demos']) for j in jobs),'workers':a.workers})
    def work(job):
        cmd=[sys.executable,'-X','utf8','-B','-u',str(RUN/'evaluate_recovery.py'),'--job-file',str(out/'jobs.json'),
             '--job-id',job['id'],'--output-dir',str(out/job['id'])]
        with (out/f"{job['id']}.log").open('x',encoding='utf8') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        assert r.returncode==0,job['id']
        result=json.loads((out/job['id']/'result.json').read_text());assert result['status']=='complete'
        print(json.dumps({'job':job['id'],'success':sum(r['success'] for r in result['records']),'episodes':len(result['records'])}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(work,jobs))
    for name,sha in hashes.items():assert digest(ROOT/name)==sha,name
    write(out/'completed.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'inputs_unchanged':True,'episodes':sum(len(j['demos']) for j in jobs)})

if __name__=='__main__':main()
