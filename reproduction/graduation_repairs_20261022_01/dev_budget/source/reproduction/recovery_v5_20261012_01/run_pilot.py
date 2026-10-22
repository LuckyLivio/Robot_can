import argparse,concurrent.futures,json,subprocess,sys
from pathlib import Path
from setup_study import ROOT,RUN,digest,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--jobs',default='pilot_jobs_fixed.json');a=p.parse_args()
    out=RUN/a.revision;out.mkdir(exist_ok=False)
    source=out/'source';source.mkdir()
    for name in ['recovery_controller.py','evaluate_recovery.py','clearance.py','gripper_geometry.json','variants.py']:
        (source/name).write_bytes((RUN/name).read_bytes())
    jobs=json.loads((RUN/a.jobs).read_text())
    write(out/'jobs.json',jobs)
    def work(job):
        cmd=[sys.executable,'-X','utf8','-B','-u',str(RUN/'evaluate_recovery.py'),'--job-file',str(out/'jobs.json'),'--job-id',job['id'],'--output-dir',str(out/job['id'])]
        with (out/f"{job['id']}.log").open('x',encoding='utf-8') as log:
            completed=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:raise RuntimeError(job['id'])
        r=json.loads((out/job['id']/'result.json').read_text())
        print(json.dumps({'id':job['id'],'complete':r['status'],'success':sum(t['success'] for t in r['records']),
                          'episodes':len(r['records']),'attempts':[t['attempts'] for t in r['records']]}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(work,jobs))
    write(out/'complete.json',{'complete':True,'code_sha256':{p.name:digest(p) for p in source.glob('*.py')}})


if __name__=='__main__':main()
