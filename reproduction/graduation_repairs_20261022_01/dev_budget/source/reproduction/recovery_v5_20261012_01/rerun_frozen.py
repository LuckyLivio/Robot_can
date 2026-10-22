"""Repeat the existing frozen experiment into a fresh directory, never overwrite."""
import argparse,datetime,json,subprocess,sys
from setup_study import ROOT,RUN,digest
def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['native','geometry'],default='native');p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    assert 1<=a.workers<=16
    lock=json.loads((RUN/'confirmation_lock.json').read_text())
    def check():
        for name,sha in lock['sha256'].items():assert digest(ROOT/name)==sha,name
    check();name='repeat_'+a.suite+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    jobs='confirmation_jobs.json' if a.suite=='native' else 'geometry_jobs.json'
    subprocess.run([sys.executable,'-X','utf8','-B','-u',str(RUN/'run_stage.py'),'--stage',name,'--jobs',jobs,'--workers',str(a.workers)],cwd=ROOT,check=True)
    check()
    subprocess.run([sys.executable,'-X','utf8','-B','-u',str(RUN/'verify_stage.py'),'--stage',name],cwd=ROOT,check=True)
    check();print('Completed exact-state repeat:',RUN/name,flush=True)
    print('This repeat reuses locked starts and is not a new independent confirmation.',flush=True)
if __name__=='__main__':main()
