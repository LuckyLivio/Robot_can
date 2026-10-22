"""Protect the previous project and define a training-only engineering pilot."""
import hashlib,json,os,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=Path(__file__).resolve().parent


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,obj):
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)


def main():
    files={}
    for folder,dirs,names in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in {'.venv','.git','.idea','__pycache__'} and Path(folder)/d!=RUN]
        for n in names:
            p=Path(folder)/n;files[p.relative_to(ROOT).as_posix()]=digest(p)
    write(RUN/'before_manifest.json',files)
    names=json.loads((ROOT/'data/splits.json').read_text())['train'][:8]
    plan={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'engineering_pilot_before_formal_registration',
          'training_demos':names,'constraints':['Preserve all old files','No training','D=2 real FIFO never cleared','400-step total limit',
                                             'Current 23D observation and fixed task goal only','No reward/contact/future in supervisor'],
          'pilot_pass':['Gripper sign and world-axis tests pass','Disabled supervisor exactly reproduces original BC/Full',
                        'At least one natural training stall or reproducible injected stall rescued to task success',
                        'At least one physically injected loss recovered to task success','Normal training successes preserved',
                        'No FIFO bypass, event chatter, unbounded retries or output overwrite'],
          'formal_if_pass':'30 dev starts x clean/noise0.05 x (fixed BC+recovery, Full seeds17/23/41+same recovery) = 240 episodes. Freeze implementation and thresholds first.',
          'confirmation':'No use of the previous 60 confirmation starts for tuning or new claims',
          'attribution':'A shared engineering recovery controller, not a learned world-model improvement'}
    write(RUN/'pilot_plan.json',plan)
    jobs=[]
    for enabled in [False,True]:
        jobs.append({'id':f'bc_{"recovery" if enabled else "original"}','method':'bc_k2','seed':None,'noise':0.,'enabled':enabled,'demos':names})
    for method,seed in [('bc_k2',None),('full',17)]:
        jobs.append({'id':f'{method}_disabled_noise','method':method,'seed':seed,'noise':.05,'enabled':False,'demos':names[:2]})
    jobs.append({'id':'bc_forced_loss','method':'bc_k2','seed':None,'noise':0.,'enabled':True,
                 'fault':'open_after_verified_lift','demos':names[:2]})
    write(RUN/'pilot_jobs.json',jobs)
    write(RUN/'pilot_code_v1.json',{p.name:digest(p) for p in RUN.glob('*.py')})
    print(json.dumps({'protected_files':len(files),'pilot_jobs':len(jobs),'training_demos':names}))


if __name__=='__main__':main()
