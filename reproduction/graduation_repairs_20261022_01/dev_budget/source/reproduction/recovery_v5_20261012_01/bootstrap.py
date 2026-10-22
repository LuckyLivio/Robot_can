"""Create an isolated prospective v5 study, never modify older artifacts."""
import os,json,hashlib,datetime
from pathlib import Path
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False)
def main():
    files={}
    for folder,dirs,names in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in {'.venv','.git','.idea','__pycache__'} and Path(folder)/d!=RUN]
        for name in names:
            p=Path(folder)/name;files[p.relative_to(ROOT).as_posix()]=digest(p)
    write(RUN/'before_manifest.json',files)
    old=ROOT/'reproduction/recovery_v4_20261012_01'
    for name in ['setup_study.py','recovery_controller.py','evaluate_recovery.py','run_pilot.py','run_stage.py','verify_stage.py']:
        with (RUN/name).open('xb') as f:f.write((old/name).read_bytes())
    plan={'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protected_files':len(files),
      'scope':'v4 AB plus T: remaining-budget-aware timing and motion allocation; S: region-validated sideways grasp clearance. No learned weight changes and no paper.',
      'variants':['v2','v4','timing_only','side_only','v5'],
      'constraints':['400 steps, D2 FIFO never cleared, max two attempts for every variant','Current 23D observation/history plus fixed task geometry only inside controller','Reward, contacts, joints, success and simulator truth are post-hoc diagnostics only','Old v1/v2/v3/v4, all weights/results preserved byte-for-byte'],
      'known_data':'All prior data and confirmations are known. Pilot use of dev failure cases is diagnostic, not independent evidence.',
      'pilot_gate':['Both new flags off reproduce v4 actions and events','Preserve 8 natural, 2 forced loss, 1 held-stall, 4 old geometry and 3 v3-regression successes','Preserve v4 known corner successes','Timing must rescue a known 400-step budget failure without extending horizon','Sideways module must rescue at least one known side-wall fixture and reject invalid regions','Individual components and their combination evaluated separately'],
      'dev_gate':{'episodes_per_variant':240,'baseline_v4_successes':231,'baseline_v2_successes':229,'min_success':234,'max_regressions_vs_v4':2,'max_regressions_vs_v2':2,'min_distinct_rescued_starts_vs_v4':2,'min_bc_clean':29,'min_bc_noise':28,'min_full_clean':85,'min_full_noise':89},
      'selection':'Gate T,S,TS separately. Among passing candidates choose most successes, then fewer v4 regressions, then fewer components, then T before S. If none passes, do not run confirmation or lower gate. Retain every attempted revision.',
      'confirmation_if_pass':{'native_seeds':list(range(1026101201,1026101261)),'demo_ids':list(range(10000000,10000060)),
          'versions':['v2','v4','timing_only','side_only','v5'],'methods':['bc_k2','full_s17','full_s23','full_s41'],'noises':[0.,.05],'episodes':2400,'bootstrap_seed':20261016,'bootstrap_unit':'60 paired starts; Full fixed training seeds averaged within each start','freeze_before_generation':True},
      'stress_confirmation_if_pass':{'native_seeds':list(range(1026101401,1026101417)),'pose_seed':1126101401,'demo_ids':list(range(11000000,11000016)),
          'versions':['v2','v4','timing_only','side_only','v5'],'groups':['source_far_alignment','destination_corner','sideways_source','sideways_destination'],'per_group':4,'force_at':80,'proposal':'stationary_open','episodes':80,'separate_from_primary':True},
      'reporting':'Development is not independent evidence. Preserve failed pilots, revisions, and all registered starts. Report benefits of shared recovery separately from world-model quality. Confirmation may reject development promotion.'}
    write(RUN/'study_plan.json',plan);print(json.dumps({'protected_files':len(files),'directory':str(RUN),'dev_gate':plan['dev_gate']}))
if __name__=='__main__':main()
