import argparse,json
from collections import Counter
from setup_study import ROOT,RUN,write
def main(stage):
    jobs=json.loads((RUN/stage/'jobs.json').read_text());groups={};seen=set();work=[(j,RUN/stage/j['id']) for j in jobs]
    if stage.startswith('dev'):
        for j in jobs:
            for v in ['v2','v4']:
                path=j['baseline_result_'+v]
                if path not in seen:seen.add(path);work.append(({**j,'version':v,'id':(ROOT/path).parent.name},(ROOT/path).parent))
    for j,folder in work:
        payload=json.loads((folder/'result.json').read_text());rows=payload['records'];last={}
        for line in (folder/'trace.jsonl').open():
            r=json.loads(line);last[r['demo']]=r
        g=groups.setdefault(j['version'],{'episodes':0,'successes':0,'failures':[],'timing_cases':[],'event_reasons':Counter(),'final_failure_modes':Counter()})
        for r in rows:
            g['episodes']+=1;g['successes']+=r['success'];g['event_reasons'].update(e['reason'] for e in r['events'])
            t=last[r['demo']];x=t['state_after'];mode=t['recovery'].get('next_mode',t['recovery']['mode'])
            rec={'job':j['id'],'demo':r['demo'],'success':r['success'],'steps':r['steps'],'final_mode':mode,'final_object_position':x[16:19],
                'final_object_quaternion':x[19:23],'ever_stable':r['stable_hold_observed'],'verified_regrasps':r['verified_regrasps'],'events':r['events']}
            if any('preserve_budget' in e['reason'] for e in r['events']):g['timing_cases'].append(rec)
            if not r['success']:g['failures'].append(rec);g['final_failure_modes'][mode]+=1
    write(RUN/f'diagnosis_{stage}.json',{'stage':stage,'groups':groups,'limitation':'Final phases identify observed termination state, not proven physical cause; contact claims require physics probes.'})
    print(json.dumps({v:{k:g[k] for k in ['episodes','successes','final_failure_modes']} for v,g in groups.items()},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True);a=p.parse_args();main(a.stage)
