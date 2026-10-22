"""Check the retained first pilot using its own controller and geometry source."""
import importlib.util,json,sys
from setup_study import ROOT,RUN,write,digest
import verify_stage
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def main():
    folder=RUN/'pilot_a';saved=json.loads((folder/'complete.json').read_text())['code_sha256']
    for n,h in saved.items():assert digest(folder/'source'/n)==h,n
    load('clearance',folder/'source/clearance.py');module=load('v5_archived_pilot_a',folder/'source/recovery_controller.py')
    base=verify_stage.controller_class
    verify_stage.controller_class=lambda v:base(v) if v in ['v2','v4'] else (module.RecoveryController,module.Config)
    jobs=json.loads((folder/'jobs.json').read_text());audits=[verify_stage.audit(folder/j['id']) for j in jobs]
    result={'passed':True,'revision':'pilot_a','source_sha256':saved,'episodes':sum(a['episodes'] for a in audits),'control_steps':sum(a['controls'] for a in audits),'audits':audits}
    write(RUN/'validation_pilot_a.json',result);print(json.dumps({k:v for k,v in result.items() if k!='audits'}),flush=True)
if __name__=='__main__':main()
