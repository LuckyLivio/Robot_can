"""Export fixed gripper geometry once; no episode outcomes used for this model."""
import json,sys
from pathlib import Path
import numpy as np,h5py
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from itertools import product
from setup_study import ROOT,RUN,write,digest
sys.path.insert(0,str(ROOT))
from common import DATA,make_env,restore_demo,pack_obs
def main():
    env=make_env(False);parts=[]
    try:
        with h5py.File(DATA) as f:obs=restore_demo(env,f['data']['demo_54'])
        robot=env.robots[0]
        fingers=robot.gripper['right'].joints
        for name,q in zip(fingers,[.03,-.03]):env.sim.data.set_joint_qpos(name,q)
        env.sim.forward();env._obs_cache.clear();env._get_observations(force_update=True);obs=env._get_observations(force_update=True)
        x=pack_obs(obs);body_rotation=Rotation.from_quat(x[3:7]).as_matrix()
        for g in range(env.sim.model.ngeom):
            name=env.sim.model.geom_id2name(g) or ''
            if 'gripper' not in name or 'collision' not in name:continue
            typ=int(env.sim.model.geom_type[g])
            if typ==7:
                mesh=int(env.sim.model.geom_dataid[g]);start=int(env.sim.model.mesh_vertadr[mesh]);n=int(env.sim.model.mesh_vertnum[mesh])
                local=env.sim.model.mesh_vert[start:start+n].copy()
            elif typ==6:local=np.array(list(product([-1,1],repeat=3)))*env.sim.model.geom_size[g]
            else:raise ValueError((name,typ))
            world=local@env.sim.data.geom_xmat[g].reshape(3,3).T+env.sim.data.geom_xpos[g]
            vertices=(world-x[:3])@body_rotation;hull=ConvexHull(vertices);points=vertices[hull.vertices]
            compact=ConvexHull(points);edges=sorted({tuple(sorted((int(a),int(b)))) for tri in compact.simplices for a,b in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]})
            parts.append({'name':name,'vertices':points.tolist(),'edges':edges,'bounds':[points.min(0).tolist(),points.max(0).tolist()]})
    finally:env.close()
    assert len(parts)==5
    assets=ROOT/'.venv/Lib/site-packages/robosuite/models/assets/grippers'
    paths=[assets/'panda_gripper.xml',assets/'meshes/panda_gripper/hand.stl',assets/'meshes/panda_gripper/finger.stl']
    write(RUN/'gripper_geometry.json',{'source':'Fixed Panda collision geometry, coordinates relative to grip-site position in observed EEF-body axes. Fingers fixed at +/-0.03m. No dynamics or outcome data.',
        'gap_m':.06,'parts':parts,'source_sha256':{p.relative_to(ROOT).as_posix():digest(p) for p in paths}})
    print(json.dumps([{'name':p['name'],'points':len(p['vertices']),'bounds':p['bounds']} for p in parts],indent=2))
if __name__=='__main__':main()
