"""Fixed-geometry candidate checks; no simulator or joint state dependency."""
import json
from pathlib import Path
import numpy as np
_DATA=json.loads(Path(__file__).with_name('gripper_geometry.json').read_text())
PARTS=[(p['name'],np.array(p['vertices']),np.array(p['edges'],dtype=int)) for p in _DATA['parts']]
CELLS=[(-.09,.29,-.49,-.01)]+[(a,b,c,d) for a,b in [(-.09,.09),(.11,.29)] for c,d in [(.04,.27),(.29,.52)]]

def side_cell(position,axis,floor):
    center=np.asarray(position)-.01*axis
    extent=.05*np.abs(axis)+.025*np.sqrt(np.maximum(0.,1-axis**2))
    if abs(axis[2])>.35 or not floor+.012<=center[2]-extent[2]<=floor+.030:return None
    for cell in CELLS:
        a,b,c,d=cell
        if min(center[0]-extent[0]-a,b-center[0]-extent[0],center[1]-extent[1]-c,d-center[1]-extent[1])>=.001:return cell
    return None

def clearance_margin(goal,rotation,cell,floor,gap=.06):
    """Clip each collision convex hull below wall top, then test signed cell bounds.

    Edge intersections at wall height are included: testing vertices alone can
    miss a hull crossing a wall. The robot arm is not represented by this model.
    """
    a,b,c,d=cell;wall=floor+.103;worst=float('inf')
    for name,original,edges in PARTS:
        vertices=original.copy()
        if 'finger1' in name:vertices[:,1]+=(gap-.06)/2
        if 'finger2' in name:vertices[:,1]-=(gap-.06)/2
        world=vertices@rotation.T+goal
        worst=min(worst,float(world[:,2].min()-(floor+.021)))
        low=world[world[:,2]<=wall]
        ends=world[edges];cross=(ends[:,0,2]-wall)*(ends[:,1,2]-wall)<0
        pairs=ends[cross]
        if len(pairs):
            t=(wall-pairs[:,0,2])/(pairs[:,1,2]-pairs[:,0,2]);intersections=pairs[:,0]+t[:,None]*(pairs[:,1]-pairs[:,0]);low=np.concatenate([low,intersections])
        if len(low):worst=min(worst,float(np.min(np.c_[low[:,0]-a,b-low[:,0],low[:,1]-c,d-low[:,1]])))
    return worst
