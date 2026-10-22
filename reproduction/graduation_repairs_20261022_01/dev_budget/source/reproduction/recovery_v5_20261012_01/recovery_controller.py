"""Observation-only recovery supervisor. No simulator, reward, or model access."""
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation
try:
    from .clearance import side_cell,clearance_margin
except ImportError:
    from clearance import side_cell,clearance_margin


@dataclass(frozen=True)
class Config:
    delay: int = 2
    stable_frames: int = 5
    stall_steps: int = 40
    stall_progress_m: float = .005
    max_attempts: int = 2
    position_gain: float = .5
    max_delta_m: float = .035
    orientation_gain: float = .5
    max_rotation_rad: float = .15
    relative_tolerance_m: float = .012
    near_object_m: float = .09
    grip_gap_min_m: float = .025
    grip_gap_max_m: float = .067
    verified_lift_m: float = .02
    approach_clearance_m: float = .20
    carry_clearance_m: float = .28
    grasp_offset_z_m: float = .012
    close_steps: int = 16
    lift_probe_m: float = .06
    stage_timeout: int = 70
    minimum_normal_steps: int = 80
    long_stall_steps: int = 65
    progress_epsilon_m: float = .008
    recovery_budget_step: int = 160
    sideways_axis_z: float = .6
    sideways_grasp_z_m: float = .008
    alignment_switch: bool = True
    region_guard: bool = True
    alignment_stall_steps: int = 18
    alignment_progress_m: float = .003
    alignment_angle_weight_m: float = .08
    alternate_alignment_timeout: int = 110
    alternate_rotation_step_rad: float = .25
    corner_distance_m: float = .065
    corner_gap_m: float = .060
    corner_offset_m: float = .006
    corner_clearance_m: float = .001
    time_budget: bool = True
    side_clearance: bool = True
    horizon_steps: int = 400
    early_stall_steps: int = 25
    descent_stall_steps: int = 18
    safe_transit_delta_m: float = .05
    side_gap_m: float = .056
    side_margin_m: float = .001


class RecoveryController:
    def __init__(self, target, floor_z, initial_state, enabled=True, config=None):
        self.cfg=config or Config();self.target=np.asarray(target,float).copy();self.floor=float(floor_z)
        self.enabled=enabled;self.history=deque(maxlen=self.cfg.stall_steps+1)
        self.rest_z=float(initial_state[18]);self.mode='NORMAL';self.age=0;self.attempts=0
        self.stable=False;self.ever_stable=False;self.lost_frames=0;self.clock=0
        self.events=[];self.held_offset=None;self.waypoint=None;self.anchor_object_z=None
        self.failed_attempts=0;self.verified_regrasps=0;self.done_cooldown=0
        self.release_until=0
        self.goal_rotation=Rotation.from_quat([1.,0.,0.,0.]).as_matrix()
        self.planned_rotation=self.goal_rotation.copy();self.grasp_kind='upright'
        self.best_distance=float(np.linalg.norm(initial_state[:3]-initial_state[16:19]))
        self.progress_step=0;self.grasp_axis=np.array([0.,0.,1.])
        self.alignment_best=float('inf');self.alignment_progress_step=0
        self.alignment_deadline=self.cfg.stage_timeout
        self.direction_switched=False;self.alternate_attempt=False
        self.grasp_offset=np.zeros(2);self.approach_gap=None;self.corner_cell=None
        self.side_offset=np.zeros(3);self.side_cell=None;self.side_plan=None
        self.descent_best=float('inf');self.descent_progress_step=0

    def enter(self, mode, reason, x):
        self.events.append({'step':self.clock,'from':self.mode,'to':mode,'reason':reason,
                            'attempt':self.attempts,'object_position':x[16:19].tolist()})
        self.mode=mode;self.age=0
        if mode=='ALIGN':
            self.goal_rotation=self.planned_rotation.copy()
            self.alignment_best=float('inf');self.alignment_progress_step=self.clock
            self.alignment_deadline=self.cfg.stage_timeout
        if mode=='RETRACT':
            self.waypoint=np.array([x[0],x[1],max(self.floor+self.cfg.approach_clearance_m,x[2])])
        if mode=='CLOSE':self.anchor_object_z=float(x[18])
        if mode=='DESCEND':
            self.descent_best=float('inf');self.descent_progress_step=self.clock
        if mode=='TEST_LIFT':
            self.waypoint=np.array([x[16],x[17],x[2]+self.cfg.lift_probe_m])
        if mode=='LIFT':
            self.held_offset=x[:3].astype(float)-x[16:19]
            self.waypoint=np.array([x[16],x[17],self.floor+self.cfg.carry_clearance_m])+self.held_offset
        if mode=='RELEASE':self.waypoint=x[:3].astype(float).copy()
        if mode=='RETREAT':self.waypoint=x[:3].astype(float)+np.array([0.,0.,.12])

    def inside_goal(self, x):
        # Known target cell (half of the .39 x .49 bin), with a 1 cm margin.
        # A normal policy may release above the cell before the object lands.
        return bool(np.all(np.abs(x[16:18]-self.target[:2])<np.array([.0875,.1125])))

    def observable_hold(self, x, previous_action):
        c=self.cfg;gap=abs(float(x[7]-x[8]));relative=x[:3]-x[16:19]
        recent=list(self.history)[-c.stable_frames:]
        consistent=len(recent)==c.stable_frames and max(np.linalg.norm(h['relative']-relative) for h in recent)<c.relative_tolerance_m
        raised=float(x[18])>self.rest_z+c.verified_lift_m
        if self.mode=='TEST_LIFT' and self.anchor_object_z is not None:
            raised=float(x[18])>self.anchor_object_z+c.verified_lift_m
        return bool(c.grip_gap_min_m<gap<c.grip_gap_max_m and
                    np.linalg.norm(relative)<c.near_object_m and consistent and raised and
                    (previous_action[6]>0 or self.stable))

    def start_attempt(self, reason, x, holding=False):
        if self.attempts>=self.cfg.max_attempts:
            self.enter('EXHAUSTED','attempt_limit',x);return
        self.attempts+=1;self.stable=False;self.lost_frames=0
        if holding:
            self.goal_rotation=Rotation.from_quat(x[3:7]).as_matrix()
        else:
            self.plan_grasp(x)
            # First retract without rotating against a nearby wall or the can.
            self.goal_rotation=Rotation.from_quat(x[3:7]).as_matrix()
        self.enter('LIFT' if holding else 'RETRACT',reason,x)

    def plan_grasp(self,x):
        self.side_offset=np.zeros(3);self.side_cell=None;self.side_plan=None
        self.grasp_offset=np.zeros(2);self.approach_gap=None;self.corner_cell=None
        self.alternate_attempt=False
        self.grasp_axis=Rotation.from_quat(x[19:23]).as_matrix()[:,2]
        if abs(self.grasp_axis[2])<self.cfg.sideways_axis_z:
            # Finger closure is perpendicular to the horizontal cylinder axis.
            self.grasp_kind='sideways'
            yaw=float(np.arctan2(self.grasp_axis[1],self.grasp_axis[0]))
        else:
            self.grasp_kind='upright'
            if x[17]>.015:
                # Fixed inner faces, including the destination-bin dividers.
                xs=[-.09,.09,.11,.29];ys=[.04,.27,.29,.52]
            else:xs=[-.09,.29];ys=[-.49,-.01]
            clearance_x=min(abs(x[16]-wall) for wall in xs)
            clearance_y=min(abs(x[17]-wall) for wall in ys)
            yaw=np.pi/2 if clearance_y<clearance_x else 0.
        # Equivalent yaw+pi gives identical finger geometry; choose the shorter turn.
        down=Rotation.from_quat([1.,0.,0.,0.]).as_matrix()
        rotations=[Rotation.from_rotvec([0,0,yaw+k*np.pi]).as_matrix()@down for k in [0,1]]
        current=Rotation.from_quat(x[3:7]).as_matrix()
        self.planned_rotation=min(rotations,key=lambda r:np.linalg.norm(Rotation.from_matrix(r@current.T).as_rotvec()))
        if self.cfg.region_guard:self.plan_guarded_corner(x,current,down)
        if self.cfg.side_clearance and self.grasp_kind=='sideways':self.plan_side_clearance(x,current)

    def plan_side_clearance(self,x,current):
        axis=self.grasp_axis;cell=side_cell(x[16:19],axis,self.floor)
        if cell is None:return
        goal=self.grasp_target(x)
        original_margin=clearance_margin(goal,self.planned_rotation,cell,self.floor,.08)
        if original_margin>=self.cfg.side_margin_m:return
        # Search height, limited axial offset and a tilt away from nearby walls.
        # Keep the same nearest yaw branch; do not impose v3's global yaw sector.
        a,b,c,d=cell;center=x[16:19]-.01*axis
        normals=[np.array([1.,0.]),np.array([-1.,0.]),np.array([0.,1.]),np.array([0.,-1.])]
        options=[]
        for height in [self.cfg.sideways_grasp_z_m,.014,.020]:
            for shift in [0.,-.015,.015]:
                offset=shift*axis+np.array([0.,0.,height-self.cfg.sideways_grasp_z_m])
                target=goal+offset
                for away in normals:
                    for degree in [0.,10.,20.,30.,35.]:
                        rotation=Rotation.from_rotvec(np.deg2rad(degree)*np.r_[-away[1],away[0],0.]).as_matrix()@self.planned_rotation
                        margin=clearance_margin(target,rotation,cell,self.floor,self.cfg.side_gap_m)
                        if margin>=self.cfg.side_margin_m:
                            cost=.03*degree/35+2*(height-self.cfg.sideways_grasp_z_m)+.5*abs(shift)
                            options.append((cost,-margin,height,shift,degree,rotation,offset))
        if not options:
            self.side_plan={'eligible':True,'feasible':False,'baseline_margin_m':original_margin}
            return
        chosen=min(options,key=lambda p:p[:5]);_,negative_margin,height,shift,degree,rotation,offset=chosen
        self.planned_rotation=rotation;self.side_offset=offset;self.side_cell=cell;self.approach_gap=self.cfg.side_gap_m
        self.side_plan={'eligible':True,'feasible':True,'baseline_margin_m':original_margin,'margin_m':-negative_margin,
            'height_m':height,'axis_offset_m':shift,'tilt_deg':degree}

    def remaining_schedule(self,x):
        """Conservative step estimates to reserve time, not a success oracle."""
        above=np.r_[x[16:18],max(self.floor+self.cfg.approach_clearance_m,x[18]+.14)]
        approach=int(np.ceil(np.linalg.norm(above-x[:3])/.012))+8
        grasp=70;transport=int(np.ceil(np.linalg.norm(x[16:18]-self.target[:2])/.010))+10
        finish=40;retry_reserve=50
        if self.mode in ['CARRY','PLACE','RELEASE','RETREAT']:return transport+finish
        if self.mode=='LIFT':return 25+transport+finish
        return approach+grasp+transport+finish+retry_reserve

    def time_pressure(self,x):
        # Preserve the existing bounded fallback for airborne / wall-top /
        # outside objects. Faster transit is allowed for verified held objects.
        if not (self.stable and self.mode in ['LIFT','CARRY','PLACE']):
            axis=Rotation.from_quat(x[19:23]).as_matrix()[:,2]
            if self.resting_cell(x) is None and side_cell(x[16:19],axis,self.floor) is None:return False
        return self.cfg.horizon_steps-self.clock<self.remaining_schedule(x)

    def monitor_descent(self,x):
        if not self.cfg.time_budget or self.mode!='DESCEND':return
        error=float(np.linalg.norm(self.grasp_target(x)-x[:3]))
        if error<self.descent_best-.003:
            self.descent_best=error;self.descent_progress_step=self.clock
        if (self.age>=25 and self.clock-self.descent_progress_step>=self.cfg.descent_stall_steps
            and error>=.012 and self.time_pressure(x) and self.attempts<self.cfg.max_attempts):
            self.failed_attempts+=1;self.start_attempt('descent_stalled_preserve_budget',x)

    def resting_cell(self,x):
        """Signed footprint containment, not distance to the nearest wall.

        Upright can origin rests 6 cm above floor. Airborne / wall-top cans and
        side-lying cans deliberately retain v2 behavior. All bounds are known
        scene geometry, never queried from the simulator during control.
        """
        axis=Rotation.from_quat(x[19:23]).as_matrix()[:,2]
        if abs(axis[2])<.90 or not self.floor+.045<=x[18]<=self.floor+.075:return None
        cells=[(-.09,.29,-.49,-.01)]
        cells += [(lo,hi,bot,top) for lo,hi in [(-.09,.09),(.11,.29)] for bot,top in [(.04,.27),(.29,.52)]]
        for cell in cells:
            lo,hi,bot,top=cell
            if min(x[16]-lo,hi-x[16],x[17]-bot,top-x[17])>=.027:return cell
        return None

    def plan_guarded_corner(self,x,current,down):
        cell=self.resting_cell(x)
        if cell is None:return
        lo,hi,bot,top=cell
        clear=np.array([min(x[16]-lo,hi-x[16]),min(x[17]-bot,top-x[17])])
        if max(clear)>=self.cfg.corner_distance_m:return
        away=np.array([1. if x[16]<(lo+hi)/2 else -1.,1. if x[17]<(bot+top)/2 else -1.])
        offset=self.cfg.corner_offset_m*away;center=x[16:18]+offset
        bounds=np.array([min(center[0]-lo,hi-center[0]),min(center[1]-bot,top-center[1])])
        half=self.cfg.corner_gap_m/2+.017
        options=[]
        for yaw in np.arange(-np.pi/4,3*np.pi/4+1e-6,np.pi/12):
            extent=np.array([.010*abs(np.cos(yaw))+half*abs(np.sin(yaw)),.010*abs(np.sin(yaw))+half*abs(np.cos(yaw))])
            margin=float(min(bounds-extent))
            # Aperture must fit the can despite the lateral grasp offset.
            closure=np.array([-np.sin(yaw),np.cos(yaw)])
            if margin<self.cfg.corner_clearance_m or .025+abs(float(offset@closure))>self.cfg.corner_gap_m/2-.001:continue
            score=margin-.01*(1-abs(float(np.array([np.cos(yaw),np.sin(yaw)])@away))/np.sqrt(2))-.0001*abs(yaw)
            options.append((score,yaw))
        if not options:return
        yaw=max(options)[1]
        rotations=[Rotation.from_rotvec([0,0,yaw+k*np.pi]).as_matrix()@down for k in [0,1]]
        self.planned_rotation=min(rotations,key=lambda r:np.linalg.norm(Rotation.from_matrix(r@current.T).as_rotvec()))
        self.grasp_offset=offset;self.approach_gap=self.cfg.corner_gap_m;self.corner_cell=cell

    def monitor_alignment(self,x):
        if not self.cfg.alignment_switch or self.mode!='ALIGN':return
        goal=np.r_[self.grasp_target(x)[:2],max(self.floor+self.cfg.approach_clearance_m,x[18]+.14)]
        angle=np.linalg.norm(Rotation.from_matrix(self.goal_rotation@Rotation.from_quat(x[3:7]).as_matrix().T).as_rotvec())
        position=float(np.linalg.norm(goal-x[:3]))
        error=position+self.cfg.alignment_angle_weight_m*angle
        if error<self.alignment_best-self.cfg.alignment_progress_m:
            self.alignment_best=error;self.alignment_progress_step=self.clock
        settled=len(self.history)>=self.cfg.alignment_stall_steps and max(np.linalg.norm(h['object']-x[16:19]) for h in list(self.history)[-self.cfg.alignment_stall_steps:])<.008
        stalled=self.age>=24 and self.clock-self.alignment_progress_step>=self.cfg.alignment_stall_steps
        # A distant but still progressing approach must not consume its second
        # attempt at the old 70-step deadline before a stall can be observed.
        if self.age>=self.cfg.stage_timeout and not stalled and settled and self.resting_cell(x) is not None:
            self.alignment_deadline=self.cfg.alternate_alignment_timeout
        if (stalled and settled and (position>=.014 or angle>=.15) and not self.direction_switched
            and self.attempts<self.cfg.max_attempts and self.resting_cell(x) is not None):
            alternate=Rotation.from_rotvec([0,0,np.pi]).as_matrix()@self.goal_rotation
            self.failed_attempts+=1
            self.start_attempt('alignment_stalled_switch_equivalent',x)
            self.planned_rotation=alternate;self.direction_switched=True;self.alternate_attempt=True

    def grasp_target(self,x):
        goal=x[16:19].astype(float).copy()
        if self.grasp_kind=='sideways':
            # The mesh origin is 1 cm above the axial midpoint (-.06..+.04).
            axis=Rotation.from_quat(x[19:23]).as_matrix()[:,2]
            goal-=.01*axis;goal[2]+=self.cfg.sideways_grasp_z_m
            goal[2]=max(goal[2],self.floor+.035)
        else:goal[2]+=self.cfg.grasp_offset_z_m
        goal[:2]+=self.grasp_offset
        goal+=self.side_offset
        return goal

    def pose_action(self, x, goal, grip):
        if self.approach_gap is not None and self.mode in ['RETRACT','ALIGN','DESCEND']:
            grip=float(np.clip((abs(float(x[7]-x[8]))-self.approach_gap)*12.,-.2,.2))
        # OSC_POSE is verified to use world-frame deltas and scales .05 m/.5 rad.
        delta=self.cfg.position_gain*(np.asarray(goal)-x[:3])
        norm=np.linalg.norm(delta)
        limit=self.cfg.max_delta_m
        if (self.cfg.time_budget and self.time_pressure(x) and self.mode in ['RETRACT','ALIGN','LIFT','CARRY']
            and (self.mode in ['RETRACT','ALIGN'] or self.stable)):
            limit=self.cfg.safe_transit_delta_m
        if norm>limit:delta*=limit/norm
        current=Rotation.from_quat(x[3:7]).as_matrix()
        rot=self.cfg.orientation_gain*Rotation.from_matrix(self.goal_rotation@current.T).as_rotvec()
        size=np.linalg.norm(rot)
        rotation_limit=self.cfg.alternate_rotation_step_rad if self.alternate_attempt and self.mode=='ALIGN' else self.cfg.max_rotation_rad
        if size>rotation_limit:rot*=rotation_limit/size
        return np.clip(np.r_[delta/.05,rot/.5,grip],-1,1).astype(np.float32)

    def choose(self, x, proposal, previous_action):
        x=np.asarray(x,np.float32);self.clock+=1;self.age+=1;c=self.cfg
        relative=x[:3]-x[16:19];distance=float(np.linalg.norm(relative))
        xygoal=float(np.linalg.norm(x[16:18]-self.target[:2]))
        self.history.append({'relative':relative.copy(),'object':x[16:19].copy(),'distance':distance,'xygoal':xygoal})
        if distance<self.best_distance-c.progress_epsilon_m:
            self.best_distance=distance;self.progress_step=self.clock
        holding=self.observable_hold(x,previous_action)
        if holding:
            self.stable=True;self.ever_stable=True;self.lost_frames=0
        elif self.stable:self.lost_frames+=1
        if self.mode=='NORMAL' and self.ever_stable and self.inside_goal(x) and proposal[6]<0:
            self.release_until=self.clock+25
        intentional=self.mode in ['RELEASE','RETREAT'] or self.clock<self.release_until
        loss=self.stable and self.lost_frames>=c.stable_frames and not intentional
        debug={'mode_before':self.mode,'stable_hold':holding,'latched_hold':self.stable,
               'gap_m':abs(float(x[7]-x[8])),'eef_object_distance_m':distance,
               'goal_distance_xy_m':xygoal,'loss_detected':bool(loss),'attempts':self.attempts}
        if not self.enabled:return proposal.copy(),{**debug,'mode':self.mode,'intervened':False}
        reachable=bool(-.24<x[16]<.45 and -.57<x[17]<.57 and x[18]>self.floor-.04)
        if self.mode not in ['NORMAL','EXHAUSTED'] and not reachable:
            self.failed_attempts+=1;self.enter('EXHAUSTED','object_outside_recovery_workspace',x)
        if self.mode=='NORMAL':
            if loss and reachable:self.start_attempt('observed_grasp_loss',x)
            elif (c.time_budget and not holding and not intentional and reachable and not self.inside_goal(x)
                  and self.clock>=c.minimum_normal_steps and distance>.12 and self.time_pressure(x)
                  and len(self.history)>=c.early_stall_steps
                  and self.history[-c.early_stall_steps]['distance']-distance<c.progress_epsilon_m
                  and float(x[18]-self.history[-c.early_stall_steps]['object'][2])<c.verified_lift_m):
                self.start_attempt('early_stall_preserve_budget',x)
            elif (not holding and not intentional and reachable and not self.inside_goal(x)
                  and self.clock>=c.minimum_normal_steps
                  and (self.clock-self.progress_step>=c.long_stall_steps or self.clock>=c.recovery_budget_step)):
                reason='long_no_progress' if self.clock-self.progress_step>=c.long_stall_steps else 'recovery_budget_guard'
                self.start_attempt(reason,x)
            elif self.age>c.stall_steps and len(self.history)>c.stall_steps and not intentional and reachable and (holding or self.clock>=c.minimum_normal_steps):
                old=self.history[0]
                if holding:
                    progress=max(float(x[18]-old['object'][2]),old['xygoal']-xygoal)
                else:
                    progress=max(old['distance']-distance,float(x[18]-old['object'][2]))
                if progress<c.stall_progress_m and (distance<.12 or self.ever_stable):
                    self.start_attempt('held_stall' if holding else 'near_object_stall',x,holding)
        elif loss and self.mode in ['LIFT','CARRY','PLACE']:
            if self.inside_goal(x):self.enter('RELEASE','object_at_goal',x)
            else:
                self.failed_attempts+=1;self.start_attempt('observed_grasp_loss',x)
        if self.corner_cell is not None and self.mode in ['RETRACT','ALIGN','DESCEND'] and self.resting_cell(x)!=self.corner_cell:
            self.plan_grasp(x)
            if self.mode!='RETRACT':self.goal_rotation=self.planned_rotation.copy()
        self.monitor_alignment(x)
        self.monitor_descent(x)
        timeout=(c.alternate_alignment_timeout if self.alternate_attempt else self.alignment_deadline) if self.mode=='ALIGN' else c.stage_timeout
        if self.mode not in ['NORMAL','EXHAUSTED','CARRY'] and self.age>timeout:
            self.failed_attempts+=1;self.start_attempt('phase_timeout',x)
        if self.mode=='CARRY' and self.age>110:
            self.failed_attempts+=1;self.start_attempt('transport_timeout',x)
        phase=self.mode
        if phase in ['NORMAL','EXHAUSTED']:
            return proposal.copy(),{**debug,'mode':self.mode,'intervened':False,'attempts':self.attempts}
        if phase=='RETRACT':
            action=self.pose_action(x,self.waypoint,-1)
            if self.age>c.delay+5 and x[2]>=self.waypoint[2]-.012:self.enter('ALIGN','retracted',x)
        elif phase=='ALIGN':
            grasp=self.grasp_target(x)
            goal=np.r_[grasp[:2],max(self.floor+c.approach_clearance_m,x[18]+.14)]
            action=self.pose_action(x,goal,-1)
            angle=np.linalg.norm(Rotation.from_matrix(self.goal_rotation@Rotation.from_quat(x[3:7]).as_matrix().T).as_rotvec())
            if self.age>c.delay+5 and np.linalg.norm(x[:3]-goal)<.014 and angle<.15:self.enter('DESCEND','aligned_above_object',x)
        elif phase=='DESCEND':
            goal=self.grasp_target(x);action=self.pose_action(x,goal,-1)
            if self.age>c.delay+5 and np.linalg.norm(x[:3]-goal)<.009:self.enter('CLOSE','at_grasp_pose',x)
        elif phase=='CLOSE':
            action=self.pose_action(x,self.grasp_target(x),1)
            if self.age>=c.close_steps+c.delay:self.enter('TEST_LIFT','close_commands_executed',x)
        elif phase=='TEST_LIFT':
            action=self.pose_action(x,self.waypoint,1)
            if self.age>=c.delay+c.stable_frames and holding:
                self.verified_regrasps+=1;self.enter('LIFT','lift_follow_confirmed',x)
            elif self.age>=35:
                self.failed_attempts+=1;self.start_attempt('lift_probe_failed',x)
        elif phase=='LIFT':
            action=self.pose_action(x,self.waypoint,1)
            if x[18]>=self.floor+c.carry_clearance_m-.016 and holding:self.enter('CARRY','clearance_reached',x)
        elif phase=='CARRY':
            goal=np.r_[self.target[:2],self.floor+c.carry_clearance_m]+self.held_offset
            action=self.pose_action(x,goal,1)
            # 4.5 cm still leaves more than 5 cm to the narrow cell edge.
            if self.age>c.delay+5 and xygoal<.045:self.enter('PLACE','above_target',x)
        elif phase=='PLACE':
            goal=np.r_[self.target[:2],self.floor+.085]+self.held_offset
            action=self.pose_action(x,goal,1)
            if self.age>c.delay+5 and xygoal<.05 and x[18]<self.floor+.10:self.enter('RELEASE','at_release_height',x)
        elif phase=='RELEASE':
            action=self.pose_action(x,self.waypoint,-1)
            if self.age>=14+c.delay:self.enter('RETREAT','release_commands_executed',x)
        elif phase=='RETREAT':
            action=self.pose_action(x,self.waypoint,-1)
            if self.age>c.delay+5 and np.linalg.norm(x[:3]-self.waypoint)<.02:
                self.stable=False;self.lost_frames=0;self.history.clear();self.enter('NORMAL','placement_sequence_finished',x)
        else:raise RuntimeError(phase)
        return action,{**debug,'mode':phase,'next_mode':self.mode,'intervened':True,'attempts':self.attempts}
