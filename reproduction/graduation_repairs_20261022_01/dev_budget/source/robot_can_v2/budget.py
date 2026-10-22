"""Experimental, observation-only timing model layered on the frozen v5 controller.

The queue model is deliberately small: D2 OSC translation commands, measured
end-effector progress and fixed phase dwell times. It never predicts success or
reads simulator internals. A disabled feature delegates to v5 without changing
its action, event or debug output. The v5 geometry and transit-speed decisions
remain in force in both variants.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from reproduction.recovery_v5_20261012_01.recovery_controller import (
    Config as V5Config,
    RecoveryController as V5,
)


@dataclass(frozen=True)
class QueueBudgetConfig(V5Config):
    """Predetermined development constants; no episode/case-specific settings."""

    queue_budget: bool = True
    queue_speed_window: int = 8
    queue_speed_min_samples: int = 4
    queue_nominal_speed_m: float = .012
    queue_min_speed_m: float = .006
    queue_max_speed_m: float = .020
    queue_max_extra_steps: int = 50
    queue_max_credit_steps: int = 25


class QueueBudgetController(V5):
    """v5 recovery with an optional queue/progress-aware *timing* estimate.

    ``set_pending_actions`` takes the two commands that will execute before a
    newly selected command, in execution order. They are already available at
    the controller boundary and contain no privileged environment information.
    Call it before each ``choose``. Missing queue input is treated as unavailable
    (no projection), rather than inventing pending motion.

    An ordinary frozen v5 ``Config`` is also accepted; new calibration fields
    then use the defaults above. ``enabled=False`` retains v5's supervisor-off
    behavior. ``QueueBudgetConfig(queue_budget=False)`` gives the exact v5
    supervisor, including its original time-budget mechanism.
    """

    def __init__(self, target, floor_z, initial_state, enabled=True, config=None):
        super().__init__(target, floor_z, initial_state, enabled, config or QueueBudgetConfig())
        defaults = QueueBudgetConfig()
        self._calibration = {
            name: getattr(self.cfg, name, getattr(defaults, name))
            for name in defaults.__dataclass_fields__
            if name.startswith('queue_')
        }
        self._budget_active = bool(
            self.enabled and self.cfg.time_budget and self._calibration['queue_budget']
        )
        if self._budget_active:
            if (self.cfg.delay, self.cfg.horizon_steps, self.cfg.max_attempts) != (2, 400, 2):
                raise ValueError('Queue budget requires the frozen D2 / 400-step / 2-attempt contract')
            counts = [self._calibration['queue_speed_min_samples'], self._calibration['queue_speed_window']]
            if not (all(isinstance(value, (int, np.integer)) for value in counts)
                    and 1 <= counts[0] <= counts[1]):
                raise ValueError('Queue speed sample counts must satisfy 1 <= min_samples <= window')
            low = self._calibration['queue_min_speed_m']
            nominal = self._calibration['queue_nominal_speed_m']
            high = self._calibration['queue_max_speed_m']
            if not (np.isfinite([low, nominal, high]).all() and 0 < low <= nominal <= high <= .05):
                raise ValueError('Queue speed bounds must be finite and satisfy 0 < min <= nominal <= max <= .05')
            bounds = [self._calibration['queue_max_extra_steps'], self._calibration['queue_max_credit_steps']]
            if not (all(isinstance(value, (int, np.integer)) for value in bounds) and min(bounds) >= 0):
                raise ValueError('Queue timing adjustment bounds must be nonnegative')
        self._eef_history = deque(maxlen=int(self._calibration['queue_speed_window']) + 1)
        self._pending_actions = None
        self._v5_transit_pressure = False
        self._last_budget = None

    def set_pending_actions(self, queue):
        """Validate/copy D2 finite 7D commands; enforce the OSC action bounds."""
        pending = np.asarray(queue, dtype=float)
        if pending.shape != (2, 7) or not np.isfinite(pending).all():
            raise ValueError('Pending actions must have finite shape (2, 7) for D2')
        self._pending_actions = np.clip(pending, -1., 1.).copy()

    @property
    def pending_actions(self):
        """A defensive copy for inspection; callers cannot change our queue."""
        return None if self._pending_actions is None else self._pending_actions.copy()

    def _phase_schedule(self, x):
        """Known phase waypoints and mandatory dwell, used for timing only.

        These estimates do not replace v5's guarded grasp planner. Unplanned
        grasps use the same upright vertical offset as v5's initial default;
        after a recovery starts, its actual planned grasp/held offset is used.
        Movement and minimum phase age overlap, so each stage uses their max.
        """
        c = self.cfg
        eef = x[:3].astype(float)
        obj = x[16:19].astype(float)
        above = np.r_[obj[:2], max(self.floor + c.approach_clearance_m, obj[2] + .14)]
        retract = np.r_[eef[:2], max(self.floor + c.approach_clearance_m, eef[2])]
        grasp = self.grasp_target(x)
        offset = self.held_offset if self.held_offset is not None else grasp - obj
        lift = np.r_[obj[:2], self.floor + c.carry_clearance_m] + offset
        carry = np.r_[self.target[:2], self.floor + c.carry_clearance_m] + offset
        place = np.r_[self.target[:2], self.floor + .085] + offset
        probe = np.r_[grasp[:2], grasp[2] + c.lift_probe_m]
        release = eef if self.mode == 'RELEASE' else place
        retreat = release + np.array([0., 0., .12])
        stages = [
            ('RETRACT', retract, c.delay + 6),
            ('ALIGN', above, c.delay + 6),
            ('DESCEND', grasp, c.delay + 6),
            ('CLOSE', grasp, c.close_steps + c.delay),
            ('TEST_LIFT', probe, c.delay + c.stable_frames),
            ('LIFT', lift, c.delay + c.stable_frames),
            ('CARRY', carry, c.delay + 6),
            ('PLACE', place, c.delay + 6),
            ('RELEASE', release, 14 + c.delay),
            ('RETREAT', retreat, c.delay + 6),
        ]
        names = [stage[0] for stage in stages]
        start = names.index(self.mode) if self.mode in names else 0
        stages = stages[start:]
        if self.mode not in ('NORMAL', 'EXHAUSTED'):
            name, goal, dwell = stages[0]
            if self.waypoint is not None and name in ('RETRACT', 'TEST_LIFT', 'LIFT', 'RELEASE', 'RETREAT'):
                goal = np.asarray(self.waypoint, dtype=float).copy()
            stages[0] = (name, goal, max(0, dwell - self.age))
        return stages

    def _speed(self, goal, current):
        c = self._calibration
        # Use consecutive observations from this phase; a retry must not join
        # samples separated by a different phase into a fictitious fast step.
        recent = []
        for phase, point in reversed(self._eef_history):
            if phase != self.mode:
                break
            recent.append(point)
        recent.reverse()
        deltas = np.diff(np.asarray(recent, dtype=float), axis=0) if len(recent) > 1 else np.empty((0, 3))
        samples = len(deltas)
        if samples:
            # Clip *before* norm so even a corrupted very large finite sample
            # cannot overflow. Such a sample never makes us over-optimistic.
            bound = c['queue_max_speed_m']
            deltas = np.clip(deltas, -bound, bound)
            norms = np.linalg.norm(deltas, axis=1)
            deltas *= np.minimum(1., bound / np.maximum(norms, 1e-12))[:, None]
            motion = float(np.median(np.linalg.norm(deltas, axis=1)))
            direction = goal - current
            length = float(np.linalg.norm(direction))
            progress = float(np.median(deltas @ (direction / length))) if length > 1e-9 else motion
        else:
            motion = progress = 0.
        nominal = c['queue_nominal_speed_m']
        if samples < c['queue_speed_min_samples']:
            effective = nominal
        else:
            # Half nominal prevents a short stationary interval from asserting
            # zero mobility; signed progress makes oscillation/nonprogress slow.
            effective = .5 * nominal + .5 * max(0., progress)
        speed = float(np.clip(effective, c['queue_min_speed_m'], c['queue_max_speed_m']))
        projection_cap = float(np.clip(motion if samples else speed, c['queue_min_speed_m'], c['queue_max_speed_m']))
        return speed, projection_cap, samples, motion, progress

    def budget_diagnostics(self, x):
        """Return a finite auditable estimate without advancing controller time."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) < 23 or not np.isfinite(x).all():
            raise ValueError('Budget observation must contain at least 23 finite values')
        base = int(V5.remaining_schedule(self, x))
        if not self._budget_active:
            return {'enabled': False, 'base_steps': base, 'estimated_steps': base}
        stages = self._phase_schedule(x)
        current = x[:3].copy()
        next_goal = next((goal for _, goal, _ in stages if np.linalg.norm(goal-current) > .001), stages[0][1])
        speed, cap, samples, motion, progress = self._speed(next_goal, current)
        raw = 0.
        mandatory = 0
        for _, goal, dwell in stages:
            raw += max(float(dwell), float(np.linalg.norm(goal-current)) / speed)
            mandatory += int(dwell)
            current = goal
        # Preserve v5's second-attempt reserve while the grasp is unverified.
        retry = 50 if self.mode in ('NORMAL', 'EXHAUSTED', 'RETRACT', 'ALIGN', 'DESCEND', 'CLOSE', 'TEST_LIFT') else 0
        raw += retry
        bounded = float(np.clip(raw, max(0, base-self._calibration['queue_max_credit_steps']),
                                base+self._calibration['queue_max_extra_steps']))
        projection = np.zeros(3)
        if self._pending_actions is not None:
            # Fixed, verified OSC output_max translation scale. Rotations and
            # grip do not masquerade as translational progress.
            for action in self._pending_actions:
                delta = .05 * action[:3]
                norm = float(np.linalg.norm(delta))
                projection += delta * min(1., cap / max(norm, 1e-12))
        distance_before = float(np.linalg.norm(next_goal-x[:3]))
        distance_after = float(np.linalg.norm(next_goal-(x[:3]+projection)))
        queue_motion_steps = (distance_after-distance_before) / speed
        queue_wait_steps = self.cfg.delay if self._pending_actions is not None else 0
        estimate = int(np.clip(math.ceil(bounded+queue_wait_steps+queue_motion_steps), 0, self.cfg.horizon_steps))
        result = {
            'enabled': True, 'base_steps': base, 'estimated_steps': estimate,
            'phase_steps_unbounded': float(raw), 'phase_steps_bounded': bounded,
            'mandatory_phase_dwell_steps': mandatory, 'retry_reserve_steps': retry,
            'speed_m_per_step': speed, 'speed_sample_count': samples,
            'observed_motion_m_per_step': motion, 'observed_progress_m_per_step': progress,
            'queue_available': self._pending_actions is not None,
            'queue_projection_m': projection.tolist(), 'queue_motion_steps': float(queue_motion_steps),
            'queue_wait_steps': queue_wait_steps, 'next_waypoint_m': np.asarray(next_goal).tolist(),
            'remaining_horizon_steps': max(0, self.cfg.horizon_steps-self.clock),
            'pressure': bool(self.cfg.horizon_steps-self.clock < estimate),
        }
        self._last_budget = result
        return result

    def remaining_schedule(self, x):
        if not self._budget_active or self._v5_transit_pressure:
            return V5.remaining_schedule(self, x)
        return self.budget_diagnostics(x)['estimated_steps']

    def pose_action(self, x, goal, grip):
        # v5 also consults its time heuristic to select a transit step limit.
        # Keep that decision exactly frozen: this experiment changes timing.
        previous = self._v5_transit_pressure
        self._v5_transit_pressure = True
        try:
            return super().pose_action(x, goal, grip)
        finally:
            self._v5_transit_pressure = previous

    def start_attempt(self, reason, x, holding=False):
        timing = reason in ('early_stall_preserve_budget', 'descent_stalled_preserve_budget')
        estimate = self.budget_diagnostics(x) if self._budget_active and timing else None
        previous_events = len(self.events)
        super().start_attempt(reason, x, holding)
        if estimate is not None and len(self.events) > previous_events:
            self.events[-1]['queue_budget'] = estimate

    def choose(self, x, proposal, previous_action):
        if not self._budget_active:
            return super().choose(x, proposal, previous_action)
        state = np.asarray(x, dtype=float)
        self._eef_history.append((self.mode, state[:3].copy()))
        action, debug = super().choose(x, proposal, previous_action)
        debug['queue_budget'] = self.budget_diagnostics(state)
        return action, debug
