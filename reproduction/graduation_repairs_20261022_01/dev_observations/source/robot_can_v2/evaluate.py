"""Explicit contracts and frozen-weight factorial recovery evaluation.

The benchmark terminates at first original success or the registered horizon.
Optional settling observations are diagnostic only and saved separately.
"""
import argparse
from collections import deque, Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from common import ROOT, DATA, load_policy, make_env, pack_obs, restore_demo
from world_model import WorldModel, score
from reproduction.recovery_v5_20261012_01.recovery_controller import RecoveryController, Config
from robot_can_v2.observations import pack_observation


def validate_job(job):
    if job.get('method') not in ('bc', 'full'):
        raise ValueError('method must be bc or full')
    if job.get('controller') not in ('none', 'v5', 'queue_budget'):
        raise ValueError('controller must be none, v5 or queue_budget')
    if job.get('semantics', 'legacy') not in ('legacy', 'canonical'):
        raise ValueError('Unknown observation semantics')
    if job.get('steps', 400) != 400 or job.get('delay', 2) != 2:
        raise ValueError('This frozen recovery study requires exactly 400 steps and D2')
    cfg = Config(**job.get('config', {}))
    if cfg.delay != job.get('delay', 2) or cfg.horizon_steps != job.get('steps', 400):
        raise ValueError('Controller delay/horizon conflicts with actual FIFO or episode budget')
    if cfg.max_attempts != 2 or cfg.stable_frames != 5:
        raise ValueError('Study must retain two attempts and five holding confirmation frames')
    if job.get('method') == 'full' and job.get('semantics', 'legacy') != 'legacy':
        raise ValueError('Historical WM was trained on legacy inputs; canonical requires matched new WM')
    names = job.get('demos', [])
    if not names or len(set(names)) != len(names):
        raise ValueError('demos must be nonempty and unique')
    if not np.isfinite(job.get('noise', 0.)) or job.get('noise', 0.) < 0:
        raise ValueError('noise must be finite and nonnegative')
    if job.get('settle_steps', 20) != 20:
        raise ValueError('Diagnostic settling length is preregistered at 20 controls')
    return cfg


def load_world(seed):
    path = ROOT / f'models/multiseed_20261012_01_full_s{seed}/model.pt'
    ck = torch.load(path, map_location='cpu', weights_only=True)
    c = ck['config']
    model = WorldModel(c['dim'], c['width'], c['future_only'])
    model.load_state_dict(ck['state_dict'])
    return model.eval()


def pending_sequences(pending, commands, horizon=8):
    prefix = np.asarray(pending, np.float32)
    if prefix.shape != (2, 7):
        raise ValueError('Expected actual D2 queue')
    return np.asarray([np.concatenate((prefix, np.repeat(a[None], horizon, axis=0)))[:horizon]
                       for a in commands], np.float32)


@torch.inference_mode()
def propose(model, x, base, pending):
    if model is None:
        return base[2].copy(), 2, None
    seqs = pending_sequences(pending, base[:4])
    states = torch.from_numpy(np.repeat(x[None], 4, axis=0))
    _, labels = model(states, torch.from_numpy(seqs))
    scores = score(labels, .5).numpy()
    raw = int(scores.argmax())
    selected = 2 if raw != 2 and float(scores[raw] - scores[2]) < .02 else raw
    return base[selected].copy(), selected, scores


def timing_summary(values):
    v = np.asarray(values, float)
    return {'n': len(v), 'p50_ms': float(np.percentile(v, 50)),
            'p95_ms': float(np.percentile(v, 95)), 'p99_ms': float(np.percentile(v, 99)),
            'max_ms': float(v.max()), 'over_50ms': int((v > 50).sum()),
            'over_50ms_fraction': float(np.mean(v > 50))}


def settling_check(env, obs, pending, rng, noise, trace, demo):
    """20 extra diagnostics, outside benchmark; preserve in-flight FIFO commands."""
    inside, speeds = [], []
    previous = pack_obs(obs)[16:19].copy()
    for k in range(20):
        pending.append(np.array([0., 0., 0., 0., 0., 0., -1.], np.float32))
        executed = pending.popleft().copy()
        if noise:
            executed[:3] += rng.normal(0, noise, 3)
        obs, *_ = env.step(np.clip(executed, -1, 1))
        position = pack_obs(obs)[16:19]
        velocity = float(np.linalg.norm(position - previous) * 20.)
        success = bool(env._check_success())
        inside.append(success); speeds.append(velocity); previous = position.copy()
        trace.write(json.dumps({'demo': demo, 'diagnostic_step': k + 1,
                                'object_position': position.tolist(), 'speed_m_s': velocity,
                                'benchmark_success': success}) + '\n')
    return {'observed': True, 'diagnostic_controls': 20, 'all_in_goal': all(inside),
            'last10_max_speed_m_s': max(speeds[-10:]),
            'stable': bool(all(inside) and max(speeds[-10:]) < .02),
            'scope': 'Post-termination open-gripper diagnostic; excluded from benchmark steps and success denominator.'}


def run(job, output):
    cfg = validate_job(job)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing existing output directory: {output}')
    policy_path = ROOT / job.get('policy', 'policy/bc_chunk_dev_v4/policy_15000.pt')
    ck = torch.load(policy_path, map_location='cpu', weights_only=True)
    actual_semantics = ck['config'].get('observation_semantics', 'legacy')
    if actual_semantics != job.get('semantics', 'legacy'):
        raise ValueError('Policy observation semantics mismatch')
    dataset = ROOT / job['dataset'] if job.get('dataset') else DATA
    with h5py.File(dataset, 'r') as f:
        if any(name not in f['data'] for name in job['demos']):
            raise ValueError('Requested demo missing from dataset')
    torch.set_num_threads(2)
    policy = load_policy(policy_path)
    model = load_world(job['seed']) if job['method'] == 'full' else None
    output.mkdir(parents=True, exist_ok=False)
    payload = {'status': 'running', 'job': job, 'records': [], 'controller_config': asdict(cfg),
               'policy_sha256': hashlib.sha256(policy_path.read_bytes()).hexdigest(),
               'input': 'PRIVILEGED_23D_STATE_AND_FIXED_TARGET',
               'timing_scope': 'Complete observation packing + BC + optional WM + recovery; physics/JSON outside timer. Parallel load can affect timings.',
               'stability_scope': '20 diagnostic controls after first benchmark success; never added to benchmark steps or independent samples.'}
    result_path = output / 'result.json'
    def save(status):
        payload['status'] = status
        result_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    save('running')
    env = None
    try:
        env = make_env(False)
        arm = env.robots[0].part_controllers['right']
        if not (arm.input_type == 'delta' and arm.input_ref_frame == 'world'
                and env.control_freq == 20 and np.array_equal(arm.output_max, [.05, .05, .05, .5, .5, .5])):
            raise ValueError('Unexpected OSC/controller units')
        with h5py.File(dataset, 'r') as data, (output / 'trace.jsonl').open('x', encoding='utf-8') as trace, \
                (output / 'settling_trace.jsonl').open('x', encoding='utf-8') as settling_trace:
            for name in job['demos']:
                obs = restore_demo(env, data['data'][name])
                if hasattr(policy, 'reset'):
                    policy.reset()
                x0 = pack_observation(obs, job.get('semantics', 'legacy'))
                target = env.target_bin_placements[env.object_to_id['can']].copy()
                cls = RecoveryController
                if job['controller'] == 'queue_budget':
                    from robot_can_v2.budget import QueueBudgetController
                    cls = QueueBudgetController
                supervisor = cls(target, env.bin2_pos[2], x0, job['controller'] != 'none', cfg)
                pending = deque([np.zeros(7, np.float32) for _ in range(cfg.delay)])
                rng = np.random.default_rng(70000 + int(name.split('_')[-1]))
                previous = np.zeros(7, np.float32)
                timings, phases, selected_counts = [], Counter(), Counter()
                success, intervened = False, 0
                start = time.perf_counter()
                for step in range(400):
                    tic = time.perf_counter()
                    x = pack_observation(obs, job.get('semantics', 'legacy'))
                    base = policy.best(torch.from_numpy(x[None]))[0].numpy()
                    proposal, selected, scores = propose(model, x, base, list(pending))
                    if hasattr(supervisor, 'set_pending_actions'):
                        supervisor.set_pending_actions(list(pending))
                    action, diagnostic = supervisor.choose(x, proposal, previous)
                    elapsed_ms = 1000 * (time.perf_counter() - tic)
                    timings.append(elapsed_ms); phases[diagnostic['mode']] += 1
                    selected_counts[selected] += 1; intervened += diagnostic['intervened']
                    row = {'demo': name, 'step': step + 1, 'state_before': x.tolist(),
                           'base_chunk': base.tolist(), 'queue_before': np.asarray(pending).tolist(),
                           'selected': selected, 'scores': None if scores is None else scores.tolist(),
                           'proposal': proposal.tolist(), 'submitted_action': action.tolist(),
                           'recovery': diagnostic, 'compute_ms': elapsed_ms}
                    pending.append(action.copy()); executed = pending.popleft().copy()
                    row['executed_action_before_noise'] = executed.tolist()
                    if job.get('noise', 0.):
                        executed[:3] += rng.normal(0, job['noise'], 3)
                    applied = np.clip(executed, -1, 1); previous = applied.copy()
                    obs, *_ = env.step(applied)
                    success = bool(env._check_success())
                    row.update(state_after=pack_observation(obs, job.get('semantics', 'legacy')).tolist(),
                               executed_action=applied.tolist(), success_after=success)
                    trace.write(json.dumps(row, allow_nan=False) + '\n')
                    if success:
                        break
                benchmark_elapsed = time.perf_counter() - start
                stability = settling_check(env, obs, pending, rng, job.get('noise', 0.), settling_trace, name) if success else {'observed': False, 'stable': False}
                record = {'demo': name, 'success': success, 'steps': step + 1,
                          'intervention_steps': int(intervened), 'attempts': supervisor.attempts,
                          'verified_regrasps': supervisor.verified_regrasps,
                          'failed_attempts': supervisor.failed_attempts, 'events': supervisor.events,
                          'phases': dict(phases), 'selection_counts': dict(selected_counts),
                          'compute_timing': timing_summary(timings), 'stability': stability,
                          'benchmark_elapsed_s': benchmark_elapsed}
                payload['records'].append(record); trace.flush(); settling_trace.flush(); save('running')
                print(json.dumps({'job': job['id'], 'demo': name, 'success': success, 'steps': step + 1}), flush=True)
        save('complete')
    except Exception as error:
        payload['error'] = repr(error); save('failed')
        raise
    finally:
        if env is not None:
            env.close()
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-file', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.job_file.read_text(encoding='utf-8'))
    job = next(j for j in jobs if j['id'] == args.job_id)
    run(job, args.output_dir)
