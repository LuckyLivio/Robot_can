"""Isolated delay evaluator with explicit model and observation contracts.

This entrypoint evaluates BC/BC-shift/world-model control without recovery.
Use the separate v2 recovery evaluator for the recovery factorial experiment.
"""
import argparse
from collections import deque
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from common import ROOT, DATA, load_policy, make_env, restore_demo
from collect_branches import candidates
from world_model import score
from .train_world import SEMANTICS, construct_model, predict, validate_model_config


def physical_cost(future, target, grasp_hint):
    """Historical multiseed prediction-only physical cost, in distance units."""
    eef, obj = future[:, :, :3], future[:, :, 16:19]
    target = torch.as_tensor(target, dtype=future.dtype, device=future.device)
    if grasp_hint:
        return ((obj - target) ** 2).sum(-1).sqrt().mean(-1) + .5 * torch.relu(.86 - obj[:, -1, 2])
    return ((eef - obj) ** 2).sum(-1).sqrt().mean(-1)


def build_delayed_sequences(candidate_chunks, pending, horizon, delay):
    """Exactly prepend all pending FIFO commands, then repeat the new command.

    Each candidate commits only its first command. With delay D, that command
    first changes the predicted state at future index D (zero based).
    """
    aa = np.asarray(candidate_chunks, dtype=np.float32)
    prefix = np.asarray(pending, dtype=np.float32).reshape(-1, 7)
    if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
        raise ValueError('delay must be a nonnegative integer')
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= delay:
        raise ValueError('horizon must exceed delay so each candidate has an observable effect')
    if aa.ndim != 3 or aa.shape[-1] != 7 or aa.shape[0] < 1 or aa.shape[1] < 1:
        raise ValueError('Candidates must have shape (candidates,chunk_horizon,7)')
    if prefix.shape != (delay, 7):
        raise ValueError(f'Pending FIFO length {len(prefix)} differs from configured delay {delay}')
    if not np.isfinite(aa).all() or not np.isfinite(prefix).all():
        raise ValueError('Candidate and pending commands must be finite')
    repeated = np.repeat(aa[:, :1, :], horizon - delay, axis=1)
    return np.concatenate([np.repeat(prefix[None], len(aa), axis=0), repeated], axis=1)


def validate_control_config(cfg, args):
    validate_model_config(cfg, horizon=args.horizon, observation_semantics=args.observation_semantics)
    if cfg['dim'] != 23:
        raise ValueError('This task evaluator requires native 23-dimensional privileged state')
    if cfg['method'] == 'prediction_only':
        if args.score_only or args.readout_weight != 0:
            raise ValueError('prediction_only requires physical cost and readout_weight=0; its label head is unused')
    elif cfg['method'] == 'direct' and not args.score_only:
        raise ValueError('DirectScore requires --score-only because it predicts no physical state')


def score_candidates(model, x, actions, cfg, target, grasp_hint, delay, *, score_only=False, readout_weight=.02, risk_weight=.5):
    """Score candidates; prediction_only bypasses the untrained scorer entirely."""
    if delay < 0 or delay >= actions.shape[1]:
        raise ValueError('Prediction horizon must exceed delay')
    if cfg['method'] == 'prediction_only' and (score_only or readout_weight != 0):
        raise ValueError('prediction_only requires physical cost and readout_weight=0')
    future, logits = predict(model, x, actions, cfg['method'])
    if future is None and not score_only:
        raise ValueError('A state-free model requires score_only')
    cost = physical_cost(future[:, delay:], target, grasp_hint) if future is not None else x.new_zeros(len(x))
    learned = None
    if logits is not None:
        learned = score(logits, risk_weight)
        cost = -learned if score_only else cost - readout_weight * learned
    return future, logits, learned, cost


def output_plan(args):
    directory = Path(args.output_dir).resolve()
    # Must run before split reads, checkpoint loads, policy construction, or env.
    if directory.exists():
        raise FileExistsError(f'Refusing to reuse evaluation output directory: {directory}')
    for parent in directory.parents:
        if parent.exists() and not parent.is_dir():
            raise NotADirectoryError(str(parent))
    if args.episodes < 1 or args.steps < 1 or args.delay < 0 or args.horizon <= args.delay or args.candidates < 1:
        raise ValueError('episodes/steps/candidates must be positive; horizon must exceed nonnegative delay')
    for key in ('noise', 'risk_weight', 'gate_margin'):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if args.readout_weight is not None and (not math.isfinite(args.readout_weight) or args.readout_weight < 0):
        raise ValueError('readout_weight must be finite and nonnegative')
    if args.shift_index is not None and args.shift_index < 0:
        raise ValueError('shift_index must be nonnegative')
    if args.fallback_index is not None and args.fallback_index < 0:
        raise ValueError('fallback_index must be nonnegative')
    split_names = json.loads((ROOT / 'data/splits.json').read_text(encoding='utf-8'))[args.split]
    if args.demos:
        if len(set(args.demos)) != len(args.demos) or not set(args.demos).issubset(split_names):
            raise ValueError('Every requested demo must be unique and belong to the selected split')
        names = list(args.demos)
    else:
        if args.episodes > len(split_names):
            raise ValueError('Requested more episodes than available in split')
        names = split_names[:args.episodes]
    return names, directory / 'result.json', {name: directory / 'traces' / f'{name}.jsonl' for name in names} if args.trace else {}


def _provenance(files):
    paths = [Path(path).resolve() for path in files]
    versions = {}
    for name in ('torch', 'numpy', 'robosuite', 'mujoco', 'h5py'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'sha256': {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in paths if path.is_file()}, 'packages': versions,
            'sequence_semantics': 'exact pending FIFO prefix, then repeat candidate first command; replan after every step',
            'prediction_only_cost': 'historical multiseed physical_cost; label head frozen during new training and never called',
            'timing_note': 'Decision timing excludes policy inference and environment stepping; not a real-time claim'}


def run(args):
    names, out, trace_paths = output_plan(args)
    out.parent.mkdir(parents=True, exist_ok=False)
    from .observations import pack_observation
    torch.set_num_threads(2)
    policy_path = Path(args.policy_path).resolve() if args.policy_path else ROOT / 'policy' / args.policy / args.policy_checkpoint
    policy_cfg = torch.load(policy_path, map_location='cpu', weights_only=True)['config']
    if policy_cfg.get('observation_semantics', 'legacy') != args.observation_semantics:
        raise ValueError('Policy observation semantics differ from requested evaluation semantics')
    if policy_cfg.get('dim') != 23:
        raise ValueError('Policy must accept 23-dimensional task state')
    policy = load_policy(policy_path)
    cfg, model, model_path = None, None, None
    if args.method == 'wm':
        model_path = Path(args.model_checkpoint).resolve() if args.model_checkpoint else ROOT / 'models' / args.model / 'model.pt'
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        cfg = checkpoint['config']
        if args.readout_weight is None:
            args.readout_weight = 0. if cfg.get('method') == 'prediction_only' else .02
        validate_control_config(cfg, args)
        model = construct_model(cfg)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        if cfg['method'] == 'prediction_only':
            model.labels.requires_grad_(False)
    elif args.readout_weight is None:
        args.readout_weight = 0.
    if args.method == 'bc_shift' and (args.shift_index if args.shift_index is not None else args.delay) >= policy.horizon:
        raise ValueError('Requested BC shift is outside policy action chunk')
    count = min(args.candidates, policy.horizon) if args.candidate_mode == 'shift' else args.candidates
    fallback_index = 0 if args.fallback == 'base' else (args.fallback_index if args.fallback_index is not None else args.delay)
    if args.method == 'wm' and not 0 <= fallback_index < count:
        raise ValueError('Fallback index is outside actual candidate set')
    sources = [DATA, ROOT / 'data/splits.json', policy_path, Path(__file__), ROOT / 'common.py', ROOT / 'world_model.py',
               ROOT / 'collect_branches.py', ROOT / 'robot_can_v2/train_world.py', ROOT / 'robot_can_v2/observations.py']
    if model_path is not None:
        sources.append(model_path)
    provenance = _provenance(sources)
    records = []
    with ExitStack() as resources:
        result_file = resources.enter_context(out.open('x', encoding='utf-8'))
        traces = {}
        for name, path in trace_paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            traces[name] = resources.enter_context(path.open('x', encoding='utf-8'))

        def save_result(status):
            payload = {'args': vars(args), 'status': status, 'resolved_demos': names, 'records': records,
                       'input': 'PRIVILEGED_STATE_DIAGNOSTIC', 'observation_semantics': args.observation_semantics,
                       'success_rate': float(np.mean([row['success'] for row in records])) if records else None,
                       'model_config': cfg, 'policy_config': policy_cfg, 'provenance': provenance,
                       'trace_schema_version': 2, 'trace_files': {name: path.relative_to(out.parent).as_posix() for name, path in trace_paths.items()}}
            result_file.seek(0)
            result_file.write(json.dumps(payload, indent=2, allow_nan=False))
            result_file.truncate()
            result_file.flush()

        save_result('running')
        env = make_env(False)
        resources.callback(env.close)
        source = resources.enter_context(h5py.File(DATA, 'r'))
        for name in names:
            obs = restore_demo(env, source['data'][name])
            target = env.target_bin_placements[env.object_to_id['can']].astype(np.float32)
            if hasattr(policy, 'reset'):
                policy.reset()
            queue = deque([np.zeros(7, np.float32) for _ in range(args.delay)])
            rng = np.random.default_rng(70000 + int(name.split('_')[1]))
            crng = np.random.default_rng(170000 + int(name.split('_')[1]))
            success, peak, grasped = False, 0., False
            selected_ids, fallback_ids, decisions = [], [], []
            started = time.perf_counter()
            for step in range(args.steps):
                x = pack_observation(obs, args.observation_semantics)
                base = policy.best(torch.from_numpy(x[None]))[0].numpy()
                pending = np.asarray(list(queue), np.float32).reshape(-1, 7).copy()
                stages_before = np.asarray(env.staged_rewards()).copy() if args.trace else None
                tic = time.perf_counter()
                raw_selected = None
                if args.method == 'bc':
                    selected, new_action = 0, base[0].copy()
                elif args.method == 'bc_shift':
                    selected = args.shift_index if args.shift_index is not None else args.delay
                    new_action = base[selected].copy()
                else:
                    aa = np.asarray([np.repeat(base[k:k + 1], base.shape[0], axis=0) for k in range(count)], np.float32) if args.candidate_mode == 'shift' else candidates(base, crng, args.candidates)
                    seqs = build_delayed_sequences(aa, pending, args.horizon, args.delay)
                    xb = torch.from_numpy(np.repeat(x[None], len(seqs), axis=0))
                    ab = torch.from_numpy(seqs)
                    holding = bool(x[18] > .88 and np.linalg.norm(x[:3] - x[16:19]) < .10)
                    with torch.no_grad():
                        future, logits, learned, cost = score_candidates(model, xb, ab, cfg, target, holding, args.delay,
                            score_only=args.score_only, readout_weight=args.readout_weight, risk_weight=args.risk_weight)
                    selected = raw_selected = int(cost.argmin())
                    if selected != fallback_index and float(cost[fallback_index] - cost[selected]) < args.gate_margin:
                        selected = fallback_index
                    new_action = aa[selected, 0].copy()
                    fallback_ids.append(fallback_index)
                decisions.append(time.perf_counter() - tic)
                selected_ids.append(selected)
                submitted = new_action.copy()
                queue.append(new_action.copy())
                executed = queue.popleft()
                executed_before_noise = executed.copy()
                if args.noise:
                    executed[:3] += rng.normal(0, args.noise, 3)
                applied = np.clip(executed, -1, 1)
                obs, _, _, _ = env.step(applied)
                stages = np.asarray(env.staged_rewards())
                peak = max(peak, float(max(stages)))
                grasped = grasped or bool(stages[1] > .3)
                success = bool(env._check_success())
                if args.trace:
                    trace = {'demo': name, 'step': step + 1, 'state_before': x.tolist(),
                             'state_after': pack_observation(obs, args.observation_semantics).tolist(),
                             'queue_before': pending.tolist(), 'base_chunk': base.tolist(), 'submitted_action': submitted.tolist(),
                             'executed_action_before_noise': executed_before_noise.tolist(), 'executed_action': applied.tolist(),
                             'stages_before': stages_before.tolist(), 'stages_after': stages.tolist(), 'success_after': success,
                             'target_position': target.tolist(), 'selected': selected, 'decision_ms': decisions[-1] * 1000}
                    if args.method == 'wm':
                        probabilities = logits.sigmoid().numpy() if logits is not None else None
                        trace.update(candidate_actions=aa[:, 0].tolist(), model_sequences=seqs.tolist(),
                                     predicted_states=future.numpy().tolist() if future is not None else None,
                                     predicted_progress=probabilities[:, :, 0].tolist() if probabilities is not None else None,
                                     predicted_risk=probabilities[:, :, 1].tolist() if probabilities is not None else None,
                                     learned_scores=learned.numpy().tolist() if learned is not None else None, costs=cost.numpy().tolist(),
                                     raw_selected=raw_selected, fallback_index=fallback_index,
                                     raw_advantage=float(cost[fallback_index] - cost[raw_selected]),
                                     selected_advantage=float(cost[fallback_index] - cost[selected]),
                                     overrode_fallback=selected != fallback_index, gate_blocked=raw_selected != selected)
                    traces[name].write(json.dumps(trace, allow_nan=False) + '\n')
                if success:
                    break
            row = {'demo': name, 'success': success, 'steps': step + 1, 'peak_stage_reward': peak,
                   'ever_grasped': grasped, 'delay': args.delay, 'fraction_base_selected': float(np.mean(np.asarray(selected_ids) == 0)),
                   'mean_decision_ms': float(np.mean(decisions) * 1000), 'elapsed_s': time.perf_counter() - started,
                   'selection_counts': {str(k): selected_ids.count(k) for k in sorted(set(selected_ids))}}
            if fallback_ids:
                fraction = float(np.mean(np.asarray(selected_ids) == np.asarray(fallback_ids)))
                row.update(fraction_fallback_selected=fraction, fraction_override=1 - fraction)
            if args.trace:
                traces[name].flush()
            records.append(row)
            save_result('running')
            print(json.dumps(row, allow_nan=False), flush=True)
        save_result('complete')
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True, help='A fresh directory; every existing directory is refused.')
    p.add_argument('--method', choices=('bc', 'bc_shift', 'wm'), default='wm')
    p.add_argument('--model', default='full_future_bc_v4')
    p.add_argument('--model-checkpoint')
    p.add_argument('--policy', default='bc_chunk_dev_v4')
    p.add_argument('--policy-checkpoint', default='policy_15000.pt')
    p.add_argument('--policy-path')
    p.add_argument('--observation-semantics', choices=SEMANTICS, default='legacy')
    p.add_argument('--split', choices=('dev', 'sealed_test'), default='dev')
    p.add_argument('--episodes', type=int, default=8)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--delay', type=int, default=2)
    p.add_argument('--shift-index', type=int)
    p.add_argument('--fallback-index', type=int)
    p.add_argument('--candidates', type=int, default=6)
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--noise', type=float, default=0.)
    p.add_argument('--risk-weight', type=float, default=.5)
    p.add_argument('--readout-weight', type=float, default=None, help='Default 0 for prediction_only; .02 for label-trained methods.')
    p.add_argument('--gate-margin', type=float, default=0.)
    p.add_argument('--fallback', choices=('base', 'shift'), default='base')
    p.add_argument('--score-only', action='store_true')
    p.add_argument('--candidate-mode', choices=('random', 'shift'), default='random')
    p.add_argument('--demos', nargs='+')
    p.add_argument('--trace', action='store_true')
    return p


if __name__ == '__main__':
    run(build_parser().parse_args())
