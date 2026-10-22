"""Safe world-model training entrypoint; legacy code and artifacts are untouched.

Run from the project root: python -m robot_can_v2.train_world --help
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

from common import ROOT
from world_model import DirectScore, WorldModel, grouped_rank_loss, score, target_score
from .metrics import ranking_metrics

METHODS = ('plain', 'full', 'direct', 'prediction_only')
SEMANTICS = ('legacy', 'canonical')


def validate_model_config(cfg, *, horizon=None, observation_semantics=None):
    """Validate native checkpoint architecture and its observation contract."""
    if not isinstance(cfg, dict) or cfg.get('method') not in METHODS:
        raise ValueError('Checkpoint must declare a supported native model method')
    for key in ('dim', 'width', 'horizon'):
        value = cfg.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f'Checkpoint {key} must be a positive integer')
    if not isinstance(cfg.get('future_only', False), bool):
        raise ValueError('Checkpoint future_only must be boolean')
    semantics = cfg.get('observation_semantics', 'legacy')
    if semantics not in SEMANTICS:
        raise ValueError('Unsupported checkpoint observation_semantics')
    if observation_semantics is not None and semantics != observation_semantics:
        raise ValueError(f'Checkpoint observation semantics {semantics!r} differ from requested {observation_semantics!r}')
    if horizon is not None and cfg['horizon'] != horizon:
        raise ValueError(f"Checkpoint horizon {cfg['horizon']} differs from execution horizon {horizon}")
    return cfg


def construct_model(cfg):
    validate_model_config(cfg)
    if cfg['method'] == 'direct':
        return DirectScore(cfg['dim'], cfg['horizon'], cfg['width'])
    return WorldModel(cfg['dim'], cfg['width'], cfg.get('future_only', False))


def predict(model, x, actions, method):
    """Prediction-only never executes the untrained label head.

    The recurrence is the native WorldModel recurrence. Keeping the label head
    in its state_dict preserves architecture compatibility with saved weights.
    """
    if method != 'prediction_only':
        return model(x, actions)
    z = (x - model.mean) / model.scale
    hidden = model.init(z)
    states = []
    for t in range(actions.shape[1]):
        hidden = model.cell(torch.cat([z, actions[:, t]], -1), hidden)
        z = z + model.delta(hidden) * .1
        states.append(z * model.scale + model.mean)
    return torch.stack(states, 1), None


def selection_criterion(method, metrics):
    key = 'normalized_state_mse' if method == 'prediction_only' else 'mean_regret'
    value = metrics[key]
    if value is None or not math.isfinite(value):
        raise ValueError(f'Checkpoint selection metric {key} must be finite')
    return float(value)


def _dataset_semantics(archive):
    if 'observation_semantics' not in archive:
        return 'legacy', 'historical_missing_metadata'
    raw = np.asarray(archive['observation_semantics'])
    if raw.ndim != 0:
        raise ValueError('observation_semantics metadata must be a scalar string')
    value = raw.item()
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if value not in SEMANTICS:
        raise ValueError(f'Unknown branch observation semantics: {value!r}')
    return value, 'explicit_npz_metadata'


def load_data(path, device, observation_semantics):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        actual, declaration = _dataset_semantics(archive)
        if actual != observation_semantics:
            raise ValueError(f'{path.name}: data semantics {actual!r} differ from requested {observation_semantics!r}; canonical data require explicit metadata')
        arrays = {key: np.asarray(archive[key]) for key in ('x', 'a', 'y', 'labels')}
        parents = sorted(set(archive['parent'].astype(str).tolist())) if 'parent' in archive else None
    x, a, y, labels = (arrays[key] for key in ('x', 'a', 'y', 'labels'))
    if x.ndim != 2 or a.ndim != 4 or a.shape[-1] != 7:
        raise ValueError('Expected x=(groups,dim), a=(groups,candidates,horizon,7)')
    n, k, h, _ = a.shape
    if min(n, k, h, x.shape[-1]) < 1 or x.shape[0] != n:
        raise ValueError('Empty or inconsistent branch dimensions')
    if y.shape != (n, k, h, x.shape[-1]) or labels.shape != (n, k, h, 2):
        raise ValueError('Branch state/label shapes do not match x and a')
    for name, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value) or not np.isfinite(value).all():
            raise ValueError(f'Branch {name} must be finite numeric data')
    if np.any(labels < 0) or np.any(labels > 1):
        raise ValueError('Branch labels must lie in [0,1]')
    if np.any(np.abs(a) > 1.00001):
        raise ValueError('Branch actions must lie in [-1,1]')
    if actual == 'canonical':
        from .observations import canonicalize_state
        for name in ('x', 'y'):
            rebuilt = canonicalize_state(arrays[name])
            if not np.allclose(arrays[name], rebuilt, atol=2e-6, rtol=1e-6):
                raise ValueError(f'{path.name}: {name} is declared canonical but violates same-frame geometry')
    tensors = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in arrays.items()}
    metadata = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'observation_semantics': actual, 'semantics_declaration': declaration, 'parents': parents}
    return tensors, metadata


@torch.no_grad()
def evaluate(model, dev, risk_weight=.5, method='plain'):
    model.eval()
    n, k, _, _ = dev['a'].shape
    predictions, truth = [], []
    square_sum, count = 0., 0
    for start in range(0, n, 32):
        end = min(start + 32, n)
        x = dev['x'][start:end, None].expand(-1, k, -1).reshape((end - start) * k, -1)
        future, logits = predict(model, x, dev['a'][start:end].flatten(0, 1), method)
        if logits is not None:
            predictions.extend(score(logits, risk_weight).cpu().tolist())
            truth.extend(target_score(dev['labels'][start:end].flatten(0, 1), risk_weight).cpu().tolist())
        if future is not None:
            squared = ((future - dev['y'][start:end].flatten(0, 1)) / model.scale).square()
            square_sum += float(squared.sum())
            count += squared.numel()
    metrics = ranking_metrics(predictions, truth, n, k) if predictions else {
        'mean_regret': None, 'strict_reversal_rate': None, 'tie_rate': None,
        'pair_error_half_ties': None, 'effective_pairs': 0, 'groups': n,
        'ranking_note': 'Not evaluated: prediction_only has no trained score head',
    }
    metrics['normalized_state_mse'] = square_sum / count if count else None
    return metrics


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def validate_args(args):
    out = Path(args.output_dir).resolve()
    # Deliberately check before loading data, creating a model, or touching CUDA.
    if out.exists():
        raise FileExistsError(f'Refusing to reuse training output directory: {out}')
    for parent in out.parents:
        if parent.exists() and not parent.is_dir():
            raise NotADirectoryError(str(parent))
    for key in ('updates', 'batch', 'width', 'eval_every', 'threads'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    for key in ('lr', 'rank_weight', 'risk_weight'):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key == 'lr' and value == 0):
            raise ValueError(f'{key} must be finite and nonnegative (lr positive)')
    if (args.train_data is None) != (args.dev_data is None):
        raise ValueError('--train-data and --dev-data must be provided together')
    if args.method not in METHODS or args.observation_semantics not in SEMANTICS:
        raise ValueError('Unsupported method or observation semantics')
    return out


def main(args):
    out = validate_args(args)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
    train_path = args.train_data or ROOT / 'data' / f'branches_train_{args.dataset}.npz'
    dev_path = args.dev_data or ROOT / 'data' / f'branches_dev_{args.dataset}.npz'
    if Path(train_path).resolve() == Path(dev_path).resolve():
        raise ValueError('Training and dev data must be distinct files')
    train, train_meta = load_data(train_path, device, args.observation_semantics)
    dev, dev_meta = load_data(dev_path, device, args.observation_semantics)
    if train['x'].shape[1:] != dev['x'].shape[1:] or train['a'].shape[1:] != dev['a'].shape[1:]:
        raise ValueError('Train/dev branch dimensions differ')
    if train_meta['parents'] is not None and dev_meta['parents'] is not None:
        if set(train_meta['parents']) & set(dev_meta['parents']):
            raise ValueError('Training and dev parent trajectories overlap')
        splits = json.loads((ROOT / 'data/splits.json').read_text(encoding='utf-8'))
        if not set(train_meta['parents']).issubset(splits['train']) or not set(dev_meta['parents']).issubset(splits['dev']):
            raise ValueError('Branch parents do not belong to their declared train/dev split')
    n, k, h, _ = train['a'].shape
    dim = train['x'].shape[-1]
    cfg = {key: value for key, value in vars(args).items()}
    cfg.update(dim=dim, horizon=h, device=str(device), entrypoint='robot_can_v2.train_world',
               input='PRIVILEGED_STATE_DIAGNOSTIC', n_groups=n, n_candidates=k,
               selection='minimum normalized dev state MSE; earliest tie' if args.method == 'prediction_only' else 'minimum dev score mean_regret; earliest tie',
               score_head_usage='frozen_and_never_called' if args.method == 'prediction_only' else 'trained',
               effective_rank_weight=args.rank_weight if args.method in ('full', 'direct') else 0.,
               batch_schedule='CPU generator seeded independently of model initialization; matched across methods',
               dataset_provenance={'train': train_meta, 'dev': dev_meta},
               source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in (Path(__file__).resolve(), ROOT / 'world_model.py', ROOT / 'robot_can_v2/metrics.py')})
    model = construct_model(cfg).to(device)
    model.mean.copy_(train['x'].mean(0))
    model.scale.copy_(train['x'].std(0, unbiased=n > 1).clamp_min(.02))
    if args.method == 'prediction_only':
        model.labels.requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    cfg.update(parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in parameters))
    _write_json(out / 'config.json', cfg)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    best, chosen, logs = float('inf'), None, []
    batch_generator = torch.Generator(device='cpu').manual_seed(args.seed)
    batch_hash = hashlib.sha256()
    started = time.perf_counter()
    with (out / 'training.jsonl').open('x', encoding='utf-8') as log:
        for step in range(1, args.updates + 1):
            model.train()
            ids_cpu = torch.randint(n, (args.batch,), generator=batch_generator)
            batch_hash.update(ids_cpu.numpy().tobytes())
            ids = ids_cpu.to(device)
            x = train['x'][ids, None].expand(-1, k, -1).reshape(args.batch * k, dim)
            actions = train['a'][ids].flatten(0, 1)
            y, labels = train['y'][ids].flatten(0, 1), train['labels'][ids].flatten(0, 1)
            future, logits = predict(model, x, actions, args.method)
            zero = x.new_zeros(())
            pred_loss = ((future - y) / model.scale).square().mean() if future is not None else zero
            progress_loss = risk_loss = ranking = zero
            if args.method != 'prediction_only':
                progress_loss = nn.functional.binary_cross_entropy_with_logits(logits[:, :, 0], labels[:, :, 0])
                risk_loss = nn.functional.binary_cross_entropy_with_logits(logits[:, :, 1], labels[:, :, 1])
                if args.method in ('full', 'direct'):
                    ranking = grouped_rank_loss(score(logits, args.risk_weight), target_score(labels, args.risk_weight), args.batch, k)
            loss = pred_loss + progress_loss + .5 * risk_loss + args.rank_weight * ranking
            if not torch.isfinite(loss):
                raise FloatingPointError('Training loss became non-finite')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 10)
            optimizer.step()
            if step % args.eval_every == 0 or step == args.updates:
                metrics = evaluate(model, dev, args.risk_weight, args.method)
                criterion = selection_criterion(args.method, metrics)
                row = {'step': step, 'loss': float(loss.detach()), 'prediction_loss': float(pred_loss.detach()),
                       'progress_loss': float(progress_loss.detach()), 'risk_loss': float(risk_loss.detach()),
                       'ranking_loss': float(ranking.detach()), 'selection_criterion': criterion,
                       'elapsed_s': time.perf_counter() - started, **metrics}
                logs.append(row)
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
                if criterion < best:
                    best = criterion
                    chosen = {'state_dict': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                              'config': cfg, 'selected_step': step, 'dev_metrics': metrics}
    with (out / 'model.pt').open('xb') as stream:
        torch.save(chosen, stream)
    _write_json(out / 'log.json', logs)
    _write_json(out / 'training_complete.json', {'status': 'complete', 'selected_step': chosen['selected_step'],
                'sha256': hashlib.sha256((out / 'model.pt').read_bytes()).hexdigest(),
                'minibatch_sha256': batch_hash.hexdigest(), 'elapsed_s': time.perf_counter() - started})
    return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='New directory only; existing directories are rejected before data/model loading.')
    parser.add_argument('--method', choices=METHODS, default='plain')
    parser.add_argument('--dataset', default='pilot_v0')
    parser.add_argument('--train-data')
    parser.add_argument('--dev-data')
    parser.add_argument('--observation-semantics', choices=SEMANTICS, default='legacy')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--updates', type=int, default=3000)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--rank-weight', type=float, default=.2)
    parser.add_argument('--risk-weight', type=float, default=.5)
    parser.add_argument('--future-only', action='store_true')
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())
