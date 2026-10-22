"""Train new, matched legacy/canonical BC policies without changing history.

Example: python -m robot_can_v2.train_policy --semantics canonical
         --output-dir reproduction/observation_fix_20261022/canonical_seed17
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import common
from robot_can_v2.observations import (
    CANONICAL_CONTRACT, OBSERVATION_SCHEMA, SEMANTICS, read_episodes,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reserve_output_dir(path: str | Path) -> Path:
    """Atomically reserve a new run directory, rejecting every existing path."""
    output = Path(path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def parameter_sha256(model: torch.nn.Module) -> str:
    """Hash trainable initialization; normalization buffers are deliberately excluded."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def make_batch_generator(seed: int) -> torch.Generator:
    """An independent CPU stream makes paired minibatches device-independent."""
    return torch.Generator(device="cpu").manual_seed(seed)


def _exclusive_json(path: Path, data: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _save_checkpoint(path: Path, state_dict: dict, config: dict, update: int, metric: float) -> None:
    # Unlike historical periodic snapshots, each final artifact is written once.
    with path.open("xb") as stream:
        torch.save({"state_dict": state_dict, "config": config,
                    "selected_update": update, "dev_nll": metric}, stream)


def _evaluate(model, x, actions, chunk: int = 2048) -> tuple[float, float]:
    """Evaluate the complete dev split in bounded batches, without random sampling."""
    model.eval()
    total_nll = total_mse = 0.0
    with torch.no_grad():
        for start in range(0, len(x), chunk):
            xx, aa = x[start:start + chunk], actions[start:start + chunk]
            total_nll += model.loss(xx, aa).item() * len(xx)
            total_mse += (model.best(xx)[:, 0] - aa[:, 0]).square().mean().item() * len(xx)
    return total_nll / len(x), total_mse / len(x)


def main(args: argparse.Namespace) -> dict:
    if min(args.updates, args.batch, args.horizon, args.width, args.mixtures,
           args.eval_every, args.threads) < 1 or args.lr <= 0:
        raise ValueError("training sizes, intervals, threads, and learning rate must be positive")
    # This precedes all data reads, hashes, model initialization, and training.
    output = reserve_output_dir(args.output_dir)
    start = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    # cuBLAS workspace must be fixed before its first CUDA operation.
    if device.type == "cuda":
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sources = {"dataset": common.DATA, "splits": common.ROOT / "data/splits.json",
               "historical_common": Path(common.__file__),
               "observations": Path(__file__).with_name("observations.py"),
               "trainer": Path(__file__)}
    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    splits = json.loads(sources["splits"].read_text(encoding="utf-8"))
    train_names, dev_names = splits["train"], splits["dev"]
    if (len(train_names) != len(set(train_names)) or len(dev_names) != len(set(dev_names))
            or set(train_names) & set(dev_names)):
        raise ValueError("train/dev episode IDs must be unique and disjoint")
    train = read_episodes("train", args.semantics)
    dev = read_episodes("dev", args.semantics)
    x, actions = common.policy_windows(train, args.horizon)
    vx, dev_actions = common.policy_windows(dev, args.horizon)
    x, actions, vx, dev_actions = [torch.from_numpy(v).to(device)
                                   for v in (x, actions, vx, dev_actions)]
    model = common.ChunkPolicy(x.shape[-1], args.horizon, args.mixtures, args.width).to(device)
    initial_parameter_hash = parameter_sha256(model)
    model.mean.copy_(x.mean(0))
    model.scale.copy_(x.std(0).clamp_min(.02))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = make_batch_generator(args.seed)
    config = vars(args).copy() | {
        "output_dir": str(output), "dim": int(x.shape[-1]), "architecture": "mlp",
        "observation_semantics": args.semantics, "observation_schema": OBSERVATION_SCHEMA,
        "observation_contract": CANONICAL_CONTRACT if args.semantics == "canonical" else "common.pack_obs/raw HDF5",
        "input": "PRIVILEGED_STATE_DIAGNOSTIC", "device": str(device),
        "n_train_windows": len(x), "n_dev_windows": len(vx),
        "data_split": {"train": train_names, "dev": dev_names},
        "source_paths": {name: str(path.resolve()) for name, path in sources.items()},
        "source_hashes": source_hashes, "initial_parameter_sha256": initial_parameter_hash,
        "batch_rng": "independent torch CPU generator seeded with run seed",
        "selection_metric": "minimum complete-dev action-chunk NLL",
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "deterministic_algorithms": True,
    }
    _exclusive_json(output / "config.json", config)
    best_nll, best_update, best_state = float("inf"), None, None
    batch_digest = hashlib.sha256()
    final_nll = None
    with (output / "log.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        for step in range(1, args.updates + 1):
            model.train()
            ids = torch.randint(len(x), (args.batch,), generator=generator, device="cpu")
            batch_digest.update(ids.numpy().astype("<i8", copy=False).tobytes())
            ids_device = ids.to(device)
            loss = model.loss(x[ids_device], actions[ids_device])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite training loss at update {step}")
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            if step % args.eval_every == 0 or step == args.updates:
                final_nll, mse = _evaluate(model, vx, dev_actions)
                if not np.isfinite(final_nll) or not np.isfinite(mse):
                    raise FloatingPointError(f"nonfinite dev metric at update {step}")
                row = {"update": step, "loss": loss.item(), "dev_nll": final_nll,
                       "dev_action_mse": mse, "gradient_norm": float(gradient_norm),
                       "elapsed_s": time.perf_counter() - start, "device": str(device),
                       "observation_semantics": args.semantics, "seed": args.seed,
                       "batch_sha256": batch_digest.hexdigest(), "source_hashes": source_hashes,
                       "data_split": {"train_count": len(train_names), "dev_count": len(dev_names),
                                      "splits_sha256": source_hashes["splits"]}}
                log.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
                if final_nll < best_nll:
                    best_nll, best_update = final_nll, step
                    best_state = {name: value.detach().cpu().clone()
                                  for name, value in model.state_dict().items()}
    _save_checkpoint(output / "best.pt", best_state, config, best_update, best_nll)
    final_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    _save_checkpoint(output / "final.pt", final_state, config, args.updates, final_nll)
    summary = {"status": "complete", "best_update": best_update, "best_dev_nll": best_nll,
               "final_update": args.updates, "final_dev_nll": final_nll,
               "batch_sha256": batch_digest.hexdigest(),
               "initial_parameter_sha256": initial_parameter_hash,
               "elapsed_s": time.perf_counter() - start,
               "artifacts_sha256": {name: file_sha256(output / name)
                                    for name in ("config.json", "log.jsonl", "best.pt", "final.pt")}}
    _exclusive_json(output / "summary.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantics", choices=SEMANTICS, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--mixtures", type=int, default=5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--updates", type=int, default=15000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
