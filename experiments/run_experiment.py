"""Full experiment pipeline: target training → shadow training → influence computation.

Usage examples
--------------
# Full pipeline, all shadow models
python -m experiments.run_experiment --dataset cifar10 --cuda 0

# Full pipeline, single shadow model
python -m experiments.run_experiment --dataset cifar10 --cuda 0 --shadow_id 2

# Skip target training (already done), run all shadows
python -m experiments.run_experiment --dataset cifar10 --cuda 0 --skip_target

# Write outputs to a custom root directory
python -m experiments.run_experiment --dataset cifar10 --cuda 0 --output_dir /scratch/mia_runs
"""

import argparse
import gc
import os
import sys
import types
from pathlib import Path

import yaml
import torch

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_target import train_target
from training.train_shadow import train_shadow
from training.compute_influence import compute_influence
from experiments.analyze import run as run_analysis
from data.loader import get_dataset


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_config(dataset: str) -> types.SimpleNamespace:
    config_dir = os.path.join("config")
    yaml_path = os.path.join(config_dir, f"{dataset}.yaml")
    yml_path = os.path.join(config_dir, f"{dataset}.yml")
    config_path = yaml_path if os.path.exists(yaml_path) else yml_path
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Missing config for dataset '{dataset}' (expected {yaml_path} or {yml_path})."
        )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return types.SimpleNamespace(**cfg)


def _make_exp_name(args) -> str:
    """Build a reproducible experiment name from the key hyperparameters."""
    parts = [
        f"ep{args.epochs}",
        f"lr{args.lr}",
        f"bs{args.batch_size}",
        f"seed{args.seed}",
    ]
    if hasattr(args, "pkeep"):
        parts.append(f"pk{args.pkeep}")
    if hasattr(args, "n_shadow_models"):
        parts.append(f"ns{args.n_shadow_models}")
    return "_".join(parts)


def _make_args(cli: argparse.Namespace) -> types.SimpleNamespace:
    """Merge config file with CLI flags; attach exp_dir."""
    cfg = _load_config(cli.dataset)
    args = types.SimpleNamespace(**vars(cfg))
    args.dataset = cli.dataset
    get_dataset(args)  # sets args.in_channels, data_mean, data_std, num_classes
    exp_name = cli.exp_name if cli.exp_name else _make_exp_name(args)
    args.exp_dir = str(Path(cli.output_dir) / exp_name / cli.dataset)
    _save_config(args)
    return args


def _save_config(args: types.SimpleNamespace) -> None:
    """Persist the resolved config to exp_dir/config.yaml so analysis can reconstruct it."""
    os.makedirs(args.exp_dir, exist_ok=True)
    cfg_out = os.path.join(args.exp_dir, "config.yaml")
    if not os.path.exists(cfg_out):
        with open(cfg_out, "w") as f:
            yaml.dump(vars(args), f, default_flow_style=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end MIA experiment: target → shadows → influence"
    )
    parser.add_argument("--dataset", default="cifar10",
                        help="Dataset name (must match config/<name>.yaml)")
    parser.add_argument("--cuda", type=int, default=0,
                        help="CUDA device index (-1 for CPU)")
    parser.add_argument("--output_dir", default="outputs",
                        help="Root directory for all outputs (default: outputs/)")
    parser.add_argument("--exp_name", default=None,
                        help="Experiment name (subdirectory under output_dir). "
                             "If omitted, auto-generated from hyperparameters. "
                             "Pass the same name to resume a crashed run.")
    parser.add_argument("--shadow_id", type=int, default=None,
                        help="Run a single shadow model ID instead of all")
    parser.add_argument("--skip_target", action="store_true",
                        help="Skip target model training (use existing checkpoint)")
    parser.add_argument("--skip_shadows", action="store_true",
                        help="Skip shadow training + influence (target only)")
    parser.add_argument("--num_buckets", type=int, default=10,
                        help="Number of quantile buckets for Stage 3 analysis")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _run_target(args, device: torch.device) -> None:
    target_path = os.path.join(args.exp_dir, "target_model.pt")
    if os.path.exists(target_path):
        print(f"[target] target_model.pt already exists — skipping training.")
        return
    print(f"[target] === Training target model ===")
    train_target(args, device)


def _run_shadow_one(args, shadow_id: int, device: torch.device) -> None:
    shadow_dir = os.path.join(args.exp_dir, "shadows", str(shadow_id))
    model_path = os.path.join(shadow_dir, "shadow_model.pt")
    ckpt_path = os.path.join(shadow_dir, "checkpoint.pt")
    lira_path = os.path.join(shadow_dir, "lira_stats.npy")

    # Train (resume if checkpoint exists, skip if already complete)
    if os.path.exists(model_path) and not os.path.exists(ckpt_path):
        print(f"[shadow {shadow_id}] shadow_model.pt already exists — skipping training.")
    else:
        label = "Resuming" if os.path.exists(ckpt_path) else "Training"
        print(f"[shadow {shadow_id}] === {label} shadow model ===")
        train_shadow(args, shadow_id, device)

    # Influence
    if os.path.exists(lira_path):
        print(f"[shadow {shadow_id}] lira_stats.npy already exists — skipping influence.")
    else:
        print(f"[shadow {shadow_id}] === Computing influence ===")
        compute_influence(args, shadow_id, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cli = _parse_args()
    args = _make_args(cli)

    print("[debug] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("[debug] torch.version.cuda:", torch.version.cuda)
    print("[debug] torch.cuda.is_available():", torch.cuda.is_available())
    print("[debug] torch.cuda.device_count():", torch.cuda.device_count())

    if torch.cuda.is_available() and cli.cuda >= 0:
        device = torch.device(f"cuda:{cli.cuda}")
        print("[debug] using", device, "name:", torch.cuda.get_device_name(cli.cuda))
    else:
        device = torch.device("cpu")

    print(f"Experiment directory : {args.exp_dir}")
    print(f"Device               : {device}")
    print(f"Config               : {vars(args)}\n")

    # --- Stage 1: target ---
    if not cli.skip_target:
        print(f"\n{'='*60}\nStage 1: Target model\n{'='*60}")
        _run_target(args, device)
    else:
        print("[target] --skip_target set — skipping target training.")

    # --- Stage 2: shadows + influence ---
    if not cli.skip_shadows:
        shadow_ids = (
            [cli.shadow_id] if cli.shadow_id is not None
            else list(range(args.n_shadow_models))
        )
        for sid in shadow_ids:
            print(f"\n{'='*60}\nStage 2: Shadow model {sid} / {args.n_shadow_models - 1}\n{'='*60}")
            _run_shadow_one(args, sid, device)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        print("[shadows] --skip_shadows set — skipping shadow training.")

    # --- Stage 3: influence vs LiRA bucketing analysis ---
    # Skip when --skip_shadows is set or when only a single shadow was run
    # (partial run — not all shadows are necessarily complete yet).
    if not cli.skip_shadows and cli.shadow_id is None:
        print(f"\n{'='*60}\nStage 3: Influence vs LiRA analysis\n{'='*60}")
        run_analysis(exp_dir=args.exp_dir, dataset=cli.dataset, num_buckets=cli.num_buckets)

    print("\nAll done.")


if __name__ == "__main__":
    main()
