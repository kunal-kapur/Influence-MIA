"""Entry point: train shadow models and compute influence functions.

Usage examples
--------------
# Train + compute influence for shadow model 0
python -m experiments.run_shadow --shadow_id 0 --cuda 0

# Train + compute influence for all shadow models
python -m experiments.run_shadow --all --cuda 0

# Write outputs to a custom directory (experiment name is auto-generated)
python -m experiments.run_shadow --all --cuda 0 --output_dir /scratch/mia_runs
"""

import argparse
import os
import types
from pathlib import Path

import yaml
import torch

from training.train_shadow import train_shadow
from training.compute_influence import compute_influence


def _load_config(dataset: str) -> types.SimpleNamespace:
    """Load config/{dataset}.yaml (fallback to .yml) and return as a SimpleNamespace."""
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
    cfg.setdefault("warmup_epochs", 1)
    cfg.setdefault("temperature", 1.0)
    cfg.setdefault("margin_weight", 1.0)
    return types.SimpleNamespace(**cfg)


def _make_exp_name(args) -> str:
    """Build a human-readable experiment name from the key hyperparameters."""
    parts = [
        f"ep{args.epochs}",
        f"lr{args.lr}",
        f"bs{args.batch_size}",
        f"seed{args.seed}",
    ]
    if hasattr(args, "warmup_epochs"):
        parts.append(f"wu{args.warmup_epochs}")
    if hasattr(args, "temperature"):
        parts.append(f"T{args.temperature}")
    if hasattr(args, "margin_weight"):
        parts.append(f"mw{args.margin_weight}")
    if hasattr(args, "pkeep"):
        parts.append(f"pk{args.pkeep}")
    if hasattr(args, "n_shadow_models"):
        parts.append(f"ns{args.n_shadow_models}")
    return "_".join(parts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shadow model training + influence computation")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset name (must match a config/<name>.yaml file)")
    parser.add_argument("--cuda", type=int, default=0,
                        help="CUDA device index (ignored if no GPU available)")
    parser.add_argument("--shadow_id", type=int, default=0,
                        help="Shadow model index to train/evaluate")
    parser.add_argument("--all", action="store_true",
                        help="Run all shadow models 0 .. n_shadow_models-1")
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Root directory for all experiment outputs (default: outputs/)",
    )
    return parser.parse_args()


def _run_one(args, shadow_id: int, device: torch.device) -> None:
    """Train shadow model and compute influence for a single shadow_id,
    skipping steps whose output files already exist."""
    shadow_dir = os.path.join(args.exp_dir, "shadows", str(shadow_id))
    model_path = os.path.join(shadow_dir, "shadow_model.pt")
    ckpt_path = os.path.join(shadow_dir, "checkpoint.pt")
    lira_path = os.path.join(shadow_dir, "lira_stats.npy")

    # --- Train ---
    # Run training if the final model is missing; a checkpoint means training
    # was interrupted and should be resumed.
    if os.path.exists(model_path) and not os.path.exists(ckpt_path):
        print(f"[shadow {shadow_id}] shadow_model.pt already exists — skipping training.")
    else:
        if os.path.exists(ckpt_path):
            print(f"[shadow {shadow_id}] === Resuming shadow model training ===")
        else:
            print(f"[shadow {shadow_id}] === Training shadow model ===")
        train_shadow(args, shadow_id, device)


    if os.path.exists(lira_path):
        print(f"[shadow {shadow_id}] lira_stats.npy already exists — skipping influence computation.")
    else:
        print(f"[shadow {shadow_id}] === Computing influence ===")
        compute_influence(args, shadow_id, device)


def main() -> None:
    cli = _parse_args()

    # Load config, then overlay CLI-provided values
    cfg = _load_config(cli.dataset)
    args = types.SimpleNamespace(**vars(cfg))
    args.dataset = cli.dataset

    # Build experiment directory: <output_dir>/<exp_name>/<dataset>/
    exp_name = _make_exp_name(args)
    args.exp_dir = str(Path(cli.output_dir) / exp_name / cli.dataset)
    print(f"Experiment directory: {args.exp_dir}")

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cli.cuda}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    if cli.all:
        shadow_ids = list(range(args.n_shadow_models))
    else:
        shadow_ids = [cli.shadow_id]

    for sid in shadow_ids:
        print(f"\n{'='*60}")
        print(f"Shadow model {sid} / {args.n_shadow_models - 1}")
        print(f"{'='*60}")
        _run_one(args, sid, device)

    print("\nAll done.")


if __name__ == "__main__":
    main()
