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

import numpy as np
import yaml
import torch

from training.train_shadow import train_shadow
from training.compute_influence import compute_influence
from training.select_pivot import select_pivot_data


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
    cfg.setdefault("T2", 20)
    cfg.setdefault("k_pivot", 100)
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


def _ensure_pivot(args, device: torch.device) -> np.ndarray:
    """Compute or load pivot indices (one-time per experiment)."""
    from data.loader import get_dataset, offline_data_split
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import ConcatDataset
    from models.resnet import ResNet18_Influence
    from utils.io import load_model

    pivot_path = os.path.join(args.exp_dir, "pivot_indices.npy")
    if os.path.exists(pivot_path):
        indices = np.load(pivot_path).astype(np.int64)
        print(f"[pivot] Loaded {len(indices)} pivot indices from {pivot_path}")
        return indices

    # Build shadow pool no-aug
    get_dataset(args)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(args.data_mean, args.data_std),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform,
    )
    shadow_pool_no_aug = offline_data_split(
        ConcatDataset([train_ds, test_ds]), args.seed, "shadow"
    )

    target_model = ResNet18_Influence(num_classes=args.num_classes).to(device)
    target_model = load_model(
        target_model, os.path.join(args.exp_dir, "target_model.pt"), device
    )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    return select_pivot_data(
        target_model=target_model,
        shadow_pool_no_aug=shadow_pool_no_aug,
        k_per_class=int(getattr(args, "k_pivot", 100)),
        num_classes=args.num_classes,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        exp_dir=args.exp_dir,
    )


def _run_one(args, shadow_id: int, device: torch.device,
             pivot_indices: "np.ndarray") -> None:
    """Train shadow model pair and compute influence for a single shadow_id,
    skipping steps whose output files already exist."""
    shadow_dir    = os.path.join(args.exp_dir, "shadows", str(shadow_id))
    out_path      = os.path.join(shadow_dir, "shadow_model_out.pt")
    in_path       = os.path.join(shadow_dir, "shadow_model_in.pt")
    ckpt_path     = os.path.join(shadow_dir, "checkpoint.pt")
    lira_path     = os.path.join(shadow_dir, "lira_stats.npy")

    # --- Train ---
    both_done = os.path.exists(out_path) and os.path.exists(in_path)
    if both_done and not os.path.exists(ckpt_path):
        print(f"[shadow {shadow_id}] OUT+IN models exist — skipping training.")
    else:
        if os.path.exists(ckpt_path):
            print(f"[shadow {shadow_id}] === Resuming shadow model training ===")
        else:
            print(f"[shadow {shadow_id}] === Training shadow model pair ===")
        train_shadow(args, shadow_id, device, pivot_indices=pivot_indices)

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

    pivot_indices = _ensure_pivot(args, device)

    for sid in shadow_ids:
        print(f"\n{'='*60}")
        print(f"Shadow model {sid} / {args.n_shadow_models - 1}")
        print(f"{'='*60}")
        _run_one(args, sid, device, pivot_indices=pivot_indices)

    print("\nAll done.")


if __name__ == "__main__":
    main()
