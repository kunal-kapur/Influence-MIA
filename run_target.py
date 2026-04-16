"""Entry point for training the target model on CIFAR-10."""
import argparse
import sys
import yaml
import torch
from pathlib import Path
from argparse import Namespace

sys.path.insert(0, str(Path(__file__).parent))

from training import train_target


def parse_args():
    parser = argparse.ArgumentParser(description="Train target model for MIA")
    parser.add_argument("--dataset", default="cifar10", help="Dataset name")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device index (-1 for CPU)")
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Root directory for all experiment outputs (default: outputs/)",
    )
    return parser.parse_args()


def load_config(dataset):
    config_dir = Path(__file__).parent / "config"
    yaml_path = config_dir / f"{dataset}.yaml"
    yml_path = config_dir / f"{dataset}.yml"
    config_path = yaml_path if yaml_path.exists() else yml_path
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config for dataset '{dataset}' (expected {yaml_path} or {yml_path})."
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("warmup_epochs", 1)
    cfg.setdefault("temperature", 1.0)
    cfg.setdefault("margin_weight", 1.0)
    return Namespace(**cfg)


def make_exp_name(args) -> str:
    """Build a human-readable experiment name from the key hyperparameters."""
    parts = [
        f"ep{args.epochs}",
        f"lr{args.lr}",
        f"bs{args.batch_size}",
        f"seed{args.seed}",
    ]
    # Include shadow/imitation hparams if present in config
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


def main():
    cli = parse_args()

    args = load_config(cli.dataset)
    args.dataset = cli.dataset

    # Build experiment directory: <output_dir>/<exp_name>/<dataset>/
    exp_name = make_exp_name(args)
    args.exp_dir = str(Path(cli.output_dir) / exp_name / cli.dataset)
    print(f"Experiment directory: {args.exp_dir}")

    if torch.cuda.is_available() and cli.cuda >= 0:
        gpu_count = torch.cuda.device_count()
        if cli.cuda >= gpu_count:
            print(
                f"Requested cuda:{cli.cuda}, but only {gpu_count} GPU(s) visible. Falling back to cuda:0."
            )
            cli.cuda = 0
        args.device = f"cuda:{cli.cuda}"
        gpu_name = torch.cuda.get_device_name(cli.cuda)
        print(f"CUDA available: {gpu_count} GPU(s). Using {args.device} ({gpu_name}).")
    else:
        args.device = "cpu"
        if cli.cuda >= 0:
            print("CUDA not available; using CPU.")

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Config: {vars(args)}\n")

    train_target(args, device)


if __name__ == "__main__":
    main()
