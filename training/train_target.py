import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from data import load_dataset
from models import ResNet18_Influence
from training.trainer import train
from utils import save_model, save_array


def train_target(args, device):
    # 1. Load the 20,000-sample target pool
    target_pool = load_dataset(args, data_type="target")

    # 2. Split 80/20 into train / val
    n_train = int(0.8 * len(target_pool))
    n_val = len(target_pool) - n_train

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(target_pool, [n_train, n_val], generator=generator)

    # Capture the underlying indices relative to target_pool
    train_indices = np.array(train_ds.indices)
    val_indices = np.array(val_ds.indices)

    # 3. DataLoaders
    # Hyperparameters are configured in cifar10.yaml (lr, batch_size, optimizer, scheduler, dropout, num_workers, etc.)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # 4. Model
    model = ResNet18_Influence(num_classes=args.num_classes).to(device)

    # 5. Train
    model = train(model, train_loader, val_loader, args, device)

    # 6. Save model
    model_path = f"{args.exp_dir}/target_model.pt"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    # 7. Save indices
    train_idx_path = f"{args.exp_dir}/target_train_indices.npy"
    val_idx_path = f"{args.exp_dir}/target_val_indices.npy"
    save_array(train_indices, train_idx_path)
    save_array(val_indices, val_idx_path)
    print(f"Indices saved to {train_idx_path} and {val_idx_path}")

