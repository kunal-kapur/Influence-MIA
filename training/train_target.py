import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data import load_dataset
from models import ResNet18_Influence
from training.trainer import train_one_epoch, evaluate, build_optimizer, build_scheduler
from utils import save_model, save_array


def _save_checkpoint(out_dir, epoch, model, optimizer, scheduler, best_val_acc):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
    }
    tmp_path = os.path.join(out_dir, "target_checkpoint.pt.tmp")
    ckpt_path = os.path.join(out_dir, "target_checkpoint.pt")
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, ckpt_path)


def _load_checkpoint(ckpt_path, model, optimizer, scheduler, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"] + 1, ckpt["best_val_acc"]


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

    # 4. Model, optimizer, scheduler
    model = ResNet18_Influence(num_classes=args.num_classes).to(device)
    optimizer = build_optimizer(args, model.parameters())
    scheduler = build_scheduler(args, optimizer)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    criterion = nn.CrossEntropyLoss()
    model_path = os.path.join(args.exp_dir, "target_model.pt")
    ckpt_path = os.path.join(args.exp_dir, "target_checkpoint.pt")

    best_val_acc = 0.0
    start_epoch = 1

    # 5. Resume from checkpoint if one exists
    if os.path.exists(ckpt_path):
        print(f"[target] Resuming from checkpoint {ckpt_path}")
        start_epoch, best_val_acc = _load_checkpoint(
            ckpt_path, model, optimizer, scheduler, device
        )
        print(f"[target] Resumed at epoch {start_epoch}, best_val_acc so far: {best_val_acc:.4f}")

    # 6. Training loop
    eval_interval = 5
    val_loss, val_acc = float("nan"), float("nan")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        if scheduler is not None:
            scheduler.step()

        if epoch % eval_interval == 0 or epoch == args.epochs:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, model_path)

        print(
            f"[target] Epoch [{epoch:3d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        _save_checkpoint(args.exp_dir, epoch, model, optimizer, scheduler, best_val_acc)

    # Ensure final model is saved (covers case where last epoch wasn't an eval epoch)
    if not os.path.exists(model_path):
        save_model(model, model_path)

    print(f"[target] Best val accuracy: {best_val_acc:.4f}  Model saved to {model_path}")

    # Remove checkpoint once training completes cleanly
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("[target] Epoch checkpoint removed (training complete)")

    # 7. Save indices
    train_idx_path = f"{args.exp_dir}/target_train_indices.npy"
    val_idx_path = f"{args.exp_dir}/target_val_indices.npy"
    save_array(train_indices, train_idx_path)
    save_array(val_indices, val_idx_path)
    print(f"Indices saved to {train_idx_path} and {val_idx_path}")
