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
    os.makedirs(args.exp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load the 20 000-sample target pool  (D in the threat model)
    # ------------------------------------------------------------------
    target_pool = load_dataset(args, data_type="target")
    pool_size = len(target_pool)

    # ------------------------------------------------------------------
    # 2. Split D into D_train (80 %) and D_nonmember (20 %)
    # ------------------------------------------------------------------
    n_train = int(0.8 * pool_size)
    n_nonmember = pool_size - n_train

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, nonmember_ds = random_split(
        target_pool, [n_train, n_nonmember], generator=generator
    )

    # target_pool is a torch Subset over the full CIFAR-10 train+test concat.
    # We keep metadata in full-dataset global index space.
    target_pool_global_indices = np.asarray(target_pool.indices, dtype=np.int64)
    train_pool_indices = np.array(train_ds.indices, dtype=np.int64)
    nonmember_pool_indices = np.array(nonmember_ds.indices, dtype=np.int64)

    # Convert local target-pool indices to global dataset indices.
    train_global_indices = target_pool_global_indices[train_pool_indices]
    target_nonmember_global_indices = target_pool_global_indices[nonmember_pool_indices]

    # ------------------------------------------------------------------
    # 3. Build a balanced query set D_query = D_a ∪ D_b in global index space.
    #    D_a ⊂ D_train                  (member queries)
    #    D_b ⊂ (D \ D_train)            (non-member queries inside shared pool D)
    #    |D_a| = |D_b| = min(|D_train|, |D \ D_train|)
    # ------------------------------------------------------------------
    n_query_half = min(len(train_global_indices), len(target_nonmember_global_indices))

    rng = np.random.default_rng(args.seed)
    da_global_indices = rng.choice(train_global_indices, n_query_half, replace=False)
    db_global_indices = rng.choice(target_nonmember_global_indices, n_query_half, replace=False)

    # query_global_indices are full-dataset indices in fixed order:
    # [Da members ... Db non-members].
    query_global_indices = np.concatenate([da_global_indices, db_global_indices])
    n_query = len(query_global_indices)

    # Dataset integrity checks for the MIA threat-model assumptions.
    assert n_train + n_nonmember <= pool_size, (
        f"target split sizes exceed shared pool: train={n_train}, "
        f"nonmember={n_nonmember}, pool={pool_size}"
    )
    assert n_query <= pool_size, (
        f"query set size {n_query} exceeds shared pool size {pool_size}."
    )
    query_outside_pool = np.setdiff1d(
        query_global_indices,
        target_pool_global_indices,
        assume_unique=False,
    )
    assert len(query_outside_pool) == 0, (
        "query set must be drawn exclusively from shared pool D; "
        f"found {len(query_outside_pool)} points outside D"
    )

    # ground_truth[i] = 1 if query point i is a target member, else 0.
    ground_truth = np.zeros(n_query, dtype=np.int32)
    ground_truth[:n_query_half] = 1  # first half are Da (members)

    # ------------------------------------------------------------------
    # 4. DataLoaders for training
    # ------------------------------------------------------------------
    use_cuda = device.type == "cuda"
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader    = DataLoader(train_ds,     shuffle=True,  **loader_kwargs)
    val_loader      = DataLoader(nonmember_ds, shuffle=False, **loader_kwargs)

    # ------------------------------------------------------------------
    # 5. Model, optimizer, scheduler
    # ------------------------------------------------------------------
    model = ResNet18_Influence(
        num_classes=args.num_classes,
        in_channels=getattr(args, "in_channels", 3),
    ).to(device)
    optimizer = build_optimizer(args, model.parameters())
    scheduler = build_scheduler(args, optimizer)

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler() if use_amp else None

    criterion  = nn.CrossEntropyLoss()
    model_path = os.path.join(args.exp_dir, "target_model.pt")
    ckpt_path  = os.path.join(args.exp_dir, "target_checkpoint.pt")

    best_val_acc = 0.0
    start_epoch  = 1

    # ------------------------------------------------------------------
    # 6. Resume from checkpoint if one exists
    # ------------------------------------------------------------------
    if os.path.exists(ckpt_path):
        print(f"[target] Resuming from checkpoint {ckpt_path}")
        start_epoch, best_val_acc = _load_checkpoint(
            ckpt_path, model, optimizer, scheduler, device
        )
        print(f"[target] Resumed at epoch {start_epoch}, best_val_acc so far: {best_val_acc:.4f}")

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
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

    if not os.path.exists(model_path):
        save_model(model, model_path)

    print(f"[target] Best val accuracy: {best_val_acc:.4f}  Model saved to {model_path}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print("[target] Epoch checkpoint removed (training complete)")

    # ------------------------------------------------------------------
    # 8. Save metadata
    #
    # target_train_indices.npy  — global indices of D_train  (n_train,)
    # query_indices.npy         — global indices of D_query  (2*n_query_half,)
    #                             order: [Da members | Db in-pool non-members]
    # ground_truth.npy          — 1/0 membership labels aligned to query_indices (n_query,)
    # ------------------------------------------------------------------
    save_array(train_global_indices, os.path.join(args.exp_dir, "target_train_indices.npy"))
    save_array(query_global_indices, os.path.join(args.exp_dir, "query_indices.npy"))
    save_array(ground_truth,         os.path.join(args.exp_dir, "ground_truth.npy"))

    print(
        f"[target] Saved metadata:\n"
        f"  target_train_indices : {n_train} points (global indices)\n"
        f"  query_indices        : {n_query} points  "
        f"({n_query_half} members + {n_query_half} in-pool non-members)\n"
        f"  ground_truth         : {ground_truth.sum()} members, "
        f"{(ground_truth == 0).sum()} non-members"
    )
