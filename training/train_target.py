import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

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

    # Pool-relative indices (0 .. pool_size-1)
    train_pool_indices    = np.array(train_ds.indices,     dtype=np.int64)
    nonmember_pool_indices = np.array(nonmember_ds.indices, dtype=np.int64)

    # ------------------------------------------------------------------
    # 3. Build a balanced query set  D_query = D_a ∪ D_b
    #    D_a ⊂ D_train      (member     queries)
    #    D_b ⊂ D \ D_train  (non-member queries)
    #    |D_a| = |D_b| = min(n_train, n_nonmember)
    #
    # Also build a reserved set (2x query size) from the remaining points,
    # disjoint from D_query.  The reserved set is used to fit GMM buckets
    # without leaking query-point labels into the model-selection step.
    # ------------------------------------------------------------------
    n_query_half = min(n_train, n_nonmember)

    rng = np.random.default_rng(args.seed)

    # --- query set ---
    da_pool_indices = rng.choice(train_pool_indices,     n_query_half, replace=False)
    db_pool_indices = rng.choice(nonmember_pool_indices, n_query_half, replace=False)

    # query_pool_indices: pool-relative positions of each query point,
    # in the order [Da members ... Db non-members].
    query_pool_indices = np.concatenate([da_pool_indices, db_pool_indices])  # (2*n_query_half,)
    n_query = len(query_pool_indices)

    # ground_truth[i] = 1 if query point i is a member of D_train, else 0.
    ground_truth = np.zeros(n_query, dtype=np.int32)
    ground_truth[:n_query_half] = 1  # first half are Da (members)

    # --- reserved set (2x query size, disjoint from query set) ---
    # The reserved set has 2*n_query points total (4*n_query_half), drawn
    # from the pool points not already used in D_query.
    n_reserved_half = 2 * n_query_half  # reserved = 2x query size
    train_remaining     = np.setdiff1d(train_pool_indices,     da_pool_indices)
    nonmember_remaining = np.setdiff1d(nonmember_pool_indices, db_pool_indices)
    # Cap to available pool remainder (graceful fallback for small pools).
    n_reserved_half = min(n_reserved_half, len(train_remaining), len(nonmember_remaining))

    ra_pool_indices = rng.choice(train_remaining,     n_reserved_half, replace=False)
    rb_pool_indices = rng.choice(nonmember_remaining, n_reserved_half, replace=False)

    reserved_pool_indices = np.concatenate([ra_pool_indices, rb_pool_indices])
    n_reserved = len(reserved_pool_indices)
    reserved_ground_truth = np.zeros(n_reserved, dtype=np.int32)
    reserved_ground_truth[:n_reserved_half] = 1

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
    model = ResNet18_Influence(num_classes=args.num_classes).to(device)
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
    # target_train_indices.npy    — pool-relative indices of D_train  (n_train,)
    # query_indices.npy           — pool-relative indices of D_query  (2*n_query_half,)
    #                               order: [Da members | Db non-members]
    # ground_truth.npy            — 1/0 membership labels aligned to query_indices
    # reserved_indices.npy        — pool-relative indices of reserved set (2*n_reserved_half,)
    #                               order: [Ra members | Rb non-members]
    # reserved_ground_truth.npy   — 1/0 membership labels aligned to reserved_indices
    # ------------------------------------------------------------------
    save_array(train_pool_indices,    os.path.join(args.exp_dir, "target_train_indices.npy"))
    save_array(query_pool_indices,    os.path.join(args.exp_dir, "query_indices.npy"))
    save_array(ground_truth,          os.path.join(args.exp_dir, "ground_truth.npy"))
    save_array(reserved_pool_indices, os.path.join(args.exp_dir, "reserved_indices.npy"))
    save_array(reserved_ground_truth, os.path.join(args.exp_dir, "reserved_ground_truth.npy"))

    print(
        f"[target] Saved metadata:\n"
        f"  target_train_indices : {n_train} points\n"
        f"  query_indices        : {n_query} points  "
        f"({n_query_half} members + {n_query_half} non-members)\n"
        f"  ground_truth         : {ground_truth.sum()} members, "
        f"{(ground_truth == 0).sum()} non-members\n"
        f"  reserved_indices     : {n_reserved} points  "
        f"({n_reserved_half} members + {n_reserved_half} non-members)\n"
        f"  reserved_ground_truth: {reserved_ground_truth.sum()} members, "
        f"{(reserved_ground_truth == 0).sum()} non-members"
    )
