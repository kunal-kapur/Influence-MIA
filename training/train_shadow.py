import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from tqdm import tqdm

from data.loader import load_dataset, get_dataset
from models.resnet import ResNet18_Influence
from training.trainer import evaluate
from utils.io import save_model, load_model, save_array


def _build_shared_pool(args):
    """Concatenate target and shadow splits (~40,000 samples) with augmentation."""
    get_dataset(args)  # sets args.data_mean, args.data_std, args.num_classes
    target_split = load_dataset(args, data_type="target")
    shadow_split = load_dataset(args, data_type="shadow")
    return ConcatDataset([target_split, shadow_split])


def _compute_in_mask(n_shadow_models, pool_size, pkeep, shadow_id):
    """Staggered LiRA scheme: returns boolean IN mask for shadow_id."""
    np.random.seed(2025)
    keep_matrix = np.random.uniform(0, 1, size=(n_shadow_models, pool_size))
    order_matrix = keep_matrix.argsort(0)
    keep_matrix = order_matrix < int(pkeep * n_shadow_models)
    return keep_matrix[shadow_id]  # shape (pool_size,)


def _shadow_dir(args, shadow_id):
    return os.path.join(args.exp_dir, "shadows", str(shadow_id))


def _save_checkpoint(out_dir, epoch, student, optimizer, scheduler,
                     best_val_acc, warmup_model):
    """Atomically write a mid-training checkpoint."""
    ckpt = {
        "epoch": epoch,
        "student_state": student.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_val_acc": best_val_acc,
        "warmup_model_state": warmup_model.state_dict() if warmup_model is not None else None,
    }
    tmp_path = os.path.join(out_dir, "checkpoint.pt.tmp")
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, ckpt_path)  # atomic on POSIX


def _load_checkpoint(ckpt_path, student, optimizer, scheduler, device, num_classes):
    """Load checkpoint and return (start_epoch, best_val_acc, warmup_model)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["student_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    best_val_acc = ckpt["best_val_acc"]

    warmup_model = None
    if ckpt["warmup_model_state"] is not None:
        warmup_model = ResNet18_Influence(num_classes=num_classes)
        warmup_model.load_state_dict(ckpt["warmup_model_state"])

    start_epoch = ckpt["epoch"] + 1  # resume from next epoch
    return start_epoch, best_val_acc, warmup_model


def train_shadow(args, shadow_id, device):
    """Train a single shadow model for the given shadow_id using IMIA Algorithm 1.

    Phase 1 (warmup):  CE loss only for epochs 1 .. warmup_epochs-1
    Phase 2 (imitate): MSE distillation against frozen target model, blended
                       with CE loss, for epochs warmup_epochs .. epochs

    Resumes automatically from checkpoint.pt if training was interrupted.

    Outputs
    -------
    {args.exp_dir}/shadows/{shadow_id}/in_mask.npy
    {args.exp_dir}/shadows/{shadow_id}/shadow_model.pt
    {args.exp_dir}/shadows/{shadow_id}/warmup_model.pt  (if warmup ran)
    {args.exp_dir}/shadows/{shadow_id}/checkpoint.pt    (overwritten each epoch)
    """
    out_dir = _shadow_dir(args, shadow_id)
    model_path = os.path.join(out_dir, "shadow_model.pt")
    mask_path = os.path.join(out_dir, "in_mask.npy")
    warmup_path = os.path.join(out_dir, "warmup_model.pt")
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")

    # --- 1. Shared pool (~40,000 samples) ---
    shared_pool = _build_shared_pool(args)
    pool_size = len(shared_pool)

    # --- 2. IN mask ---
    in_mask = _compute_in_mask(
        n_shadow_models=args.n_shadow_models,
        pool_size=pool_size,
        pkeep=args.pkeep,
        shadow_id=shadow_id,
    )
    os.makedirs(out_dir, exist_ok=True)
    save_array(in_mask, mask_path)
    print(f"[shadow {shadow_id}] IN mask saved to {mask_path}  "
          f"(IN={in_mask.sum()}, OUT={(~in_mask).sum()})")

    # --- 3. Train / val subsets ---
    all_indices = np.arange(pool_size)
    in_indices = all_indices[in_mask]
    out_indices = all_indices[~in_mask]

    train_ds = Subset(shared_pool, in_indices.tolist())
    val_ds = Subset(shared_pool, out_indices.tolist())

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- Setup ---
    # Load frozen teacher (target model)
    teacher_model = ResNet18_Influence(num_classes=args.num_classes).to(device)
    teacher_path = os.path.join(args.exp_dir, "target_model.pt")
    teacher_model = load_model(teacher_model, teacher_path, device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    print(f"[shadow {shadow_id}] Teacher loaded from {teacher_path}")

    # Fresh student
    student = ResNet18_Influence(num_classes=args.num_classes).to(device)

    optimizer = optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()

    alpha = 0.5  # weight between imitation loss and CE loss

    warmup_model = None
    best_model = None
    best_val_acc = 0.0
    start_epoch = 1

    # --- Resume from checkpoint if one exists ---
    if os.path.exists(ckpt_path):
        print(f"[shadow {shadow_id}] Resuming from checkpoint {ckpt_path}")
        start_epoch, best_val_acc, warmup_model = _load_checkpoint(
            ckpt_path, student, optimizer, scheduler, device, args.num_classes
        )
        print(f"[shadow {shadow_id}] Resumed at epoch {start_epoch}, "
              f"best_val_acc so far: {best_val_acc:.4f}")
        # Restore best_model from shadow_model.pt if it already exists
        if os.path.exists(model_path):
            best_model = ResNet18_Influence(num_classes=args.num_classes)
            best_model = load_model(best_model, model_path, torch.device("cpu"))

    # --- Training loop (Algorithm 1) ---
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        pbar = tqdm(train_dl, leave=False,
                    desc=f"[shadow {shadow_id}] Epoch {epoch}/{args.epochs}")

        for x, y in pbar:
            if x.size(0) == 1:
                continue

            x, y = x.to(device), y.to(device)

            student_logits = student(x)
            ce_loss = ce_criterion(student_logits, y)

            if epoch < args.warmup_epochs:
                # Phase 1: warmup — CE loss only
                loss = ce_loss
                pbar.set_postfix_str(f"loss: {loss.item():.3f}")
            else:
                # Phase 2: imitation — MSE distillation against frozen teacher
                with torch.no_grad():
                    teacher_logits = teacher_model(x)

                # Temperature scaling
                s_logits = student_logits / args.temperature
                t_logits = teacher_logits / args.temperature

                # Weight matrix: ones everywhere, margin_weight at true class
                # and at the max incorrect class of teacher
                batch_idx = torch.arange(t_logits.size(0), device=device)
                tmp = t_logits.clone()
                tmp[batch_idx, y] = float("-inf")
                max_incorrect = tmp.argmax(dim=1)

                weight_matrix = torch.ones_like(s_logits)
                weight_matrix[batch_idx, y] = args.margin_weight
                weight_matrix[batch_idx, max_incorrect] = args.margin_weight

                imitate_loss = mse_criterion(
                    s_logits * weight_matrix,
                    t_logits * weight_matrix,
                )
                imitate_loss = imitate_loss / weight_matrix.mean()

                loss = alpha * imitate_loss + (1.0 - alpha) * ce_loss
                pbar.set_postfix_str(
                    f"loss: {loss.item():.3f}  "
                    f"ce: {ce_loss.item():.3f}  "
                    f"imitate: {imitate_loss.item():.3f}"
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save warmup checkpoint at the exact end of warmup phase
        if epoch + 1 == args.warmup_epochs:
            warmup_model = copy.deepcopy(student)
            print(f"[shadow {shadow_id}] Warmup checkpoint saved at epoch {epoch + 1}")

        # Track best model by val accuracy only after warmup
        if epoch >= args.warmup_epochs:
            val_acc = evaluate(student, val_dl, ce_criterion, device)[1]
            print(f"[shadow {shadow_id}] Epoch [{epoch:3d}/{args.epochs}]  "
                  f"val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(student)
                # Write best model immediately so a crash between epochs
                # doesn't lose the best weights seen so far
                save_model(best_model, model_path)

        scheduler.step()

        # Save epoch checkpoint (overwrites previous; atomic rename)
        _save_checkpoint(out_dir, epoch, student, optimizer, scheduler,
                         best_val_acc, warmup_model)

    # Fall back to last student state if no post-warmup epoch ran
    if best_model is None:
        best_model = copy.deepcopy(student)

    # --- Quality gate ---
    if best_val_acc < args.imitate_acc:
        print(f"[shadow {shadow_id}] WARNING: best_val_acc={best_val_acc:.4f} < "
              f"imitate_acc={args.imitate_acc:.4f}. Discarding model, returning None.")
        return None

    # --- Save final outputs ---
    # best_model.pt is already written incrementally; do a final save to be explicit
    save_model(best_model, model_path)
    print(f"[shadow {shadow_id}] Best val accuracy: {best_val_acc:.4f}  "
          f"Model saved to {model_path}")

    if warmup_model is not None:
        save_model(warmup_model, warmup_path)
        print(f"[shadow {shadow_id}] Warmup model saved to {warmup_path}")

    # Remove epoch checkpoint once training is cleanly complete
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"[shadow {shadow_id}] Epoch checkpoint removed (training complete)")

    return best_model


class _SkipSingleton:
    """Wraps a DataLoader and skips batches of size 1."""

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for batch in self._loader:
            inputs = batch[0]
            if inputs.size(0) == 1:
                continue
            yield batch

    def __len__(self):
        return len(self._loader)
