"""Shadow model training on the shadow pool using CE loss only.

Threat model / data layout
--------------------------
target pool  (D)       : 20 000 points — the candidate set the target model
                         was trained on.  The query set D_query ⊂ D is a
                         balanced subset saved by train_target.py.

shadow pool            : 20 000 *disjoint* points used only for shadow training.
                         Shadows never see target-pool data during training.

query set (D_query)    : fixed balanced set of n_query points (members + non-
                         members from D) saved as query_indices.npy by
                         train_target.py.

Outputs (per shadow_id)
-----------------------
{exp_dir}/shadows/{shadow_id}/shadow_model.pt
{exp_dir}/shadows/{shadow_id}/checkpoint.pt     (overwritten each epoch)
"""

import copy
import gc
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.loader import load_dataset, get_dataset
from models.resnet import ResNet18_Influence
from training.trainer import evaluate, build_optimizer, build_scheduler
from utils.io import save_model, load_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_shadow_pool(args):
    """Return the shadow pool with augmentation.

    Shadow models train exclusively on the shadow pool (disjoint from the
    target pool).  Never pass data_type='target' here — that would let shadow
    models see target-pool data, violating the threat model.
    """
    get_dataset(args)  # sets args.data_mean, args.data_std, args.num_classes
    return load_dataset(args, data_type="shadow")


def _shadow_dir(args, shadow_id):
    return os.path.join(args.exp_dir, "shadows", str(shadow_id))


def _save_checkpoint(out_dir, epoch, student, optimizer, scheduler, best_val_acc):
    ckpt = {
        "epoch": epoch,
        "student_state": student.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
    }
    tmp_path = os.path.join(out_dir, "checkpoint.pt.tmp")
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, ckpt_path)


def _load_checkpoint(ckpt_path, student, optimizer, scheduler, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["student_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    best_val_acc = ckpt["best_val_acc"]
    return ckpt["epoch"] + 1, best_val_acc


def _flush_state_dict(state_dict, path, num_classes):
    m = ResNet18_Influence(num_classes=num_classes)
    m.load_state_dict(state_dict)
    save_model(m, path)
    del m


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train_shadow(args, shadow_id, device):
    """Train a single shadow model on the shadow pool using CE loss only."""
    out_dir    = _shadow_dir(args, shadow_id)
    model_path = os.path.join(out_dir, "shadow_model.pt")
    ckpt_path  = os.path.join(out_dir, "checkpoint.pt")

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Shadow pool — training data (disjoint from target pool)
    # ------------------------------------------------------------------
    shadow_pool = _build_shadow_pool(args)
    shadow_pool_size = len(shadow_pool)

    # Shadow training uses its own 50/50 IN / OUT split over the shadow pool.
    np.random.seed(2025 + shadow_id)
    shadow_all = np.arange(shadow_pool_size)
    shadow_in_indices  = np.random.choice(
        shadow_all,
        int(args.pkeep * shadow_pool_size),
        replace=False,
    )
    shadow_out_indices = np.setdiff1d(shadow_all, shadow_in_indices)

    train_ds = Subset(shadow_pool, shadow_in_indices.tolist())
    val_ds   = Subset(shadow_pool, shadow_out_indices.tolist())

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 2. Student model + optimiser
    # ------------------------------------------------------------------
    student   = ResNet18_Influence(num_classes=args.num_classes).to(device)
    optimizer = build_optimizer(args, student.parameters())
    scheduler = build_scheduler(args, optimizer)

    ce_criterion = nn.CrossEntropyLoss()

    best_model_state = None
    best_val_acc     = 0.0
    start_epoch      = 1

    # ------------------------------------------------------------------
    # 3. Resume from checkpoint
    # ------------------------------------------------------------------
    if os.path.exists(ckpt_path):
        print(f"[shadow {shadow_id}] Resuming from checkpoint {ckpt_path}")
        start_epoch, best_val_acc = _load_checkpoint(
            ckpt_path, student, optimizer, scheduler, device
        )
        print(f"[shadow {shadow_id}] Resumed at epoch {start_epoch}, "
              f"best_val_acc so far: {best_val_acc:.4f}")
        if os.path.exists(model_path):
            tmp = ResNet18_Influence(num_classes=args.num_classes)
            tmp = load_model(tmp, model_path, torch.device("cpu"))
            best_model_state = tmp.state_dict()
            del tmp

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        running_loss = 0.0
        n_batches = 0

        for x, y in train_dl:
            if x.size(0) == 1:
                continue
            x, y = x.to(device), y.to(device)

            loss = ce_criterion(student(x), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

        if n_batches > 0:
            print(f"[shadow {shadow_id}] Epoch [{epoch:3d}/{args.epochs}] "
                  f"train_loss={running_loss/n_batches:.4f}")

        val_acc = evaluate(student, val_dl, ce_criterion, device)[1]
        print(f"[shadow {shadow_id}] Epoch [{epoch:3d}/{args.epochs}]  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            best_model_state = copy.deepcopy(student.cpu().state_dict())
            student.to(device)
            _flush_state_dict(best_model_state, model_path, args.num_classes)

        if scheduler is not None:
            scheduler.step()

        _save_checkpoint(out_dir, epoch, student, optimizer, scheduler, best_val_acc)

    if best_model_state is None:
        best_model_state = copy.deepcopy(student.cpu().state_dict())
        student.to(device)

    del student, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Quality gate
    # ------------------------------------------------------------------
    if best_val_acc < args.imitate_acc:
        print(f"[shadow {shadow_id}] WARNING: best_val_acc={best_val_acc:.4f} < "
              f"imitate_acc={args.imitate_acc:.4f}. Discarding model, returning None.")
        return None

    # ------------------------------------------------------------------
    # 6. Save final outputs
    # ------------------------------------------------------------------
    _flush_state_dict(best_model_state, model_path, args.num_classes)
    print(f"[shadow {shadow_id}] Best val accuracy: {best_val_acc:.4f}  "
          f"Model saved to {model_path}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"[shadow {shadow_id}] Epoch checkpoint removed (training complete)")

    return None


class _SkipSingleton:
    """Wraps a DataLoader and skips batches of size 1."""

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for batch in self._loader:
            if batch[0].size(0) == 1:
                continue
            yield batch

    def __len__(self):
        return len(self._loader)
