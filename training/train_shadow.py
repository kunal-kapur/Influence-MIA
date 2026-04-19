"""Shadow model training (IMIA Algorithm 1).

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
{exp_dir}/shadows/{shadow_id}/shadow_model_out.pt   — imitative OUT model (f_out)
{exp_dir}/shadows/{shadow_id}/shadow_model_in.pt    — imitative IN model  (f_in_pivot)
{exp_dir}/shadows/{shadow_id}/shadow_model.pt       — alias of shadow_model_out.pt
{exp_dir}/shadows/{shadow_id}/warmup_model.pt       — checkpoint at end of epoch T_warmup
{exp_dir}/shadows/{shadow_id}/checkpoint.pt         — overwritten each epoch (for resumption)
"""

import copy
import gc
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset

from data.loader import load_dataset, get_dataset, offline_data_split
from models.resnet import ResNet18_Influence
from training.trainer import evaluate, build_optimizer, build_scheduler
from training.plot_training import plot_shadow_curves
from utils.io import save_model, load_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_shadow_pool(args):
    """Shadow pool WITH augmentation — used for Phase 1 training."""
    get_dataset(args)
    return load_dataset(args, data_type="shadow")


def _build_shadow_pool_no_aug(args):
    """Shadow pool WITHOUT augmentation — used for Phase 2 pivot training."""
    get_dataset(args)
    mean = args.data_mean
    std  = args.data_std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform,
    )
    return offline_data_split(ConcatDataset([train_ds, test_ds]), args.seed, "shadow")


def _shadow_dir(args, shadow_id):
    return os.path.join(args.exp_dir, "shadows", str(shadow_id))


def _save_checkpoint(out_dir, epoch, student, optimizer, scheduler,
                     best_val_acc, warmup_state_dict):
    ckpt = {
        "epoch": epoch,
        "student_state": student.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
        "warmup_state_dict": warmup_state_dict,
    }
    tmp_path  = os.path.join(out_dir, "checkpoint.pt.tmp")
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, ckpt_path)


def _load_checkpoint(ckpt_path, student, optimizer, scheduler, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["student_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"] + 1, ckpt["best_val_acc"], ckpt.get("warmup_state_dict")


def _flush_state_dict(state_dict, path, num_classes):
    m = ResNet18_Influence(num_classes=num_classes)
    m.load_state_dict(state_dict)
    save_model(m, path)
    del m


# ---------------------------------------------------------------------------
# Imitation loss (Equation 2)
# ---------------------------------------------------------------------------

def imitation_loss(student_logits, teacher_logits, labels, num_classes):
    """Weighted MSE on log-softmax outputs (Eq. 2).

    Upweights the true class and the teacher's most-confident wrong class.
    """
    sqrt_c = num_classes ** 0.5
    w_high = sqrt_c + 1.0 / (num_classes + 2.0 * sqrt_c)
    w_low  = 1.0 / (num_classes + 2.0 * sqrt_c)

    log_s = F.log_softmax(student_logits, dim=1)  # (B, C)
    log_t = F.log_softmax(teacher_logits, dim=1)  # (B, C)

    with torch.no_grad():
        t_probs = log_t.exp()
        mask = torch.zeros_like(t_probs)
        mask.scatter_(1, labels.unsqueeze(1), float("-inf"))
        wrong_class = (t_probs + mask).argmax(dim=1)  # (B,)

    W = torch.full_like(log_s, w_low)
    W.scatter_(1, labels.unsqueeze(1), w_high)
    W.scatter_(1, wrong_class.unsqueeze(1), w_high)

    return (W * (log_s - log_t) ** 2).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train_shadow(args, shadow_id, device, pivot_indices=None):
    """Train a single shadow model pair (IMIA Algorithm 1).

    Phase 1 — Imitative OUT model:
        Epochs 1..T_warmup   : CE loss
        Epoch T_warmup       : save warmup checkpoint
        Epochs T_warmup+1..T1: imitation loss (Eq. 2)
        Save as shadow_model_out.pt

    Phase 2 — Imitative IN model:
        Reload warmup checkpoint (NOT f_out)
        Epochs 1..T2         : CE loss on D_pivot
        Save as shadow_model_in.pt

    Args:
        pivot_indices: indices into shadow_pool_no_aug for Phase 2 training.
                       If None, loaded from {exp_dir}/pivot_indices.npy.
    """
    out_dir        = _shadow_dir(args, shadow_id)
    out_model_path = os.path.join(out_dir, "shadow_model_out.pt")
    in_model_path  = os.path.join(out_dir, "shadow_model_in.pt")
    compat_path    = os.path.join(out_dir, "shadow_model.pt")
    warmup_path    = os.path.join(out_dir, "warmup_model.pt")
    ckpt_path      = os.path.join(out_dir, "checkpoint.pt")

    T1       = int(args.epochs)
    T_warmup = int(getattr(args, "warmup_epochs", 1))
    T2       = int(getattr(args, "T2", 20))
    if T_warmup < 1:
        raise ValueError(f"warmup_epochs must be >= 1, got {T_warmup}")

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Shadow pool WITH augmentation — Phase 1 training data
    # ------------------------------------------------------------------
    shadow_pool      = _build_shadow_pool(args)
    shadow_pool_size = len(shadow_pool)

    np.random.seed(2025 + shadow_id)
    shadow_all        = np.arange(shadow_pool_size)
    shadow_in_indices = np.random.choice(
        shadow_all, int(args.pkeep * shadow_pool_size), replace=False
    )
    shadow_out_indices = np.setdiff1d(shadow_all, shadow_in_indices)

    train_ds = Subset(shadow_pool, shadow_in_indices.tolist())
    val_ds   = Subset(shadow_pool, shadow_out_indices.tolist())

    use_pin  = device.type == "cuda"
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_pin,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_pin,
    )

    # ------------------------------------------------------------------
    # 2. Shadow pool WITHOUT augmentation — Phase 2 pivot data
    # ------------------------------------------------------------------
    shadow_pool_no_aug = _build_shadow_pool_no_aug(args)

    if pivot_indices is None:
        pivot_path = os.path.join(args.exp_dir, "pivot_indices.npy")
        if not os.path.exists(pivot_path):
            raise FileNotFoundError(
                f"pivot_indices.npy not found at {pivot_path}. "
                "Run select_pivot_data() first."
            )
        pivot_indices = np.load(pivot_path).astype(np.int64)
        print(f"[shadow {shadow_id}] Loaded {len(pivot_indices)} pivot indices.")

    pivot_ds = Subset(shadow_pool_no_aug, pivot_indices.tolist())
    pivot_dl = DataLoader(
        pivot_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_pin,
    )

    # ------------------------------------------------------------------
    # 3. Frozen target model (teacher for imitation loss)
    # ------------------------------------------------------------------
    teacher_path  = os.path.join(args.exp_dir, "target_model.pt")
    teacher_model = ResNet18_Influence(num_classes=args.num_classes).to(device)
    teacher_model = load_model(teacher_model, teacher_path, device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    print(f"[shadow {shadow_id}] Teacher loaded from {teacher_path}")

    # ------------------------------------------------------------------
    # 4. Student model + optimiser
    # ------------------------------------------------------------------
    student   = ResNet18_Influence(num_classes=args.num_classes).to(device)
    optimizer = build_optimizer(args, student.parameters())
    scheduler = build_scheduler(args, optimizer)

    ce_criterion = nn.CrossEntropyLoss()

    warmup_state_dict = None
    best_model_state  = None
    best_val_acc      = 0.0
    start_epoch       = 1

    history_p1 = {
        "train_loss":   [],
        "loss_type":    [],  # "CE" or "imitate" per epoch
        "val_acc":      [],  # list of (epoch, value)
        "warmup_epoch": T_warmup,
    }
    history_p2 = {"train_loss": []}

    # ------------------------------------------------------------------
    # 5. Resume from checkpoint
    # ------------------------------------------------------------------
    if os.path.exists(ckpt_path):
        print(f"[shadow {shadow_id}] Resuming from checkpoint {ckpt_path}")
        start_epoch, best_val_acc, warmup_state_dict = _load_checkpoint(
            ckpt_path, student, optimizer, scheduler, device
        )
        print(f"[shadow {shadow_id}] Resumed at epoch {start_epoch}, "
              f"best_val_acc so far: {best_val_acc:.4f}")
        if os.path.exists(out_model_path):
            tmp = ResNet18_Influence(num_classes=args.num_classes)
            tmp = load_model(tmp, out_model_path, torch.device("cpu"))
            best_model_state = tmp.state_dict()
            del tmp

    # ------------------------------------------------------------------
    # 6. Phase 1 — Imitative OUT model (Algorithm 1, Phase 1)
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, T1 + 1):
        student.train()
        running_loss = 0.0
        n_batches    = 0

        for x, y in train_dl:
            if x.size(0) == 1:
                continue
            x, y = x.to(device), y.to(device)

            student_logits = student(x)

            if epoch <= T_warmup:
                loss = ce_criterion(student_logits, y)
            else:
                with torch.no_grad():
                    teacher_logits = teacher_model(x)
                loss = imitation_loss(
                    student_logits, teacher_logits, y, args.num_classes
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

        phase_name = "CE" if epoch <= T_warmup else "imitate"
        epoch_loss = running_loss / n_batches if n_batches > 0 else float("nan")
        history_p1["train_loss"].append(epoch_loss)
        history_p1["loss_type"].append(phase_name)
        if n_batches > 0:
            print(f"[shadow {shadow_id}] Phase1 Epoch [{epoch:3d}/{T1}] "
                  f"({phase_name}) train_loss={epoch_loss:.4f}")

        # Save warmup checkpoint at the END of epoch T_warmup exactly
        if epoch == T_warmup:
            warmup_state_dict = copy.deepcopy(student.cpu().state_dict())
            student.to(device)
            _flush_state_dict(warmup_state_dict, warmup_path, args.num_classes)
            print(f"[shadow {shadow_id}] Warmup checkpoint saved at epoch {epoch}")

        # Track best OUT model quality on val set (post-warmup only)
        if epoch > T_warmup:
            val_acc = evaluate(student, val_dl, ce_criterion, device)[1]
            history_p1["val_acc"].append((epoch, val_acc))
            print(f"[shadow {shadow_id}] Phase1 Epoch [{epoch:3d}/{T1}]  val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_model_state = copy.deepcopy(student.cpu().state_dict())
                student.to(device)
                _flush_state_dict(best_model_state, out_model_path, args.num_classes)

        if scheduler is not None:
            scheduler.step()

        _save_checkpoint(out_dir, epoch, student, optimizer, scheduler,
                         best_val_acc, warmup_state_dict)

    if best_model_state is None:
        best_model_state = copy.deepcopy(student.cpu().state_dict())
        student.to(device)

    # ------------------------------------------------------------------
    # 7. Quality gate — applied to the OUT model
    # ------------------------------------------------------------------
    if best_val_acc < args.imitate_acc:
        print(f"[shadow {shadow_id}] WARNING: best_val_acc={best_val_acc:.4f} < "
              f"imitate_acc={args.imitate_acc:.4f}. Discarding shadow pair.")
        del student, teacher_model, optimizer, scheduler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return None

    # Final save of OUT model + backward-compat alias
    _flush_state_dict(best_model_state, out_model_path, args.num_classes)
    _flush_state_dict(best_model_state, compat_path, args.num_classes)
    print(f"[shadow {shadow_id}] OUT model saved → {out_model_path}")

    # ------------------------------------------------------------------
    # 8. Phase 2 — Imitative IN model (Algorithm 1, Phase 2)
    #    Resume from warmup snapshot, NOT from f_out.
    # ------------------------------------------------------------------
    if warmup_state_dict is None:
        raise RuntimeError(
            f"[shadow {shadow_id}] warmup_state_dict is None — cannot start Phase 2. "
            "Ensure T_warmup <= T1."
        )

    student.load_state_dict(warmup_state_dict)
    student.to(device)

    optimizer_p2 = build_optimizer(args, student.parameters())

    print(f"[shadow {shadow_id}] Phase 2: training IN model for {T2} epochs "
          f"on pivot set ({len(pivot_ds)} instances).")

    for epoch in range(1, T2 + 1):
        student.train()
        running_loss = 0.0
        n_batches    = 0

        for x, y in pivot_dl:
            if x.size(0) == 1:
                continue
            x, y = x.to(device), y.to(device)
            loss = ce_criterion(student(x), y)
            optimizer_p2.zero_grad()
            loss.backward()
            optimizer_p2.step()
            running_loss += loss.item()
            n_batches    += 1

        epoch_loss_p2 = running_loss / n_batches if n_batches > 0 else float("nan")
        history_p2["train_loss"].append(epoch_loss_p2)
        if n_batches > 0:
            print(f"[shadow {shadow_id}] Phase2 Epoch [{epoch:3d}/{T2}] "
                  f"train_loss={epoch_loss_p2:.4f}")

    in_state = copy.deepcopy(student.cpu().state_dict())
    _flush_state_dict(in_state, in_model_path, args.num_classes)
    print(f"[shadow {shadow_id}] IN  model saved → {in_model_path}")

    plot_shadow_curves(history_p1, history_p2, shadow_id, args.exp_dir)

    # ------------------------------------------------------------------
    # 9. Cleanup
    # ------------------------------------------------------------------
    del student, teacher_model, optimizer, optimizer_p2, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"[shadow {shadow_id}] Epoch checkpoint removed (training complete)")

    return None
