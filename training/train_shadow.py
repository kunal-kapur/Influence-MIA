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

in_mask[k][i]          : True iff query point i (a target-pool point) was
                         included in shadow k's *shadow-pool* training subset.
                         Because shadow and target pools are disjoint this mask
                         is generated synthetically via a staggered LiRA scheme
                         over the query indices — it does NOT reflect actual
                         data overlap but instead gives each query point a
                         controlled fraction of IN / OUT observations across
                         shadows, which is what LiRA requires.

Outputs (per shadow_id)
-----------------------
{exp_dir}/shadows/{shadow_id}/in_mask.npy       — bool  (n_query,)
{exp_dir}/shadows/{shadow_id}/shadow_model.pt
{exp_dir}/shadows/{shadow_id}/warmup_model.pt   (if warmup ran)
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
from utils.io import save_model, load_model, save_array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_query_indices(exp_dir: str) -> np.ndarray:
    """Load query_indices.npy written by train_target.py.

    Returns pool-relative indices of shape (n_query,).
    Raises FileNotFoundError with a clear message if missing.
    """
    path = os.path.join(exp_dir, "query_indices.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"query_indices.npy not found at {path}. "
            "Run train_target.py first — it writes this file."
        )
    return np.load(path).astype(np.int64)


def _build_shadow_pool(args):
    """Return the shadow pool with augmentation.

    Shadow models train exclusively on the shadow pool (disjoint from the
    target pool).  Never pass data_type='target' here — that would let shadow
    models see target-pool data, violating the threat model.
    """
    get_dataset(args)  # sets args.data_mean, args.data_std, args.num_classes
    return load_dataset(args, data_type="shadow")


def _compute_in_mask(n_shadow_models: int, n_query: int, pkeep: float,
                     shadow_id: int) -> np.ndarray:
    """Staggered LiRA in_mask over the query set.

    Returns a boolean array of shape (n_query,) where True means
    'query point i is treated as IN for shadow k'.

    Because the shadow pool is disjoint from the target pool, actual data
    overlap is impossible.  Instead we use the standard LiRA staggered
    scheme to assign synthetic IN / OUT labels across shadows so that each
    query point has ~pkeep * n_shadow_models IN observations — sufficient
    for Gaussian IN / OUT distribution fitting in the attack.
    """
    np.random.seed(2025)
    keep_matrix = np.random.uniform(0, 1, size=(n_shadow_models, n_query))
    order_matrix = keep_matrix.argsort(0)
    keep_matrix  = order_matrix < int(pkeep * n_shadow_models)
    return keep_matrix[shadow_id]  # shape (n_query,)


def _shadow_dir(args, shadow_id):
    return os.path.join(args.exp_dir, "shadows", str(shadow_id))


def _save_checkpoint(out_dir, epoch, student, optimizer, scheduler,
                     best_val_acc, warmup_model):
    ckpt = {
        "epoch": epoch,
        "student_state": student.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_acc": best_val_acc,
        "warmup_model_state": warmup_model.state_dict() if warmup_model is not None else None,
    }
    tmp_path = os.path.join(out_dir, "checkpoint.pt.tmp")
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, ckpt_path)


def _load_checkpoint(ckpt_path, student, optimizer, scheduler, device, num_classes):
    ckpt = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(ckpt["student_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    best_val_acc = ckpt["best_val_acc"]

    warmup_model = None
    if ckpt["warmup_model_state"] is not None:
        warmup_model = ResNet18_Influence(num_classes=num_classes)
        warmup_model.load_state_dict(ckpt["warmup_model_state"])

    return ckpt["epoch"] + 1, best_val_acc, warmup_model


def _flush_state_dict(state_dict, path, num_classes):
    m = ResNet18_Influence(num_classes=num_classes)
    m.load_state_dict(state_dict)
    save_model(m, path)
    del m


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train_shadow(args, shadow_id, device):
    """Train a single shadow model (IMIA Algorithm 1) on the shadow pool.

    Phase 1 (warmup):  CE loss only, epochs 1 .. warmup_epochs-1
    Phase 2 (imitate): MSE distillation against frozen target model blended
                       with CE loss, epochs warmup_epochs .. epochs

    in_mask is a synthetic staggered-LiRA membership mask over the query set
    (not over the shadow pool) — see module docstring for the rationale.
    """
    out_dir    = _shadow_dir(args, shadow_id)
    model_path = os.path.join(out_dir, "shadow_model.pt")
    mask_path  = os.path.join(out_dir, "in_mask.npy")
    warmup_path = os.path.join(out_dir, "warmup_model.pt")
    ckpt_path  = os.path.join(out_dir, "checkpoint.pt")

    warmup_epochs = int(getattr(args, "warmup_epochs", 1))
    temperature   = float(getattr(args, "temperature", 1.0))
    margin_weight = float(getattr(args, "margin_weight", 1.0))
    if warmup_epochs < 1:
        raise ValueError(f"warmup_epochs must be >= 1, got {warmup_epochs}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if margin_weight <= 0:
        raise ValueError(f"margin_weight must be > 0, got {margin_weight}")

    # ------------------------------------------------------------------
    # 1. Load query indices (written by train_target) — needed for in_mask
    # ------------------------------------------------------------------
    query_pool_indices = _load_query_indices(args.exp_dir)
    n_query = len(query_pool_indices)

    # ------------------------------------------------------------------
    # 2. in_mask over query set  (n_query,)
    # ------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    in_mask = _compute_in_mask(
        n_shadow_models=args.n_shadow_models,
        n_query=n_query,
        pkeep=args.pkeep,
        shadow_id=shadow_id,
    )
    assert in_mask.shape == (n_query,), (
        f"[shadow {shadow_id}] in_mask shape {in_mask.shape} != (n_query={n_query},)"
    )
    save_array(in_mask, mask_path)
    print(f"[shadow {shadow_id}] in_mask saved  "
          f"(IN={in_mask.sum()}, OUT={(~in_mask).sum()}, n_query={n_query})")

    # ------------------------------------------------------------------
    # 3. Shadow pool — training data (disjoint from target pool)
    # ------------------------------------------------------------------
    shadow_pool = _build_shadow_pool(args)
    shadow_pool_size = len(shadow_pool)

    # Shadow training uses its own 50/50 IN / OUT split over the shadow pool.
    # This split is independent of in_mask (which is over the query set).
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
    # 4. Frozen teacher (target model)
    # ------------------------------------------------------------------
    teacher_path  = os.path.join(args.exp_dir, "target_model.pt")
    teacher_model = ResNet18_Influence(num_classes=args.num_classes).to(device)
    teacher_model = load_model(teacher_model, teacher_path, device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    print(f"[shadow {shadow_id}] Teacher loaded from {teacher_path}")

    # ------------------------------------------------------------------
    # 5. Student model + optimiser
    # ------------------------------------------------------------------
    student   = ResNet18_Influence(num_classes=args.num_classes).to(device)
    optimizer = build_optimizer(args, student.parameters())
    scheduler = build_scheduler(args, optimizer)

    ce_criterion  = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    alpha = 0.5

    warmup_model_state = None
    best_model_state   = None
    best_val_acc       = 0.0
    start_epoch        = 1

    # ------------------------------------------------------------------
    # 6. Resume from checkpoint
    # ------------------------------------------------------------------
    if os.path.exists(ckpt_path):
        print(f"[shadow {shadow_id}] Resuming from checkpoint {ckpt_path}")
        start_epoch, best_val_acc, warmup_model = _load_checkpoint(
            ckpt_path, student, optimizer, scheduler, device, args.num_classes
        )
        if warmup_model is not None:
            warmup_model_state = warmup_model.state_dict()
            del warmup_model
        print(f"[shadow {shadow_id}] Resumed at epoch {start_epoch}, "
              f"best_val_acc so far: {best_val_acc:.4f}")
        if os.path.exists(model_path):
            tmp = ResNet18_Influence(num_classes=args.num_classes)
            tmp = load_model(tmp, model_path, torch.device("cpu"))
            best_model_state = tmp.state_dict()
            del tmp

    # ------------------------------------------------------------------
    # 7. Training loop (Algorithm 1)
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        running_loss = running_ce_loss = running_imitate_loss = 0.0
        n_batches = 0

        for x, y in train_dl:
            if x.size(0) == 1:
                continue
            x, y = x.to(device), y.to(device)

            student_logits = student(x)
            ce_loss = ce_criterion(student_logits, y)

            if epoch < warmup_epochs:
                loss = ce_loss
            else:
                with torch.no_grad():
                    teacher_logits = teacher_model(x)

                s_logits = student_logits / temperature
                t_logits = teacher_logits / temperature

                batch_idx = torch.arange(t_logits.size(0), device=device)
                tmp_t = t_logits.clone()
                tmp_t[batch_idx, y] = float("-inf")
                max_incorrect = tmp_t.argmax(dim=1)

                weight_matrix = torch.ones_like(s_logits)
                weight_matrix[batch_idx, y]            = margin_weight
                weight_matrix[batch_idx, max_incorrect] = margin_weight

                imitate_loss = mse_criterion(
                    s_logits * weight_matrix,
                    t_logits * weight_matrix,
                ) / weight_matrix.mean()

                loss = alpha * imitate_loss + (1.0 - alpha) * ce_loss
                running_imitate_loss += imitate_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss    += loss.item()
            running_ce_loss += ce_loss.item()
            n_batches       += 1

        if n_batches > 0:
            if epoch < warmup_epochs:
                print(f"[shadow {shadow_id}] Epoch [{epoch:3d}/{args.epochs}] "
                      f"train_loss={running_loss/n_batches:.4f} "
                      f"ce_loss={running_ce_loss/n_batches:.4f}")
            else:
                print(f"[shadow {shadow_id}] Epoch [{epoch:3d}/{args.epochs}] "
                      f"train_loss={running_loss/n_batches:.4f} "
                      f"ce_loss={running_ce_loss/n_batches:.4f} "
                      f"imitate_loss={running_imitate_loss/n_batches:.4f}")

        if epoch + 1 == args.warmup_epochs:
            warmup_model_state = copy.deepcopy(student.cpu().state_dict())
            student.to(device)
            print(f"[shadow {shadow_id}] Warmup checkpoint saved at epoch {epoch + 1}")

        if epoch >= warmup_epochs:
            val_acc = evaluate(student, val_dl, ce_criterion, device)[1]
            print(f"[shadow {shadow_id}] Epoch [{epoch:3d}/{args.epochs}]  val_acc={val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc     = val_acc
                best_model_state = copy.deepcopy(student.cpu().state_dict())
                student.to(device)
                _flush_state_dict(best_model_state, model_path, args.num_classes)

        if scheduler is not None:
            scheduler.step()

        warmup_model_for_ckpt = None
        if warmup_model_state is not None:
            warmup_model_for_ckpt = ResNet18_Influence(num_classes=args.num_classes)
            warmup_model_for_ckpt.load_state_dict(warmup_model_state)
        _save_checkpoint(out_dir, epoch, student, optimizer, scheduler,
                         best_val_acc, warmup_model_for_ckpt)
        del warmup_model_for_ckpt

    if best_model_state is None:
        best_model_state = copy.deepcopy(student.cpu().state_dict())
        student.to(device)

    del student, teacher_model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 8. Quality gate
    # ------------------------------------------------------------------
    if best_val_acc < args.imitate_acc:
        print(f"[shadow {shadow_id}] WARNING: best_val_acc={best_val_acc:.4f} < "
              f"imitate_acc={args.imitate_acc:.4f}. Discarding model, returning None.")
        return None

    # ------------------------------------------------------------------
    # 9. Save final outputs
    # ------------------------------------------------------------------
    _flush_state_dict(best_model_state, model_path, args.num_classes)
    print(f"[shadow {shadow_id}] Best val accuracy: {best_val_acc:.4f}  "
          f"Model saved to {model_path}")

    if warmup_model_state is not None:
        _flush_state_dict(warmup_model_state, warmup_path, args.num_classes)
        print(f"[shadow {shadow_id}] Warmup model saved to {warmup_path}")

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
