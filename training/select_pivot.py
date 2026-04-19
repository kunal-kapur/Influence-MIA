"""Pivot set selection for IMIA (Algorithm 2, SelectPivot step).

Selects k instances per class with the lowest CE loss under the frozen target
model — these are the points the target model is most confident on, so Phase 2
IN-model fine-tuning converges quickly.

Output: {exp_dir}/pivot_indices.npy  — indices into shadow_pool_no_aug
"""

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def select_pivot_data(
    target_model,
    shadow_pool_no_aug,
    k_per_class: int,
    num_classes: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    exp_dir: str,
    force_recompute: bool = False,
) -> np.ndarray:
    """Return indices (into shadow_pool_no_aug) of the pivot set D_pivot.

    For each class, selects up to k_per_class instances with the lowest CE loss
    under the frozen target model (i.e., the ones it's already most confident
    about).  Saves result to {exp_dir}/pivot_indices.npy and caches it.
    """
    pivot_path = os.path.join(exp_dir, "pivot_indices.npy")

    if not force_recompute and os.path.exists(pivot_path):
        indices = np.load(pivot_path).astype(np.int64)
        print(f"[pivot] Loaded {len(indices)} pivot indices from {pivot_path}")
        return indices

    loader = DataLoader(
        shadow_pool_no_aug,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    target_model.eval()
    all_losses = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = target_model(x)
            losses = F.cross_entropy(logits, y, reduction="none")
            all_losses.append(losses.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_losses = np.concatenate(all_losses)  # (N,)
    all_labels = np.concatenate(all_labels)  # (N,)
    N = len(all_losses)

    pivot_indices = []
    for c in range(num_classes):
        class_mask = np.where(all_labels == c)[0]
        if len(class_mask) == 0:
            logger.warning(f"[pivot] Class {c} has no instances in shadow pool — skipping.")
            continue
        if len(class_mask) < k_per_class:
            logger.warning(
                f"[pivot] Class {c} has only {len(class_mask)} instances "
                f"(< k_per_class={k_per_class}). Using all."
            )
        class_losses = all_losses[class_mask]
        k = min(k_per_class, len(class_mask))
        # argsort ascending: smallest loss = most confident
        top_k_local = np.argsort(class_losses)[:k]
        pivot_indices.append(class_mask[top_k_local])

    pivot_indices = np.concatenate(pivot_indices).astype(np.int64)

    os.makedirs(exp_dir, exist_ok=True)
    np.save(pivot_path, pivot_indices)
    print(f"[pivot] Selected {len(pivot_indices)} pivot instances "
          f"({k_per_class} per class × {num_classes} classes) → {pivot_path}")

    return pivot_indices
