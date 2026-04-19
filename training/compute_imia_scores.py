"""IMIA inference — Algorithm 2 (non-adaptive setting).

For each query point (x, y) computes the Lambda membership score:

    Lambda_i = (s_obs - s_bar_out)^2 - (s_obs - s_bar_in)^2

where:
    s_obs     = phi(f_theta(x)_y)           target model score on query
    s_bar_out = mean over f_out models of phi(f_out(x)_y)
    s_bar_in  = mean over (f_in models × proxy instances in D_pivot with label y)
                of phi(f_in(u)_v)

phi is the scaled confidence score (Eq. 1):
    phi(f(x)_y) = log(f(x)_y) - log(max_{y'!=y} f(x)_{y'})

Outputs
-------
{exp_dir}/imia_scores.npy   — Lambda scores, shape (n_query,)
{exp_dir}/ground_truth.npy  — already written by train_target.py (not overwritten)
"""

import gc
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset

from data.loader import get_dataset, offline_data_split
from models.resnet import ResNet18_Influence
from utils.io import load_model, save_array


# ---------------------------------------------------------------------------
# phi — scaled confidence score (Equation 1)
# ---------------------------------------------------------------------------

def phi(model, x, y, device):
    """Scalar phi score for a single instance (x, y).

    phi(f(x)_y) = log(f(x)_y) - log(max_{y' != y} f(x)_{y'})

    Args:
        x: image tensor, shape (C, H, W) — will be unsqueezed to (1, C, H, W)
        y: integer class label
    Returns:
        float scalar
    """
    model.eval()
    with torch.no_grad():
        logits = model(x.unsqueeze(0).to(device))          # (1, C)
        probs  = F.softmax(logits, dim=1).squeeze(0)       # (C,)

    p_true = probs[y].clamp(min=1e-7)

    # mask out true class to find max wrong-class prob
    probs_masked    = probs.clone()
    probs_masked[y] = 0.0
    p_wrong = probs_masked.max().clamp(min=1e-7)

    return (torch.log(p_true) - torch.log(p_wrong)).item()


def _phi_batch(model, x_batch, y_batch, device):
    """Vectorised phi for a batch.

    Args:
        x_batch: (B, C, H, W)
        y_batch: (B,) int64 tensor
    Returns:
        numpy array (B,)
    """
    model.eval()
    with torch.no_grad():
        logits = model(x_batch.to(device))                     # (B, C)
        probs  = F.softmax(logits, dim=1)                      # (B, C)

    B = probs.size(0)
    p_true = probs[torch.arange(B), y_batch.to(device)].clamp(min=1e-7)

    probs_masked = probs.clone()
    probs_masked[torch.arange(B), y_batch.to(device)] = 0.0
    p_wrong = probs_masked.max(dim=1).values.clamp(min=1e-7)

    return (torch.log(p_true) - torch.log(p_wrong)).cpu().numpy()


# ---------------------------------------------------------------------------
# Augmented phi (Section 3.2 / CIFAR augmentation)
# ---------------------------------------------------------------------------

def augmented_phi(model, x, y, num_aug, transform_aug, device):
    """Average phi over num_aug random augmentations of x.

    Args:
        x: image tensor (C, H, W), already normalised
        y: integer class label
        num_aug: number of random augmentations (1 = no augmentation)
        transform_aug: callable that takes a PIL image or tensor and returns
                       an augmented normalised tensor — caller must supply
    Returns:
        float scalar
    """
    if num_aug <= 1:
        return phi(model, x, y, device)

    scores = [phi(model, transform_aug(x), y, device) for _ in range(num_aug)]
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _build_target_pool_no_aug(args):
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
    return offline_data_split(ConcatDataset([train_ds, test_ds]), args.seed, "target")


def _build_shadow_pool_no_aug(args):
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


def _make_aug_transform(args):
    """Random crop + horizontal flip on a normalised tensor (for CIFAR)."""
    mean = args.data_mean
    std  = args.data_std
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_imia_scores(args, device, num_aug=1):
    """Run IMIA Algorithm 2 inference and return (Lambda, ground_truth).

    Args:
        args: config namespace (same as used during training).
        device: torch device.
        num_aug: number of random augmentations per query/proxy instance.
                 Set to 18 for CIFAR-10/100 as per the paper.

    Returns:
        Lambda      : (n_query,) float64 array — higher = more likely member
        ground_truth: (n_query,) int32 array  — 1=member, 0=non-member
    """
    # ------------------------------------------------------------------
    # 1. Query metadata
    # ------------------------------------------------------------------
    query_pool_indices = np.load(
        os.path.join(args.exp_dir, "query_indices.npy")
    ).astype(np.int64)
    ground_truth = np.load(
        os.path.join(args.exp_dir, "ground_truth.npy")
    ).astype(np.int32)
    n_query = len(query_pool_indices)
    print(f"[IMIA] n_query = {n_query}")

    # ------------------------------------------------------------------
    # 2. Target pool (no aug) → query dataset
    # ------------------------------------------------------------------
    get_dataset(args)
    target_pool_no_aug = _build_target_pool_no_aug(args)
    query_ds = Subset(target_pool_no_aug, query_pool_indices.tolist())

    # ------------------------------------------------------------------
    # 3. Frozen target model
    # ------------------------------------------------------------------
    target_model = ResNet18_Influence(num_classes=args.num_classes).to(device)
    target_model = load_model(
        target_model, os.path.join(args.exp_dir, "target_model.pt"), device
    )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 4. Shadow pool (no aug) → pivot dataset indexed per class
    # ------------------------------------------------------------------
    shadow_pool_no_aug = _build_shadow_pool_no_aug(args)

    pivot_indices = np.load(
        os.path.join(args.exp_dir, "pivot_indices.npy")
    ).astype(np.int64)
    pivot_ds = Subset(shadow_pool_no_aug, pivot_indices.tolist())

    # Build per-class lists of (image_tensor, label) for fast proxy lookup
    pivot_by_class = {c: [] for c in range(args.num_classes)}
    for local_idx in range(len(pivot_ds)):
        img, lbl = pivot_ds[local_idx]
        pivot_by_class[int(lbl)].append((img, int(lbl)))
    for c in range(args.num_classes):
        if len(pivot_by_class[c]) == 0:
            print(f"[IMIA] WARNING: pivot set has no instances for class {c}")

    # ------------------------------------------------------------------
    # 5. Load shadow model pairs
    # ------------------------------------------------------------------
    n_shadows  = int(getattr(args, "n_shadow_models", 10))
    f_out_list = []
    f_in_list  = []

    for k in range(n_shadows):
        shadow_dir   = os.path.join(args.exp_dir, "shadows", str(k))
        out_path     = os.path.join(shadow_dir, "shadow_model_out.pt")
        in_path      = os.path.join(shadow_dir, "shadow_model_in.pt")

        if not os.path.exists(out_path) or not os.path.exists(in_path):
            print(f"[IMIA] WARNING: shadow {k} missing OUT or IN model — skipping.")
            continue

        f_out = ResNet18_Influence(num_classes=args.num_classes).to(device)
        f_out = load_model(f_out, out_path, device)
        f_out.eval()
        for p in f_out.parameters():
            p.requires_grad_(False)

        f_in = ResNet18_Influence(num_classes=args.num_classes).to(device)
        f_in = load_model(f_in, in_path, device)
        f_in.eval()
        for p in f_in.parameters():
            p.requires_grad_(False)

        f_out_list.append(f_out)
        f_in_list.append(f_in)

    N = len(f_out_list)
    if N == 0:
        raise RuntimeError("[IMIA] No valid shadow model pairs found. Train shadows first.")
    print(f"[IMIA] Loaded {N} shadow model pairs.")

    # ------------------------------------------------------------------
    # 6. Augmentation transform (identity when num_aug == 1)
    # ------------------------------------------------------------------
    transform_aug = _make_aug_transform(args) if num_aug > 1 else None

    def _aug_phi(model, x, y):
        if num_aug <= 1:
            return phi(model, x, y, device)
        return augmented_phi(model, x, y, num_aug, transform_aug, device)

    # ------------------------------------------------------------------
    # 7. Target model scores on all query points
    # ------------------------------------------------------------------
    print("[IMIA] Computing target model phi scores on query set...")
    s_obs = np.empty(n_query, dtype=np.float64)
    query_loader = DataLoader(
        query_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    offset = 0
    for x_batch, y_batch in query_loader:
        bs = x_batch.size(0)
        s_obs[offset:offset + bs] = _phi_batch(target_model, x_batch, y_batch, device)
        offset += bs

    # ------------------------------------------------------------------
    # 8. Per-query OUT and IN scores
    # ------------------------------------------------------------------
    # s_bar_out[i] = mean over f_out_k of augmented_phi(f_out_k, x_i, y_i)
    # s_bar_in[i]  = mean over (f_in_k, proxy (u,v) where v==y_i)
    #                of augmented_phi(f_in_k, u, v)

    s_bar_out = np.zeros(n_query, dtype=np.float64)
    s_bar_in  = np.zeros(n_query, dtype=np.float64)

    # Accumulate OUT scores: loop queries in batches through each f_out
    for k, f_out_k in enumerate(f_out_list):
        print(f"[IMIA] Computing OUT scores for shadow {k}/{N-1}...")
        offset = 0
        for x_batch, y_batch in query_loader:
            bs    = x_batch.size(0)
            batch_scores = _phi_batch(f_out_k, x_batch, y_batch, device)
            s_bar_out[offset:offset + bs] += batch_scores
            offset += bs
    s_bar_out /= N

    # Accumulate IN scores: for each query use class-matched proxies
    print("[IMIA] Computing IN scores via pivot proxies...")
    # Pre-compute per-class IN scores for each f_in (model × class)
    # proxy_scores[k][c] = list of phi scores from f_in_k on class-c proxies
    proxy_scores = {}
    for k, f_in_k in enumerate(f_in_list):
        proxy_scores[k] = {}
        for c in range(args.num_classes):
            proxies = pivot_by_class[c]
            if len(proxies) == 0:
                proxy_scores[k][c] = np.array([0.0])
                continue
            scores_c = [_aug_phi(f_in_k, u, v) for u, v in proxies]
            proxy_scores[k][c] = np.array(scores_c, dtype=np.float64)

    # For each query point, look up its class and average across models × proxies
    for i in range(n_query):
        _, y_i = query_ds[i]
        y_i    = int(y_i)
        all_in_scores = []
        for k in range(N):
            all_in_scores.extend(proxy_scores[k][y_i].tolist())
        s_bar_in[i] = float(np.mean(all_in_scores)) if all_in_scores else 0.0

    # ------------------------------------------------------------------
    # 9. Lambda scores
    # ------------------------------------------------------------------
    Lambda = (s_obs - s_bar_out) ** 2 - (s_obs - s_bar_in) ** 2

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    scores_path = os.path.join(args.exp_dir, "imia_scores.npy")
    save_array(Lambda.astype(np.float64), scores_path)
    print(f"[IMIA] Lambda scores saved → {scores_path}  shape={Lambda.shape}")

    del f_out_list, f_in_list, target_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return Lambda, ground_truth
