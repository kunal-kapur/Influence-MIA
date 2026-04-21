"""Compute influence matrices and LiRA statistics for a trained shadow model.

All per-point outputs are aligned to query_indices.npy (written by train_target).
Index i in every saved array refers to query point query_indices[i].

Outputs (per shadow_id)
-----------------------
shadows/{shadow_id}/H_inv.npy       — inverse Hessian          (D×D)
shadows/{shadow_id}/C_lira.npy      — LiRA influence scores    (n_query,)
shadows/{shadow_id}/C_loss.npy      — CE-loss influence scores  (n_query,)
shadows/{shadow_id}/lira_stats.npy  — scaled logits            (n_query,)
"""

import gc
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset

from data.loader import get_dataset, offline_data_split
from models.resnet import ResNet18_Influence
from utils.io import load_model, load_array, save_array


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_query_indices(exp_dir: str) -> np.ndarray:
    """Pool-relative indices of the query set, shape (n_query,)."""
    path = os.path.join(exp_dir, "query_indices.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"query_indices.npy not found at {path}. "
            "Run train_target.py first."
        )
    return np.load(path).astype(np.int64)


def _load_reserved_indices(exp_dir: str) -> Optional[np.ndarray]:
    """Pool-relative indices of the reserved set, shape (n_reserved,).

    Returns None if the file is absent (old experiments lack a reserved set).
    """
    path = os.path.join(exp_dir, "reserved_indices.npy")
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.int64)


def _build_target_pool_no_aug(args):
    """Load the target pool (D) with normalisation only — no augmentation.

    Must use data_type='target' so that pool-relative indices in
    query_indices.npy resolve to the correct samples.
    """
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
    """Load the shadow pool with normalisation only.

    Used to reconstruct the shadow model's actual training subset for the
    Hessian computation — we must use the same pool the model trained on.
    """
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


# ---------------------------------------------------------------------------
# Last-layer Hessian
# ---------------------------------------------------------------------------

def _extract_features(model, x):
    bs  = x.size(0)
    out = F.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.layer4(out)
    out = F.avg_pool2d(out, 4)
    phi = out.view(bs, -1)
    ones = torch.ones(bs, 1, device=x.device, dtype=phi.dtype)
    return torch.cat([phi, ones], dim=1)  # (bs, F+1)


def _compute_last_layer_hessian(model, loader, device, damping=1e-4):
    """Empirical Fisher Hessian on the last linear layer.

    H = (1/N) * Σ_i  P_cov_i ⊗ φ_i φ_i^T  +  damping * I

    Returns H_inv  (D×D float32 tensor on device).
    """
    model.eval()
    num_classes  = model.linear.out_features
    num_features = model.linear.in_features + 1
    D = num_classes * num_features

    H = torch.zeros(D, D, device=device, dtype=torch.float32)
    N = 0

    with torch.no_grad():
        for x, y in loader:
            x  = x.to(device)
            bs = x.size(0)

            phi_aug = _extract_features(model, x)
            logits  = phi_aug[:, :-1] @ model.linear.weight.T + model.linear.bias
            probs   = torch.softmax(logits, dim=1)

            P_cov = torch.diag_embed(probs) - torch.einsum('bi,bj->bij', probs, probs)
            H += torch.einsum('bij,bk,bl->ikjl', P_cov, phi_aug, phi_aug).reshape(D, D)
            N += bs

    H /= N
    H += damping * torch.eye(D, device=device, dtype=torch.float32)
    return torch.linalg.inv(H)


# ---------------------------------------------------------------------------
# Influence matrices
# ---------------------------------------------------------------------------

def _collect_gradients(model, loader, train_dataset_size, device):
    """Per-sample CE and LiRA gradients for every point in loader.

    Returns (G_lira, G_loss) both (N, D) float32 on device.
    """
    model.eval()
    num_classes  = model.linear.out_features
    num_features = model.linear.in_features + 1
    D = num_classes * num_features

    G_lira_list, G_loss_list = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            bs   = x.size(0)

            phi_aug = _extract_features(model, x)
            logits  = phi_aug[:, :-1] @ model.linear.weight.T + model.linear.bias
            probs   = torch.softmax(logits, dim=1)

            p_true    = probs.gather(1, y.unsqueeze(1)).clamp(1e-7, 1.0 - 1e-7)
            y_onehot  = F.one_hot(y, num_classes=num_classes).float()
            d_logits  = probs - y_onehot
            d_logits_lira = d_logits / (p_true - 1.0)

            G_loss_list.append(
                torch.einsum('nc,nf->ncf', d_logits,      phi_aug).reshape(bs, D)
            )
            G_lira_list.append(
                torch.einsum('nc,nf->ncf', d_logits_lira, phi_aug).reshape(bs, D)
            )

    G_loss = torch.cat(G_loss_list, dim=0) / train_dataset_size
    G_lira = torch.cat(G_lira_list, dim=0) / train_dataset_size
    return G_lira, G_loss


def _compute_influence_matrices(model, loader, H_inv, train_dataset_size, device):
    """Column norms of the N×N cross-influence matrices.

    C_lira = -(G_lira @ H_inv @ G_loss^T) / N  → col norms  (N,)
    C_loss = -(G_loss @ H_inv @ G_loss^T) / N  → col norms  (N,)
    """
    G_lira, G_loss = _collect_gradients(model, loader, train_dataset_size, device)
    N = G_lira.shape[0]

    H_inv_Gl_T = H_inv @ G_loss.T           # (D, N)
    C_lira = torch.linalg.norm(-(G_lira @ H_inv_Gl_T) / N, dim=0).cpu().numpy()
    C_loss = torch.linalg.norm(-(G_loss @ H_inv_Gl_T) / N, dim=0).cpu().numpy()
    return C_lira, C_loss


# ---------------------------------------------------------------------------
# LiRA statistics
# ---------------------------------------------------------------------------

def _get_lira_statistics(model, loader, device):
    """t_i = log(p_{y_i} / (1 - p_{y_i})) for every point in loader.

    Returns (N,) numpy array.
    """
    model.eval()
    stats = []
    with torch.no_grad():
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)
            p_true = probs.gather(1, y.unsqueeze(1)).squeeze(1).clamp(1e-7, 1.0 - 1e-7)
            stats.append(torch.log(p_true / (1.0 - p_true)).cpu().numpy())
    return np.concatenate(stats)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_influence(args, shadow_id, device):
    """Compute and save influence matrices + LiRA stats for one shadow model.

    All outputs are aligned to query_indices.npy:
      lira_stats[i]  →  query_indices[i]
      C_lira[i]      →  query_indices[i]
      C_loss[i]      →  query_indices[i]

    If reserved_indices.npy exists, identical artifacts are also computed for
    the reserved set and saved with a _reserved suffix.
    """
    out_dir    = os.path.join(args.exp_dir, "shadows", str(shadow_id))
    hinv_path  = os.path.join(out_dir, "H_inv.npy")
    clira_path = os.path.join(out_dir, "C_lira.npy")
    closs_path = os.path.join(out_dir, "C_loss.npy")
    lira_path  = os.path.join(out_dir, "lira_stats.npy")

    clira_res_path = os.path.join(out_dir, "C_lira_reserved.npy")
    closs_res_path = os.path.join(out_dir, "C_loss_reserved.npy")
    lira_res_path  = os.path.join(out_dir, "lira_stats_reserved.npy")

    # ------------------------------------------------------------------
    # 1. Query indices — defines the N points all outputs are aligned to
    # ------------------------------------------------------------------
    get_dataset(args)
    query_pool_indices = _load_query_indices(args.exp_dir)
    n_query = len(query_pool_indices)
    print(f"[shadow {shadow_id}] Query set size: n_query={n_query}")

    reserved_pool_indices = _load_reserved_indices(args.exp_dir)
    n_reserved = len(reserved_pool_indices) if reserved_pool_indices is not None else 0
    if reserved_pool_indices is not None:
        print(f"[shadow {shadow_id}] Reserved set size: n_reserved={n_reserved}")
    else:
        print(f"[shadow {shadow_id}] No reserved_indices.npy found — skipping reserved set.")

    # ------------------------------------------------------------------
    # 2. Target pool (no aug) — for evaluating the shadow model on D_query
    #    and the reserved set (both index into the same target pool)
    # ------------------------------------------------------------------
    target_pool_no_aug = _build_target_pool_no_aug(args)

    all_indices = query_pool_indices
    if reserved_pool_indices is not None:
        all_indices = np.concatenate([query_pool_indices, reserved_pool_indices])

    # Verify pool size matches what query_indices was built against
    assert len(target_pool_no_aug) >= all_indices.max() + 1, (
        f"[shadow {shadow_id}] target pool size {len(target_pool_no_aug)} is smaller than "
        f"max index {all_indices.max()}. Seed or dataset changed."
    )

    # Subset to exactly the query points, in query_indices order
    query_ds = Subset(target_pool_no_aug, query_pool_indices.tolist())
    assert len(query_ds) == n_query

    query_loader = DataLoader(
        query_ds,
        batch_size=args.batch_size,
        shuffle=False,          # must stay False — order defines the index space
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if reserved_pool_indices is not None:
        reserved_ds = Subset(target_pool_no_aug, reserved_pool_indices.tolist())
        reserved_loader = DataLoader(
            reserved_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        reserved_loader = None

    # ------------------------------------------------------------------
    # 4. Shadow pool (no aug) — needed for Hessian (shadow's training data)
    # ------------------------------------------------------------------
    shadow_pool_no_aug  = _build_shadow_pool_no_aug(args)
    shadow_pool_size    = len(shadow_pool_no_aug)

    # Reconstruct the same shadow training subset used in train_shadow
    np.random.seed(2025 + shadow_id)
    shadow_in_indices = np.random.choice(
        shadow_pool_size, int(args.pkeep * shadow_pool_size), replace=False
    )
    train_dataset_size = len(shadow_in_indices)
    print(f"[shadow {shadow_id}] Shadow training subset size: {train_dataset_size}")

    # ------------------------------------------------------------------
    # 5. Load shadow model
    # ------------------------------------------------------------------
    model_path = os.path.join(out_dir, "shadow_model.pt")
    model = ResNet18_Influence(num_classes=args.num_classes).to(device)
    model = load_model(model, model_path, device)
    model.eval()
    print(f"[shadow {shadow_id}] Model loaded from {model_path}")

    # ------------------------------------------------------------------
    # 6. Hessian (on shadow model's actual training subset)
    # ------------------------------------------------------------------
    expected_D = (model.linear.in_features + 1) * model.linear.out_features
    H_inv = None
    if os.path.exists(hinv_path):
        cached = load_array(hinv_path)
        if cached.shape == (expected_D, expected_D):
            print(f"[shadow {shadow_id}] H_inv loaded from cache")
            H_inv = torch.tensor(cached, dtype=torch.float32, device=device)
        else:
            print(f"[shadow {shadow_id}] H_inv shape {cached.shape} != expected "
                  f"({expected_D},{expected_D}) — recomputing.")

    if H_inv is None:
        hessian_sample = min(3000, train_dataset_size)
        hessian_indices = np.random.choice(
            shadow_in_indices, hessian_sample, replace=False
        ).tolist()
        hessian_loader = DataLoader(
            Subset(shadow_pool_no_aug, hessian_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print(f"[shadow {shadow_id}] Computing Hessian on {hessian_sample} shadow-IN samples...")
        H_inv = _compute_last_layer_hessian(model, hessian_loader, device, damping=1e-4)
        save_array(H_inv.cpu().numpy(), hinv_path)
        print(f"[shadow {shadow_id}] H_inv saved  shape={H_inv.shape}")

    # ------------------------------------------------------------------
    # 7. Influence matrices — evaluated on the query set
    # ------------------------------------------------------------------
    if os.path.exists(clira_path) and os.path.exists(closs_path):
        print(f"[shadow {shadow_id}] Influence matrices already exist, skipping.")
        C_lira = load_array(clira_path)
        C_loss = load_array(closs_path)
    else:
        print(f"[shadow {shadow_id}] Computing influence matrices over query set...")
        C_lira, C_loss = _compute_influence_matrices(
            model, query_loader, H_inv,
            train_dataset_size=train_dataset_size,
            device=device,
        )
        assert C_lira.shape == (n_query,), (
            f"[shadow {shadow_id}] C_lira shape {C_lira.shape} != (n_query={n_query},)"
        )
        save_array(C_lira, clira_path)
        save_array(C_loss, closs_path)
        print(f"[shadow {shadow_id}] C_lira saved  shape={C_lira.shape}")
        print(f"[shadow {shadow_id}] C_loss saved  shape={C_loss.shape}")

    # ------------------------------------------------------------------
    # 8. LiRA statistics — evaluated on the query set
    # ------------------------------------------------------------------
    if os.path.exists(lira_path):
        print(f"[shadow {shadow_id}] lira_stats already exists, skipping.")
    else:
        print(f"[shadow {shadow_id}] Computing LiRA statistics over query set...")
        lira_stats = _get_lira_statistics(model, query_loader, device)
        assert lira_stats.shape == (n_query,), (
            f"[shadow {shadow_id}] lira_stats shape {lira_stats.shape} != (n_query={n_query},)"
        )
        save_array(lira_stats, lira_path)
        print(f"[shadow {shadow_id}] lira_stats saved  shape={lira_stats.shape}")

    # ------------------------------------------------------------------
    # 9. Reserved set — influence matrices and LiRA stats
    #    Only computed when reserved_indices.npy exists (new experiments).
    # ------------------------------------------------------------------
    if reserved_loader is not None:
        if os.path.exists(clira_res_path) and os.path.exists(closs_res_path):
            print(f"[shadow {shadow_id}] Reserved influence matrices already exist, skipping.")
        else:
            print(f"[shadow {shadow_id}] Computing influence matrices over reserved set...")
            C_lira_res, C_loss_res = _compute_influence_matrices(
                model, reserved_loader, H_inv,
                train_dataset_size=train_dataset_size,
                device=device,
            )
            assert C_lira_res.shape == (n_reserved,), (
                f"[shadow {shadow_id}] C_lira_reserved shape {C_lira_res.shape} "
                f"!= (n_reserved={n_reserved},)"
            )
            save_array(C_lira_res, clira_res_path)
            save_array(C_loss_res, closs_res_path)
            print(f"[shadow {shadow_id}] C_lira_reserved saved  shape={C_lira_res.shape}")
            print(f"[shadow {shadow_id}] C_loss_reserved saved  shape={C_loss_res.shape}")

        if os.path.exists(lira_res_path):
            print(f"[shadow {shadow_id}] lira_stats_reserved already exists, skipping.")
        else:
            print(f"[shadow {shadow_id}] Computing LiRA statistics over reserved set...")
            lira_stats_res = _get_lira_statistics(model, reserved_loader, device)
            assert lira_stats_res.shape == (n_reserved,), (
                f"[shadow {shadow_id}] lira_stats_reserved shape {lira_stats_res.shape} "
                f"!= (n_reserved={n_reserved},)"
            )
            save_array(lira_stats_res, lira_res_path)
            print(f"[shadow {shadow_id}] lira_stats_reserved saved  shape={lira_stats_res.shape}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[shadow {shadow_id}] compute_influence done.")
