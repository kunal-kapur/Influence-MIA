"""Compute influence matrices and LiRA statistics for a trained shadow model.

The Hessian and influence computations are over the *last linear layer* only,
treating the penultimate feature vector as a fixed representation.  This keeps
the Hessian size manageable (513×513 for ResNet18 with bias) while remaining a
valid local approximation of influence.

Outputs (per shadow_id)
-----------------------
outputs/{dataset}/shadows/{shadow_id}/H_inv.npy      — inverse Hessian  (D×D)
outputs/{dataset}/shadows/{shadow_id}/C_lira.npy     — LiRA influence   (N,)  column norms of N×N matrix
outputs/{dataset}/shadows/{shadow_id}/C_loss.npy     — loss influence   (N,)  column norms of N×N matrix
outputs/{dataset}/shadows/{shadow_id}/lira_stats.npy — scaled logits    (N,)
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
from utils.io import load_model, load_array, save_array


# ---------------------------------------------------------------------------
# Helpers: load the shared pool with *no augmentation* (deterministic features)
# ---------------------------------------------------------------------------

def _build_shared_pool_no_aug(args):
    """Load the shared pool (target/shadow) with only normalisation."""
    get_dataset(args)  # ensures data_mean / data_std / num_classes are set
    mean = args.data_mean
    std = args.data_std

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform,
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    full_dataset = ConcatDataset([train_ds, test_ds])
    return offline_data_split(full_dataset, args.seed, "target")


# ---------------------------------------------------------------------------
# Last-layer Hessian
# ---------------------------------------------------------------------------

def _extract_features(model, x):
    """Return augmented features [phi, 1] for the last linear layer (bias-aware)."""
    bs = x.size(0)
    out = F.relu(model.bn1(model.conv1(x)))
    out = model.layer1(out)
    out = model.layer2(out)
    out = model.layer3(out)
    out = model.layer4(out)
    out = F.avg_pool2d(out, 4)
    phi = out.view(bs, -1)  # (bs, F)
    ones = torch.ones(bs, 1, device=x.device, dtype=phi.dtype)
    return torch.cat([phi, ones], dim=1)  # (bs, F+1)


def _compute_last_layer_hessian(model, loader, device, damping=1e-4):
    """Exact empirical Fisher Hessian of the CE loss w.r.t. the last linear
    layer (weights + bias), using augmented features [phi, 1].

    H = (1/N) * sum_i  kron(phi_aug_i phi_aug_i^T, P_cov_i)  +  damping * I

    where P_cov_i = diag(p_i) - p_i p_i^T is the correct curvature block for
    softmax + CE. D = (F+1) * C.

    Returns: H_inv  (numpy array, shape [D, D])
    """
    model.eval()

    last_linear = model.linear
    in_features = last_linear.in_features + 1  # +1 for bias
    num_classes = last_linear.out_features
    D = in_features * num_classes

    H = torch.zeros(D, D, dtype=torch.float64)
    N = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            bs = x.size(0)

            phi_aug = _extract_features(model, x)  # (bs, F+1)
            logits = phi_aug[:, :-1] @ model.linear.weight.T + model.linear.bias
            probs = torch.softmax(logits, dim=1)  # (bs, C)

            # Per-sample gradient via kron(phi_aug, p - e_y): shape (bs, D)
            residual = probs.clone()
            residual[torch.arange(bs), y] -= 1.0
            grad = torch.bmm(
                residual.unsqueeze(2),    # (bs, C, 1)
                phi_aug.unsqueeze(1),     # (bs, 1, F+1)
            ).reshape(bs, D)

            grad_cpu = grad.double().cpu()
            H += grad_cpu.T @ grad_cpu
            N += bs

    H /= N
    H += damping * torch.eye(D, dtype=torch.float64)
    H_inv = torch.linalg.inv(H).numpy()
    return H_inv


# ---------------------------------------------------------------------------
# Influence matrices
# ---------------------------------------------------------------------------

def _compute_all_gradients(model, loader, train_dataset_size, device):
    """Compute per-sample gradients of CE loss and LiRA statistic for every
    point in `loader`, using augmented features [phi, 1] (bias-aware).

    LiRA gradient uses the sign-corrected formula:
        d(t)/d(logits) = (p - e_y) / (p_true - 1.0)
    which divides by (p_true - 1) < 0, matching the proper sign convention.

    Returns: (G_lira, G_loss)  both (N, D) numpy arrays.
    """
    model.eval()
    in_features = model.linear.in_features + 1  # +1 for bias
    num_classes = model.linear.out_features
    D = in_features * num_classes

    G_lira_list = []
    G_loss_list = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            bs = x.size(0)

            phi_aug = _extract_features(model, x)  # (bs, F+1)
            logits = phi_aug[:, :-1] @ model.linear.weight.T + model.linear.bias
            probs = torch.softmax(logits, dim=1)  # (bs, C)

            p_true = probs.gather(1, y.unsqueeze(1)).squeeze(1)  # (bs,)
            p_true = torch.clamp(p_true, 1e-7, 1.0 - 1e-7)

            # CE loss gradient: (p - e_y) ⊗ phi_aug
            residual_loss = probs.clone()
            residual_loss[torch.arange(bs), y] -= 1.0
            grad_loss = torch.bmm(
                residual_loss.unsqueeze(2),
                phi_aug.unsqueeze(1),
            ).reshape(bs, D).double().cpu() / train_dataset_size

            # LiRA gradient: (p - e_y) / (p_true - 1) ⊗ phi_aug
            # (p_true - 1) is always negative, which flips the sign correctly
            residual_lira = residual_loss / (p_true - 1.0).unsqueeze(1)
            grad_lira = torch.bmm(
                residual_lira.unsqueeze(2),
                phi_aug.unsqueeze(1),
            ).reshape(bs, D).double().cpu() / train_dataset_size

            G_lira_list.append(grad_lira.numpy())
            G_loss_list.append(grad_loss.numpy())

    return np.concatenate(G_lira_list, axis=0), np.concatenate(G_loss_list, axis=0)


def _compute_influence_matrices(model, loader, H_inv, train_dataset_size, device):
    """Compute the full N×N cross-influence matrix and return column norms.

    C_lira[j] = ||column j of (-G_lira @ H_inv @ G_loss^T / N)||
              = vulnerability score for training point j, measuring how
                much removing j shifts the LiRA statistic across all N points.

    C_loss[j] is the same but using G_loss for both sides.

    Returns: (C_lira, C_loss)  both 1-D numpy arrays of length N (column norms).
    """
    G_lira, G_loss = _compute_all_gradients(model, loader, train_dataset_size, device)
    # G_lira, G_loss: (N, D)

    H_inv_np = np.array(H_inv, dtype=np.float64)

    # H_inv @ G^T  ->  (D, N)
    HinvGloss_T = H_inv_np @ G_loss.T   # (D, N)
    HinvGlira_T = H_inv_np @ G_lira.T   # (D, N)

    # Cross-influence matrices: (N, N)
    # C_lira_mat[i, j] = -G_lira[i] @ H_inv @ G_loss[j] / N
    N = G_lira.shape[0]
    C_lira_mat = -(G_lira @ HinvGloss_T) / N   # (N, N)
    C_loss_mat = -(G_loss @ HinvGloss_T) / N   # (N, N)

    # Column norms: vulnerability score per training point j
    C_lira = np.linalg.norm(C_lira_mat, axis=0)  # (N,)
    C_loss = np.linalg.norm(C_loss_mat, axis=0)  # (N,)

    return C_lira, C_loss


# ---------------------------------------------------------------------------
# LiRA statistics (scaled logits)
# ---------------------------------------------------------------------------

def _get_lira_statistics(model, loader, device):
    """Compute the LiRA scaled logit for every point in the loader.

    t_i = log( p_{y_i} / (1 - p_{y_i}) )

    Returns a 1-D numpy array of length N.
    """
    model.eval()
    stats = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            p_true = probs.gather(1, y.unsqueeze(1)).squeeze(1)
            p_true = torch.clamp(p_true, 1e-7, 1.0 - 1e-7)
            t = torch.log(p_true / (1.0 - p_true))
            stats.append(t.cpu().numpy())
    return np.concatenate(stats)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_influence(args, shadow_id, device):
    """Compute and save influence matrices + LiRA stats for one shadow model.

    Parameters
    ----------
    args    : namespace with exp_dir, dataset, data_dir, seed, batch_size, n_shadow_models
    shadow_id : int
    device  : torch.device
    """
    out_dir = os.path.join(args.exp_dir, "shadows", str(shadow_id))
    mask_path = os.path.join(out_dir, "in_mask.npy")
    hinv_path = os.path.join(out_dir, "H_inv.npy")
    clira_path = os.path.join(out_dir, "C_lira.npy")
    closs_path = os.path.join(out_dir, "C_loss.npy")
    lira_path = os.path.join(out_dir, "lira_stats.npy")

    # --- 1. Shared pool (no augmentation) ---
    get_dataset(args)
    shared_pool = _build_shared_pool_no_aug(args)
    pool_size = len(shared_pool)

    # --- 2. IN mask ---
    in_mask = load_array(mask_path).astype(bool)
    train_dataset_size = int(in_mask.sum())
    print(f"[shadow {shadow_id}] IN size: {train_dataset_size}  "
          f"OUT size: {pool_size - train_dataset_size}")

    # --- 3. Full-pool DataLoader (shuffle=False for deterministic ordering) ---
    # Hyperparameters are configured in cifar10.yaml (lr, batch_size, optimizer, scheduler, dropout, num_workers, etc.)
    full_loader = DataLoader(
        shared_pool,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- 4. Load shadow model ---
    model_path = os.path.join(out_dir, "shadow_model.pt")
    model = ResNet18_Influence(num_classes=args.num_classes).to(device)
    model = load_model(model, model_path, device)
    model.eval()
    print(f"[shadow {shadow_id}] Model loaded from {model_path}")

    # --- 5. Hessian ---
    if os.path.exists(hinv_path):
        print(f"[shadow {shadow_id}] H_inv already exists, loading...")
        H_inv = load_array(hinv_path)
    else:
        print(f"[shadow {shadow_id}] Computing Hessian...")
        H_inv = _compute_last_layer_hessian(model, full_loader, device, damping=1e-4)
        save_array(H_inv, hinv_path)
        print(f"[shadow {shadow_id}] H_inv saved to {hinv_path}  shape={H_inv.shape}")

    # --- 6. Influence matrices ---
    if os.path.exists(clira_path) and os.path.exists(closs_path):
        print(f"[shadow {shadow_id}] Influence matrices already exist, skipping.")
        C_lira = load_array(clira_path)
        C_loss = load_array(closs_path)
    else:
        print(f"[shadow {shadow_id}] Computing influence matrices...")
        C_lira, C_loss = _compute_influence_matrices(
            model, full_loader, H_inv,
            train_dataset_size=train_dataset_size,
            device=device,
        )
        save_array(C_lira, clira_path)
        save_array(C_loss, closs_path)
        print(f"[shadow {shadow_id}] C_lira saved to {clira_path}")
        print(f"[shadow {shadow_id}] C_loss saved to {closs_path}")

    # --- 7. LiRA statistics ---
    if os.path.exists(lira_path):
        print(f"[shadow {shadow_id}] lira_stats already exists, skipping.")
    else:
        print(f"[shadow {shadow_id}] Computing LiRA statistics...")
        lira_stats = _get_lira_statistics(model, full_loader, device)
        save_array(lira_stats, lira_path)
        print(f"[shadow {shadow_id}] lira_stats saved to {lira_path}  "
              f"shape={lira_stats.shape}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[shadow {shadow_id}] compute_influence done.")
