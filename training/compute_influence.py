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
    """Exact empirical Fisher Hessian on GPU in float32 using einsum, matching
    the alternate implementation's approach for memory efficiency.

    H = (1/N) * sum_i  einsum(P_cov_i, phi_aug_i, phi_aug_i)  +  damping * I

    Accumulates H on GPU in float32; D = (F+1)*C = 5130.

    Returns: H_inv  (torch.Tensor float32 on device, shape [D, D])
    """
    model.eval()
    num_classes = model.linear.out_features
    num_features = model.linear.in_features + 1  # +1 for bias
    D = num_classes * num_features

    H = torch.zeros(D, D, device=device, dtype=torch.float32)
    N = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            bs = x.size(0)

            phi_aug = _extract_features(model, x)           # (bs, F+1)
            logits = phi_aug[:, :-1] @ model.linear.weight.T + model.linear.bias
            probs = torch.softmax(logits, dim=1)             # (bs, C)

            # Exact curvature: P_cov = diag(p) - p p^T
            P_cov = torch.diag_embed(probs) - torch.einsum('bi,bj->bij', probs, probs)  # (bs, C, C)
            H += torch.einsum('bij,bk,bl->ikjl', P_cov, phi_aug, phi_aug).reshape(D, D)
            N += bs

    H /= N
    H += damping * torch.eye(D, device=device, dtype=torch.float32)
    return torch.linalg.inv(H)  # (D, D) float32 on device


# ---------------------------------------------------------------------------
# Influence matrices
# ---------------------------------------------------------------------------

def _collect_gradients(model, loader, train_dataset_size, device):
    """Collect per-sample CE and LiRA gradients on GPU in float32.

    LiRA gradient: (p - e_y) / (p_true - 1)  — sign-correct convention.

    Returns: (G_lira, G_loss)  both (N, D) float32 tensors on device.
    """
    model.eval()
    num_classes = model.linear.out_features
    num_features = model.linear.in_features + 1
    D = num_classes * num_features

    G_lira_list = []
    G_loss_list = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            bs = x.size(0)

            phi_aug = _extract_features(model, x)            # (bs, F+1)
            logits = phi_aug[:, :-1] @ model.linear.weight.T + model.linear.bias
            probs = torch.softmax(logits, dim=1)              # (bs, C)

            p_true = probs.gather(1, y.unsqueeze(1)).clamp(1e-7, 1.0 - 1e-7)  # (bs, 1)

            y_onehot = F.one_hot(y, num_classes=num_classes).float()
            d_logits = probs - y_onehot                       # (bs, C)  CE gradient
            d_logits_lira = d_logits / (p_true - 1.0)        # (bs, C)  LiRA gradient

            G_loss_list.append(torch.einsum('nc,nf->ncf', d_logits,      phi_aug).reshape(bs, D))
            G_lira_list.append(torch.einsum('nc,nf->ncf', d_logits_lira, phi_aug).reshape(bs, D))

    G_loss = torch.cat(G_loss_list, dim=0) / train_dataset_size  # (N, D)
    G_lira = torch.cat(G_lira_list, dim=0) / train_dataset_size  # (N, D)
    return G_lira, G_loss


def _compute_influence_matrices(model, loader, H_inv, train_dataset_size, device):
    """Compute the full N×N cross-influence matrices on GPU in float32, then
    return column norms as the per-point vulnerability scores.

    C_lira = -(G_lira @ H_inv @ G_loss^T) / N   →  column norms  (N,)
    C_loss = -(G_loss @ H_inv @ G_loss^T) / N   →  column norms  (N,)

    At N=20k and D=5130, the (D,N) intermediate is ~400 MB float32 and the
    full (N,N) matrix is ~1.5 GB float32 — fits comfortably on an A10 GPU.

    Returns: (C_lira, C_loss)  both 1-D numpy arrays of length N.
    """
    G_lira, G_loss = _collect_gradients(model, loader, train_dataset_size, device)
    N = G_lira.shape[0]

    # H_inv @ G_loss^T: reuse for both matrices  (D, N)
    H_inv_Gl_T = H_inv @ G_loss.T

    C_lira_mat = -(G_lira @ H_inv_Gl_T) / N   # (N, N)
    C_loss_mat = -(G_loss @ H_inv_Gl_T) / N   # (N, N)

    C_lira = torch.linalg.norm(C_lira_mat, dim=0).cpu().numpy()  # (N,)
    C_loss = torch.linalg.norm(C_loss_mat, dim=0).cpu().numpy()  # (N,)

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

    # --- 5. Hessian (subsample IN points, keep H_inv on GPU as float32) ---
    expected_D = (model.linear.in_features + 1) * model.linear.out_features
    H_inv = None
    if os.path.exists(hinv_path):
        cached = load_array(hinv_path)
        if cached.shape != (expected_D, expected_D):
            print(f"[shadow {shadow_id}] H_inv shape {cached.shape} != expected "
                  f"({expected_D},{expected_D}) — recomputing.")
        else:
            print(f"[shadow {shadow_id}] H_inv already exists, loading...")
            H_inv = torch.tensor(cached, dtype=torch.float32, device=device)
    if H_inv is None:
        hessian_sample = min(3000, train_dataset_size)
        in_indices = np.where(in_mask)[0]
        hessian_indices = np.random.choice(in_indices, hessian_sample, replace=False).tolist()
        hessian_loader = DataLoader(
            Subset(shared_pool, hessian_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print(f"[shadow {shadow_id}] Computing Hessian on {hessian_sample} IN samples...")
        H_inv = _compute_last_layer_hessian(model, hessian_loader, device, damping=1e-4)
        save_array(H_inv.cpu().numpy(), hinv_path)
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
