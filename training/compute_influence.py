"""Compute influence matrices and LiRA statistics for a trained shadow model.

The Hessian and influence computations are over the *last linear layer* only,
treating the penultimate feature vector as a fixed representation.  This keeps
the Hessian size manageable (512×512 for ResNet18) while remaining a valid
local approximation of influence.

Outputs (per shadow_id)
-----------------------
outputs/{dataset}/shadows/{shadow_id}/H_inv.npy      — inverse Hessian  (D×D)
outputs/{dataset}/shadows/{shadow_id}/C_lira.npy     — LiRA influence   (N,)
outputs/{dataset}/shadows/{shadow_id}/C_loss.npy     — loss influence   (N,)
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

def _compute_last_layer_hessian(model, loader, device, damping=1e-4):
    """Gauss-Newton / empirical Fisher Hessian of the CE loss w.r.t. the last
    linear layer weights, accumulated over all batches.

    H = (1/N) * sum_i  J_i^T J_i  +  damping * I

    where J_i is the Jacobian of the per-sample CE loss w.r.t. the last layer
    weight matrix, flattened to a vector.

    For a linear layer  out = phi @ W^T  (no bias for simplicity, or bias
    absorbed), the per-sample gradient is  kron(phi_i, (p_i - e_{y_i})).
    The Gauss-Newton approximation gives H as the empirical covariance of
    these gradients.

    Returns: H_inv  (torch.Tensor, shape [D, D])  where D = in_features * num_classes
    """
    model.eval()

    # Identify the last linear layer and its dimensions
    last_linear = model.linear
    in_features = last_linear.in_features
    num_classes = last_linear.out_features
    D = in_features * num_classes

    H = torch.zeros(D, D, dtype=torch.float64)
    N = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            bs = x.size(0)

            # Forward pass up to penultimate layer (feature extractor)
            out = F.relu(model.bn1(model.conv1(x)))
            out = model.layer1(out)
            out = model.layer2(out)
            out = model.layer3(out)
            out = model.layer4(out)
            out = F.avg_pool2d(out, 4)
            phi = out.view(bs, -1)  # (bs, in_features)

            # Softmax probabilities
            logits = model.linear(phi)
            probs = torch.softmax(logits, dim=1)  # (bs, C)

            # Residual: p - e_y
            residual = probs.clone()
            residual[torch.arange(bs), y] -= 1.0  # (bs, C)

            # Per-sample gradient (flattened): kron(phi_i, residual_i)
            # Shape: (bs, C, in_features) -> (bs, D)
            grad = torch.bmm(
                residual.unsqueeze(2),   # (bs, C, 1)
                phi.unsqueeze(1),        # (bs, 1, in_features)
            ).reshape(bs, D)             # (bs, D)

            # Accumulate outer products (move to CPU to keep memory on GPU low)
            grad_cpu = grad.double().cpu()
            H += grad_cpu.T @ grad_cpu
            N += bs

    H /= N
    H += damping * torch.eye(D, dtype=torch.float64)
    return H


# ---------------------------------------------------------------------------
# Influence matrices
# ---------------------------------------------------------------------------

def _compute_influence_matrices(model, loader, H_inv, train_dataset_size, device):
    """Compute per-sample influence scores for every point in `loader`.

    Two variants:
      C_lira[i]  — influence on the *scaled-logit* (LiRA statistic) of sample i
                   when sample i is removed from the training set.
      C_loss[i]  — influence on the *CE loss* of sample i.

    Both use the standard influence-function formula:
        I(z, z_test) = -grad_test^T H^{-1} grad_train

    Here we take the self-influence (z == z_test) approximation which is the
    quantity used in MIA: "how much does removing point i affect the model's
    behaviour on point i itself?"

    Returns: (C_lira, C_loss)  both 1-D numpy arrays of length N.
    """
    model.eval()
    last_linear = model.linear
    in_features = last_linear.in_features
    num_classes = last_linear.out_features
    D = in_features * num_classes

    H_inv_t = torch.tensor(H_inv, dtype=torch.float64)  # (D, D) on CPU

    C_lira_list = []
    C_loss_list = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            bs = x.size(0)

            # Feature extraction
            out = F.relu(model.bn1(model.conv1(x)))
            out = model.layer1(out)
            out = model.layer2(out)
            out = model.layer3(out)
            out = model.layer4(out)
            out = F.avg_pool2d(out, 4)
            phi = out.view(bs, -1)  # (bs, F)

            logits = model.linear(phi)
            probs = torch.softmax(logits, dim=1)  # (bs, C)

            # --- Gradient of CE loss w.r.t. last-layer weights ---
            residual_loss = probs.clone()
            residual_loss[torch.arange(bs), y] -= 1.0
            # Scale by 1/N_train as in the paper convention
            grad_loss = torch.bmm(
                residual_loss.unsqueeze(2),
                phi.unsqueeze(1),
            ).reshape(bs, D).double().cpu() / train_dataset_size  # (bs, D)

            # --- Gradient of LiRA scaled-logit w.r.t. last-layer weights ---
            # scaled logit t_i = log(p_{y_i} / (1 - p_{y_i}))
            # dt/dp_{y_i} = 1 / (p_{y_i} * (1 - p_{y_i}))
            p_true = probs.gather(1, y.unsqueeze(1)).squeeze(1)  # (bs,)
            p_true = torch.clamp(p_true, 1e-7, 1.0 - 1e-7)
            dt_dp = 1.0 / (p_true * (1.0 - p_true))  # (bs,)

            # dp_{y_i}/dlogits = p_{y_i}*(delta_{k,y_i} - p_k)
            # d(lira)/d(logit_k) = dt_dp * p_{y_i} * (delta_{k,y_i} - p_k)
            residual_lira = -p_true.unsqueeze(1) * probs.clone()  # (bs, C)
            residual_lira[torch.arange(bs), y] += p_true          # add p_{y_i}
            residual_lira = residual_lira * (dt_dp * p_true).unsqueeze(1)

            grad_lira = torch.bmm(
                residual_lira.unsqueeze(2),
                phi.unsqueeze(1),
            ).reshape(bs, D).double().cpu() / train_dataset_size  # (bs, D)

            # --- Self-influence: -g_test^T H^{-1} g_train ---
            # Here g_test == g_train (self-influence)
            Hinv_g_loss = grad_loss @ H_inv_t.T   # (bs, D)
            Hinv_g_lira = grad_lira @ H_inv_t.T   # (bs, D)

            c_loss = -(grad_loss * Hinv_g_loss).sum(dim=1)   # (bs,)
            c_lira = -(grad_lira * Hinv_g_lira).sum(dim=1)   # (bs,)

            C_lira_list.append(c_lira.numpy())
            C_loss_list.append(c_loss.numpy())

    return np.concatenate(C_lira_list), np.concatenate(C_loss_list)


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
        H = _compute_last_layer_hessian(model, full_loader, device, damping=1e-4)
        print(f"[shadow {shadow_id}] Inverting Hessian ({H.shape[0]}×{H.shape[1]})...")
        H_inv = torch.linalg.inv(H).numpy()
        save_array(H_inv, hinv_path)
        print(f"[shadow {shadow_id}] H_inv saved to {hinv_path}")

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
