"""Compute influence matrices and LiRA statistics for a trained shadow model.

All per-point outputs are aligned to query_indices.npy (written by train_target).
query_indices.npy stores global indices over CIFAR-10 train+test concat, but
all query points are required to lie inside the shared target/shadow pool D.
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
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.loader import get_dataset, load_dataset
from models.resnet import ResNet18_Influence
from utils.io import load_model, load_array, save_array


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_query_indices(exp_dir: str) -> np.ndarray:
    """Global dataset indices of the query set, shape (n_query,)."""
    path = os.path.join(exp_dir, "query_indices.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"query_indices.npy not found at {path}. "
            "Run train_target.py first."
        )
    return np.load(path).astype(np.int64)



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
    out = F.adaptive_avg_pool2d(out, (1, 1))
    phi = out.view(bs, -1)
    ones = torch.ones(bs, 1, device=x.device, dtype=phi.dtype)
    return torch.cat([phi, ones], dim=1)  # (bs, F+1)


def _compute_last_layer_hessian(model, loader, device, damping=1e-4):
    """Empirical Fisher Hessian on the last linear layer.

    H = (1/N) * Σ_i  P_cov_i ⊗ φ_i φ_i^T  +  damping * I

    Returns H_inv  (D×D float32 tensor on CPU).
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
            
            # Compute blocks without allocating full DxD on GPU
            outer_phi = torch.bmm(phi_aug.unsqueeze(2), phi_aug.unsqueeze(1))
            for i in range(num_classes):
                H_i_unreshaped = torch.einsum('bj,bkl->jkl', P_cov[:, i, :], outer_phi)
                H_i = H_i_unreshaped.permute(1, 0, 2).reshape(num_features, D)
                H[i*num_features : (i+1)*num_features, :] += H_i
                del H_i_unreshaped
                del H_i
            del outer_phi

            N += bs

    H /= N
    H = H.cpu()
    # Keep Hessian numerically symmetric before inversion.
    # We do this efficiently to prevent a 3x 10.5 GB spike from H + H.T
    H_T = H.T.clone()
    H.add_(H_T).mul_(0.5)
    del H_T

    # Serialize H to disk to completely free it from CPU RAM during inversion
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False) as f:
        H_tmp_path = f.name
    torch.save(H, H_tmp_path)
    del H  # Free 10.5 GB

    damping_candidates = [float(damping), float(damping) * 10.0, float(damping) * 100.0]

    for damp in damping_candidates:
        H_damped = torch.load(H_tmp_path)
        H_damped.diagonal().add_(damp)
        try:
            # Prefer Cholesky inverse for SPD matrices
            chol = torch.linalg.cholesky(H_damped)
            del H_damped  # immediately free 10.5 GB

            H_inv = torch.cholesky_inverse(chol)
            del chol  # immediately free 10.5 GB

            if not torch.isfinite(H_inv).all():
                del H_inv
                raise RuntimeError("H_inv contains non-finite values")
            if damp != float(damping):
                warnings.warn(
                    f"Hessian inversion required stronger damping={damp:.1e} "
                    f"(requested {float(damping):.1e})."
                )
            os.remove(H_tmp_path)
            return H_inv
        except RuntimeError:
            if 'H_damped' in locals(): del H_damped
            if 'chol' in locals(): del chol
            continue

    # Last-resort fallback: heavily damped pseudo-inverse.
    fallback_damp = float(damping) * 1000.0
    warnings.warn(
        f"Falling back to pseudo-inverse with damping={fallback_damp:.1e}. "
        "Results may be less reliable; consider increasing Hessian sample size."
    )
    H_fallback = torch.load(H_tmp_path)
    H_fallback.diagonal().add_(fallback_damp)
    H_inv = torch.linalg.pinv(H_fallback)
    del H_fallback
    os.remove(H_tmp_path)
    return H_inv


def _artifacts_match_query_alignment(meta_path, query_indices, train_dataset_size):
    """Check whether cached influence artifacts match current query alignment."""
    if not os.path.exists(meta_path):
        return False
    try:
        meta = np.load(meta_path)
        cached_q = meta["query_indices"].astype(np.int64)
        cached_train_size = int(meta["train_dataset_size"])
        return (
            cached_train_size == int(train_dataset_size)
            and cached_q.shape == query_indices.shape
            and np.array_equal(cached_q, query_indices)
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Influence matrices
# ---------------------------------------------------------------------------

def _collect_gradients(model, loader, train_dataset_size, device):
    """Per-sample CE and LiRA gradients for every point in loader.

    Returns (G_lira, G_loss) both (N, D) float32 on CPU.
    """
    model.eval()
    num_classes  = model.linear.out_features
    num_features = model.linear.in_features + 1
    D = num_classes * num_features

    N_total = len(loader.dataset)
    G_loss = torch.empty((N_total, D), dtype=torch.float32, device='cpu')
    G_lira = torch.empty((N_total, D), dtype=torch.float32, device='cpu')

    idx = 0
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

            G_loss[idx:idx+bs] = torch.einsum('nc,nf->ncf', d_logits, phi_aug).reshape(bs, D).cpu()
            G_lira[idx:idx+bs] = torch.einsum('nc,nf->ncf', d_logits_lira, phi_aug).reshape(bs, D).cpu()
            idx += bs

    G_loss.div_(train_dataset_size)
    G_lira.div_(train_dataset_size)
    return G_lira, G_loss


def _compute_influence_matrices(model, loader, H_inv, train_dataset_size, device):
    """Column norms of the N×N cross-influence matrices.

    C_lira = -(G_lira @ H_inv @ G_loss^T) / N  → col norms  (N,)
    C_loss = -(G_loss @ H_inv @ G_loss^T) / N  → col norms  (N,)
    """
    G_lira, G_loss = _collect_gradients(model, loader, train_dataset_size, device)
    N = G_lira.shape[0]

    C_lira = torch.zeros(N, dtype=torch.float32)
    C_loss = torch.zeros(N, dtype=torch.float32)

    H_inv_gpu = None
    if device.type == 'cuda':
        try:
            H_inv_gpu = H_inv.to(device)
        except RuntimeError:
            warnings.warn("Not enough VRAM for H_inv. Falling back to CPU ops.")
            H_inv_gpu = None
            torch.cuda.empty_cache()

    chunk_size = 2000
    sub_chunk = 5000

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        
        if H_inv_gpu is not None:
            U_chunk = (G_loss[i:end].to(device) @ H_inv_gpu).cpu()
        else:
            U_chunk = G_loss[i:end] @ H_inv

        M_chunk_lira = torch.zeros((N, end - i), dtype=torch.float32)
        M_chunk_loss = torch.zeros((N, end - i), dtype=torch.float32)
        
        if device.type == 'cuda':
            try:
                U_chunk_gpu = U_chunk.T.to(device)
                for j in range(0, N, sub_chunk):
                    j_end = min(j + sub_chunk, N)
                    M_chunk_lira[j:j_end] = (G_lira[j:j_end].to(device) @ U_chunk_gpu).cpu()
                    M_chunk_loss[j:j_end] = (G_loss[j:j_end].to(device) @ U_chunk_gpu).cpu()
                del U_chunk_gpu
            except RuntimeError:
                torch.cuda.empty_cache()
                U_chunk_T = U_chunk.T
                for j in range(0, N, sub_chunk):
                    j_end = min(j + sub_chunk, N)
                    M_chunk_lira[j:j_end] = G_lira[j:j_end] @ U_chunk_T
                    M_chunk_loss[j:j_end] = G_loss[j:j_end] @ U_chunk_T
        else:
            U_chunk_T = U_chunk.T
            for j in range(0, N, sub_chunk):
                j_end = min(j + sub_chunk, N)
                M_chunk_lira[j:j_end] = G_lira[j:j_end] @ U_chunk_T
                M_chunk_loss[j:j_end] = G_loss[j:j_end] @ U_chunk_T

        C_lira[i:end] = torch.linalg.norm(M_chunk_lira, dim=0) / N
        C_loss[i:end] = torch.linalg.norm(M_chunk_loss, dim=0) / N

    if H_inv_gpu is not None:
        del H_inv_gpu
        torch.cuda.empty_cache()

    return C_lira.numpy(), C_loss.numpy()


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
    """
    out_dir    = os.path.join(args.exp_dir, "shadows", str(shadow_id))
    hinv_path  = os.path.join(out_dir, "H_inv.npy")
    clira_path = os.path.join(out_dir, "C_lira.npy")
    closs_path = os.path.join(out_dir, "C_loss.npy")
    lira_path  = os.path.join(out_dir, "lira_stats.npy")
    meta_path  = os.path.join(out_dir, "influence_meta.npz")

    # ------------------------------------------------------------------
    # 1. Query indices — defines the N points all outputs are aligned to
    # ------------------------------------------------------------------
    get_dataset(args)
    query_pool_indices = _load_query_indices(args.exp_dir)
    n_query = len(query_pool_indices)
    print(f"[shadow {shadow_id}] Query set size: n_query={n_query}")

    # ------------------------------------------------------------------
    # 2. Full eval dataset (no aug) — for evaluating the shadow model on D_query
    # ------------------------------------------------------------------
    full_eval_no_aug = load_dataset(args, use_augmentation=False, return_full=True)

    # Verify full dataset size matches what query_indices was built against
    assert len(full_eval_no_aug) >= query_pool_indices.max() + 1, (
        f"[shadow {shadow_id}] full eval dataset size {len(full_eval_no_aug)} is smaller than "
        f"max query index {query_pool_indices.max()}. Seed or dataset changed."
    )

    # Subset to exactly the query points, in query_indices order
    query_ds = Subset(full_eval_no_aug, query_pool_indices.tolist())
    assert len(query_ds) == n_query

    query_loader = DataLoader(
        query_ds,
        batch_size=args.batch_size,
        shuffle=False,          # must stay False — order defines the index space
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------
    # 4. Shadow pool (no aug) — needed for Hessian (shadow's training data)
    # ------------------------------------------------------------------
    shadow_pool_no_aug  = load_dataset(args, data_type="target", use_augmentation=False)
    shadow_pool_size    = len(shadow_pool_no_aug)
    shadow_pool_global_indices = np.asarray(shadow_pool_no_aug.indices, dtype=np.int64)

    outside_shadow_pool = np.setdiff1d(
        query_pool_indices,
        shadow_pool_global_indices,
        assume_unique=False,
    )
    assert len(outside_shadow_pool) == 0, (
        f"[shadow {shadow_id}] query set contains {len(outside_shadow_pool)} points outside "
        "the shared pool D. Query non-members must be drawn from D\\D_train."
    )

    # Reconstruct the same shadow training subset used in train_shadow
    np.random.seed(2025 + shadow_id)
    if hasattr(args, "shadow_train_size") and args.shadow_train_size is not None:
        num_shadow_train = args.shadow_train_size
    else:
        num_shadow_train = int(args.pkeep * shadow_pool_size)

    shadow_in_indices = np.random.choice(
        shadow_pool_size, num_shadow_train, replace=False
    )
    train_dataset_size = len(shadow_in_indices)

    shadow_in_global = shadow_pool_global_indices[shadow_in_indices]
    query_in_shadow_mask = np.isin(query_pool_indices, shadow_in_global)
    num_query_in_shadow = int(query_in_shadow_mask.sum())
    num_query_out_shadow = int(n_query - num_query_in_shadow)
    print(
        f"[shadow {shadow_id}] Query coverage in this shadow: "
        f"IN={num_query_in_shadow}, OUT={num_query_out_shadow}"
    )
    assert num_query_in_shadow > 0 and num_query_out_shadow > 0, (
        f"[shadow {shadow_id}] degenerate IN/OUT split over query set: "
        f"IN={num_query_in_shadow}, OUT={num_query_out_shadow}."
    )

    print(f"[shadow {shadow_id}] Shadow training subset size: {train_dataset_size}")

    # If query ordering or train subset size changed, force recomputation of
    # query-aligned artifacts to prevent silent index drift.
    has_valid_alignment_meta = _artifacts_match_query_alignment(
        meta_path, query_pool_indices, train_dataset_size
    )
    if not has_valid_alignment_meta:
        for p in (clira_path, closs_path, lira_path):
            if os.path.exists(p):
                os.remove(p)
                print(f"[shadow {shadow_id}] Removed stale artifact: {p}")

    # ------------------------------------------------------------------
    # 5. Load shadow model
    # ------------------------------------------------------------------
    model_path = os.path.join(out_dir, "shadow_model.pt")
    model = ResNet18_Influence(
        num_classes=args.num_classes,
        in_channels=getattr(args, "in_channels", 3),
    ).to(device)
    model = load_model(model, model_path, device)
    model.eval()
    print(f"[shadow {shadow_id}] Model loaded from {model_path}")

    # ------------------------------------------------------------------
    # 6. Hessian (on shadow model's actual training subset)
    # ------------------------------------------------------------------
    expected_D = (model.linear.in_features + 1) * model.linear.out_features
    H_inv = None
    if os.path.exists(hinv_path) and has_valid_alignment_meta:
        cached = load_array(hinv_path)
        if cached.shape == (expected_D, expected_D):
            print(f"[shadow {shadow_id}] H_inv loaded from cache")
            H_inv = torch.tensor(cached, dtype=torch.float32, device='cpu')
        else:
            print(f"[shadow {shadow_id}] H_inv shape {cached.shape} != expected "
                  f"({expected_D},{expected_D}) — recomputing.")
    elif os.path.exists(hinv_path) and not has_valid_alignment_meta:
        print(
            f"[shadow {shadow_id}] Recomputing H_inv because alignment metadata is "
            "missing or stale."
        )

    if H_inv is None:
        hessian_sample = min(3000, train_dataset_size)
        hessian_indices = np.random.choice(
            shadow_in_indices, hessian_sample, replace=False
        ).tolist()
        hessian_loader = DataLoader(
            Subset(shadow_pool_no_aug, hessian_indices),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
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

    np.savez_compressed(
        meta_path,
        query_indices=query_pool_indices.astype(np.int64),
        train_dataset_size=np.array(train_dataset_size, dtype=np.int64),
    )
    print(f"[shadow {shadow_id}] alignment metadata saved: {meta_path}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[shadow {shadow_id}] compute_influence done.")
