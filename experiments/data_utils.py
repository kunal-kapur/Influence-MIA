"""File I/O: loading shadow artifacts, query metadata, and target model evaluation."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import split_dataset_for_type


def load_query_metadata(exp_dir: str):
    """Load query_indices.npy and ground_truth.npy written by train_target.

    Returns
    -------
    query_global_indices : (n_query,) int64
    ground_truth         : (n_query,) int32  — 1=member, 0=non-member
    """
    qi_path = os.path.join(exp_dir, "query_indices.npy")
    gt_path = os.path.join(exp_dir, "ground_truth.npy")

    if not os.path.exists(qi_path):
        raise FileNotFoundError(
            f"query_indices.npy not found at {qi_path}. Run train_target.py first."
        )
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"ground_truth.npy not found at {gt_path}. Run train_target.py first."
        )

    query_global_indices = np.load(qi_path).astype(np.int64)
    ground_truth         = np.load(gt_path).astype(np.int32)

    assert query_global_indices.shape == ground_truth.shape, (
        f"query_indices shape {query_global_indices.shape} != "
        f"ground_truth shape {ground_truth.shape}."
    )
    members    = int(ground_truth.sum())
    nonmembers = int((ground_truth == 0).sum())
    assert members == nonmembers, (
        f"Query set is not balanced: {members} members vs {nonmembers} non-members."
    )
    return query_global_indices, ground_truth


def load_shadow_data(
    exp_dir: str,
    n_shadow_models: int,
    n_query: int,
    query_global_indices: np.ndarray,
):
    """Load per-shadow arrays aligned to query_indices.

    Returns
    -------
    lira_stats : list of (n_query,) arrays
    C_lira     : list of (n_query,) arrays
    C_loss     : list of (n_query,) arrays  (None if file absent)
    """
    lira_stats, C_lira, C_loss = [], [], []
    for k in range(n_shadow_models):
        d          = os.path.join(exp_dir, "shadows", str(k))
        lira_path  = os.path.join(d, "lira_stats.npy")
        clira_path = os.path.join(d, "C_lira.npy")
        closs_path = os.path.join(d, "C_loss.npy")
        meta_path  = os.path.join(d, "influence_meta.npz")

        if not os.path.exists(lira_path):
            print(f"  [shadow {k}] lira_stats.npy missing — skipping.")
            continue

        ls = np.load(lira_path)
        cl = np.load(clira_path)

        assert ls.shape == (n_query,), (
            f"[shadow {k}] lira_stats shape {ls.shape} != (n_query={n_query},). "
            "Shadow was evaluated on a different query set — re-run compute_influence."
        )
        assert cl.shape == (n_query,), (
            f"[shadow {k}] C_lira shape {cl.shape} != (n_query={n_query},). "
            "Re-run compute_influence for this shadow."
        )

        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            if "query_indices" in meta:
                cached_q = meta["query_indices"].astype(np.int64)
                assert cached_q.shape == query_global_indices.shape and np.array_equal(
                    cached_q, query_global_indices
                ), (
                    f"[shadow {k}] influence_meta query_indices mismatch with current "
                    "query_indices.npy. Re-run compute_influence for this shadow."
                )

        lira_stats.append(ls)
        C_lira.append(cl)
        C_loss.append(np.load(closs_path) if os.path.exists(closs_path) else None)

    return lira_stats, C_lira, C_loss


def build_target_pool_global_indices(cfg: dict) -> np.ndarray:
    """Reconstruct shared pool D global indices exactly as in training."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["data_mean"], cfg["data_std"]),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=cfg["data_dir"], train=True, download=False, transform=transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=cfg["data_dir"], train=False, download=False, transform=transform
    )
    target_pool = split_dataset_for_type(
        ConcatDataset([train_ds, test_ds]),
        seed=cfg["seed"],
        data_type="target",
        shared_pool_size=cfg.get("shared_pool_size"),
    )
    return np.asarray(target_pool.indices, dtype=np.int64)


def assert_query_has_in_out_shadow_coverage(cfg: dict, query_global_indices: np.ndarray) -> None:
    """Ensure each query point has at least one IN and one OUT shadow model."""
    target_pool_global_indices = build_target_pool_global_indices(cfg)
    outside_pool = np.setdiff1d(
        query_global_indices,
        target_pool_global_indices,
        assume_unique=False,
    )
    assert len(outside_pool) == 0, (
        "Query set contains points outside shared pool D; "
        f"found {len(outside_pool)} such points."
    )

    pool_size = len(target_pool_global_indices)
    sorter = np.argsort(target_pool_global_indices)
    sorted_globals = target_pool_global_indices[sorter]
    pos = np.searchsorted(sorted_globals, query_global_indices)
    assert np.all(sorted_globals[pos] == query_global_indices), (
        "Failed to map some query points into shared pool D."
    )
    query_local_indices = sorter[pos]

    n_shadow_models = int(cfg["n_shadow_models"])
    in_size = int(cfg["pkeep"] * pool_size)
    assert in_size > 0 and in_size < pool_size, (
        f"Invalid shadow IN subset size {in_size} for pool size {pool_size}."
    )

    in_counts = np.zeros(len(query_local_indices), dtype=np.int32)
    for shadow_id in range(n_shadow_models):
        rs = np.random.RandomState(2025 + shadow_id)
        shadow_in_local = rs.choice(pool_size, in_size, replace=False)
        in_mask = np.zeros(pool_size, dtype=bool)
        in_mask[shadow_in_local] = True
        in_counts += in_mask[query_local_indices].astype(np.int32)

    out_counts = n_shadow_models - in_counts
    min_in  = int(in_counts.min())
    min_out = int(out_counts.min())
    print(
        "  Query IN/OUT shadow coverage: "
        f"min_IN={min_in}, min_OUT={min_out}, "
        f"avg_IN={in_counts.mean():.2f}, avg_OUT={out_counts.mean():.2f}"
    )

    assert min_in > 0, (
        "At least one query point has 0 IN shadows, which invalidates LiRA IN-distribution fitting."
    )
    assert min_out > 0, (
        "At least one query point has 0 OUT shadows, which invalidates LiRA OUT-distribution fitting."
    )


def compute_target_lira_scores(
    exp_dir: str,
    cfg: dict,
    query_global_indices: np.ndarray,
) -> np.ndarray:
    """Evaluate the target model on D_query and return scaled logits.

    Returns (n_query,) array of log(p_true / (1 - p_true)).
    """
    from models.resnet import ResNet18_Influence

    target_path = os.path.join(exp_dir, "target_model.pt")
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Target model not found at {target_path}. Run train_target.py first."
        )
    assert os.path.basename(target_path) == "target_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ResNet18_Influence(num_classes=int(cfg["num_classes"])).to(device)
    state_dict = torch.load(target_path, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["data_mean"], cfg["data_std"]),
    ])
    train_ds = torchvision.datasets.CIFAR10(
        root=cfg["data_dir"], train=True, download=False, transform=transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=cfg["data_dir"], train=False, download=False, transform=transform
    )
    full_dataset = ConcatDataset([train_ds, test_ds])
    query_ds = Subset(full_dataset, query_global_indices.tolist())
    loader   = DataLoader(
        query_ds,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    scores = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            probs  = torch.softmax(logits, dim=1)
            p_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            p_true = torch.clamp(p_true, min=1e-7, max=1.0 - 1e-7)
            scores.append(torch.log(p_true / (1.0 - p_true)).cpu().numpy())

    result  = np.concatenate(scores)
    n_query = len(query_global_indices)
    assert result.shape == (n_query,), (
        f"Target score array shape {result.shape} != (n_query={n_query},). "
        "Pool reconstruction or query_indices may be misaligned."
    )
    return result
