"""Post-hoc analysis: influence scores vs MIA vulnerability bucketing.

For each query point we have, across K shadow models:
  - lira_stats[k][i]  : scaled logit  t_i  for shadow k  (N,)
  - C_lira[k][i]      : influence on LiRA statistic        (N,)
  - C_loss[k][i]      : influence on CE loss                (N,)

The MIA attack score for point i (expected discrepancy) is:
    mia_score[i] = target_score[i] - mean_k(lira_stats[k][i])

This directly measures how much more confident the target model is relative
to the shadow models' baseline expectation, without requiring IN/OUT Gaussian
fitting (which is invalid when shadow pools are disjoint from the query set).

We bucket points by each influence score and compute:
  - Pearson correlation with mia_score
  - TPR @ 0% FPR, TPR @ 0.1% FPR, TPR @ 1% FPR, and Balanced Accuracy per bucket

Usage
-----
python -m experiments.analyze --exp_dir outputs/<exp_name>/cifar10
python -m experiments.analyze --exp_dir outputs/<exp_name>/cifar10 --num_buckets 10
"""

import argparse
import csv
import os
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import yaml
from sklearn.metrics import roc_curve
from sklearn.mixture import GaussianMixture
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Load all shadow artifacts
# ---------------------------------------------------------------------------

def _load_shadow_data(exp_dir: str, n_shadow_models: int, n_query: int):
    """Load per-shadow arrays aligned to query_indices.

    Returns
    -------
    lira_stats : list of (n_query,) arrays  — shadow scaled logits
    C_lira     : list of (n_query,) arrays
    C_loss     : list of (n_query,) arrays  (None if file absent)
    """
    lira_stats, C_lira, C_loss = [], [], []
    for k in range(n_shadow_models):
        d          = os.path.join(exp_dir, "shadows", str(k))
        lira_path  = os.path.join(d, "lira_stats.npy")
        clira_path = os.path.join(d, "C_lira.npy")
        closs_path = os.path.join(d, "C_loss.npy")

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

        lira_stats.append(ls)
        C_lira.append(cl)
        C_loss.append(np.load(closs_path) if os.path.exists(closs_path) else None)

    return lira_stats, C_lira, C_loss


# ---------------------------------------------------------------------------
# Query-set metadata
# ---------------------------------------------------------------------------

def _load_query_metadata(exp_dir: str):
    """Load query_indices.npy and ground_truth.npy written by train_target.

    Returns
    -------
    query_pool_indices : (n_query,) int64 — pool-relative index of each query point
    ground_truth       : (n_query,) int32 — 1 = member, 0 = non-member
    """
    qi_path = os.path.join(exp_dir, "query_indices.npy")
    gt_path = os.path.join(exp_dir, "ground_truth.npy")

    if not os.path.exists(qi_path):
        raise FileNotFoundError(
            f"query_indices.npy not found at {qi_path}. "
            "Run train_target.py first."
        )
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"ground_truth.npy not found at {gt_path}. "
            "Run train_target.py first."
        )

    query_pool_indices = np.load(qi_path).astype(np.int64)
    ground_truth       = np.load(gt_path).astype(np.int32)

    assert query_pool_indices.shape == ground_truth.shape, (
        f"query_indices shape {query_pool_indices.shape} != "
        f"ground_truth shape {ground_truth.shape}."
    )
    members    = int(ground_truth.sum())
    nonmembers = int((ground_truth == 0).sum())
    assert members == nonmembers, (
        f"Query set is not balanced: {members} members vs {nonmembers} non-members."
    )
    return query_pool_indices, ground_truth


# ---------------------------------------------------------------------------
# Target model evaluation on the query set
# ---------------------------------------------------------------------------

def _compute_target_lira_scores(exp_dir: str, cfg: dict,
                                 query_pool_indices: np.ndarray) -> np.ndarray:
    """Evaluate the target model on D_query and return scaled logits.

    Returns (n_query,) array of log(p_true / (1 - p_true)).

    The loader iterates over query_pool_indices in order so that output[i]
    corresponds to query point i — the same ordering used by lira_stats,
    C_lira, in_mask, and ground_truth.
    """
    from models.resnet import ResNet18_Influence
    from data.loader import offline_data_split

    target_path = os.path.join(exp_dir, "target_model.pt")
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Target model not found at {target_path}. Run train_target.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ResNet18_Influence().to(device)
    state_dict = torch.load(target_path, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    mean = cfg["data_mean"]
    std  = cfg["data_std"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root=cfg["data_dir"], train=True, download=False, transform=transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=cfg["data_dir"], train=False, download=False, transform=transform
    )
    full_dataset = ConcatDataset([train_ds, test_ds])

    # Reconstruct the target pool — must match the seed used during training
    target_pool = offline_data_split(full_dataset, cfg["seed"], "target")

    # Subset to the query points in query_indices order (no shuffling)
    query_ds = Subset(target_pool, query_pool_indices.tolist())
    loader   = DataLoader(query_ds, batch_size=256, shuffle=False, num_workers=2)

    scores = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits  = model(inputs)
            probs   = torch.softmax(logits, dim=1)
            p_true  = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            p_true  = torch.clamp(p_true, min=1e-7, max=1.0 - 1e-7)
            scores.append(torch.log(p_true / (1.0 - p_true)).cpu().numpy())

    result = np.concatenate(scores)
    n_query = len(query_pool_indices)
    assert result.shape == (n_query,), (
        f"Target score array shape {result.shape} != (n_query={n_query},). "
        "Pool reconstruction or query_indices may be misaligned."
    )
    return result


# ---------------------------------------------------------------------------
# Expected discrepancy MIA score
# ---------------------------------------------------------------------------

def _compute_mia_scores(lira_stats, target_scores: np.ndarray) -> np.ndarray:
    """Per-point expected discrepancy attack scores.

    mia_score[i] = target_score[i] - mean_k(shadow_score[k][i])

    This measures how much more confident the target model is on point i
    relative to the shadow models' average, without requiring IN/OUT Gaussian
    fitting (invalid when shadow pools are disjoint from the query set).
    """
    shadow_mean = np.stack(lira_stats, axis=0).mean(axis=0)  # (N,)
    return target_scores - shadow_mean


def _aggregate_influence(C_list):
    """Mean influence score across all shadows per query point."""
    K = len(C_list)
    return np.stack([C_list[k] for k in range(K)], axis=0).mean(axis=0)


def _gmm_aggregate_mia_scores(
    influence_scores: np.ndarray,
    mia_scores: np.ndarray,
    num_buckets: int = 10,
) -> np.ndarray:
    """Bucket-wise GMM aggregation of MIA scores with BIC-based model selection.

    For each influence bucket, fits both a 1-component and 2-component GMM and
    uses BIC to decide which is appropriate:
      - BIC(1) <= BIC(2): no separable signal; assign p_in = 0.5 for all points.
      - BIC(2) <  BIC(1): separable IN/OUT distributions; extract P(IN) via the
                           component with the higher mean.

    Points in buckets too small to fit a GMM (< 10) fall back to a
    rank-normalised version of their raw mia_score.
    """
    N = len(mia_scores)
    gmm_probs = np.full(N, np.nan)

    quantiles  = np.quantile(influence_scores, np.linspace(0, 1, num_buckets + 1)[1:-1])
    bucket_ids = np.digitize(influence_scores, quantiles)

    for b in range(num_buckets):
        idx = np.where(bucket_ids == b)[0]
        if len(idx) < 10:
            # fallback: rank-normalise raw scores within this tiny bucket
            ranks = np.argsort(np.argsort(mia_scores[idx])).astype(float)
            gmm_probs[idx] = ranks / max(len(idx) - 1, 1)
            continue

        X = mia_scores[idx].reshape(-1, 1)
        gmm_1 = GaussianMixture(n_components=1, reg_covar=1e-3, random_state=0).fit(X)
        gmm_2 = GaussianMixture(n_components=2, reg_covar=1e-3, random_state=0).fit(X)

        if gmm_1.bic(X) <= gmm_2.bic(X):
            print(f"  [Bucket {b}] BIC favours 1-component — defaulting to p_in=0.5 "
                  f"(no separable membership signal, n={len(idx)})")
            gmm_probs[idx] = 0.5
        else:
            in_component = int(np.argmax(gmm_2.means_.ravel()))
            gmm_probs[idx] = gmm_2.predict_proba(X)[:, in_component]

    return gmm_probs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _balanced_accuracy_from_roc(fpr, tpr):
    return float(((tpr + (1.0 - fpr)) / 2.0).max())


def _tpr_at_fpr(fpr, tpr, max_fpr=0.01):
    valid = np.where(fpr <= max_fpr)[0]
    return float(tpr[valid[-1]]) if len(valid) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_bucket_mia_hist(mia_scores, ground_truth, bucket_indices, bucket_id,
                           out_dir, title_prefix):
    fig, ax = plt.subplots(figsize=(6, 4))
    idx_in  = bucket_indices[ground_truth[bucket_indices] == 1]
    idx_out = bucket_indices[ground_truth[bucket_indices] == 0]
    bins = np.linspace(mia_scores.min(), mia_scores.max(), 40)
    ax.hist(mia_scores[idx_in],  bins=bins, alpha=0.6, label="member",     color="steelblue")
    ax.hist(mia_scores[idx_out], bins=bins, alpha=0.6, label="non-member", color="tomato")
    ax.set_title(f"{title_prefix} — bucket {bucket_id}")
    ax.set_xlabel("MIA score (expected discrepancy)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fname = os.path.join(out_dir, f"hist_{title_prefix}_bucket{bucket_id}.png")
    fig.savefig(fname, dpi=100)
    plt.close(fig)


def _plot_bucket_tpr_comparison(scores_dict, mia_scores, ground_truth, out_dir, num_buckets):
    score_names = list(scores_dict.keys())
    bucket_tprs = {name: [] for name in score_names}

    ref_scores = next(iter(scores_dict.values()))
    quantiles  = np.quantile(ref_scores, np.linspace(0, 1, num_buckets + 1)[1:-1])
    bucket_ids = np.digitize(ref_scores, quantiles)

    for b in range(num_buckets):
        idx = np.where(bucket_ids == b)[0]
        if len(idx) < 10:
            for name in score_names:
                bucket_tprs[name].append(float("nan"))
            continue
        fpr, tpr, _ = roc_curve(ground_truth[idx], mia_scores[idx])
        tpr_val = _tpr_at_fpr(fpr, tpr, max_fpr=0.01)
        for name in score_names:
            bucket_tprs[name].append(tpr_val * 100)

    x     = np.arange(num_buckets)
    width = 0.8 / max(len(score_names), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(score_names):
        ax.bar(x + j * width, bucket_tprs[name], width, label=name, alpha=0.8)
    ax.set_xlabel("Quintile bucket (by influence score)")
    ax.set_ylabel("TPR @ 1% FPR (%)")
    ax.set_title("MIA TPR@1%FPR by influence bucket")
    ax.set_xticks(x + width * (len(score_names) - 1) / 2)
    ax.set_xticklabels([f"Q{b}" for b in range(num_buckets)])
    ax.legend()
    fig.tight_layout()
    fname = os.path.join(out_dir, "bucket_tpr_comparison.png")
    fig.savefig(fname, dpi=100)
    plt.close(fig)
    print(f"  Saved bucket TPR comparison to {fname}")


def _plot_score_vs_mia(score_name, scores, mia_scores, ground_truth, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(scores[ground_truth == 0], mia_scores[ground_truth == 0],
               s=2, alpha=0.3, color="tomato",    label="non-member")
    ax.scatter(scores[ground_truth == 1], mia_scores[ground_truth == 1],
               s=2, alpha=0.3, color="steelblue", label="member")
    ax.set_xlabel(score_name)
    ax.set_ylabel("MIA score (expected discrepancy)")
    ax.set_title(f"{score_name} vs MIA score")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"scatter_{score_name}_vs_mia.png"), dpi=100)
    plt.close(fig)


def _plot_gmm_bucket_components(influence_scores, mia_scores, ground_truth, out_dir,
                                num_buckets=10, min_points=10):
    """Plot one fitted GMM per influence bucket and save each plot separately.

    Each bucket plot shows:
    - the member vs non-member score distributions inside the bucket,
    - the fitted 2-component GMM density,
    - each individual Gaussian component density,
    - vertical markers at the component means.

    The plots are written to a dedicated subdirectory so each bucket can be
    inspected independently.
    """
    bucket_plot_dir = os.path.join(out_dir, "gmm_bucket_components")
    os.makedirs(bucket_plot_dir, exist_ok=True)
    csv_rows = []
    quantiles = np.quantile(influence_scores, np.linspace(0, 1, num_buckets + 1)[1:-1])
    bucket_ids = np.digitize(influence_scores, quantiles)

    for b in range(num_buckets):
        idx = np.where(bucket_ids == b)[0]
        if len(idx) == 0:
            continue

        bucket_scores = mia_scores[idx]
        bucket_members = ground_truth[idx] == 1
        bucket_non_members = ~bucket_members
        member_scores = bucket_scores[bucket_members]
        non_member_scores = bucket_scores[bucket_non_members]

        score_min = float(bucket_scores.min())
        score_max = float(bucket_scores.max())
        if np.isclose(score_min, score_max):
            score_min -= 0.5
            score_max += 0.5

        x_grid = np.linspace(score_min, score_max, 300)

        fig, ax = plt.subplots(figsize=(7.2, 4.8))

        if len(idx) >= min_points:
            X = bucket_scores.reshape(-1, 1)
            gmm_1 = GaussianMixture(n_components=1, reg_covar=1e-3, random_state=0).fit(X)
            gmm_2 = GaussianMixture(n_components=2, reg_covar=1e-3, random_state=0).fit(X)

            if gmm_1.bic(X) <= gmm_2.bic(X):
                # No separable signal — show the 1-component fit, assign p_in=0.5
                gmm = gmm_1
                weights = gmm.weights_.ravel()
                means = gmm.means_.ravel()
                variances = gmm.covariances_.reshape(-1)
                component_pdfs = []
                for component_id in range(1):
                    var = float(max(variances[component_id], 1e-6))
                    mean = float(means[component_id])
                    weight = float(weights[component_id])
                    pdf = weight * (1.0 / np.sqrt(2.0 * np.pi * var)) * np.exp(-0.5 * ((x_grid - mean) ** 2) / var)
                    component_pdfs.append(pdf)
                total_pdf = np.sum(component_pdfs, axis=0)
                in_component = None
                p_in = np.full(len(idx), 0.5)
                p_out = np.full(len(idx), 0.5)
            else:
                gmm = gmm_2
                weights = gmm.weights_.ravel()
                means = gmm.means_.ravel()
                variances = gmm.covariances_.reshape(-1)
                component_pdfs = []
                for component_id in range(2):
                    var = float(max(variances[component_id], 1e-6))
                    mean = float(means[component_id])
                    weight = float(weights[component_id])
                    pdf = weight * (1.0 / np.sqrt(2.0 * np.pi * var)) * np.exp(-0.5 * ((x_grid - mean) ** 2) / var)
                    component_pdfs.append(pdf)
                total_pdf = np.sum(component_pdfs, axis=0)
                in_component = int(np.argmax(means))
                posterior = gmm.predict_proba(X)
                p_in = posterior[:, in_component]
                p_out = 1.0 - p_in
        else:
            gmm = None
            total_pdf = None
            component_pdfs = []
            means = np.array([])
            in_component = None
            p_in = np.full(len(idx), np.nan, dtype=float)
            p_out = np.full(len(idx), np.nan, dtype=float)

            if len(idx) > 0:
                ranks = np.argsort(np.argsort(bucket_scores)).astype(float)
                p_in = ranks / max(len(idx) - 1, 1)
                p_out = 1.0 - p_in

        predicted_is_in = p_in >= 0.5

        bins = np.linspace(score_min, score_max, 28)
        ax.hist(non_member_scores, bins=bins, density=True, alpha=0.45,
                color="tomato", label="out / non-member")
        ax.hist(member_scores, bins=bins, density=True, alpha=0.45,
                color="steelblue", label="in / member")

        if total_pdf is not None:
            ax.plot(x_grid, total_pdf, color="black", linewidth=2.0, label="GMM total")
            component_colors = ["darkorange", "purple"]
            for component_id, pdf in enumerate(component_pdfs):
                label = f"Gaussian {component_id + 1}"
                if in_component is not None and component_id == in_component:
                    label += " (higher mean / IN)"
                ax.plot(x_grid, pdf, linestyle="--", linewidth=1.8,
                        color=component_colors[component_id], label=label)
                ax.axvline(float(means[component_id]), color=component_colors[component_id],
                           linestyle=":", linewidth=1.0)
        else:
            ax.text(0.5, 0.9, "Too few points for GMM", transform=ax.transAxes,
                    ha="center", va="top", fontsize=9)

        ax.set_title(
            f"Bucket {b}  n={len(idx)}  in={int(bucket_members.sum())}  out={int(bucket_non_members.sum())}"
        )
        ax.set_xlabel("MIA score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="best")

        fig.tight_layout()
        out_path = os.path.join(bucket_plot_dir, f"bucket_{b:02d}_gmm_components.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

        bic_n_components = 1 if in_component is None and len(idx) >= min_points else (2 if len(idx) >= min_points else None)
        for local_pos, point_idx in enumerate(idx):
            csv_rows.append({
                "point_index": int(point_idx),
                "bucket": int(b),
                "bucket_size": int(len(idx)),
                "bic_n_components": bic_n_components,
                "mia_score": float(bucket_scores[local_pos]),
                "ground_truth": int(ground_truth[point_idx]),
                "ground_truth_label": "in" if ground_truth[point_idx] == 1 else "out",
                "predicted_label": "in" if bool(predicted_is_in[local_pos]) else "out",
                "p_in": float(p_in[local_pos]),
                "p_out": float(p_out[local_pos]),
            })

    csv_rows.sort(key=lambda row: (row["bucket"], row["mia_score"], row["point_index"]))
    csv_path = os.path.join(bucket_plot_dir, "bucket_points_sorted.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "point_index",
                "bucket",
                "bucket_size",
                "bic_n_components",
                "mia_score",
                "ground_truth",
                "ground_truth_label",
                "predicted_label",
                "p_in",
                "p_out",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"  Saved bucket-wise GMM component plots to {bucket_plot_dir}")
    print(f"  Saved sorted bucket-point CSV to {csv_path}")


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_score(score_name, scores, mia_scores, ground_truth, out_dir, num_buckets=10):
    corr = np.corrcoef(scores, mia_scores)[0, 1]
    print(f"\n  [{score_name}] Pearson corr(score, MIA score): {corr:.3f}")

    quantiles  = np.quantile(scores, np.linspace(0, 1, num_buckets + 1)[1:-1])
    bucket_ids = np.digitize(scores, quantiles)

    print(f"  [{score_name}] Bucketed TPR@0%FPR, TPR@0.1%FPR, TPR@1%FPR and Balanced Acc by quintile:")
    for b in range(num_buckets):
        idx = np.where(bucket_ids == b)[0]
        if len(idx) < 10:
            print(f"    Bucket {b}: too few points ({len(idx)}), skipping")
            continue
        fpr, tpr, _ = roc_curve(ground_truth[idx], mia_scores[idx])
        tpr_0pct  = _tpr_at_fpr(fpr, tpr, max_fpr=0.0)
        tpr_01pct = _tpr_at_fpr(fpr, tpr, max_fpr=0.001)
        tpr_1pct  = _tpr_at_fpr(fpr, tpr, max_fpr=0.01)
        bal_acc   = _balanced_accuracy_from_roc(fpr, tpr)
        print(f"    Bucket {b}: size={len(idx):4d}, "
              f"TPR@0%FPR={tpr_0pct*100:5.2f}%, "
              f"TPR@0.1%FPR={tpr_01pct*100:5.2f}%, "
              f"TPR@1%FPR={tpr_1pct*100:5.2f}%, "
              f"Balanced Acc={bal_acc*100:5.2f}%")
        _plot_bucket_mia_hist(mia_scores, ground_truth, idx, b, out_dir, score_name)

    _plot_score_vs_mia(score_name, scores, mia_scores, ground_truth, out_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Influence vs LiRA vulnerability analysis")
    parser.add_argument("--dataset",     default="cifar10")
    parser.add_argument("--exp_dir",     required=True,
                        help="Experiment directory (e.g. outputs/<exp>/cifar10)")
    parser.add_argument("--num_buckets", type=int, default=10,
                        help="Number of quantile buckets (default: 5 = quintiles)")
    return parser.parse_args()


def run(exp_dir: str, dataset: str = "cifar10", num_buckets: int = 5) -> None:
    config_dir = "config"
    yaml_path  = os.path.join(config_dir, f"{dataset}.yaml")
    yml_path   = os.path.join(config_dir, f"{dataset}.yml")
    cfg_path   = yaml_path if os.path.exists(yaml_path) else yml_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    n_shadow_models = cfg["n_shadow_models"]

    out_dir = os.path.join(exp_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Analysis output directory: {out_dir}")

    # ------------------------------------------------------------------
    # 1. Query-set metadata — single source of truth for N and ground truth
    # ------------------------------------------------------------------
    print("\nLoading query metadata...")
    query_pool_indices, ground_truth = _load_query_metadata(exp_dir)
    n_query  = len(query_pool_indices)
    members  = int(ground_truth.sum())
    nonmembers = int((ground_truth == 0).sum())
    print(f"  n_query={n_query}  members={members}  non-members={nonmembers}")

    # ------------------------------------------------------------------
    # 2. Shadow artifacts — all must have shape (n_query,)
    # ------------------------------------------------------------------
    print(f"\nLoading {n_shadow_models} shadow models from {exp_dir}/shadows/...")
    lira_stats, C_lira_list, C_loss_list = _load_shadow_data(
        exp_dir, n_shadow_models, n_query=n_query
    )
    K = len(lira_stats)
    print(f"  Loaded {K} shadows successfully.")

    if K < 2:
        print("  Need at least 2 complete shadows — skipping analysis.")
        return

    # ------------------------------------------------------------------
    # 3. Target model scores on D_query
    # ------------------------------------------------------------------
    print("\nEvaluating target model on query set...")
    target_scores = _compute_target_lira_scores(exp_dir, cfg, query_pool_indices)
    print(f"  target_scores: mean={target_scores.mean():.3f}, std={target_scores.std():.3f}")

    # ------------------------------------------------------------------
    # 4. Expected discrepancy MIA scores
    # ------------------------------------------------------------------
    print("\nComputing MIA scores (expected discrepancy)...")
    mia_scores = _compute_mia_scores(lira_stats, target_scores)

    fpr_all, tpr_all, _ = roc_curve(ground_truth.astype(int), mia_scores)
    tpr_0pct_all  = _tpr_at_fpr(fpr_all, tpr_all, max_fpr=0.0)
    tpr_01pct_all = _tpr_at_fpr(fpr_all, tpr_all, max_fpr=0.001)
    tpr_1pct_all  = _tpr_at_fpr(fpr_all, tpr_all, max_fpr=0.01)
    bal_acc_all   = _balanced_accuracy_from_roc(fpr_all, tpr_all)
    print(f"  MIA global: TPR@0%FPR={tpr_0pct_all*100:.2f}%, "
          f"TPR@0.1%FPR={tpr_01pct_all*100:.2f}%, "
          f"TPR@1%FPR={tpr_1pct_all*100:.2f}%, Balanced Acc={bal_acc_all*100:.2f}%")

    # ------------------------------------------------------------------
    # 5. Influence scores
    # ------------------------------------------------------------------
    print("\nAggregating influence scores across shadows...")
    scores_C_lira = _aggregate_influence(C_lira_list)
    has_closs     = all(c is not None for c in C_loss_list)
    scores_C_loss = _aggregate_influence(C_loss_list) if has_closs else None

    scores_dict = {"C_lira": scores_C_lira}
    if scores_C_loss is not None:
        scores_dict["C_loss"] = scores_C_loss

    for name, scores in scores_dict.items():
        analyze_score(name, scores, mia_scores, ground_truth, out_dir, num_buckets)

    _plot_bucket_tpr_comparison(scores_dict, mia_scores, ground_truth, out_dir, num_buckets)

    # ------------------------------------------------------------------
    # 6. Bucket-wise GMM aggregation → global MIA
    # ------------------------------------------------------------------
    print("\nRunning bucket-wise GMM aggregation (C_lira buckets)...")
    gmm_probs = _gmm_aggregate_mia_scores(scores_C_lira, mia_scores, num_buckets)

    fpr_gmm, tpr_gmm, _ = roc_curve(ground_truth.astype(int), gmm_probs)
    tpr_0pct_gmm  = _tpr_at_fpr(fpr_gmm, tpr_gmm, max_fpr=0.0)
    tpr_01pct_gmm = _tpr_at_fpr(fpr_gmm, tpr_gmm, max_fpr=0.001)
    tpr_1pct_gmm  = _tpr_at_fpr(fpr_gmm, tpr_gmm, max_fpr=0.01)
    bal_acc_gmm   = _balanced_accuracy_from_roc(fpr_gmm, tpr_gmm)
    print(f"  GMM-aggregated MIA: TPR@0%FPR={tpr_0pct_gmm*100:.2f}%, "
          f"TPR@0.1%FPR={tpr_01pct_gmm*100:.2f}%, "
          f"TPR@1%FPR={tpr_1pct_gmm*100:.2f}%, "
          f"Balanced Acc={bal_acc_gmm*100:.2f}%")

    _plot_gmm_bucket_components(scores_C_lira, mia_scores, ground_truth, out_dir, num_buckets)

    # ------------------------------------------------------------------
    # 7. Save analysis outputs
    # ------------------------------------------------------------------
    save_dict = dict(
        mia_scores=mia_scores,
        gmm_probs=gmm_probs,
        ground_truth=ground_truth,
        scores_C_lira=scores_C_lira,
        query_pool_indices=query_pool_indices,
    )
    if scores_C_loss is not None:
        save_dict["scores_C_loss"] = scores_C_loss

    out_path = os.path.join(out_dir, "influence_vs_mia.npz")
    np.savez_compressed(out_path, **save_dict)
    print(f"\n  Saved analysis data to {out_path}")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  MIA (raw):         TPR@0%FPR={tpr_0pct_all*100:.2f}%  "
          f"TPR@0.1%FPR={tpr_01pct_all*100:.2f}%  "
          f"TPR@1%FPR={tpr_1pct_all*100:.2f}%  "
          f"Balanced Acc={bal_acc_all*100:.2f}%")
    print(f"  MIA (GMM-bucketed):TPR@0%FPR={tpr_0pct_gmm*100:.2f}%  "
          f"TPR@0.1%FPR={tpr_01pct_gmm*100:.2f}%  "
          f"TPR@1%FPR={tpr_1pct_gmm*100:.2f}%  "
          f"Balanced Acc={bal_acc_gmm*100:.2f}%")
    print(f"{'='*60}\n")


def main():
    cli = _parse_args()
    run(exp_dir=cli.exp_dir, dataset=cli.dataset, num_buckets=cli.num_buckets)


if __name__ == "__main__":
    main()
