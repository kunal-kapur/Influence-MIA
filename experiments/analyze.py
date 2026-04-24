"""Post-hoc analysis: influence scores vs MIA vulnerability bucketing.

For each query point we have, across K shadow models:
  - lira_stats[k][i]  : scaled logit  t_i  for shadow k  (N,)
  - C_lira[k][i]      : influence on LiRA statistic        (N,)
  - C_loss[k][i]      : influence on CE loss                (N,)

The MIA attack score for point i (expected discrepancy) is:
    mia_score[i] = target_score[i] - mean_k(lira_stats[k][i])

This directly measures how much more confident the target model is relative
to the shadow models' baseline expectation.  The bucket boundaries and GMM
parameters are calibrated on a reference subset first, and the query points
are assigned to those fixed buckets afterwards.

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
from scipy.stats import norm
from sklearn.metrics import roc_curve
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import offline_data_split


# ---------------------------------------------------------------------------
# Constrained zero-mean / right-shifted 2-component mixture
# ---------------------------------------------------------------------------

def fit_fixed_zero_rightshift_mixture(
    x,
    max_iter=200,
    tol=1e-6,
    min_var=1e-4,
    mu_in_min=0.5,
    min_pi_in=0.02,
    reliability_margin=0.25,
):
    """Fit a constrained 1D mixture:

        p(x) = pi_out * N(x; 0, var_out) + pi_in * N(x; mu_in, var_in)

    mu_out is fixed at 0.  mu_in is clamped to >= mu_in_min after each M-step.
    Returns a dict with posteriors, parameters, convergence flag, reliability
    flag, and the posterior-equality threshold.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    # --- initialise ---
    mu_in  = max(float(np.percentile(x, 75)), mu_in_min)
    var_out = max(float(np.var(x)) * 0.5, min_var)
    var_in  = max(float(np.var(x)) * 0.5, min_var)
    pi_out  = 0.5
    pi_in   = 0.5

    prev_loglik = -np.inf
    converged   = False

    for _ in range(max_iter):
        # E-step
        log_p_out = np.log(pi_out + 1e-300) + norm.logpdf(x, 0.0, np.sqrt(var_out))
        log_p_in  = np.log(pi_in  + 1e-300) + norm.logpdf(x, mu_in, np.sqrt(var_in))

        log_sum = np.logaddexp(log_p_out, log_p_in)
        r_out   = np.exp(log_p_out - log_sum)
        r_in    = 1.0 - r_out

        loglik = float(log_sum.sum())

        # M-step
        n_out = r_out.sum()
        n_in  = r_in.sum()
        total = n_out + n_in

        pi_out = n_out / total
        pi_in  = n_in  / total

        # mu_out stays 0; update mu_in, clamp
        mu_in_unconstrained = float(np.dot(r_in, x) / (n_in + 1e-300))
        mu_in = max(mu_in_unconstrained, mu_in_min)

        var_out = max(float(np.dot(r_out, x ** 2) / (n_out + 1e-300)), min_var)
        var_in  = max(float(np.dot(r_in, (x - mu_in) ** 2) / (n_in + 1e-300)), min_var)

        if abs(loglik - prev_loglik) < tol:
            converged = True
            break
        prev_loglik = loglik

    # --- posterior-equality threshold (pi_out*N(t;0,s_out) == pi_in*N(t;mu_in,s_in)) ---
    # Solve analytically by scanning the grid (closed form is messy with unequal variances)
    t_grid = np.linspace(0.0, mu_in * 2.5, 2000)
    d_out  = pi_out * norm.pdf(t_grid, 0.0, np.sqrt(var_out))
    d_in   = pi_in  * norm.pdf(t_grid, mu_in, np.sqrt(var_in))
    sign_changes = np.where(np.diff(np.sign(d_in - d_out)))[0]
    threshold = None
    if len(sign_changes) > 0:
        i = sign_changes[0]
        # linear interpolation
        f0 = (d_in - d_out)[i]
        f1 = (d_in - d_out)[i + 1]
        threshold = float(t_grid[i] - f0 * (t_grid[i + 1] - t_grid[i]) / (f1 - f0))

    # --- reliability ---
    reliable = True
    reason   = "ok"
    if mu_in <= mu_in_min + reliability_margin:
        reliable = False
        reason   = f"mu_in={mu_in:.3f} too close to mu_in_min={mu_in_min}"
    elif pi_in < min_pi_in:
        reliable = False
        reason   = f"pi_in={pi_in:.3f} < min_pi_in={min_pi_in}"
    elif var_out <= min_var * 2 or var_in <= min_var * 2:
        reliable = False
        reason   = f"variance degenerate (var_out={var_out:.4f}, var_in={var_in:.4f})"

    return {
        "pi_out":       pi_out,
        "pi_in":        pi_in,
        "mu_out":       0.0,
        "mu_in":        mu_in,
        "var_out":      var_out,
        "var_in":       var_in,
        "posterior_in":  r_in,
        "posterior_out": r_out,
        "loglik":       loglik,
        "converged":    converged,
        "reliable":     reliable,
        "reason":       reason,
        "threshold":    threshold,
    }


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


def _build_no_aug_split(cfg: dict, data_type: str):
    mean = cfg["data_mean"]
    std = cfg["data_std"]
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
    return offline_data_split(full_dataset, cfg["seed"], data_type)


def _select_reference_subset(reference_split, reference_fraction: float, seed: int):
    if not 0.0 < float(reference_fraction) <= 1.0:
        raise ValueError(
            f"gmm_reference_fraction must be in (0, 1], got {reference_fraction}"
        )

    total = len(reference_split)
    subset_size = max(1, min(total, int(round(total * float(reference_fraction)))))
    if subset_size == total:
        subset_indices = np.arange(total, dtype=np.int64)
    else:
        rng = np.random.RandomState(seed)
        subset_indices = np.sort(
            rng.choice(total, size=subset_size, replace=False)
        ).astype(np.int64)
    return Subset(reference_split, subset_indices.tolist()), subset_indices


def _load_model_from_path(model_path: str, device: torch.device):
    from models.resnet import ResNet18_Influence

    model = ResNet18_Influence().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model


def _compute_scaled_logit_scores(model, loader, device, return_labels: bool = False):
    scores = []
    labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            p_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            p_true = torch.clamp(p_true, min=1e-7, max=1.0 - 1e-7)
            scores.append(torch.log(p_true / (1.0 - p_true)).cpu().numpy())
            if return_labels:
                labels.append(targets.cpu().numpy().astype(np.int32))

    result = np.concatenate(scores)
    if return_labels:
        return result, np.concatenate(labels)
    return result


def _compute_target_scores_for_dataset(exp_dir: str, cfg: dict, dataset,
                                       return_labels: bool = False):
    target_path = os.path.join(exp_dir, "target_model.pt")
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"Target model not found at {target_path}. Run train_target.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_path(target_path, device)
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
    )
    result = _compute_scaled_logit_scores(model, loader, device, return_labels=return_labels)
    del model
    return result


def _compute_shadow_reference_scores(exp_dir: str, cfg: dict, reference_dataset, n_shadow_models: int):
    from training.compute_influence import _compute_influence_matrices, _get_lira_statistics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shadow_pool_no_aug = _build_no_aug_split(cfg, "shadow")
    train_dataset_size = int(cfg["pkeep"] * len(shadow_pool_no_aug))
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 2)),
    )

    lira_stats_list, c_lira_list, c_loss_list = [], [], []
    for k in range(n_shadow_models):
        shadow_dir = os.path.join(exp_dir, "shadows", str(k))
        model_path = os.path.join(shadow_dir, "shadow_model_out.pt")
        hinv_path = os.path.join(shadow_dir, "H_inv.npy")

        if not os.path.exists(model_path):
            print(f"  [reference shadow {k}] missing {model_path} — skipping.")
            continue
        if not os.path.exists(hinv_path):
            raise FileNotFoundError(
                f"[reference shadow {k}] missing {hinv_path}. Run compute_influence first."
            )

        model = _load_model_from_path(model_path, device)
        H_inv = torch.tensor(np.load(hinv_path), dtype=torch.float32, device=device)

        C_lira, C_loss = _compute_influence_matrices(
            model,
            reference_loader,
            H_inv,
            train_dataset_size=train_dataset_size,
            device=device,
        )
        lira_stats = _get_lira_statistics(model, reference_loader, device)

        lira_stats_list.append(lira_stats)
        c_lira_list.append(C_lira)
        c_loss_list.append(C_loss)

        del model, H_inv, C_lira, C_loss, lira_stats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return lira_stats_list, c_lira_list, c_loss_list


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
    from data.loader import offline_data_split

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
    result = _compute_target_scores_for_dataset(exp_dir, cfg, query_ds)
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


def _bucketize_scores(scores: np.ndarray, bucket_edges: np.ndarray) -> np.ndarray:
    return np.digitize(scores, bucket_edges)


def _fit_reference_bucket_gmms(influence_scores: np.ndarray, mia_scores: np.ndarray,
                               num_buckets: int = 10, min_points: int = 10,
                               verbose: bool = True):
    bucket_edges = np.quantile(influence_scores, np.linspace(0, 1, num_buckets + 1)[1:-1])
    bucket_ids = _bucketize_scores(influence_scores, bucket_edges)
    fits = []

    for b in range(num_buckets):
        idx = np.where(bucket_ids == b)[0]
        if len(idx) < min_points:
            if verbose:
                print(f"  [Reference bucket {b}] n={len(idx)}: too few points for mixture; using fallback p_in=0.5")
            fits.append(None)
            continue

        fit = fit_fixed_zero_rightshift_mixture(mia_scores[idx])
        if verbose:
            print(
                f"  [Reference bucket {b}] n={len(idx)}: "
                f"mu_in={fit['mu_in']:.3f}  "
                f"sd_out={np.sqrt(fit['var_out']):.3f}  "
                f"sd_in={np.sqrt(fit['var_in']):.3f}  "
                f"pi_in={fit['pi_in']:.3f}  "
                f"reliable={fit['reliable']}  "
                f"converged={fit['converged']}"
            )
            if not fit["reliable"]:
                print(f"    -> unreliable: {fit['reason']} — query points in this bucket will default to p_in=0.5")
        fits.append(fit)

    return bucket_edges, bucket_ids, fits


def _posterior_in_from_fit(fit, mia_scores: np.ndarray) -> np.ndarray:
    mia_scores = np.asarray(mia_scores, dtype=float)
    log_p_out = np.log(fit["pi_out"] + 1e-300) + norm.logpdf(mia_scores, 0.0, np.sqrt(fit["var_out"]))
    log_p_in = np.log(fit["pi_in"] + 1e-300) + norm.logpdf(mia_scores, fit["mu_in"], np.sqrt(fit["var_in"]))
    return np.exp(log_p_in - np.logaddexp(log_p_out, log_p_in))


def _apply_reference_bucket_gmms(influence_scores: np.ndarray, mia_scores: np.ndarray,
                                 bucket_edges: np.ndarray, bucket_fits):
    bucket_ids = _bucketize_scores(influence_scores, bucket_edges)
    gmm_probs = np.full(len(mia_scores), 0.5, dtype=float)

    for b, fit in enumerate(bucket_fits):
        idx = np.where(bucket_ids == b)[0]
        if len(idx) == 0:
            continue
        if fit is None or not fit["reliable"]:
            gmm_probs[idx] = 0.5
        else:
            gmm_probs[idx] = _posterior_in_from_fit(fit, mia_scores[idx])

    return gmm_probs, bucket_ids


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _balanced_accuracy_from_roc(fpr, tpr):
    return float(((tpr + (1.0 - fpr)) / 2.0).max())


def _tpr_at_fpr(fpr, tpr, max_fpr=0.01):
    valid = np.where(fpr <= max_fpr)[0]
    return float(tpr[valid[-1]]) if len(valid) > 0 else float("nan")


def _compute_metrics_from_scores(ground_truth: np.ndarray, scores: np.ndarray):
    fpr, tpr, _ = roc_curve(ground_truth.astype(int), scores)
    return {
        "tpr_0": _tpr_at_fpr(fpr, tpr, max_fpr=0.0),
        "tpr_01": _tpr_at_fpr(fpr, tpr, max_fpr=0.001),
        "tpr_1": _tpr_at_fpr(fpr, tpr, max_fpr=0.01),
        "bal_acc": _balanced_accuracy_from_roc(fpr, tpr),
    }


def _prepare_binary_membership_labels(labels):
    if labels is None:
        return None
    labels = np.asarray(labels).astype(np.int32)
    uniq = np.unique(labels)
    if uniq.size == 2 and np.array_equal(uniq, np.array([0, 1], dtype=np.int32)):
        return labels
    return None


def _compute_unsupervised_fit_quality(bucket_ids: np.ndarray, bucket_fits) -> dict:
    total = int(len(bucket_ids))
    if total == 0:
        return {
            "fit_quality": float("-inf"),
            "reliable_fraction": 0.0,
            "mean_reliable_loglik": float("-inf"),
        }

    reliable_points = 0
    reliable_loglik = []
    for b, fit in enumerate(bucket_fits):
        n_b = int((bucket_ids == b).sum())
        if n_b == 0:
            continue
        if fit is None or not fit.get("reliable", False):
            continue
        reliable_points += n_b
        reliable_loglik.append(float(fit.get("loglik", float("-inf"))) / max(n_b, 1))

    reliable_fraction = float(reliable_points / total)
    mean_reliable_loglik = (
        float(np.mean(reliable_loglik)) if len(reliable_loglik) > 0 else float("-inf")
    )
    # Prioritize reliable coverage; log-likelihood is a tie-breaker.
    fit_quality = reliable_fraction + 1e-3 * mean_reliable_loglik
    return {
        "fit_quality": fit_quality,
        "reliable_fraction": reliable_fraction,
        "mean_reliable_loglik": mean_reliable_loglik,
    }


def _select_best_reference_bucketing(
    reference_scores: np.ndarray,
    reference_mia_scores: np.ndarray,
    reference_ground_truth,
    bucket_candidates,
    selection_metric: str,
    min_buckets: int = 2,
    max_buckets: int = 100,
    max_evals: int = 16,
    selection_influence_scores: np.ndarray = None,
    selection_mia_scores: np.ndarray = None,
    selection_ground_truth: np.ndarray = None,
):
    metric_key = {
        "tpr@0%fpr": "tpr_0",
        "tpr@0.1%fpr": "tpr_01",
        "tpr@1%fpr": "tpr_1",
        "balanced_acc": "bal_acc",
        "bal_acc": "bal_acc",
    }.get(str(selection_metric).strip().lower())
    if metric_key is None and str(selection_metric).strip().lower() not in {"fit_quality", "unsupervised"}:
        raise ValueError(
            "gmm_selection_metric must be one of: "
            "tpr@0%fpr, tpr@0.1%fpr, tpr@1%fpr, balanced_acc, fit_quality"
        )

    binary_reference_labels = _prepare_binary_membership_labels(reference_ground_truth)
    binary_selection_labels = _prepare_binary_membership_labels(selection_ground_truth)

    supervised_selection = metric_key is not None and (
        binary_selection_labels is not None or binary_reference_labels is not None
    )

    if metric_key is not None and not supervised_selection:
        print(
            "  [Reference tune] no binary membership labels available for reference set; "
            "falling back to unsupervised fit_quality selection."
        )

    min_buckets = max(2, int(min_buckets))
    max_avg_safe = max(2, int(len(reference_scores) // 10))
    max_buckets = min(int(max_buckets), max_avg_safe)
    if min_buckets > max_buckets:
        min_buckets = max_buckets

    seed_candidates = sorted(set(int(x) for x in bucket_candidates))
    seed_candidates = [x for x in seed_candidates if min_buckets <= x <= max_buckets]
    if not seed_candidates:
        midpoint = max(min_buckets, min(max_buckets, (min_buckets + max_buckets) // 2))
        seed_candidates = [midpoint]

    tried = []
    cache = {}
    best = None
    eval_count = 0

    def _evaluate(nb: int):
        nonlocal best, eval_count
        nb = int(nb)
        if nb < min_buckets or nb > max_buckets:
            return None
        if nb in cache:
            return cache[nb]
        if eval_count >= int(max_evals):
            return None

        print(f"\n  [Reference tune] trying num_buckets={nb}...")
        bucket_edges, _, bucket_fits = _fit_reference_bucket_gmms(
            reference_scores,
            reference_mia_scores,
            num_buckets=nb,
            verbose=False,
        )
        ref_probs, ref_bucket_ids = _apply_reference_bucket_gmms(
            reference_scores,
            reference_mia_scores,
            bucket_edges,
            bucket_fits,
        )

        if supervised_selection:
            if (
                binary_selection_labels is not None
                and selection_influence_scores is not None
                and selection_mia_scores is not None
            ):
                selection_probs, _ = _apply_reference_bucket_gmms(
                    selection_influence_scores,
                    selection_mia_scores,
                    bucket_edges,
                    bucket_fits,
                )
                metrics = _compute_metrics_from_scores(binary_selection_labels, selection_probs)
            else:
                metrics = _compute_metrics_from_scores(binary_reference_labels, ref_probs)
        else:
            metrics = _compute_unsupervised_fit_quality(ref_bucket_ids, bucket_fits)
        candidate = {
            "num_buckets": nb,
            "bucket_edges": bucket_edges,
            "bucket_fits": bucket_fits,
            "metrics": metrics,
        }
        cache[nb] = candidate
        tried.append((nb, metrics))
        eval_count += 1
        if supervised_selection:
            print(
                "    metrics: "
                f"TPR@0%FPR={metrics['tpr_0']*100:.2f}%  "
                f"TPR@0.1%FPR={metrics['tpr_01']*100:.2f}%  "
                f"TPR@1%FPR={metrics['tpr_1']*100:.2f}%  "
                f"Balanced Acc={metrics['bal_acc']*100:.2f}%"
            )
            score_value = metrics[metric_key]
        else:
            print(
                "    fit quality: "
                f"reliable_fraction={metrics['reliable_fraction']*100:.2f}%  "
                f"mean_reliable_loglik={metrics['mean_reliable_loglik']:.4f}  "
                f"objective={metrics['fit_quality']:.6f}"
            )
            score_value = metrics["fit_quality"]

        if best is None:
            best = candidate
        else:
            best_score = best["metrics"][metric_key] if supervised_selection else best["metrics"]["fit_quality"]
            if score_value > best_score:
                best = candidate

        if best is candidate:
            pass
        return candidate

    for nb in seed_candidates:
        _evaluate(nb)
        if eval_count >= int(max_evals):
            break

    if best is not None and eval_count < int(max_evals):
        search_span = max(max_buckets - min_buckets, 1)
        step = max(1, search_span // 4)
        while step >= 1 and eval_count < int(max_evals):
            improved = True
            while improved and eval_count < int(max_evals):
                improved = False
                current_best_nb = int(best["num_buckets"])
                for nb in (current_best_nb - step, current_best_nb + step):
                    prev_best_nb = int(best["num_buckets"])
                    cand = _evaluate(nb)
                    if cand is None:
                        continue
                    if int(best["num_buckets"]) != prev_best_nb:
                        improved = True
            step //= 2

    if best is None:
        raise RuntimeError("No valid bucket candidates for GMM tuning.")

    print("\n  [Reference tune] summary:")
    for nb, metrics in tried:
        if supervised_selection:
            print(
                f"    buckets={nb:2d}: "
                f"TPR@0%FPR={metrics['tpr_0']*100:.2f}%  "
                f"TPR@0.1%FPR={metrics['tpr_01']*100:.2f}%  "
                f"TPR@1%FPR={metrics['tpr_1']*100:.2f}%  "
                f"Balanced Acc={metrics['bal_acc']*100:.2f}%"
            )
        else:
            print(
                f"    buckets={nb:2d}: "
                f"reliable_fraction={metrics['reliable_fraction']*100:.2f}%  "
                f"mean_reliable_loglik={metrics['mean_reliable_loglik']:.4f}  "
                f"objective={metrics['fit_quality']:.6f}"
            )

    selected_by = selection_metric if supervised_selection else "fit_quality"
    print(
        f"  [Reference tune] selected num_buckets={best['num_buckets']} "
        f"by {selected_by}"
    )

    return best

    return best


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


def _plot_bucket_tpr_comparison(scores_dict, mia_scores, ground_truth, out_dir, num_buckets,
                                bucket_edges=None):
    score_names = list(scores_dict.keys())
    bucket_tprs = {name: [] for name in score_names}

    ref_scores = next(iter(scores_dict.values()))
    if bucket_edges is None:
        bucket_edges = np.quantile(ref_scores, np.linspace(0, 1, num_buckets + 1)[1:-1])
    bucket_ids = np.digitize(ref_scores, bucket_edges)

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


def _plot_gmm_bucket_components(influence_scores, mia_scores, ground_truth, bucket_edges,
                                bucket_fits, out_dir, num_buckets=10):
    """Plot query points against the reference-fitted constrained mixtures."""
    bucket_plot_dir = os.path.join(out_dir, "gmm_bucket_components")
    os.makedirs(bucket_plot_dir, exist_ok=True)
    csv_rows = []
    bucket_ids = np.digitize(influence_scores, bucket_edges)

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

        # extend grid to comfortably show the zero-mean OUT component
        grid_lo = min(score_min, -3.0 * np.sqrt(float(np.var(bucket_scores))))
        grid_hi = max(score_max, 1.0)
        x_grid = np.linspace(grid_lo, grid_hi, 400)

        fig, ax = plt.subplots(figsize=(7.2, 4.8))

        fit = bucket_fits[b] if b < len(bucket_fits) else None
        if fit is not None:
            thr_str = f"{fit['threshold']:.3f}" if fit["threshold"] is not None else "none"
            print(
                f"  [Bucket {b}] n={len(idx)}: "
                f"mu_in={fit['mu_in']:.3f}  "
                f"sd_out={np.sqrt(fit['var_out']):.3f}  "
                f"sd_in={np.sqrt(fit['var_in']):.3f}  "
                f"pi_in={fit['pi_in']:.3f}  "
                f"threshold={thr_str}  "
                f"reliable={fit['reliable']}  converged={fit['converged']}"
            )
            if not fit["reliable"]:
                print(f"    -> unreliable: {fit['reason']}")

            pdf_out   = fit["pi_out"] * norm.pdf(x_grid, 0.0, np.sqrt(fit["var_out"]))
            pdf_in    = fit["pi_in"]  * norm.pdf(x_grid, fit["mu_in"], np.sqrt(fit["var_in"]))
            total_pdf = pdf_out + pdf_in

            p_in  = _posterior_in_from_fit(fit, bucket_scores)
            p_out = 1.0 - p_in
        else:
            fit       = None
            total_pdf = None
            pdf_out   = None
            pdf_in    = None
            p_in  = np.full(len(idx), np.nan, dtype=float)
            p_out = np.full(len(idx), np.nan, dtype=float)

        predicted_is_in = p_in >= 0.5

        bins = np.linspace(score_min, score_max, 28)
        ax.hist(non_member_scores, bins=bins, density=True, alpha=0.45,
                color="tomato", label="out / non-member")
        ax.hist(member_scores, bins=bins, density=True, alpha=0.45,
                color="steelblue", label="in / member")

        if total_pdf is not None:
            ax.plot(x_grid, total_pdf, color="black",      linewidth=2.0, label="mixture total")
            ax.plot(x_grid, pdf_out,   color="darkorange",  linewidth=1.8,
                    linestyle="--", label=f"OUT  μ=0  σ={np.sqrt(fit['var_out']):.2f}")
            ax.plot(x_grid, pdf_in,    color="purple",      linewidth=1.8,
                    linestyle="--", label=f"IN   μ={fit['mu_in']:.2f}  σ={np.sqrt(fit['var_in']):.2f}")

            ax.axvline(0.0,           color="darkorange", linestyle=":", linewidth=1.2)
            ax.axvline(fit["mu_in"],  color="purple",     linestyle=":", linewidth=1.2)

            if fit["threshold"] is not None:
                ax.axvline(fit["threshold"], color="gray", linestyle="-.", linewidth=1.0,
                           label=f"threshold={fit['threshold']:.2f}")

            reliability_tag = "" if fit["reliable"] else f"  [UNRELIABLE: {fit['reason']}]"
        else:
            ax.text(0.5, 0.9, "Too few points for mixture", transform=ax.transAxes,
                    ha="center", va="top", fontsize=9)
            reliability_tag = ""

        title = (
            f"Bucket {b}  n={len(idx)}  "
            f"in={int(bucket_members.sum())}  out={int(bucket_non_members.sum())}"
            + reliability_tag
        )
        ax.set_title(title, fontsize=8 if reliability_tag else 10)
        ax.set_xlabel("MIA score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, loc="best")

        fig.tight_layout()
        out_path = os.path.join(bucket_plot_dir, f"bucket_{b:02d}_gmm_components.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

        mu_in_val   = float(fit["mu_in"])  if fit else float("nan")
        pi_in_val   = float(fit["pi_in"])  if fit else float("nan")
        reliable    = bool(fit["reliable"]) if fit else False
        threshold   = fit["threshold"]     if fit else None
        for local_pos, point_idx in enumerate(idx):
            csv_rows.append({
                "point_index":    int(point_idx),
                "bucket":         int(b),
                "bucket_size":    int(len(idx)),
                "mu_in":          mu_in_val,
                "pi_in":          pi_in_val,
                "reliable":       reliable,
                "threshold":      float(threshold) if threshold is not None else float("nan"),
                "mia_score":      float(bucket_scores[local_pos]),
                "ground_truth":   int(ground_truth[point_idx]),
                "ground_truth_label": "in" if ground_truth[point_idx] == 1 else "out",
                "predicted_label": "in" if bool(predicted_is_in[local_pos]) else "out",
                "p_in":           float(p_in[local_pos]),
                "p_out":          float(p_out[local_pos]),
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
                "mu_in",
                "pi_in",
                "reliable",
                "threshold",
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

def analyze_score(score_name, scores, mia_scores, ground_truth, out_dir, num_buckets=10,
                  bucket_edges=None):
    corr = np.corrcoef(scores, mia_scores)[0, 1]
    print(f"\n  [{score_name}] Pearson corr(score, MIA score): {corr:.3f}")

    if bucket_edges is None:
        bucket_edges = np.quantile(scores, np.linspace(0, 1, num_buckets + 1)[1:-1])
    bucket_ids = np.digitize(scores, bucket_edges)

    print(f"  [{score_name}] Bucketed metrics (within-bucket and scan-from-bucket vs full dataset):")
    for b in range(num_buckets):
        idx = np.where(bucket_ids == b)[0]
        if len(idx) < 10:
            print(f"    Bucket {b}: too few points ({len(idx)}), skipping")
            continue

        # Within-bucket ROC
        fpr_b, tpr_b, _ = roc_curve(ground_truth[idx], mia_scores[idx])
        tpr_0pct_b  = _tpr_at_fpr(fpr_b, tpr_b, max_fpr=0.0)
        tpr_01pct_b = _tpr_at_fpr(fpr_b, tpr_b, max_fpr=0.001)
        tpr_1pct_b  = _tpr_at_fpr(fpr_b, tpr_b, max_fpr=0.01)
        bal_acc_b   = _balanced_accuracy_from_roc(fpr_b, tpr_b)

        # Scan-from-bucket: attacker only audits samples in buckets b..num_buckets-1.
        # Non-audited samples are predicted non-member regardless of MIA score.
        # FPR/TPR denominators are still full-dataset member/non-member counts.
        scan_mask = (bucket_ids >= b)
        n_members     = int(ground_truth.sum())
        n_nonmembers  = len(ground_truth) - n_members
        # Use full-dataset ROC but zero out scores for non-scanned samples so they
        # rank last (always predicted non-member).
        masked_scores = np.where(scan_mask, mia_scores, mia_scores.min() - 1.0)
        fpr_s, tpr_s, _ = roc_curve(ground_truth, masked_scores)
        tpr_0pct_s  = _tpr_at_fpr(fpr_s, tpr_s, max_fpr=0.0)
        tpr_01pct_s = _tpr_at_fpr(fpr_s, tpr_s, max_fpr=0.001)
        tpr_1pct_s  = _tpr_at_fpr(fpr_s, tpr_s, max_fpr=0.01)
        bal_acc_s   = _balanced_accuracy_from_roc(fpr_s, tpr_s)

        print(f"    Bucket {b}: size={len(idx):4d}")
        print(f"      [within-bucket ]  TPR@0%FPR={tpr_0pct_b*100:5.2f}%  TPR@0.1%FPR={tpr_01pct_b*100:5.2f}%  TPR@1%FPR={tpr_1pct_b*100:5.2f}%  Balanced Acc={bal_acc_b*100:5.2f}%")
        print(f"      [scan from b.. ]  TPR@0%FPR={tpr_0pct_s*100:5.2f}%  TPR@0.1%FPR={tpr_01pct_s*100:5.2f}%  TPR@1%FPR={tpr_1pct_s*100:5.2f}%  Balanced Acc={bal_acc_s*100:5.2f}%")
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
    reference_fraction = float(cfg.get("gmm_reference_fraction", 0.25))
    bucket_candidates = cfg.get("gmm_bucket_candidates", [5, 10, 15, 20, 25])
    bucket_candidates = sorted(set(int(x) for x in bucket_candidates + [num_buckets]))
    selection_metrics = cfg.get("gmm_selection_metrics", ["tpr@0%fpr", "tpr@0.1%fpr"])
    selection_metrics = [str(m).strip().lower() for m in selection_metrics]
    # Keep order, drop duplicates.
    selection_metrics = list(dict.fromkeys(selection_metrics))
    search_max_evals = int(cfg.get("gmm_bucket_search_max_evals", 16))
    min_buckets_cfg = int(cfg.get("gmm_bucket_min", 2))
    max_buckets_cfg = int(cfg.get("gmm_bucket_max", 100))

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
    # 2. Reference calibration subset for bucket / GMM fitting
    # ------------------------------------------------------------------
    print(f"\nLoading reference split for calibration (fraction={reference_fraction:.2f})...")
    reference_pool_no_aug = _build_no_aug_split(cfg, "reference")
    reference_subset, reference_subset_indices = _select_reference_subset(
        reference_pool_no_aug, reference_fraction, cfg["seed"]
    )
    print(
        f"  reference split size={len(reference_pool_no_aug)}  "
        f"calibration subset={len(reference_subset)}"
    )

    # ------------------------------------------------------------------
    # 3. Shadow artifacts — all query caches must have shape (n_query,)
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
    # 4. Target model scores on D_query
    # ------------------------------------------------------------------
    print("\nEvaluating target model on query set...")
    target_scores = _compute_target_lira_scores(exp_dir, cfg, query_pool_indices)
    print(f"  target_scores: mean={target_scores.mean():.3f}, std={target_scores.std():.3f}")

    # ------------------------------------------------------------------
    # 5. Expected discrepancy MIA scores
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
    # 6. Reference-calibrated bucket fits
    # ------------------------------------------------------------------
    print("\nComputing reference-calibrated bucket fits...")
    reference_target_scores = _compute_target_scores_for_dataset(exp_dir, cfg, reference_subset)
    reference_ground_truth = None
    reference_lira_stats, reference_c_lira_list, reference_c_loss_list = _compute_shadow_reference_scores(
        exp_dir, cfg, reference_subset, n_shadow_models
    )
    if len(reference_lira_stats) < 2:
        print("  Need at least 2 complete reference shadows — skipping GMM calibration.")
        return

    reference_mia_scores = _compute_mia_scores(reference_lira_stats, reference_target_scores)

    # ------------------------------------------------------------------
    # 7. Influence scores
    # ------------------------------------------------------------------
    print("\nAggregating influence scores across shadows...")
    scores_C_lira = _aggregate_influence(C_lira_list)
    has_closs     = all(c is not None for c in C_loss_list)
    scores_C_loss = _aggregate_influence(C_loss_list) if has_closs else None

    reference_scores_C_lira = _aggregate_influence(reference_c_lira_list)
    has_reference_closs = all(c is not None for c in reference_c_loss_list)
    reference_scores_C_loss = _aggregate_influence(reference_c_loss_list) if has_reference_closs else None

    scores_dict = {"C_lira": scores_C_lira}
    if scores_C_loss is not None:
        scores_dict["C_loss"] = scores_C_loss

    reference_scores_dict = {"C_lira": reference_scores_C_lira}
    if reference_scores_C_loss is not None:
        reference_scores_dict["C_loss"] = reference_scores_C_loss

    for name, scores in scores_dict.items():
        ref_scores = reference_scores_dict.get(name)
        ref_bucket_edges = None
        if ref_scores is not None:
            ref_bucket_edges = np.quantile(ref_scores, np.linspace(0, 1, num_buckets + 1)[1:-1])
        analyze_score(name, scores, mia_scores, ground_truth, out_dir, num_buckets, bucket_edges=ref_bucket_edges)

    primary_bucket_name = "C_loss" if (scores_C_loss is not None and reference_scores_C_loss is not None) else "C_lira"
    gmm_bucket_name = primary_bucket_name
    gmm_bucket_scores = scores_dict[gmm_bucket_name]
    reference_gmm_bucket_scores = reference_scores_dict[gmm_bucket_name]

    if gmm_bucket_name != "C_loss":
        print("\nC_loss not available for all shadows; using C_lira for GMM buckets.")

    dual_results = {}
    first_result = None
    for selection_metric in selection_metrics:
        print(
            "\nTuning bucket count on reference-fitted GMMs "
            f"for {gmm_bucket_name} by QUERY aggregated metric={selection_metric}..."
        )
        best_bucketing = _select_best_reference_bucketing(
            reference_gmm_bucket_scores,
            reference_mia_scores,
            reference_ground_truth=reference_ground_truth,
            bucket_candidates=bucket_candidates,
            selection_metric=selection_metric,
            min_buckets=min_buckets_cfg,
            max_buckets=max_buckets_cfg,
            max_evals=search_max_evals,
            selection_influence_scores=gmm_bucket_scores,
            selection_mia_scores=mia_scores,
            selection_ground_truth=ground_truth,
        )

        active_num_buckets = int(best_bucketing["num_buckets"])
        reference_bucket_edges = best_bucketing["bucket_edges"]
        bucket_fits = best_bucketing["bucket_fits"]

        gmm_probs, query_bucket_ids = _apply_reference_bucket_gmms(
            gmm_bucket_scores,
            mia_scores,
            reference_bucket_edges,
            bucket_fits,
        )

        metrics_gmm = _compute_metrics_from_scores(ground_truth.astype(int), gmm_probs)
        tpr_0pct_gmm = metrics_gmm["tpr_0"]
        tpr_01pct_gmm = metrics_gmm["tpr_01"]
        tpr_1pct_gmm = metrics_gmm["tpr_1"]
        bal_acc_gmm = metrics_gmm["bal_acc"]

        print(f"  GMM-aggregated MIA ({selection_metric}): "
              f"TPR@0%FPR={tpr_0pct_gmm*100:.2f}%, "
              f"TPR@0.1%FPR={tpr_01pct_gmm*100:.2f}%, "
              f"TPR@1%FPR={tpr_1pct_gmm*100:.2f}%, "
              f"Balanced Acc={bal_acc_gmm*100:.2f}%")

        metric_tag = selection_metric.replace("@", "at").replace("%", "pct").replace(".", "p").replace("/", "_")
        metric_out_dir = os.path.join(out_dir, f"gmm_{metric_tag}")
        _plot_gmm_bucket_components(
            gmm_bucket_scores,
            mia_scores,
            ground_truth,
            reference_bucket_edges,
            bucket_fits,
            metric_out_dir,
            active_num_buckets,
        )

        dual_results[selection_metric] = {
            "gmm_probs": gmm_probs,
            "query_bucket_ids": query_bucket_ids,
            "bucket_edges": reference_bucket_edges,
            "num_buckets": active_num_buckets,
            "metrics": metrics_gmm,
        }
        if first_result is None:
            first_result = (selection_metric, dual_results[selection_metric])

    # Keep one representative bucket-comparison plot using first optimization.
    first_metric, first_payload = first_result
    _plot_bucket_tpr_comparison(
        scores_dict,
        mia_scores,
        ground_truth,
        out_dir,
        int(first_payload["num_buckets"]),
        bucket_edges=first_payload["bucket_edges"],
    )

    # ------------------------------------------------------------------
    # 9. Save analysis outputs
    # ------------------------------------------------------------------
    save_dict = dict(
        mia_scores=mia_scores,
        gmm_probs=first_payload["gmm_probs"],
        ground_truth=ground_truth,
        scores_C_lira=scores_C_lira,
        query_pool_indices=query_pool_indices,
        scores_C_lira_reference=reference_scores_C_lira,
        gmm_bucket_edges=first_payload["bucket_edges"],
        gmm_reference_fraction=np.array(reference_fraction, dtype=np.float32),
        gmm_bucket_name=np.array(gmm_bucket_name),
        gmm_selection_metric=np.array(first_metric),
        gmm_selected_num_buckets=np.array(first_payload["num_buckets"], dtype=np.int32),
        gmm_query_bucket_ids=first_payload["query_bucket_ids"],
        reference_subset_size=np.array(len(reference_subset), dtype=np.int32),
    )
    if scores_C_loss is not None:
        save_dict["scores_C_loss"] = scores_C_loss
    if reference_scores_C_loss is not None:
        save_dict["scores_C_loss_reference"] = reference_scores_C_loss

    for metric_name, payload in dual_results.items():
        metric_tag = metric_name.replace("@", "at").replace("%", "pct").replace(".", "p").replace("/", "_")
        save_dict[f"gmm_probs_{metric_tag}"] = payload["gmm_probs"]
        save_dict[f"gmm_bucket_edges_{metric_tag}"] = payload["bucket_edges"]
        save_dict[f"gmm_selected_num_buckets_{metric_tag}"] = np.array(payload["num_buckets"], dtype=np.int32)
        save_dict[f"gmm_query_bucket_ids_{metric_tag}"] = payload["query_bucket_ids"]

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
    for metric_name, payload in dual_results.items():
        m = payload["metrics"]
        print(f"  MIA (GMM-bucketed, selected by {metric_name}):"
              f"TPR@0%FPR={m['tpr_0' ]*100:.2f}%  "
              f"TPR@0.1%FPR={m['tpr_01' ]*100:.2f}%  "
              f"TPR@1%FPR={m['tpr_1' ]*100:.2f}%  "
              f"Balanced Acc={m['bal_acc' ]*100:.2f}%")
    print(f"{'='*60}\n")


def main():
    cli = _parse_args()
    run(exp_dir=cli.exp_dir, dataset=cli.dataset, num_buckets=cli.num_buckets)


if __name__ == "__main__":
    main()
