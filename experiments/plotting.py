"""All matplotlib plotting functions for influence vs MIA analysis."""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

from .metrics import fit_fixed_zero_rightshift_mixture, tpr_at_fpr


def plot_bucket_mia_hist(mia_scores, ground_truth, bucket_indices, bucket_id,
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


def plot_bucket_tpr_comparison(scores_dict, mia_scores, ground_truth, out_dir, num_buckets):
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
        tpr_val = tpr_at_fpr(fpr, tpr, max_fpr=0.01)
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


def plot_score_vs_mia(score_name, scores, mia_scores, ground_truth, out_dir):
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


def plot_gmm_bucket_components(influence_scores, mia_scores, ground_truth, out_dir,
                               num_buckets=10, min_points=10):
    """Plot one fitted constrained mixture per influence bucket, and write a CSV."""
    bucket_plot_dir = os.path.join(out_dir, "gmm_bucket_components")
    os.makedirs(bucket_plot_dir, exist_ok=True)
    csv_rows = []
    quantiles  = np.quantile(influence_scores, np.linspace(0, 1, num_buckets + 1)[1:-1])
    bucket_ids = np.digitize(influence_scores, quantiles)

    for b in range(num_buckets):
        idx = np.where(bucket_ids == b)[0]
        if len(idx) == 0:
            continue

        bucket_scores      = mia_scores[idx]
        bucket_members     = ground_truth[idx] == 1
        bucket_non_members = ~bucket_members
        member_scores      = bucket_scores[bucket_members]
        non_member_scores  = bucket_scores[bucket_non_members]

        score_min = float(bucket_scores.min())
        score_max = float(bucket_scores.max())
        if np.isclose(score_min, score_max):
            score_min -= 0.5
            score_max += 0.5

        grid_lo = min(score_min, -3.0 * np.sqrt(float(np.var(bucket_scores))))
        grid_hi = max(score_max, 1.0)
        x_grid  = np.linspace(grid_lo, grid_hi, 400)

        fig, ax = plt.subplots(figsize=(7.2, 4.8))

        if len(idx) >= min_points:
            fit = fit_fixed_zero_rightshift_mixture(bucket_scores)
            thr_str = f"{fit['threshold']:.3f}" if fit["threshold"] is not None else "none"
            print(
                f"  [Bucket {b}] n={len(idx)}: "
                f"mu_in={fit['mu_in']:.3f}  "
                f"sd_out={np.sqrt(fit['var_out']):.3f}  "
                f"sd_in={np.sqrt(fit['var_in']):.3f}  "
                f"pi_in={fit['pi_in']:.3f}  "
                f"bhattacharyya={fit['bhattacharyya']:.3f}  "
                f"threshold={thr_str}  "
                f"reliable={fit['reliable']}  converged={fit['converged']}"
            )
            if not fit["reliable"]:
                print(f"    -> unreliable: {fit['reason']}")

            if len(bucket_members) >= 4 and bucket_members.sum() >= 2 and (~bucket_members).sum() >= 2:
                try:
                    real_auc = roc_auc_score(bucket_members.astype(int), fit["llr"])
                    rng_perm = np.random.RandomState(42)
                    perm_aucs = []
                    for _ in range(200):
                        shuffled = rng_perm.permutation(bucket_members.astype(int))
                        perm_aucs.append(roc_auc_score(shuffled, fit["llr"]))
                    perm_p = float(np.mean(np.array(perm_aucs) >= real_auc))
                    print(
                        f"    -> permutation test: real AUC={real_auc:.3f}  "
                        f"perm mean={np.mean(perm_aucs):.3f}  p={perm_p:.3f}"
                        + ("  [NO SIGNAL]" if perm_p > 0.1 else "")
                    )
                except Exception:
                    pass

            pdf_out   = fit["pi_out"] * norm.pdf(x_grid, 0.0, np.sqrt(fit["var_out"]))
            pdf_in    = fit["pi_in"]  * norm.pdf(x_grid, fit["mu_in"], np.sqrt(fit["var_in"]))
            total_pdf = pdf_out + pdf_in
            p_in      = fit["posterior_in"]
            p_out     = fit["posterior_out"]
            llr_vals  = fit["llr"]
        else:
            fit       = None
            total_pdf = None
            pdf_out   = None
            pdf_in    = None
            p_in  = np.full(len(idx), np.nan, dtype=float)
            p_out = np.full(len(idx), np.nan, dtype=float)
            llr_vals = np.zeros(len(idx), dtype=float)
            if len(idx) > 0:
                ranks = np.argsort(np.argsort(bucket_scores)).astype(float)
                p_in  = ranks / max(len(idx) - 1, 1)
                p_out = 1.0 - p_in

        predicted_is_in = llr_vals > 0.0

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

        mu_in_val  = float(fit["mu_in"])         if fit else float("nan")
        pi_in_val  = float(fit["pi_in"])         if fit else float("nan")
        bhatta_val = float(fit["bhattacharyya"]) if fit else float("nan")
        reliable   = bool(fit["reliable"])        if fit else False
        threshold  = fit["threshold"]             if fit else None
        for local_pos, point_idx in enumerate(idx):
            csv_rows.append({
                "point_index":        int(point_idx),
                "bucket":             int(b),
                "bucket_size":        int(len(idx)),
                "mu_in":              mu_in_val,
                "pi_in":              pi_in_val,
                "bhattacharyya":      bhatta_val,
                "reliable":           reliable,
                "threshold":          float(threshold) if threshold is not None else float("nan"),
                "mia_score":          float(bucket_scores[local_pos]),
                "llr":                float(llr_vals[local_pos]),
                "ground_truth":       int(ground_truth[point_idx]),
                "ground_truth_label": "in" if ground_truth[point_idx] == 1 else "out",
                "predicted_label":    "in" if bool(predicted_is_in[local_pos]) else "out",
                "p_in":               float(p_in[local_pos]),
                "p_out":              float(p_out[local_pos]),
            })

    csv_rows.sort(key=lambda row: (row["bucket"], row["mia_score"], row["point_index"]))
    csv_path = os.path.join(bucket_plot_dir, "bucket_points_sorted.csv")
    fieldnames = [
        "point_index", "bucket", "bucket_size", "mu_in", "pi_in", "bhattacharyya",
        "reliable", "threshold", "mia_score", "llr", "ground_truth",
        "ground_truth_label", "predicted_label", "p_in", "p_out",
    ]
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"  Saved bucket-wise GMM component plots to {bucket_plot_dir}")
    print(f"  Saved sorted bucket-point CSV to {csv_path}")
