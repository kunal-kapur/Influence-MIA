"""Post-hoc analysis: influence scores vs MIA vulnerability bucketing.

For each query point we have, across K shadow models:
  - lira_stats[k][i]  : scaled logit  t_i  for shadow k  (N,)
  - C_lira[k][i]      : influence on LiRA statistic        (N,)
  - C_loss[k][i]      : influence on CE loss                (N,)

The MIA attack score for point i (expected discrepancy) is:
    mia_score[i] = target_score[i] - mean_k(lira_stats[k][i])

Usage
-----
python -m experiments.analyze --exp_dir outputs/<exp_name>/cifar10
python -m experiments.analyze --exp_dir outputs/<exp_name>/cifar10 --num_buckets 10
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import yaml
from sklearn.metrics import roc_curve

from .data_utils import (
    assert_query_has_in_out_shadow_coverage,
    compute_target_lira_scores,
    load_query_metadata,
    load_shadow_data,
)
from .metrics import (
    adaptive_gmm_bucket_search,
    aggregate_influence,
    balanced_accuracy_from_roc,
    compute_mia_scores,
    tpr_at_fpr,
)
from .plotting import (
    plot_bucket_mia_hist,
    plot_bucket_tpr_comparison,
    plot_gmm_bucket_components,
    plot_score_vs_mia,
)


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
        tpr_0pct  = tpr_at_fpr(fpr, tpr, max_fpr=0.0)
        tpr_01pct = tpr_at_fpr(fpr, tpr, max_fpr=0.001)
        tpr_1pct  = tpr_at_fpr(fpr, tpr, max_fpr=0.01)
        bal_acc   = balanced_accuracy_from_roc(fpr, tpr)
        print(f"    Bucket {b}: size={len(idx):4d}, "
              f"TPR@0%FPR={tpr_0pct*100:5.2f}%, "
              f"TPR@0.1%FPR={tpr_01pct*100:5.2f}%, "
              f"TPR@1%FPR={tpr_1pct*100:5.2f}%, "
              f"Balanced Acc={bal_acc*100:5.2f}%")
        plot_bucket_mia_hist(mia_scores, ground_truth, idx, b, out_dir, score_name)

    plot_score_vs_mia(score_name, scores, mia_scores, ground_truth, out_dir)


def _parse_args():
    parser = argparse.ArgumentParser(description="Influence vs LiRA vulnerability analysis")
    parser.add_argument("--dataset",         default="cifar10")
    parser.add_argument("--exp_dir",         required=True,
                        help="Experiment directory (e.g. outputs/<exp>/cifar10)")
    parser.add_argument("--num_buckets",     type=int, default=10,
                        help="Number of quantile buckets (default: 5 = quintiles)")
    parser.add_argument("--min_gmm_buckets", type=int, default=4,
                        help="Minimum bucket count for adaptive GMM search")
    parser.add_argument("--max_gmm_buckets", type=int, default=50,
                        help="Maximum bucket count for adaptive GMM search")
    return parser.parse_args()


def run(
    exp_dir: str,
    dataset: str = "cifar10",
    num_buckets: int = 5,
    min_gmm_buckets: int = 4,
    max_gmm_buckets: int = 20,
) -> None:
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
    # 1. Query-set metadata
    # ------------------------------------------------------------------
    print("\nLoading query metadata...")
    query_global_indices, ground_truth = load_query_metadata(exp_dir)
    n_query    = len(query_global_indices)
    members    = int(ground_truth.sum())
    nonmembers = int((ground_truth == 0).sum())
    print(f"  n_query={n_query}  members={members}  non-members={nonmembers}")
    assert_query_has_in_out_shadow_coverage(cfg, query_global_indices)

    # ------------------------------------------------------------------
    # 2. Shadow artifacts
    # ------------------------------------------------------------------
    print(f"\nLoading {n_shadow_models} shadow models from {exp_dir}/shadows/...")
    lira_stats, C_lira_list, C_loss_list = load_shadow_data(
        exp_dir,
        n_shadow_models,
        n_query=n_query,
        query_global_indices=query_global_indices,
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
    target_scores = compute_target_lira_scores(exp_dir, cfg, query_global_indices)
    print(f"  target_scores: mean={target_scores.mean():.3f}, std={target_scores.std():.3f}")

    # ------------------------------------------------------------------
    # 4. Expected discrepancy MIA scores
    # ------------------------------------------------------------------
    print("\nComputing MIA scores (expected discrepancy)...")
    mia_scores = compute_mia_scores(lira_stats, target_scores)

    fpr_all, tpr_all, _ = roc_curve(ground_truth.astype(int), mia_scores)
    tpr_0pct_all  = tpr_at_fpr(fpr_all, tpr_all, max_fpr=0.0)
    tpr_01pct_all = tpr_at_fpr(fpr_all, tpr_all, max_fpr=0.001)
    tpr_1pct_all  = tpr_at_fpr(fpr_all, tpr_all, max_fpr=0.01)
    bal_acc_all   = balanced_accuracy_from_roc(fpr_all, tpr_all)
    print(f"  MIA global: TPR@0%FPR={tpr_0pct_all*100:.2f}%, "
          f"TPR@0.1%FPR={tpr_01pct_all*100:.2f}%, "
          f"TPR@1%FPR={tpr_1pct_all*100:.2f}%, Balanced Acc={bal_acc_all*100:.2f}%")

    # Outlier/variance check
    shadow_matrix = np.stack(lira_stats, axis=0)
    per_point_var = shadow_matrix.var(axis=0)
    top_k_check   = min(20, int(ground_truth.sum()))
    sorted_by_score = np.argsort(mia_scores)[::-1]
    top_k_in_idx = [i for i in sorted_by_score if ground_truth[i] == 1][:top_k_check]
    rest_in_idx  = [i for i in range(len(ground_truth))
                    if ground_truth[i] == 1 and i not in set(top_k_in_idx)]
    if top_k_in_idx and rest_in_idx:
        var_top  = float(per_point_var[top_k_in_idx].mean())
        var_rest = float(per_point_var[rest_in_idx].mean())
        print(
            f"  [Variance check] Top-{top_k_check} IN members: "
            f"mean shadow var={var_top:.4f}  vs  rest of IN: mean shadow var={var_rest:.4f}"
        )
        if var_top > var_rest * 1.5:
            print(
                "    -> Top-ranked IN members have significantly higher shadow variance "
                "(>1.5x rest). Outlier/noise hypothesis confirmed — "
                "these high MIA scores are likely variance artefacts."
            )

    # ------------------------------------------------------------------
    # 5. Influence scores
    # ------------------------------------------------------------------
    print("\nAggregating influence scores across shadows...")
    scores_C_lira = aggregate_influence(C_lira_list)
    has_closs     = all(c is not None for c in C_loss_list)
    scores_C_loss = aggregate_influence(C_loss_list) if has_closs else None

    scores_dict = {"C_lira": scores_C_lira}
    if scores_C_loss is not None:
        scores_dict["C_loss"] = scores_C_loss

    for name, scores in scores_dict.items():
        analyze_score(name, scores, mia_scores, ground_truth, out_dir, num_buckets)

    plot_bucket_tpr_comparison(scores_dict, mia_scores, ground_truth, out_dir, num_buckets)

    # ------------------------------------------------------------------
    # 6. Adaptive bucket-wise GMM aggregation
    # ------------------------------------------------------------------
    print("\nRunning adaptive GMM bucket search...")
    if scores_C_loss is None:
        raise RuntimeError(
            "C_loss is required for GMM bucketing but was not found in shadow artifacts. "
            "Re-run influence computation to produce C_loss.npy for all shadows."
        )

    _all_gmm_results, best_0pct, best_001pct, best_1pct = adaptive_gmm_bucket_search(
        ground_truth=ground_truth,
        mia_scores=mia_scores,
        influence_scores=scores_C_loss,
        min_buckets=min_gmm_buckets,
        max_buckets=max_gmm_buckets,
    )

    print(f"  Evaluated {len(_all_gmm_results)} bucket counts via staged search.")
    evaluated_bucket_counts = [int(r["num_buckets"]) for r in _all_gmm_results]
    print(f"  Evaluated bucket counts: {evaluated_bucket_counts}")
    print("  Per-count staged-search metrics:")
    for res in _all_gmm_results:
        m = res["metrics"]
        print(
            f"    buckets={int(res['num_buckets']):2d}  "
            f"TPR@0%FPR={m['tpr_0pct']*100:6.2f}%  "
            f"TPR@0.1%FPR={m['tpr_01pct']*100:6.2f}%  "
            f"TPR@1%FPR={m['tpr_1pct']*100:6.2f}%  "
            f"Balanced Acc={m['bal_acc']*100:6.2f}%"
        )

    print(
        "  Best loss-bucketing for FPR=0%: "
        f"buckets={best_0pct['num_buckets']}  "
        f"TPR@0%FPR={best_0pct['metrics']['tpr_0pct']*100:.2f}%  "
        f"TPR@0.1%FPR={best_0pct['metrics']['tpr_01pct']*100:.2f}%"
    )
    print(
        "  Best loss-bucketing for FPR=0.1%: "
        f"buckets={best_001pct['num_buckets']}  "
        f"TPR@0%FPR={best_001pct['metrics']['tpr_0pct']*100:.2f}%  "
        f"TPR@0.1%FPR={best_001pct['metrics']['tpr_01pct']*100:.2f}%"
    )
    print(
        "  Best loss-bucketing for FPR=1%: "
        f"buckets={best_1pct['num_buckets']}  "
        f"TPR@1%FPR={best_1pct['metrics']['tpr_1pct']*100:.2f}%"
    )

    gmm_probs_0pct   = best_0pct["gmm_probs"]
    gmm_probs_001pct = best_001pct["gmm_probs"]

    # Precision@K sanity check
    base_rate = float(ground_truth.mean())
    llr_best  = best_0pct["gmm_probs"]
    sorted_by_llr = np.argsort(llr_best)[::-1]
    sorted_labels = ground_truth[sorted_by_llr]
    print("\n  [Precision@K sanity check] (base rate = {:.3f})".format(base_rate))
    for k in [10, 50, 100]:
        if k <= len(sorted_labels):
            prec = float(sorted_labels[:k].mean())
            flag = "  [OK]" if prec > base_rate * 1.5 else (
                "  [WARNING: near base rate — check GMM component labels]"
            )
            print(f"    Precision@{k:3d} = {prec:.3f}{flag}")

    plot_gmm_bucket_components(
        scores_C_loss,
        mia_scores,
        ground_truth,
        out_dir,
        best_0pct["num_buckets"],
    )

    # ------------------------------------------------------------------
    # 7. Save analysis outputs
    # ------------------------------------------------------------------
    save_dict = dict(
        mia_scores=mia_scores,
        gmm_probs_best_fpr0=gmm_probs_0pct,
        gmm_probs_best_fpr01=gmm_probs_001pct,
        ground_truth=ground_truth,
        scores_C_lira=scores_C_lira,
        query_global_indices=query_global_indices,
        best_gmm_fpr0_score=np.array("C_loss"),
        best_gmm_fpr0_buckets=np.array(best_0pct["num_buckets"], dtype=np.int64),
        best_gmm_fpr01_score=np.array("C_loss"),
        best_gmm_fpr01_buckets=np.array(best_001pct["num_buckets"], dtype=np.int64),
        best_gmm_fpr1_score=np.array("C_loss"),
        best_gmm_fpr1_buckets=np.array(best_1pct["num_buckets"], dtype=np.int64),
    )
    save_dict["gmm_probs_best_fpr001"]    = gmm_probs_001pct
    save_dict["best_gmm_fpr001_score"]    = np.array("C_loss")
    save_dict["best_gmm_fpr001_buckets"]  = np.array(best_001pct["num_buckets"], dtype=np.int64)
    if scores_C_loss is not None:
        save_dict["scores_C_loss"] = scores_C_loss

    out_path = os.path.join(out_dir, "influence_vs_mia.npz")
    np.savez_compressed(out_path, **save_dict)
    print(f"\n  Saved analysis data to {out_path}")

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(
        f"  MIA (raw):         TPR@0%FPR={tpr_0pct_all*100:.2f}%  "
        f"TPR@0.1%FPR={tpr_01pct_all*100:.2f}%  "
        f"TPR@1%FPR={tpr_1pct_all*100:.2f}%  "
        f"Balanced Acc={bal_acc_all*100:.2f}%"
    )
    print(
        f"  LLR-GMM best@FPR0:   score=C_loss  buckets={best_0pct['num_buckets']}  "
        f"TPR@0%FPR={best_0pct['metrics']['tpr_0pct']*100:.2f}%  "
        f"TPR@0.1%FPR={best_0pct['metrics']['tpr_01pct']*100:.2f}%  "
        f"TPR@1%FPR={best_0pct['metrics']['tpr_1pct']*100:.2f}%  "
        f"Balanced Acc={best_0pct['metrics']['bal_acc']*100:.2f}%"
    )
    print(
        f"  LLR-GMM best@FPR0.1: score=C_loss  buckets={best_001pct['num_buckets']}  "
        f"TPR@0%FPR={best_001pct['metrics']['tpr_0pct']*100:.2f}%  "
        f"TPR@0.1%FPR={best_001pct['metrics']['tpr_01pct']*100:.2f}%  "
        f"TPR@1%FPR={best_001pct['metrics']['tpr_1pct']*100:.2f}%  "
        f"Balanced Acc={best_001pct['metrics']['bal_acc']*100:.2f}%"
    )
    print(f"{'='*60}\n")


def main():
    cli = _parse_args()
    run(
        exp_dir=cli.exp_dir,
        dataset=cli.dataset,
        num_buckets=cli.num_buckets,
        min_gmm_buckets=cli.min_gmm_buckets,
        max_gmm_buckets=cli.max_gmm_buckets,
    )


if __name__ == "__main__":
    main()
