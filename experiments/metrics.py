"""Pure math/stats: GMM fitting, ROC helpers, MIA score aggregation."""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve


def fit_rightshift_mixture(
    x,
    max_iter=200,
    tol=1e-6,
    min_var=1e-4,
    mu_in_min=0.5,
    min_pi_in=0.02,
    reliability_margin=0.25,
):
    """Fit a constrained 1D mixture:

        p(x) = pi_out * N(x; mu_out, var_out) + pi_in * N(x; mu_in, var_in)

    Both mu_out and mu_in are free parameters updated each M-step.
    mu_in is clamped to >= mu_in_min after each M-step.
    Returns a dict with posteriors, parameters, convergence flag, reliability
    flag, and the posterior-equality threshold.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    mu_out  = float(np.percentile(x, 25))
    mu_in   = max(float(np.percentile(x, 75)), mu_in_min)
    var_out = max(float(np.var(x)) * 0.5, min_var)
    var_in  = max(float(np.var(x)) * 0.5, min_var)
    pi_out  = 0.5
    pi_in   = 0.5

    prev_loglik = -np.inf
    converged   = False

    for _ in range(max_iter):
        log_p_out = np.log(pi_out + 1e-300) + norm.logpdf(x, mu_out, np.sqrt(var_out))
        log_p_in  = np.log(pi_in  + 1e-300) + norm.logpdf(x, mu_in,  np.sqrt(var_in))

        log_sum = np.logaddexp(log_p_out, log_p_in)
        r_out   = np.exp(log_p_out - log_sum)
        r_in    = 1.0 - r_out

        loglik = float(log_sum.sum())

        n_out = r_out.sum()
        n_in  = r_in.sum()
        total = n_out + n_in

        pi_out = n_out / total
        pi_in  = n_in  / total

        mu_out = float(np.dot(r_out, x) / (n_out + 1e-300))
        mu_in_unconstrained = float(np.dot(r_in, x) / (n_in + 1e-300))
        mu_in = max(mu_in_unconstrained, mu_in_min)

        var_out = max(float(np.dot(r_out, (x - mu_out) ** 2) / (n_out + 1e-300)), min_var)
        var_in  = max(float(np.dot(r_in,  (x - mu_in)  ** 2) / (n_in  + 1e-300)), min_var)

        if abs(loglik - prev_loglik) < tol:
            converged = True
            break
        prev_loglik = loglik

    t_lo   = min(mu_out, mu_in)
    t_hi   = max(mu_out, mu_in) * 2.5
    t_grid = np.linspace(t_lo, t_hi, 2000)
    d_out  = pi_out * norm.pdf(t_grid, mu_out, np.sqrt(var_out))
    d_in   = pi_in  * norm.pdf(t_grid, mu_in,  np.sqrt(var_in))
    sign_changes = np.where(np.diff(np.sign(d_in - d_out)))[0]
    threshold = None
    if len(sign_changes) > 0:
        i = sign_changes[0]
        f0 = (d_in - d_out)[i]
        f1 = (d_in - d_out)[i + 1]
        threshold = float(t_grid[i] - f0 * (t_grid[i + 1] - t_grid[i]) / (f1 - f0))

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

    log_p_in_density  = norm.logpdf(x, mu_in,  np.sqrt(var_in))
    log_p_out_density = norm.logpdf(x, mu_out, np.sqrt(var_out))
    llr = log_p_in_density - log_p_out_density

    delta_mu = mu_in - mu_out
    bc_term1 = 0.25 * (delta_mu ** 2) / (var_in + var_out)
    bc_term2 = 0.5 * np.log((var_in + var_out) / (2.0 * np.sqrt(var_in * var_out)))
    bhattacharyya = bc_term1 + bc_term2

    return {
        "pi_out":         pi_out,
        "pi_in":          pi_in,
        "mu_out":         mu_out,
        "mu_in":          mu_in,
        "var_out":        var_out,
        "var_in":         var_in,
        "posterior_in":   r_in,
        "posterior_out":  r_out,
        "llr":            llr,
        "bhattacharyya":  bhattacharyya,
        "loglik":         loglik,
        "converged":      converged,
        "reliable":       reliable,
        "reason":         reason,
        "threshold":      threshold,
    }


def fit_fixed_zero_rightshift_mixture(*args, **kwargs):
    """Deprecated alias — use fit_rightshift_mixture instead."""
    return fit_rightshift_mixture(*args, **kwargs)


def tpr_at_fpr(fpr, tpr, max_fpr=0.01):
    valid = np.where(fpr <= max_fpr)[0]
    return float(tpr[valid[-1]]) if len(valid) > 0 else float("nan")


def balanced_accuracy_from_roc(fpr, tpr):
    return float(((tpr + (1.0 - fpr)) / 2.0).max())


def evaluate_attack_metrics(ground_truth: np.ndarray, scores: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(ground_truth.astype(int), scores)
    return {
        "tpr_0pct":  tpr_at_fpr(fpr, tpr, max_fpr=0.0),
        "tpr_01pct": tpr_at_fpr(fpr, tpr, max_fpr=0.001),
        "tpr_1pct":  tpr_at_fpr(fpr, tpr, max_fpr=0.01),
        "bal_acc":   balanced_accuracy_from_roc(fpr, tpr),
    }


def compute_mia_scores(lira_stats, target_scores: np.ndarray) -> np.ndarray:
    """Per-point expected discrepancy: target_score - mean_shadow_score."""
    shadow_mean = np.stack(lira_stats, axis=0).mean(axis=0)
    return target_scores - shadow_mean


def aggregate_influence(C_list) -> np.ndarray:
    """Mean influence score across all shadows per query point."""
    return np.stack([C_list[k] for k in range(len(C_list))], axis=0).mean(axis=0)


def gmm_aggregate_mia_scores(
    influence_scores: np.ndarray,
    mia_scores: np.ndarray,
    num_buckets: int = 10,
    verbose: bool = True,
) -> np.ndarray:
    """Bucket-wise GMM aggregation using Log-Likelihood Ratios (LLR).

    Returns a global LLR array comparable across all buckets.
    Unreliable buckets produce LLR ≈ 0; tiny buckets fall back to centred ranks.
    """
    N = len(mia_scores)
    llr_scores = np.zeros(N, dtype=float)

    quantiles  = np.quantile(influence_scores, np.linspace(0, 1, num_buckets + 1)[1:-1])
    bucket_ids = np.digitize(influence_scores, quantiles)

    for b in range(num_buckets):
        idx = np.where(bucket_ids == b)[0]
        if len(idx) < 10:
            ranks = np.argsort(np.argsort(mia_scores[idx])).astype(float)
            centred = ranks - (len(idx) - 1) / 2.0
            llr_scores[idx] = centred / max((len(idx) - 1) / 2.0, 1.0)
            continue

        fit = fit_rightshift_mixture(mia_scores[idx])
        if verbose:
            print(
                f"  [Bucket {b}] n={len(idx)}: "
                f"mu_in={fit['mu_in']:.3f}  "
                f"sd_out={np.sqrt(fit['var_out']):.3f}  "
                f"sd_in={np.sqrt(fit['var_in']):.3f}  "
                f"pi_in={fit['pi_in']:.3f}  "
                f"bhattacharyya={fit['bhattacharyya']:.3f}  "
                f"reliable={fit['reliable']}  "
                f"converged={fit['converged']}"
            )
        if not fit["reliable"]:
            if verbose:
                print(f"    -> unreliable: {fit['reason']} — LLR set to 0 (uninformative)")
        else:
            llr_scores[idx] = fit["llr"]

    return llr_scores


def adaptive_gmm_bucket_search(
    ground_truth: np.ndarray,
    mia_scores: np.ndarray,
    influence_scores: np.ndarray,
    min_buckets: int,
    max_buckets: int,
) -> tuple:
    """Find strong loss-bucketing counts without a pure grid sweep.

    Uses staged search: coarse sampling, local refinement around top seeds,
    and boundary probing.

    Returns (results, best_for_0pct, best_for_001pct, best_for_1pct).
    """
    n_query = len(mia_scores)
    max_reasonable = max(2, min(max_buckets, n_query // 10))
    min_reasonable = max(2, min(min_buckets, max_reasonable))

    if min_reasonable > max_reasonable:
        min_reasonable = max_reasonable

    evaluated = {}

    def _eval(num_buckets: int):
        num_buckets = int(num_buckets)
        if num_buckets in evaluated:
            return evaluated[num_buckets]
        gmm_probs = gmm_aggregate_mia_scores(
            influence_scores=influence_scores,
            mia_scores=mia_scores,
            num_buckets=num_buckets,
            verbose=False,
        )
        metrics = evaluate_attack_metrics(ground_truth, gmm_probs)
        result = {"num_buckets": num_buckets, "metrics": metrics, "gmm_probs": gmm_probs}
        evaluated[num_buckets] = result
        return result

    search_span = max_reasonable - min_reasonable + 1

    if search_span <= 8:
        for nb in range(min_reasonable, max_reasonable + 1):
            _eval(nb)
    else:
        coarse_points = np.linspace(min_reasonable, max_reasonable, num=8)
        coarse_candidates = {int(round(v)) for v in coarse_points}
        coarse_candidates.update({min_reasonable, max_reasonable})
        for nb in sorted(coarse_candidates):
            _eval(nb)

        def _key_0pct(res):
            m = res["metrics"]
            return (m["tpr_0pct"], m["tpr_01pct"], m["tpr_1pct"], m["bal_acc"])

        def _key_001pct(res):
            m = res["metrics"]
            return (m["tpr_01pct"], m["tpr_1pct"], m["tpr_0pct"], m["bal_acc"])

        coarse_results = list(evaluated.values())
        seeds = set()
        for res in sorted(coarse_results, key=_key_0pct, reverse=True)[:2]:
            seeds.add(res["num_buckets"])
        for res in sorted(coarse_results, key=_key_001pct, reverse=True)[:2]:
            seeds.add(res["num_buckets"])

        for seed in sorted(seeds):
            for radius in (1, 2, 3):
                for nb in (seed - radius, seed + radius):
                    if min_reasonable <= nb <= max_reasonable:
                        _eval(nb)

        current_best_0   = max(evaluated.values(), key=_key_0pct)
        current_best_001 = max(evaluated.values(), key=_key_001pct)

        def _key_1pct(res):
            m = res["metrics"]
            return (m["tpr_1pct"], m["tpr_01pct"], m["tpr_0pct"], m["bal_acc"])

        current_best_1 = max(evaluated.values(), key=_key_1pct)

        edge_seeds = {
            current_best_0["num_buckets"],
            current_best_001["num_buckets"],
            current_best_1["num_buckets"],
        }
        for seed in edge_seeds:
            if seed <= min_reasonable + 1:
                for nb in range(min_reasonable, min(min_reasonable + 5, max_reasonable + 1)):
                    _eval(nb)
            if seed >= max_reasonable - 1:
                for nb in range(max(min_reasonable, max_reasonable - 4), max_reasonable + 1):
                    _eval(nb)

    all_results = [evaluated[k] for k in sorted(evaluated.keys())]

    def _key_0pct(res):
        m = res["metrics"]
        return (m["tpr_0pct"], m["tpr_01pct"], m["tpr_1pct"], m["bal_acc"])

    def _key_001pct(res):
        m = res["metrics"]
        return (m["tpr_01pct"], m["tpr_1pct"], m["tpr_0pct"], m["bal_acc"])

    def _key_1pct(res):
        m = res["metrics"]
        return (m["tpr_1pct"], m["tpr_01pct"], m["tpr_0pct"], m["bal_acc"])

    best_for_0pct   = max(all_results, key=_key_0pct)
    best_for_001pct = max(all_results, key=_key_001pct)
    best_for_1pct   = max(all_results, key=_key_1pct)

    return all_results, best_for_0pct, best_for_001pct, best_for_1pct
