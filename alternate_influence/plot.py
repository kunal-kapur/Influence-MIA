import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ATTACK_DIR = "/scratch/scholar/kapur16/attacks/run_20260227_111315"

def main():
    probs_json = os.path.join(ATTACK_DIR, "posterior_probs.json")
    attack_data_path = os.path.join(ATTACK_DIR, "attack_data.npz")

    with open(probs_json, "r") as f:
        prob_dict = json.load(f)

    attack_data = np.load(attack_data_path)
    ground_truth = attack_data["ground_truth"]
    query_indices = attack_data["query_indices"]
    label_by_index = {
        int(query_index): int(label)
        for query_index, label in zip(query_indices, ground_truth)
    }

    labels = []
    scores = []
    for k, v in prob_dict.items():
        idx = int(k)
        if idx not in label_by_index:
            continue
        labels.append(label_by_index[idx])
        scores.append(float(v))

    labels = np.array(labels, dtype=np.int8)
    scores = np.array(scores, dtype=np.float32)

    print(f"Using {len(scores)} points with ground-truth membership labels.")

    in_scores = scores[labels == 1]
    out_scores = scores[labels == 0]

    print("\n=== Basic stats ===")
    print(f"Members     : mean={in_scores.mean():.4f}, std={in_scores.std():.4f}")
    print(f"Non-members : mean={out_scores.mean():.4f}, std={out_scores.std():.4f}")

    auc = roc_auc_score(labels, scores)
    print(f"\nAUC-ROC: {auc:.4f}")

    fpr, tpr, thr = roc_curve(labels, scores)

    def report_at(target_fpr):
        idx = (fpr <= target_fpr).nonzero()[0]
        if len(idx) == 0:
            print(f"FPR≈{target_fpr:.4e}: no point on curve <= this FPR")
            return
        j = idx[-1]
        print(f"FPR≈{fpr[j]:.4e}, TPR={tpr[j]:.4f}, thresh={thr[j]:.4f}")

    print("\n=== TPR at different FPRs ===")
    for tf in [1e-1, 1e-2, 1e-3]:
        report_at(tf)

if __name__ == "__main__":
    main()
