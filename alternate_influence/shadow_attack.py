import os
import json
import gc
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import numpy as np
from scipy.stats import norm
import torchvision
from global_variables import DATA_DIR, TRANSFORM_TRAIN, TRANSFORM_TEST
import torchvision.datasets as datasets
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from resnet_influence import ResNet18_Influence
from train_utils import TrainingConfig, train_shadow_model

def evaluate_target_model(model_path, query_indices, device):
    print(f"Loading Target Model from {model_path}...")
    
    model = ResNet18_Influence()
    
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
        
    model.to(device)
    model.eval()
    

    full_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=TRANSFORM_TRAIN)
    
    query_dataset = Subset(full_dataset, query_indices)
    dataloader = DataLoader(query_dataset, batch_size=128, shuffle=False)
    
    target_scores = []
    
    print("Evaluating Query Points on Target Model...")
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            
            # Extract true class probability
            p_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            
            # CLAMP to prevent inf / -inf
            p_true = torch.clamp(p_true, min=1e-7, max=1.0 - 1e-7)
            
            # Compute LiRA score exactly as done in shadow models
            scores = torch.log(p_true / (1.0 - p_true)).cpu().numpy()
            target_scores.extend(scores)
            
    return np.array(target_scores)



def train_single_shadow_model(subset_indices, device, training_config, checkpoint_path=None):
    """Physically trains ONE ResNet on the specified subset of CIFAR-10.
    
    Args:
        subset_indices: List of training sample indices
        device: torch device
        training_config: TrainingConfig object from target model metadata
        checkpoint_path: Optional path to save/resume checkpoint
    
    Returns:
        Trained model in eval mode
    """
    model = ResNet18_Influence().to(device)
    trained_model = train_shadow_model(model, subset_indices, training_config, device, checkpoint_path)
    return trained_model


def _shadow_cache_metadata_path(shadow_models_dir):
    return os.path.join(shadow_models_dir, 'cache_metadata.npz')


def _normalized_confident_memberships(confident_memberships, num_queries):
    if confident_memberships is None:
        return np.full(num_queries, -1, dtype=np.int8)
    return np.asarray(confident_memberships, dtype=np.int8)


def _shadow_cache_matches(metadata_path, query_indices, confident_memberships, num_models, target_train_size):
    if not os.path.exists(metadata_path):
        return False, 'cache metadata is missing'

    cache_metadata = np.load(metadata_path)
    cached_query_indices = cache_metadata['query_indices']
    cached_confident_memberships = cache_metadata['confident_memberships']
    cached_num_models = int(cache_metadata['num_models'][0])
    cached_target_train_size = int(cache_metadata['target_train_size'][0])

    expected_query_indices = np.asarray(query_indices, dtype=np.int64)
    expected_confident_memberships = _normalized_confident_memberships(
        confident_memberships,
        len(query_indices)
    )

    if cached_num_models != num_models:
        return False, f'cached num_models={cached_num_models} does not match requested {num_models}'
    if cached_target_train_size != target_train_size:
        return False, (
            f'cached target_train_size={cached_target_train_size} does not match '
            f'requested {target_train_size}'
        )
    if not np.array_equal(cached_query_indices, expected_query_indices):
        return False, 'cached query indices do not match the current attack run'
    if not np.array_equal(cached_confident_memberships, expected_confident_memberships):
        return False, 'cached anchors do not match the current run'

    return True, None


def _write_shadow_cache_metadata(shadow_models_dir, query_indices, confident_memberships, num_models, target_train_size):
    np.savez(
        _shadow_cache_metadata_path(shadow_models_dir),
        query_indices=np.asarray(query_indices, dtype=np.int64),
        confident_memberships=_normalized_confident_memberships(
            confident_memberships,
            len(query_indices)
        ),
        num_models=np.array([num_models], dtype=np.int64),
        target_train_size=np.array([target_train_size], dtype=np.int64),
    )

def train_shadow_models(query_indices, num_models, total_dataset_size, target_train_size, 
                       training_config, confident_memberships=None, shadow_models_dir=None):
    """
    Train shadow models using the same configuration as the target model.
    Supports resuming from partially completed training runs.
    
    Args:
        query_indices: Indices of query points
        num_models: Number of shadow models to train
        total_dataset_size: Total size of CIFAR-10 training set
        target_train_size: Size of target model's training set
        training_config: TrainingConfig object from target model metadata
        confident_memberships: Optional array of anchored memberships (-1 for unknown, 0/1 for known)
        shadow_models_dir: Directory to save shadow models (enables checkpointing if provided)
    
    Returns:
        Tuple of (shadow_models, shadow_datasets_m, shadow_subsets, any_new_models_trained)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shadow_models = []
    shadow_datasets_m = [] 
    shadow_subsets = []
    any_new_models_trained = False
    
    all_indices = set(range(total_dataset_size))
    background_pool = list(all_indices - set(query_indices))
    
    # Check if we're resuming from a previous run
    membership_matrix_path = None
    cache_metadata_path = None
    allow_cache_reuse = True
    if shadow_models_dir is not None:
        os.makedirs(shadow_models_dir, exist_ok=True)
        membership_matrix_path = os.path.join(shadow_models_dir, 'membership_matrix.npy')
        cache_metadata_path = _shadow_cache_metadata_path(shadow_models_dir)

        shadow_model_paths = [
            os.path.join(shadow_models_dir, f"shadow_{k}.pth")
            for k in range(num_models)
        ]
        has_existing_cache = os.path.exists(membership_matrix_path) or any(
            os.path.exists(path) for path in shadow_model_paths
        )
        if has_existing_cache:
            cache_matches, mismatch_reason = _shadow_cache_matches(
                cache_metadata_path,
                query_indices,
                confident_memberships,
                num_models,
                target_train_size,
            )
            if not cache_matches:
                print(
                    f"  -> Ignoring existing shadow cache in {shadow_models_dir}: {mismatch_reason}."
                )
                allow_cache_reuse = False

        _write_shadow_cache_metadata(
            shadow_models_dir,
            query_indices,
            confident_memberships,
            num_models,
            target_train_size,
        )
    
    # Load or create membership matrix
    if allow_cache_reuse and membership_matrix_path and os.path.exists(membership_matrix_path):
        print(f"  -> Loading existing membership matrix from {membership_matrix_path}")
        existing_matrix = np.load(membership_matrix_path)
        existing_num_models = existing_matrix.shape[0]
        
        if existing_num_models == num_models:
            # Perfect match, use as is
            membership_matrix = existing_matrix
        elif existing_num_models < num_models:
            # Need to extend the matrix with additional models
            print(f"  -> Extending membership matrix from {existing_num_models} to {num_models} models")
            membership_matrix = np.zeros((num_models, len(query_indices)), dtype=int)
            membership_matrix[:existing_num_models, :] = existing_matrix
            
            # For each query point, compute how many additional IN assignments are needed
            # to maintain ~50% IN across all num_models
            for i, idx in enumerate(query_indices):
                if confident_memberships is not None and confident_memberships[i] != -1:
                    # Anchored point - set all new models to the anchored value
                    membership_matrix[existing_num_models:, i] = confident_memberships[i]
                else:
                    # Count how many existing models have this query point as IN
                    existing_in_count = np.sum(existing_matrix[:, i])
                    # Target is roughly half of all models
                    target_in_count = num_models // 2
                    # How many more models should have this as IN?
                    additional_in_needed = max(0, target_in_count - existing_in_count)
                    
                    # Randomly select from the new models to add IN
                    num_new_models = num_models - existing_num_models
                    additional_in_needed = min(additional_in_needed, num_new_models)
                    
                    if additional_in_needed > 0:
                        new_models_with_in = np.random.choice(
                            num_new_models, 
                            size=additional_in_needed, 
                            replace=False
                        ) + existing_num_models
                        membership_matrix[new_models_with_in, i] = 1
            
            # Save the extended matrix
            np.save(membership_matrix_path, membership_matrix)
            print(f"  -> Saved extended membership matrix to {membership_matrix_path}")
        else:
            # More models in file than requested - use first num_models
            print(f"  -> WARNING: Existing matrix has {existing_num_models} models but only {num_models} requested")
            print(f"  -> Using first {num_models} models from existing matrix")
            membership_matrix = existing_matrix[:num_models, :]
    else:
        # Pre-compute membership matrix to ensure proper distribution
        # Each query point should be IN for exactly half the models
        membership_matrix = np.zeros((num_models, len(query_indices)), dtype=int)
        
        for i, idx in enumerate(query_indices):
            if confident_memberships is not None and confident_memberships[i] != -1:
                # If this query point is anchored, all models must respect it
                membership_matrix[:, i] = confident_memberships[i]
            else:
                # Randomly select half the models to have this query point as IN
                models_with_in = np.random.choice(num_models, size=num_models // 2, replace=False)
                membership_matrix[models_with_in, i] = 1
        
        # Save membership matrix for resumability
        if membership_matrix_path:
            np.save(membership_matrix_path, membership_matrix)
            print(f"  -> Saved membership matrix to {membership_matrix_path}")
    
    # Now train each shadow model based on the pre-computed membership matrix
    for k in range(num_models):
        # Check if this shadow model already exists
        if shadow_models_dir is not None and allow_cache_reuse:
            shadow_model_path = os.path.join(shadow_models_dir, f"shadow_{k}.pth")
            if os.path.exists(shadow_model_path):
                print(f"  -> Loading existing Shadow Model {k+1}/{num_models} from {shadow_model_path}")
                # Load the model
                model = ResNet18_Influence()
                checkpoint = torch.load(shadow_model_path, map_location='cpu', weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                shadow_models.append(model)
                shadow_datasets_m.append(checkpoint['m_k'])
                shadow_subsets.append(checkpoint['subset_indices'])
                continue
        
        print(f"  -> Training Shadow Model {k+1}/{num_models}...")
        any_new_models_trained = True
        
        m_k = membership_matrix[k]
        subset_indices = []
        
        # Add query points that are IN for this model
        for i, idx in enumerate(query_indices):
            if m_k[i] == 1:
                subset_indices.append(int(idx))
                
        num_background_needed = target_train_size - len(subset_indices)
        
        # Deterministic seed per model to ensure reproducibility
        # This ensures model k always gets the same background samples
        rng = np.random.RandomState(seed=1000 + k)
        background_sample = rng.choice(background_pool, num_background_needed, replace=False)
        subset_indices.extend(background_sample.tolist())
        
        shadow_subsets.append(subset_indices)
        
        # Train with checkpointing support
        checkpoint_path = None
        if shadow_models_dir is not None:
            checkpoint_path = os.path.join(shadow_models_dir, f"shadow_{k}.pth")
        
        trained_model = train_single_shadow_model(subset_indices, device, training_config, checkpoint_path)
        
        shadow_models.append(trained_model.cpu()) # this makes copy of model in CPU
        shadow_datasets_m.append(m_k)
        
        # Save the completed shadow model immediately
        if shadow_models_dir is not None:
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'm_k': m_k,
                'subset_indices': subset_indices
            }, checkpoint_path)
            print(f"     Saved shadow model {k+1} to {checkpoint_path}")
        
        del trained_model 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return shadow_models, np.array(shadow_datasets_m), shadow_subsets, any_new_models_trained


def precompute_influence_matrices(shadow_models, shadow_subsets, query_indices, device,
                                  hessian_sample_size=3000, save_dir=None):
    """
    Calculate Hessian and both Influence matrices (LiRA-vs-loss and loss-vs-loss).

    Matrices are written to disk one at a time (save_dir/C_k.npy, C_loss_k.npy) to
    avoid accumulating [K, N, N] arrays in RAM.  Only t_bases (tiny) is kept in memory.

    Returns t_bases as [K, N] numpy array.  C_matrices and C_loss_matrices are NOT
    returned — callers should load them from disk via save_dir.
    """
    t_bases = []

    full_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=TRANSFORM_TEST)

    for k, model in enumerate(shadow_models):
        print(f"  -> Precomputing Influence for Shadow Model {k+1}/{len(shadow_models)}...")
        model = model.to(device)

        train_indices = shadow_subsets[k]
        sample_size = min(hessian_sample_size, len(train_indices))
        hessian_indices = np.random.choice(train_indices, sample_size, replace=False)
        hessian_loader = DataLoader(
            Subset(full_dataset, hessian_indices), batch_size=256, shuffle=False, num_workers=0
        )

        H = model.compute_last_layer_hessian(hessian_loader, device)
        H_inv = torch.linalg.inv(H)

        query_loader = DataLoader(
            Subset(full_dataset, [int(idx) for idx in query_indices]), batch_size=256, shuffle=False
        )

        train_size = len(train_indices)
        C, C_loss = model.compute_influence_matrices(query_loader, H_inv, train_size, device)

        # Write each matrix to disk immediately — never accumulate [K, N, N] in RAM
        if save_dir is not None:
            np.save(os.path.join(save_dir, f"C_{k}.npy"),      C.cpu().numpy())
            np.save(os.path.join(save_dir, f"C_loss_{k}.npy"), C_loss.cpu().numpy())

        t_bases.append(model.get_lira_statistics(query_loader, device).cpu().numpy())

        model = model.cpu()
        del H, H_inv, C, C_loss
        torch.cuda.empty_cache()

    print("done precomputing influence matrices.")
    return np.array(t_bases)  # [K, N]


def _stack_influence_matrices_to_npz(save_dir, num_models, influence_cache_path, t_bases):
    """
    Memory-map individual C_k.npy / C_loss_k.npy files and write them into a single
    compressed npz without ever holding more than one matrix in RAM at a time.
    """
    print("  Stacking influence matrices into compressed cache...")
    # Peek at shape from first file
    c0 = np.load(os.path.join(save_dir, "C_0.npy"))
    N = c0.shape[0]

    C_stack      = np.empty((num_models, N, N), dtype=np.float32)
    C_loss_stack = np.empty((num_models, N, N), dtype=np.float32)

    for k in range(num_models):
        C_stack[k]      = np.load(os.path.join(save_dir, f"C_{k}.npy"))
        C_loss_stack[k] = np.load(os.path.join(save_dir, f"C_loss_{k}.npy"))

    np.savez_compressed(
        influence_cache_path,
        C_matrices=C_stack,
        C_loss_matrices=C_loss_stack,
        t_bases=t_bases,
    )
    del C_stack, C_loss_stack

    # Clean up per-model shard files
    for k in range(num_models):
        os.remove(os.path.join(save_dir, f"C_{k}.npy"))
        os.remove(os.path.join(save_dir, f"C_loss_{k}.npy"))
    print(f"  Saved influence cache to {influence_cache_path}")

def plot_calibration(scores_dict, ground_truth, attack_dir):
    """
    Plot marginal calibration and ROC curves for one or more score sets.

    scores_dict: {label: 1-D array of scores (higher = more likely member)}
    Saves two PNG files to attack_dir:
      - calibration_histogram.png  (score distribution, members vs non-members)
      - roc_curve.png              (ROC with low-FPR inset)
    """
    members = ground_truth == 1
    non_members = ~members

    # --- Histogram ---
    fig, axes = plt.subplots(1, len(scores_dict), figsize=(5 * len(scores_dict), 4), squeeze=False)
    for ax, (label, scores) in zip(axes[0], scores_dict.items()):
        ax.hist(scores[members],     bins=60, alpha=0.6, density=True, label='Member',     color='steelblue')
        ax.hist(scores[non_members], bins=60, alpha=0.6, density=True, label='Non-Member', color='salmon')
        ax.set_title(f'{label} score distribution')
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(attack_dir, 'calibration_histogram.png'), dpi=120)
    plt.close(fig)

    # --- ROC curve ---
    fig, (ax_full, ax_low) = plt.subplots(1, 2, figsize=(12, 5))
    for label, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(ground_truth, scores)
        ax_full.plot(fpr, tpr, label=label)
        # Low-FPR inset (0–2%)
        mask = fpr <= 0.02
        ax_low.plot(fpr[mask], tpr[mask], label=label)

    for ax, title in [(ax_full, 'Full ROC'), (ax_low, 'ROC — FPR ≤ 2%')]:
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(title)
        ax.legend()
        ax.axvline(0.001, color='gray', linestyle=':', linewidth=0.8, label='0.1% FPR')
        ax.axvline(0.01,  color='gray', linestyle='--', linewidth=0.8, label='1% FPR')

    fig.tight_layout()
    fig.savefig(os.path.join(attack_dir, 'roc_curve.png'), dpi=120)
    plt.close(fig)
    print(f"Saved calibration plots to {attack_dir}/")


def _balanced_accuracy_from_roc(fpr_curve, tpr_curve):
    """Return the maximum balanced accuracy achievable across all ROC thresholds."""
    ba_values = (tpr_curve + (1.0 - fpr_curve)) / 2.0
    return float(np.max(ba_values))


def run_lira_baseline(target_scores, t_bases, m_actual, ground_truth, attack_dir):
    """
    Standard LiRA (Likelihood Ratio Attack) baseline — no MCMC, no influence functions.

    For each query point i:
      - Collect shadow logit scores where point was IN vs OUT
      - Fit Gaussian (mean, std) to each set
      - Score = log p(target_score | IN Gaussian) - log p(target_score | OUT Gaussian)
    A higher score means the model thinks the point is a member.

    Args:
        target_scores: Logit scores from the target model, shape (N,)
        t_bases:       Shadow model logit scores, shape (K, N)
        m_actual:      Membership matrix, shape (K, N), values 0 or 1
        ground_truth:  Ground truth membership, shape (N,)
        attack_dir:    Directory to save results

    Returns:
        lira_scores:    Per-point LiRA log-likelihood ratio scores, shape (N,)
        tpr_at_01pct:   TPR at 0.1% FPR
        tpr_at_1pct:    TPR at 1% FPR
        balanced_acc:   Max balanced accuracy across all thresholds
    """
    N = len(target_scores)
    K = len(t_bases)
    global_std = np.std(t_bases, ddof=1) + 1e-8

    lira_scores = np.zeros(N, dtype=np.float64)

    for i in range(N):
        in_scores  = t_bases[m_actual[:, i] == 1, i]
        out_scores = t_bases[m_actual[:, i] == 0, i]

        mu_in  = np.mean(in_scores)  if len(in_scores)  > 0 else 0.0
        mu_out = np.mean(out_scores) if len(out_scores) > 0 else 0.0
        std_in  = np.std(in_scores,  ddof=1) if len(in_scores)  > 1 else global_std
        std_out = np.std(out_scores, ddof=1) if len(out_scores) > 1 else global_std
        std_in  = std_in  if std_in  > 0 else global_std
        std_out = std_out if std_out > 0 else global_std

        log_p_in  = norm.logpdf(target_scores[i], loc=mu_in,  scale=std_in)
        log_p_out = norm.logpdf(target_scores[i], loc=mu_out, scale=std_out)
        lira_scores[i] = log_p_in - log_p_out

    # Convert log-LR to a [0, 1] posterior-like probability via sigmoid for comparability
    lira_probs = 1.0 / (1.0 + np.exp(-lira_scores))

    # Save results
    lira_results = {str(int(i)): float(lira_probs[i]) for i in range(N)}
    with open(os.path.join(attack_dir, "lira_baseline_probs.json"), "w") as f:
        json.dump(lira_results, f, indent=2)
    np.save(os.path.join(attack_dir, "lira_baseline_scores.npy"), lira_scores)

    # Metrics
    fpr_curve, tpr_curve, _ = roc_curve(ground_truth, lira_probs)
    valid_01 = np.where(fpr_curve <= 0.001)[0]
    tpr_at_01pct = tpr_curve[valid_01[-1]] if len(valid_01) > 0 else float('nan')
    valid_1 = np.where(fpr_curve <= 0.01)[0]
    tpr_at_1pct = tpr_curve[valid_1[-1]] if len(valid_1) > 0 else float('nan')
    balanced_acc = _balanced_accuracy_from_roc(fpr_curve, tpr_curve)

    return lira_scores, tpr_at_01pct, tpr_at_1pct, balanced_acc

def compute_grad_norms_last_layer(model_path, query_indices, device):
    """
    Compute per-query L2 norm of the gradient of cross-entropy loss
    w.r.t. the last linear layer of the target model.
    Returns an array of shape [N] in the order of query_indices.
    """
    from torchvision import datasets

    model = ResNet18_Influence().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    full_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=TRANSFORM_TEST)
    query_subset = Subset(full_dataset, [int(idx) for idx in query_indices])
    loader = DataLoader(query_subset, batch_size=64, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    all_scores = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses = criterion(logits, y)  # [B]

        for i in range(x.size(0)):
            model.zero_grad(set_to_none=True)
            losses[i].backward(retain_graph=True)

            grads = []
            for p in model.linear.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().reshape(-1))
            if grads:
                g_flat = torch.cat(grads)
                all_scores.append(g_flat.norm().item())
            else:
                all_scores.append(0.0)

    return np.array(all_scores, dtype=np.float64)






def save_attack_inputs(attack_dir, query_indices, ground_truth):
    """Save the query set and labels needed for offline evaluation."""
    np.savez(
        os.path.join(attack_dir, 'attack_data.npz'),
        query_indices=np.asarray(query_indices, dtype=np.int64),
        ground_truth=np.asarray(ground_truth, dtype=np.int8),
    )
    print(f"Saved attack inputs to {attack_dir}")


def load_attack_inputs(attack_dir):
    """Load query indices and ground-truth labels from a previous run."""
    attack_data_path = os.path.join(attack_dir, 'attack_data.npz')
    if not os.path.exists(attack_data_path):
        raise FileNotFoundError(
            f"Missing attack data file: {attack_data_path}. "
            "Run without --reuse-attack-run once to build shadow models/influence matrices first."
        )

    attack_data = np.load(attack_data_path)
    query_indices = attack_data['query_indices']
    ground_truth = attack_data['ground_truth']
    return query_indices, ground_truth

def setup_attack_directory(base_dir="attacks", model_name=None):
    """Creates or resumes an attack directory named after the target model."""
    attack_dir = os.path.join(base_dir, model_name)
    os.makedirs(attack_dir, exist_ok=True)
    return attack_dir

def load_target_metadata(meta_path):
    """Loads the exact indices the target model was trained on."""
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return set(meta['train_indices'])

def setup_query_points(meta_path, num_queries=1000, total_dataset_size=50000, member_percentage=0.5):
    """
    Randomly selects a set of query points to attack.
    Args:
        member_percentage: Percentage of query points that should be true members (default: 0.5 for 50%)
    Returns:
        query_indices: The absolute indices in the CIFAR-10 dataset to attack.
        ground_truth: Array of 1s (Member) and 0s (Non-Member) for evaluation.
    """
    # Figure out what the target model actually saw
    true_members = load_target_metadata(meta_path)
    all_indices = set(range(total_dataset_size))
    true_non_members = all_indices - true_members
    
    # Calculate number of members and non-members based on percentage
    num_members = int(num_queries * member_percentage)
    num_non_members = num_queries - num_members
    
    sampled_members = np.random.choice(list(true_members), num_members, replace=False)
    sampled_non_members = np.random.choice(list(true_non_members), num_non_members, replace=False)
    
    query_indices = np.concatenate([sampled_members, sampled_non_members])
    np.random.shuffle(query_indices)
    
    ground_truth = np.array([1 if idx in true_members else 0 for idx in query_indices])
    return query_indices, ground_truth


def per_point_influence_norms(model, dataloader, criterion, device):
    model.eval()
    norms = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, feats = model(x, return_features=True)
            loss = criterion(logits, y)
            # gradient wrt last layer weights (per batch)
            grads = torch.autograd.grad(loss, model.linear.parameters(), retain_graph=False, create_graph=False)
            g_flat = torch.cat([g.reshape(g.size(0), -1) for g in grads], dim=1)  # [B, P]
            batch_norms = g_flat.norm(dim=1).cpu().numpy()
            norms.extend(batch_norms)
    return np.array(norms)

def plot_bucket_lira_hist(
    lira_scores: np.ndarray,
    ground_truth: np.ndarray,
    bucket_indices: np.ndarray,
    bucket_id: int,
    attack_dir: str,
    title_prefix: str = "",
) -> None:
    """
    Plot histogram of LiRA log-LR scores in a given bucket,
    split by members vs non-members.
    """
    bucket_scores = lira_scores[bucket_indices]
    bucket_gt     = ground_truth[bucket_indices]

    in_vals  = bucket_scores[bucket_gt == 1]
    out_vals = bucket_scores[bucket_gt == 0]

    plt.figure(figsize=(6, 4))
    all_vals = np.concatenate([in_vals, out_vals])
    bins = np.linspace(all_vals.min(), all_vals.max(), 41) if len(all_vals) > 1 else 40

    if len(out_vals) > 0:
        plt.hist(out_vals, bins=bins, alpha=0.6, label="Non-members",
                 color="tab:blue", density=True)
    if len(in_vals) > 0:
        plt.hist(in_vals, bins=bins, alpha=0.6, label="Members",
                 color="tab:orange", density=True)

    plt.xlabel("LiRA log-likelihood ratio")
    plt.ylabel("Density")
    prefix = f"{title_prefix} " if title_prefix else ""
    plt.title(f"{prefix}Bucket {bucket_id} (n={len(bucket_indices)})")
    plt.legend()
    plt.tight_layout()

    plots_dir = os.path.join(attack_dir, "bucket_plots")
    os.makedirs(plots_dir, exist_ok=True)
    safe_prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in title_prefix.lower())
    out_path = os.path.join(plots_dir, f"{safe_prefix}_bucket_{bucket_id}_lira_hist.png")
    plt.savefig(out_path)
    plt.close()


def _plot_bucket_tpr_comparison(
    scores_dict: dict,
    lira_scores: np.ndarray,
    ground_truth: np.ndarray,
    attack_dir: str,
    num_buckets: int = 5,
) -> None:
    """
    Bar chart comparing TPR@1%FPR across score-quantile buckets for multiple score types.
    One grouped bar per bucket, one bar-group colour per score type.
    """
    bucket_labels = [f"Q{b}" for b in range(num_buckets)]
    x = np.arange(num_buckets)
    width = 0.8 / max(len(scores_dict), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (score_name, scores) in enumerate(scores_dict.items()):
        quantiles = np.quantile(scores, np.linspace(0, 1, num_buckets + 1)[1:-1])
        bucket_ids = np.digitize(scores, quantiles)
        tprs = []
        for b in range(num_buckets):
            idx = np.where(bucket_ids == b)[0]
            if len(idx) < 10:
                tprs.append(float("nan"))
                continue
            fpr, tpr, _ = roc_curve(ground_truth[idx], lira_scores[idx])
            valid = np.where(fpr <= 0.01)[0]
            tprs.append(tpr[valid[-1]] * 100 if len(valid) > 0 else float("nan"))
        offset = (i - (len(scores_dict) - 1) / 2) * width
        ax.bar(x + offset, tprs, width, label=score_name)

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel("Score quintile bucket")
    ax.set_ylabel("TPR @ 1% FPR (%)")
    ax.set_title("LiRA TPR@1%FPR by influence-score quintile")
    ax.legend()
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Random (1%)")
    fig.tight_layout()
    out_path = os.path.join(attack_dir, "bucket_tpr_comparison.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved bucket TPR comparison plot to {out_path}")


def analyze_influence_vs_lira(
    attack_dir: str,
    lira_scores: np.ndarray,
    ground_truth: np.ndarray,
    target_model_path: str,
    device,
):
    print("\n[Analysis] Influence vs LiRA vulnerability")

    precomputed_dir = os.path.join(attack_dir, "precomputed_matrices")
    influence_cache_path = os.path.join(precomputed_dir, "influence_data.npz")
    attack_data_path = os.path.join(attack_dir, "attack_data.npz")

    if not (os.path.exists(influence_cache_path) and os.path.exists(attack_data_path)):
        print("  Missing influence or attack artifacts; skipping analysis.")
        return

    influence_data = np.load(influence_cache_path)
    C_matrices = influence_data["C_matrices"]          # [K, N, N]
    C_loss_matrices = influence_data["C_loss_matrices"] if "C_loss_matrices" in influence_data else None

    attack_data = np.load(attack_data_path)
    query_indices = attack_data["query_indices"]
    ground_truth = attack_data["ground_truth"]

    # --- Score: LiRA-vs-loss column norm ---
    C_mean_lira = C_matrices.mean(axis=0)
    scores_C_lira = np.linalg.norm(C_mean_lira, axis=0)
    scores_C_lira = (scores_C_lira - scores_C_lira.mean()) / (scores_C_lira.std() + 1e-8)

    # --- Score: loss-vs-loss column norm ---
    if C_loss_matrices is not None:
        C_mean_loss = C_loss_matrices.mean(axis=0)
        scores_C_loss = np.linalg.norm(C_mean_loss, axis=0)
        scores_C_loss = (scores_C_loss - scores_C_loss.mean()) / (scores_C_loss.std() + 1e-8)
    else:
        print("  WARNING: C_loss_matrices not found in cache; loss-only score unavailable.")
        scores_C_loss = None

    # --- Score: gradient norm on target model ---
    print("  Computing gradient-norm scores on target model...")
    scores_grad = compute_grad_norms_last_layer(target_model_path, query_indices, device)
    scores_grad = (scores_grad - scores_grad.mean()) / (scores_grad.std() + 1e-8)

    scores_dict = {"C-LiRA (lira-vs-loss)": scores_C_lira, "Grad-norm last layer": scores_grad}
    if scores_C_loss is not None:
        scores_dict["C-Loss (loss-vs-loss)"] = scores_C_loss

    def analyze_score(score_name: str, scores: np.ndarray) -> None:
        corr = np.corrcoef(scores, lira_scores)[0, 1]
        print(f"\n  [{score_name}] Pearson corr(score, LiRA log-LR): {corr:.3f}")

        num_buckets = 5
        quantiles = np.quantile(scores, np.linspace(0, 1, num_buckets + 1)[1:-1])
        bucket_ids = np.digitize(scores, quantiles)

        print(f"  [{score_name}] Bucketed TPR@1%FPR and Balanced Accuracy by quintile:")
        for b in range(num_buckets):
            idx = np.where(bucket_ids == b)[0]
            if len(idx) < 10:
                print(f"    Bucket {b}: too few points ({len(idx)}), skipping")
                continue
            fpr, tpr, _ = roc_curve(ground_truth[idx], lira_scores[idx])
            valid = np.where(fpr <= 0.01)[0]
            tpr_1pct = tpr[valid[-1]] if len(valid) > 0 else float("nan")
            bal_acc = _balanced_accuracy_from_roc(fpr, tpr)
            print(f"    Bucket {b}: size={len(idx)}, TPR@1%FPR={tpr_1pct * 100:.2f}%, Balanced Acc={bal_acc * 100:.2f}%")
            plot_bucket_lira_hist(
                lira_scores=lira_scores,
                ground_truth=ground_truth,
                bucket_indices=idx,
                bucket_id=b,
                attack_dir=attack_dir,
                title_prefix=score_name,
            )

    for name, scores in scores_dict.items():
        analyze_score(name, scores)

    _plot_bucket_tpr_comparison(scores_dict, lira_scores, ground_truth, attack_dir)

    save_dict = dict(
        scores_C_lira=scores_C_lira,
        scores_grad=scores_grad,
        lira_scores=lira_scores,
        ground_truth=ground_truth,
        query_indices=query_indices,
    )
    if scores_C_loss is not None:
        save_dict["scores_C_loss"] = scores_C_loss

    out_path = os.path.join(attack_dir, "influence_vs_lira_multi.npz")
    np.savez_compressed(out_path, **save_dict)
    print(f"\n  Saved extended influence analysis data to {out_path}")

def main():
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description='LiRA-based Membership Inference Attack')
    parser.add_argument('--num_queries', type=int, default=1000)
    parser.add_argument('--num_shadow_models', type=int, default=16)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--use-final-model', action='store_true')
    parser.add_argument('--member-percentage', type=float, default=0.5)
    parser.add_argument('--shadow-epochs', type=int, default=None)
    parser.add_argument('--attack-dir', type=str, default="attacks")
    parser.add_argument('--reuse-attack-run', type=str, default=None)
    parser.add_argument('--analyze-influence', action='store_true',
                    help='Run influence vs LiRA vulnerability analysis')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Set up attack directory and load metadata (same as before)
    reuse_mode = args.reuse_attack_run is not None
    checkpoint_dir = args.checkpoint_dir
    model_name = os.path.basename(checkpoint_dir.rstrip("/"))

    if reuse_mode:
        attack_dir = args.reuse_attack_run
        if not os.path.exists(attack_dir):
            raise FileNotFoundError(f"Reuse run directory not found: {attack_dir}")
    else:
        attack_dir = setup_attack_directory(base_dir=args.attack_dir, model_name=model_name)

    meta_path = os.path.join(checkpoint_dir, "training_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    model_suffix = "final" if args.use_final_model else "best"
    model_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(f"_{model_suffix}.pth")]
    if not model_files:
        raise FileNotFoundError(f"No model file found with suffix '{model_suffix}' in {checkpoint_dir}")
    target_model_path = os.path.join(checkpoint_dir, model_files[0])

    print(f"Loading target model from: {target_model_path}")
    print(f"Loading metadata from: {meta_path}")

    # 3. Query selection and attack inputs
    attack_data_path = os.path.join(attack_dir, 'attack_data.npz')
    if reuse_mode or os.path.exists(attack_data_path):
        query_indices, ground_truth = load_attack_inputs(attack_dir)
        NUM_QUERIES = len(query_indices)
        if not reuse_mode:
            print(f"Resuming existing run — loaded {NUM_QUERIES} query points from {attack_data_path}")
    else:
        NUM_QUERIES = args.num_queries
        query_indices, ground_truth = setup_query_points(
            meta_path,
            num_queries=NUM_QUERIES,
            member_percentage=args.member_percentage,
        )
        save_attack_inputs(attack_dir, query_indices, ground_truth)

    # 4. Load training config, train/reuse shadow models, precompute influence matrices
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    TOTAL_DATASET_SIZE = meta['total_cifar10_train_size']
    TARGET_TRAIN_SIZE = meta['num_samples_used']

    print(f"\nTarget model was trained on {TARGET_TRAIN_SIZE} samples out of {TOTAL_DATASET_SIZE}.")
    print(f"Starting attack on {len(query_indices)} points.")
    print(f"Number of true members in query set: {np.sum(ground_truth)}")

    shadow_models_dir = os.path.join(attack_dir, "shadow_models")
    precomputed_dir = os.path.join(attack_dir, "precomputed_matrices")
    os.makedirs(shadow_models_dir, exist_ok=True)
    os.makedirs(precomputed_dir, exist_ok=True)

    influence_cache_path = os.path.join(precomputed_dir, "influence_data.npz")
    m_actual_path = os.path.join(shadow_models_dir, "m_actual.npy")

    if reuse_mode:
        if not os.path.exists(influence_cache_path):
            raise FileNotFoundError(
                f"Missing influence cache at {influence_cache_path}. "
                "Run without --reuse-attack-run once to build shadow models/influence matrices first."
            )
        if not os.path.exists(m_actual_path):
            raise FileNotFoundError(
                f"Missing membership matrix at {m_actual_path}. "
                "Run without --reuse-attack-run once to build shadow models first."
            )

        print(f"Reusing cached shadow artifacts from {attack_dir}...")
        influence_data = np.load(influence_cache_path)
        C_matrices = influence_data['C_matrices']   # still available for later analysis
        t_bases = influence_data['t_bases']
        m_actual = np.load(m_actual_path)
    else:
        print(f"\nLoading training configuration from metadata...")
        training_config = TrainingConfig.from_metadata(meta_path)
        if args.shadow_epochs is not None:
            print(f"  Overriding epochs: {training_config.epochs} -> {args.shadow_epochs}")
            training_config.epochs = args.shadow_epochs

        print(f"  Shadow models will use:")
        print(f"    - Optimizer: {training_config.optimizer_type.upper()}")
        print(f"    - Learning rate: {training_config.lr}")
        print(f"    - Epochs: {training_config.epochs}")
        print(f"    - Batch size: {training_config.batch_size}")
        print(f"    - Scheduler: {training_config.scheduler_type}")

        shadow_models, m_actual, shadow_subsets, any_new_models_trained = train_shadow_models(
            query_indices=query_indices,
            num_models=args.num_shadow_models,
            total_dataset_size=TOTAL_DATASET_SIZE,
            target_train_size=TARGET_TRAIN_SIZE,
            training_config=training_config,
            confident_memberships=None,
            shadow_models_dir=shadow_models_dir
        )

        np.save(m_actual_path, m_actual)

        for k, model in enumerate(shadow_models):
            torch.save({
                'model_state_dict': model.state_dict(),
                'm_k': m_actual[k],
                'subset_indices': shadow_subsets[k]
            }, os.path.join(shadow_models_dir, f"shadow_{k}.pth"))

        print("Done training/loading shadow models. Now precomputing influence matrices...")
        t_bases = precompute_influence_matrices(
            shadow_models=shadow_models,
            shadow_subsets=shadow_subsets,
            query_indices=query_indices,
            device=device,
            hessian_sample_size=5000,
            save_dir=precomputed_dir,
        )
        del shadow_models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _stack_influence_matrices_to_npz(
            save_dir=precomputed_dir,
            num_models=args.num_shadow_models,
            influence_cache_path=influence_cache_path,
            t_bases=t_bases,
        )

    # 5. Evaluate target model on query points
    print("Evaluating target model on query points...")
    target_scores = evaluate_target_model(target_model_path, query_indices, device)

    # 6. LiRA baseline (unchanged)
    lira_scores, lira_tpr_01pct, lira_tpr_1pct, lira_balanced_acc = run_lira_baseline(
        target_scores,
        t_bases,
        m_actual,
        ground_truth,
        attack_dir,
    )

    if args.analyze_influence:
        analyze_influence_vs_lira(
            attack_dir=attack_dir,
            lira_scores=lira_scores,
            ground_truth=ground_truth,
            target_model_path=target_model_path,
            device=device,
        )

    # 7. Calibration / plots
    plot_calibration({"LiRA": lira_scores}, ground_truth, attack_dir)

    # 8. Final printout
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"  LiRA:  TPR @ 0.1% FPR = {lira_tpr_01pct * 100:.2f}%  |  TPR @ 1% FPR = {lira_tpr_1pct * 100:.2f}%  |  Balanced Acc = {lira_balanced_acc * 100:.2f}%")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
