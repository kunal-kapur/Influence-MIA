"""Training curve plotting utilities.

All plots are saved to {exp_dir}/plots/.

Functions
---------
plot_target_curves(history, exp_dir)
    Loss + accuracy curves for the target model.

plot_shadow_curves(history_p1, history_p2, shadow_id, exp_dir)
    Phase 1 (OUT) and Phase 2 (IN) curves for a single shadow model.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _plots_dir(exp_dir: str) -> str:
    d = os.path.join(exp_dir, "plots")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Target model
# ---------------------------------------------------------------------------

def plot_target_curves(history: dict, exp_dir: str) -> None:
    """Save loss and accuracy curves for the target model.

    Args:
        history: dict with keys
            "train_loss"  : list of float, one per epoch
            "train_acc"   : list of float, one per epoch
            "val_loss"    : list of (epoch_idx, float) — only eval epochs
            "val_acc"     : list of (epoch_idx, float) — only eval epochs
        exp_dir: experiment root directory
    """
    out_dir   = _plots_dir(exp_dir)
    epochs    = list(range(1, len(history["train_loss"]) + 1))
    val_epochs = [e for e, _ in history["val_loss"]]
    val_losses = [v for _, v in history["val_loss"]]
    val_accs   = [v for _, v in history["val_acc"]]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Target model — training curves", fontsize=13)

    # Loss
    ax_loss.plot(epochs, history["train_loss"], label="train loss", color="steelblue")
    ax_loss.plot(val_epochs, val_losses, "o--", label="val loss",
                 color="darkorange", markersize=4)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Accuracy
    ax_acc.plot(epochs, history["train_acc"], label="train acc", color="steelblue")
    ax_acc.plot(val_epochs, val_accs, "o--", label="val acc",
                color="darkorange", markersize=4)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "target_training_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] Target curves saved → {path}")


# ---------------------------------------------------------------------------
# Shadow model
# ---------------------------------------------------------------------------

def plot_shadow_curves(history_p1: dict, history_p2: dict,
                       shadow_id: int, exp_dir: str) -> None:
    """Save training curves for one shadow model (both phases).

    Args:
        history_p1: dict with keys
            "train_loss"  : list of float (all T1 epochs)
            "loss_type"   : list of str, "CE" or "imitate" per epoch
            "val_acc"     : list of (epoch_idx, float) — post-warmup eval epochs
            "warmup_epoch": int — epoch at which warmup ends
        history_p2: dict with keys
            "train_loss"  : list of float (all T2 epochs)
        shadow_id: integer index
        exp_dir: experiment root directory
    """
    out_dir = _plots_dir(exp_dir)
    T1      = len(history_p1["train_loss"])
    T2      = len(history_p2["train_loss"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"Shadow {shadow_id} — training curves", fontsize=13)

    # --- Phase 1 loss ---
    ax = axes[0]
    ep1  = list(range(1, T1 + 1))
    loss = history_p1["train_loss"]
    types = history_p1.get("loss_type", ["CE"] * T1)
    wu    = history_p1.get("warmup_epoch", T1)

    ce_ep      = [e for e, t in zip(ep1, types) if t == "CE"]
    ce_loss    = [l for l, t in zip(loss, types) if t == "CE"]
    im_ep      = [e for e, t in zip(ep1, types) if t == "imitate"]
    im_loss    = [l for l, t in zip(loss, types) if t == "imitate"]

    ax.plot(ce_ep, ce_loss, color="steelblue", label="CE loss")
    if im_ep:
        ax.plot(im_ep, im_loss, color="mediumseagreen", label="imitation loss")
    if wu < T1:
        ax.axvline(wu, color="gray", linestyle="--", linewidth=1, label=f"warmup end (ep {wu})")
    ax.set_title("Phase 1 — OUT model (train loss)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Phase 1 val accuracy ---
    ax = axes[1]
    if history_p1["val_acc"]:
        val_ep  = [e for e, _ in history_p1["val_acc"]]
        val_acc = [v for _, v in history_p1["val_acc"]]
        ax.plot(val_ep, val_acc, "o-", color="darkorange", markersize=4)
    ax.set_title("Phase 1 — OUT model (val accuracy)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # --- Phase 2 loss ---
    ax = axes[2]
    ep2 = list(range(1, T2 + 1))
    ax.plot(ep2, history_p2["train_loss"], color="mediumpurple", label="CE loss")
    ax.set_title("Phase 2 — IN model (train loss on pivot)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f"shadow_{shadow_id}_training_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plot] Shadow {shadow_id} curves saved → {path}")
