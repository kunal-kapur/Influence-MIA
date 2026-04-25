import torch
from torch.utils.data import ConcatDataset, random_split, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_dataset(args):
    """Return CIFAR-10 dataset class and set metadata on args."""
    args.data_mean = [0.4914, 0.4822, 0.4465]
    args.data_std = [0.2023, 0.1994, 0.2010]
    args.num_classes = 10
    return torchvision.datasets.CIFAR10


def offline_data_split(dataset, seed, data_type):
    """
    Split the full 60k CIFAR-10 dataset deterministically into four disjoint pools:

      target     : n // 3        (20 000) — candidate set D for target model
      shadow     : n // 3        (20 000) — auxiliary pool for shadow model training
      validation : n // 12       ( 5 000) — small held-out split
      reference  : remainder     (15 000) — disjoint reference data

    Sizes are fixed by n and the seed; changing either invalidates saved artifacts.
    """
    assert data_type in ("target", "shadow", "validation", "reference"), (
        f"Unknown data_type '{data_type}'. "
        "Expected one of: target, shadow, validation, reference."
    )

    n = len(dataset)

    target_size     = n // 3          # 20 000
    shadow_size     = n // 3          # 20 000
    validation_size = n // 12         #  5 000
    reference_size  = n - target_size - shadow_size - validation_size  # 15 000

    generator = torch.Generator().manual_seed(seed)
    target_split, shadow_split, validation_split, reference_split = random_split(
        dataset,
        [target_size, shadow_size, validation_size, reference_size],
        generator=generator,
    )

    return {
        "target":     target_split,
        "shadow":     shadow_split,
        "validation": validation_split,
        "reference":  reference_split,
    }[data_type]


def _maybe_shrink_target_pool(target_split, seed, shared_pool_size):
    """Optionally shrink the target split to a deterministic smaller pool."""
    if shared_pool_size is None:
        return target_split

    shared_pool_size = int(shared_pool_size)
    if shared_pool_size <= 0:
        raise ValueError(f"shared_pool_size must be positive, got {shared_pool_size}.")

    if shared_pool_size >= len(target_split):
        return target_split

    # target_split is a Subset over the full dataset. We sample directly in
    # global index space to keep downstream index mapping simple.
    global_indices = np.asarray(target_split.indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(global_indices, size=shared_pool_size, replace=False))
    return Subset(target_split.dataset, chosen.tolist())


def split_dataset_for_type(dataset, seed, data_type, shared_pool_size=None):
    """Return a deterministic split, with optional shrinking for target pool."""
    split = offline_data_split(dataset, seed, data_type)
    if data_type == "target":
        return _maybe_shrink_target_pool(split, seed, shared_pool_size)
    return split


def load_dataset(args, data_type="target"):
    """
    Load CIFAR-10 train+test, apply transforms, and return the requested split.

    Training splits (target, shadow) get data augmentation.
    Evaluation splits (validation, reference) get normalisation only.
    """
    mean = args.data_mean
    std  = args.data_std
    normalize = transforms.Normalize(mean, std)

    if data_type in ("target", "shadow"):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_ds = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )

    full_dataset = ConcatDataset([train_ds, test_ds])
    return split_dataset_for_type(
        full_dataset,
        seed=args.seed,
        data_type=data_type,
        shared_pool_size=getattr(args, "shared_pool_size", None),
    )