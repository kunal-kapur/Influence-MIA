import torch
from torch.utils.data import ConcatDataset, random_split
import torchvision
import torchvision.transforms as transforms


def get_dataset(args):
    """Return the torchvision dataset class for CIFAR-10 and set dataset
    metadata on args (data_mean, data_std, num_classes)."""
    args.data_mean = [0.4914, 0.4822, 0.4465]
    args.data_std = [0.2023, 0.1994, 0.2010]
    args.num_classes = 10
    return torchvision.datasets.CIFAR10


def offline_data_split(dataset, seed, data_type):
    """
    Split dataset into:
      - shared pool for 'target' and 'shadow'
      - small disjoint 'validation' split
      - remaining disjoint 'reference' split
    """
    assert data_type in ("target", "shadow", "validation", "reference"), (
        f"Unknown data_type: {data_type}"
    )

    n = len(dataset)

    shared_size = n // 3          # e.g. 20,000 if n = 60,000
    validation_size = n // 12     # e.g. 5,000 if n = 60,000
    reference_size = n - shared_size - validation_size

    generator = torch.Generator().manual_seed(seed)
    shared_split, validation_split, reference_split = random_split(
        dataset,
        [shared_size, validation_size, reference_size],
        generator=generator,
    )

    if data_type in ("target", "shadow"):
        return shared_split
    elif data_type == "validation":
        return validation_split
    else:  # reference
        return reference_split


def load_dataset(args, data_type="target"):
    """Load CIFAR-10 train+test (60,000 total), apply transforms, and return
    the 20,000-sample subset for the requested data_type."""
    mean = args.data_mean
    std = args.data_std

    normalize = transforms.Normalize(mean, std)

    if data_type in ("target", "shadow"):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:  # reference — no augmentation
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
    return offline_data_split(full_dataset, args.seed, data_type)
