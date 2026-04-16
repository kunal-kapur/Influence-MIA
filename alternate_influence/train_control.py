"""
Control model training script for membership inference attacks.
Trains a ResNet model on a specified percentage of the training dataset
and saves which exact data points were used.
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torch.utils.data import Subset
import numpy as np
from datetime import datetime

from resnet_influence import ResNet18_Influence
from global_variables import DATA_DIR, TRANSFORM_TEST
from train_utils import TrainingConfig, train_model, create_dataloader


def setup_device():
    """Setup GPU/CPU device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cudnn.benchmark = True
    return device


def sample_training_indices(data_percentage=100, seed=42):
    """
    Sample training indices from CIFAR-10.
    
    Args:
        data_percentage: Percentage of training data to use (1-100)
        seed: Random seed for reproducibility
        
    Returns:
        train_indices: Sorted list of training sample indices
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset to get size
    trainset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, 
                               transform=TRANSFORM_TEST)
    
    # Sample training data
    total_train_size = len(trainset)
    num_samples = max(1, int(total_train_size * data_percentage / 100))
    
    all_indices = np.arange(total_train_size)
    train_indices = np.random.choice(all_indices, size=num_samples, replace=False)
    train_indices = sorted(train_indices.tolist())
    
    return train_indices





def save_training_metadata(save_dir, data_percentage, train_indices, 
                          test_accuracy, test_loss, seed, training_config):
    """
    Save metadata about the training run including which data was used.
    
    Args:
        save_dir: Directory to save metadata
        data_percentage: Percentage of data used
        train_indices: List of training sample indices used
        test_accuracy: Final test accuracy
        test_loss: Final test loss
        seed: Random seed used
        training_config: TrainingConfig object
    """
    os.makedirs(save_dir, exist_ok=True)
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_percentage": data_percentage,
        "num_samples_used": len(train_indices),
        "total_cifar10_train_size": 50000,
        "train_indices": train_indices,  # EXACT data points used
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "seed": seed,
        "model": "ResNet18",
        "training_config": training_config.to_dict()  # Complete training configuration
    }
    
    # Save as JSON
    metadata_path = os.path.join(save_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTraining metadata saved to: {metadata_path}")
    print(f"\nSummary:")
    print(f"  - Used {len(train_indices)} samples ({data_percentage}% of training set)")
    print(f"  - Final test accuracy: {test_accuracy:.2f}%")
    print(f"  - Final test loss: {test_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train a control ResNet model')
    parser.add_argument('--data-percentage', type=float, default=100,
                       help='Percentage of training data to use (1-100)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                       help='Optimizer type (adam or sgd)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum (only used for SGD optimizer)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay (L2 penalty)')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler type')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--identifier', type=str, default=None,
                       help='Identifier for this run (default: auto-generated from parameters)')
    parser.add_argument('--model-name', type=str, default='model',
                       help='Name of the model checkpoint')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint if available')
    
    args = parser.parse_args()
    
    if not (1 <= args.data_percentage <= 100):
        raise ValueError("data_percentage must be between 1 and 100")
    
    if args.identifier is None:
        identifier = f"{int(args.data_percentage)}pct_seed{args.seed}"
    else:
        identifier = args.identifier
    
    checkpoint_dir = os.path.join('./checkpoints', identifier)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Setup
    device = setup_device()
    print(f"Using device: {device}")
    
    # Create training configuration
    training_config = TrainingConfig(
        epochs=args.epochs,
        optimizer_type=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # Sample training indices
    print(f"\nSampling {args.data_percentage}% of training data...")
    train_indices = sample_training_indices(args.data_percentage, args.seed)
    print(f"Training on {len(train_indices)} samples")
    
    # Create test loader
    test_loader = create_dataloader(
        indices=None,  # Use all test data
        batch_size=100,
        num_workers=2,
        train=False,
        shuffle=False
    )
    
    print("\nCreating ResNet18 model...")
    model = ResNet18_Influence()
    model = model.to(device)
    
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Optimizer: {args.optimizer.upper()}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Scheduler: {args.scheduler}")
    print(f"  Batch size: {args.batch_size}\n")
    
    # Callback to save best model
    best_test_acc = 0
    def save_best_callback(epoch, test_acc):
        nonlocal best_test_acc
        best_test_acc = test_acc
        model_path = os.path.join(checkpoint_dir, f"{args.model_name}_best.pth")
        torch.save(model.state_dict(), model_path)
        print(f"  -> Saved best model (accuracy: {test_acc:.2f}%)")
    
    # Determine if we should resume from latest checkpoint
    resume_from = None
    if args.resume:
        latest_checkpoint = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
        if os.path.exists(latest_checkpoint):
            resume_from = latest_checkpoint
            print(f"\\nResuming from checkpoint: {latest_checkpoint}\\n")
        else:
            print(f"\\nNo checkpoint found to resume from, starting fresh\\n")
    
    # Train the model
    _, _, final_test_loss, final_test_acc = train_model(
        model=model,
        train_indices=train_indices,
        config=training_config,
        device=device,
        verbose=True,
        test_loader=test_loader,
        save_best_callback=save_best_callback,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from
    )
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, f"{args.model_name}_final.pth")
    torch.save(model.state_dict(), final_model_path)
    
    print(f"\n{'='*60}")
    save_training_metadata(
        checkpoint_dir, 
        args.data_percentage, 
        train_indices,
        best_test_acc, 
        final_test_loss, 
        args.seed,
        training_config
    )
    print(f"{'='*60}")
    
    print(f"\nModel checkpoints saved to: {checkpoint_dir}")
    print(f"  - Best model: {args.model_name}_best.pth")
    print(f"  - Final model: {args.model_name}_final.pth")


if __name__ == "__main__":
    main()
