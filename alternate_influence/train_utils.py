"""
Shared training utilities for ResNet models.
Provides unified training functions and configuration management to ensure
shadow models use the same training configuration as the target model.
"""

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import numpy as np
from global_variables import DATA_DIR, TRANSFORM_TRAIN, TRANSFORM_TEST


class TrainingConfig:
    """Container for training configuration parameters."""
    
    def __init__(self, 
                 epochs=200,
                 optimizer_type='adam',
                 lr=0.001,
                 momentum=0.9,  # Only used if optimizer_type is 'sgd'
                 weight_decay=5e-4,
                 scheduler_type='cosine',
                 batch_size=128,
                 num_workers=2):
        self.epochs = epochs
        self.optimizer_type = optimizer_type.lower()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def to_dict(self):
        """Convert config to dictionary for JSON serialization."""
        return {
            'epochs': self.epochs,
            'optimizer_type': self.optimizer_type,
            'lr': self.lr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'scheduler_type': self.scheduler_type,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_metadata(cls, metadata_path):
        """Load training config from metadata JSON file."""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if 'training_config' not in metadata:
            raise ValueError(
                f"No 'training_config' found in metadata file: {metadata_path}\n"
                "This may be an old metadata file. Please retrain the target model."
            )
        
        return cls.from_dict(metadata['training_config'])


def save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, 
                    best_test_acc, train_indices, config, rng_states=None):
    """
    Save complete training checkpoint including RNG states.
    
    Args:
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch number (0-indexed during training)
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler (can be None)
        best_test_acc: Best test accuracy so far
        train_indices: Training indices used
        config: TrainingConfig object
        rng_states: Optional dict of RNG states (will be captured if None)
    """
    # Capture RNG states if not provided
    if rng_states is None:
        rng_states = {
            'python': None,  # Python random.getstate() if needed
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'best_test_acc': best_test_acc,
        'train_indices': train_indices,
        'config': config.to_dict(),
        'rng_states': rng_states
    }
    
    # Save to temporary file first, then rename (atomic operation)
    temp_path = checkpoint_path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Load training checkpoint and restore RNG states.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to map checkpoint to
    
    Returns:
        Dict containing: epoch, best_test_acc, train_indices, config, loaded successfully
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Restore RNG states
    if 'rng_states' in checkpoint:
        rng_states = checkpoint['rng_states']
        if rng_states['numpy'] is not None:
            np.random.set_state(rng_states['numpy'])
        if rng_states['torch'] is not None:
            torch.set_rng_state(rng_states['torch'])
        if rng_states['torch_cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_states['torch_cuda'])
    
    return {
        'epoch': checkpoint['epoch'],
        'best_test_acc': checkpoint.get('best_test_acc', 0),
        'train_indices': checkpoint.get('train_indices', None),
        'config': TrainingConfig.from_dict(checkpoint['config']) if 'config' in checkpoint else None
    }


def create_optimizer(model, config):
    """Create optimizer based on config."""
    if config.optimizer_type == 'adam':
        return optim.Adam(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
    elif config.optimizer_type == 'sgd':
        return optim.SGD(
            model.parameters(), 
            lr=config.lr, 
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer_type}")


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on config."""
    if config.scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.epochs
        )
    elif config.scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=30, 
            gamma=0.1
        )
    elif config.scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler type: {config.scheduler_type}")


def create_dataloader(indices, batch_size, num_workers, train=True, shuffle=True):
    """Create a DataLoader for specified indices."""
    transform = TRANSFORM_TRAIN if train else TRANSFORM_TEST
    dataset = datasets.CIFAR10(
        root=DATA_DIR, 
        train=train, 
        download=True, 
        transform=transform
    )
    
    if indices is not None:
        dataset = Subset(dataset, indices)
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )


def train_one_epoch(model, train_loader, criterion, optimizer, device, verbose=False):
    """Train model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: torch device
        verbose: Whether to print batch progress
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if verbose and (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx + 1}]: Loss={train_loss/(batch_idx+1):.3f}, '
                  f'Acc={100.*correct/total:.2f}%')
    
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test/validation data.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: torch device
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(model, train_indices, config, device, verbose=True, 
                test_loader=None, save_best_callback=None, checkpoint_dir=None,
                resume_from=None):
    """
    Complete training loop for a model with checkpointing support.
    
    Args:
        model: Neural network model (already on device)
        train_indices: List of training sample indices
        config: TrainingConfig object
        device: torch device
        verbose: Whether to print epoch progress
        test_loader: Optional test DataLoader for validation
        save_best_callback: Optional function(epoch, test_acc) called when best model is achieved
        checkpoint_dir: Directory to save checkpoints (saves every epoch if provided)
        resume_from: Path to checkpoint file to resume from
    
    Returns:
        Tuple of (final_train_loss, final_train_acc, final_test_loss, final_test_acc)
        If test_loader is None, test metrics will be None.
    """
    # Create data loader
    train_loader = create_dataloader(
        train_indices, 
        config.batch_size, 
        config.num_workers,
        train=True,
        shuffle=True
    )
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Initialize training state
    start_epoch = 0
    best_test_acc = 0
    final_test_loss = None
    final_test_acc = None
    
    # Resume from checkpoint if provided
    if resume_from is not None and os.path.exists(resume_from):
        if verbose:
            print(f"Resuming training from checkpoint: {resume_from}")
        checkpoint_data = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        start_epoch = checkpoint_data['epoch'] + 1  # Continue from next epoch
        best_test_acc = checkpoint_data['best_test_acc']
        if verbose:
            print(f"Resuming from epoch {start_epoch}/{config.epochs}, best_acc={best_test_acc:.2f}%")
    
    # Create checkpoint directory if needed
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(start_epoch, config.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            verbose=(verbose and epoch == start_epoch)  # Only show batch details for first epoch
        )
        
        # Evaluate on test set if provided
        if test_loader is not None:
            test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
            final_test_loss = test_loss
            final_test_acc = test_acc
            
            if verbose:
                print(f"Epoch [{epoch+1}/{config.epochs}] - "
                      f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
            
            # Track best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                if save_best_callback is not None:
                    save_best_callback(epoch + 1, test_acc)
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{config.epochs}] - Train Acc: {train_acc:.2f}%")
        
        # Save checkpoint after each epoch
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_test_acc=best_test_acc,
                train_indices=train_indices,
                config=config
            )
            
            # Also save epoch-specific checkpoint every 10 epochs (optional, for safety)
            if (epoch + 1) % 10 == 0:
                epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(
                    checkpoint_path=epoch_checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_test_acc=best_test_acc,
                    train_indices=train_indices,
                    config=config
                )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
    
    return train_loss, train_acc, final_test_loss, final_test_acc


def train_shadow_model(model, train_indices, config, device, checkpoint_path=None):
    """
    Train a shadow model without per-epoch checkpointing.
    Only saves the final model after complete training.
    
    Args:
        model: Neural network model (already on device)
        train_indices: List of training sample indices
        config: TrainingConfig object
        device: torch device
        checkpoint_path: Optional path for final model save location (passed for info only)
    
    Returns:
        Trained model in eval mode
    """
    # Train without per-epoch checkpointing
    train_model(
        model, 
        train_indices, 
        config, 
        device, 
        verbose=False,
        test_loader=None,
        save_best_callback=None,
        checkpoint_dir=None,  # No per-epoch saves
        resume_from=None
    )
    
    model.eval()
    return model
