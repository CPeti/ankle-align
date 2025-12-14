"""
Training Script for Ankle Alignment Classification

This script implements the best performing model from incremental modeling:
- ResNet18 with transfer learning
- Data augmentation
- Class-weighted loss for imbalance
- Early stopping and learning rate scheduling

Usage:
    python src/02-training.py [--epochs 50] [--batch-size 32] [--lr 0.0001]
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models

from config import (
    PROCESSED_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODEL_DIR,
    CHECKPOINT_DIR,
    MODEL_SAVE_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    WEIGHT_DECAY,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    NUM_CLASSES,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    AUGMENTATION_ENABLED,
    HORIZONTAL_FLIP_PROB,
    ROTATION_DEGREES,
    COLOR_JITTER,
    DEVICE,
)
from utils import (
    setup_logger,
    log_section,
    log_model_summary,
    log_training_config,
    log_data_summary,
    log_epoch,
    log_evaluation_results,
    clear_log_file,
)

logger = setup_logger(__name__)


# ============================================================================
# Dataset
# ============================================================================

class AnkleAlignmentDataset(Dataset):
    """
    Dataset for ankle alignment images.
    Loads images from directory structure: base_dir/label/image.png
    """
    
    def __init__(self, base_dir: Path, label_to_idx: Dict[str, int], 
                 transform=None, load_to_memory: bool = True):
        """
        Args:
            base_dir: Directory containing label subdirectories
            label_to_idx: Mapping from label names to indices
            transform: Optional transforms to apply
            load_to_memory: If True, load all images to memory for faster training
        """
        self.base_dir = Path(base_dir)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.load_to_memory = load_to_memory
        
        self.samples = []  # List of (image_path, label_idx)
        self.images = []   # Cached images if load_to_memory=True
        
        self._load_samples()
        
        if load_to_memory:
            self._load_images_to_memory()
    
    def _load_samples(self):
        """Scan directories and collect sample paths."""
        for label_name, label_idx in self.label_to_idx.items():
            label_dir = self.base_dir / label_name
            if not label_dir.exists():
                logger.warning(f"Directory not found: {label_dir}")
                continue
            
            for img_path in label_dir.glob("*.png"):
                self.samples.append((img_path, label_idx))
        
        logger.info(f"Loaded {len(self.samples)} samples from {self.base_dir}")
    
    def _load_images_to_memory(self):
        """Load all images to memory."""
        logger.info("Loading images to memory...")
        for img_path, _ in tqdm(self.samples, desc="Loading images"):
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
            # Convert to tensor format (C, H, W)
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
            self.images.append(img_tensor)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.load_to_memory:
            img = self.images[idx].clone()
        else:
            img_path, _ = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
        
        label = self.samples[idx][1]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


# ============================================================================
# Model Architecture
# ============================================================================

class ResNetTransfer(nn.Module):
    """
    Transfer learning model using pretrained ResNet18.
    
    This was the best performing model in incremental modeling experiments.
    Uses pretrained ImageNet weights and fine-tunes all layers.
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.3, 
                 freeze_backbone: bool = False):
        super(ResNetTransfer, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Optionally freeze backbone for feature extraction only
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen - only training classifier")
        
        # Replace final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen for fine-tuning")


# ============================================================================
# Training Functions
# ============================================================================

def create_transforms(training: bool = True) -> transforms.Compose:
    """Create data transforms for training or evaluation."""
    
    # ImageNet normalization (required for pretrained models)
    normalize = transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    
    if training and AUGMENTATION_ENABLED:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROB),
            transforms.RandomRotation(degrees=ROTATION_DEGREES),
            transforms.ColorJitter(
                brightness=COLOR_JITTER['brightness'],
                contrast=COLOR_JITTER['contrast'],
                saturation=COLOR_JITTER['saturation'],
                hue=COLOR_JITTER['hue']
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            normalize,
        ])
    else:
        return transforms.Compose([normalize])


def compute_class_weights(dataset: Dataset, num_classes: int, 
                          device: str) -> torch.Tensor:
    """Compute class weights for handling class imbalance."""
    labels = [sample[1] for sample in dataset.samples]
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # Inverse frequency weighting
    class_weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
    class_weights = class_weights / class_weights.sum() * num_classes
    
    logger.info("Class weights:")
    for idx, (label, weight) in enumerate(zip(IDX_TO_LABEL.values(), class_weights)):
        count = class_counts[idx]
        logger.info(f"  {label}: {weight:.3f} (count: {count})")
    
    return torch.FloatTensor(class_weights).to(device)


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: str) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer,
                    epoch: int, val_loss: float, val_acc: float,
                    path: Path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
    }, path)


# ============================================================================
# Main Training Function
# ============================================================================

def train(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOPPING_PATIENCE,
    freeze_epochs: int = 0,
    dropout: float = 0.3,
    save_checkpoints: bool = True,
    clear_logs: bool = True,
):
    """
    Main training function.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: L2 regularization weight
        patience: Early stopping patience
        freeze_epochs: Number of epochs to train with frozen backbone
        dropout: Dropout rate
        save_checkpoints: Whether to save checkpoints during training
        clear_logs: Whether to clear the log file before training
    """
    
    # Clear log file for fresh run
    if clear_logs:
        clear_log_file()
    
    # Log header
    log_section(logger, "ANKLE ALIGNMENT MODEL TRAINING")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Log training configuration
    training_config = {
        'Device': DEVICE,
        'Epochs': epochs,
        'Batch Size': batch_size,
        'Learning Rate': learning_rate,
        'Weight Decay': weight_decay,
        'Early Stopping Patience': patience,
        'Freeze Epochs': freeze_epochs,
        'Dropout': dropout,
        'Image Size': f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        'Augmentation Enabled': AUGMENTATION_ENABLED,
        'Number of Classes': NUM_CLASSES,
    }
    log_training_config(logger, training_config)
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if save_checkpoints:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    log_section(logger, "DATA LOADING")
    logger.info("Loading and preprocessing datasets...")
    
    train_transform = create_transforms(training=True)
    eval_transform = create_transforms(training=False)
    
    train_dataset = AnkleAlignmentDataset(
        TRAIN_DIR, LABEL_TO_IDX, transform=train_transform, load_to_memory=True
    )
    val_dataset = AnkleAlignmentDataset(
        VAL_DIR, LABEL_TO_IDX, transform=eval_transform, load_to_memory=True
    )
    test_dataset = AnkleAlignmentDataset(
        TEST_DIR, LABEL_TO_IDX, transform=eval_transform, load_to_memory=True
    )
    
    # Compute class distribution for training set
    train_labels = [sample[1] for sample in train_dataset.samples]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_distribution = {IDX_TO_LABEL[i]: int(class_counts[i]) for i in range(NUM_CLASSES)}
    
    # Log data summary
    log_data_summary(
        logger,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
        class_distribution=class_distribution
    )
    
    logger.info("Data loading complete.")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # Create model
    model = ResNetTransfer(
        num_classes=NUM_CLASSES, 
        dropout=dropout,
        freeze_backbone=(freeze_epochs > 0)
    )
    model = model.to(DEVICE)
    
    # Log model architecture summary
    log_model_summary(logger, model)
    
    # Loss function with class weights
    class_weights = compute_class_weights(train_dataset, NUM_CLASSES, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    log_section(logger, "TRAINING PROGRESS")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        # Unfreeze backbone after freeze_epochs
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            model.unfreeze_backbone()
            # Reset optimizer with lower learning rate for fine-tuning
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate * 0.1, weight_decay=weight_decay
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Check for improvement first to determine if this is best
        is_best = val_loss < best_val_loss
        
        # Log progress
        log_epoch(
            logger, epoch, epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            current_lr, is_best=is_best
        )
        
        # Save checkpoint
        if save_checkpoints and epoch % 10 == 0:
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:03d}.pth"
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        # Handle improvement
        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            
            # Save best model
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping triggered after {epoch} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on test set
    test_loss, test_acc, y_pred, y_true = evaluate(
        model, test_loader, criterion, DEVICE
    )
    
    # Compute detailed metrics
    from sklearn.metrics import (
        confusion_matrix, classification_report, 
        precision_score, recall_score, f1_score
    )
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=list(LABEL_TO_IDX.keys()))
    
    # Per-class accuracy
    per_class_acc = {}
    for idx, label in IDX_TO_LABEL.items():
        mask = y_true == idx
        if mask.sum() > 0:
            per_class_acc[label] = float((y_pred[mask] == y_true[mask]).mean())
    
    # Compute overall metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Log evaluation results
    metrics = {
        'Test Loss': test_loss,
        'Test Accuracy': test_acc,
        'Precision (weighted)': precision,
        'Recall (weighted)': recall,
        'F1-Score (weighted)': f1,
    }
    
    # Add per-class accuracy
    for label, acc in per_class_acc.items():
        metrics[f'Accuracy ({label})'] = acc
    
    log_evaluation_results(logger, metrics, cm, report)
    
    # Save final model with metadata
    final_model_path = MODEL_DIR / "final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': 'ResNetTransfer',
        'num_classes': NUM_CLASSES,
        'image_size': IMAGE_SIZE,
        'label_to_idx': LABEL_TO_IDX,
        'idx_to_label': IDX_TO_LABEL,
        'normalize_mean': NORMALIZE_MEAN,
        'normalize_std': NORMALIZE_STD,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'history': history,
        'training_config': {
            'epochs': epoch,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'dropout': dropout,
        }
    }, final_model_path)
    
    logger.info(f"\nFinal model saved to: {final_model_path}")
    
    # Save training history
    history_path = MODEL_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to: {history_path}")
    
    # Summary
    log_section(logger, "TRAINING COMPLETE")
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total epochs trained: {epoch}")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.4f}")
    logger.info(f"Final Test Accuracy: {test_acc:.4f}")
    logger.info(f"Final Test F1-Score: {f1:.4f}")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info("")
    
    return model, history, test_acc


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train ankle alignment classification model'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS})'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size (default: {BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=0.0001,  # Lower LR for transfer learning
        help='Learning rate (default: 0.0001)'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=WEIGHT_DECAY,
        help=f'Weight decay for L2 regularization (default: {WEIGHT_DECAY})'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=f'Early stopping patience (default: {EARLY_STOPPING_PATIENCE})'
    )
    
    parser.add_argument(
        '--freeze-epochs',
        type=int,
        default=0,
        help='Number of epochs to train with frozen backbone (default: 0)'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate (default: 0.3)'
    )
    
    parser.add_argument(
        '--no-checkpoints',
        action='store_true',
        help='Disable saving checkpoints during training'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        freeze_epochs=args.freeze_epochs,
        dropout=args.dropout,
        save_checkpoints=not args.no_checkpoints,
    )
