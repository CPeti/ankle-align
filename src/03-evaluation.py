"""
Model Evaluation Script for Ankle Alignment Classification

This script evaluates the trained model and generates comprehensive metrics:
- Overall accuracy, precision, recall, F1 score
- Per-class metrics
- Confusion matrix visualization
- ROC curves and AUC scores
- Misclassification analysis
- Detailed evaluation report

Usage:
    python src/03-evaluation.py [--model-path models/final_model.pth]
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from config import (
    PROCESSED_DIR,
    TEST_DIR,
    MODEL_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    NUM_CLASSES,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    DEVICE,
)
from utils import setup_logger, log_section, log_evaluation_results

logger = setup_logger(__name__)


# ============================================================================
# Dataset (same as training)
# ============================================================================

class AnkleAlignmentDataset(Dataset):
    """Dataset for ankle alignment images."""
    
    def __init__(self, base_dir: Path, label_to_idx: Dict[str, int], 
                 transform=None, load_to_memory: bool = True):
        self.base_dir = Path(base_dir)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.load_to_memory = load_to_memory
        
        self.samples = []
        self.images = []
        
        self._load_samples()
        
        if load_to_memory:
            self._load_images_to_memory()
    
    def _load_samples(self):
        for label_name, label_idx in self.label_to_idx.items():
            label_dir = self.base_dir / label_name
            if not label_dir.exists():
                continue
            
            for img_path in label_dir.glob("*.png"):
                self.samples.append((img_path, label_idx))
    
    def _load_images_to_memory(self):
        for img_path, _ in self.samples:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
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
    
    def get_image_path(self, idx) -> Path:
        """Get the file path of an image by index."""
        return self.samples[idx][0]


# ============================================================================
# Model Architecture (same as training)
# ============================================================================

class ResNetTransfer(nn.Module):
    """Transfer learning model using pretrained ResNet18."""
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super(ResNetTransfer, self).__init__()
        self.backbone = models.resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================================
# Evaluation Functions
# ============================================================================

def load_model(model_path: Path, device: str) -> Tuple[nn.Module, dict]:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model configuration
    num_classes = checkpoint.get('num_classes', NUM_CLASSES)
    
    # Create model
    model = ResNetTransfer(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"  Training epochs: {checkpoint.get('training_config', {}).get('epochs', 'N/A')}")
    logger.info(f"  Best val accuracy: {checkpoint.get('best_val_acc', 'N/A'):.4f}")
    
    return model, checkpoint


def get_predictions(model: nn.Module, loader: DataLoader, device: str) -> Dict:
    """Get model predictions for all samples."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs)
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    y_probs: np.ndarray, class_names: list) -> Dict:
    """Compute comprehensive evaluation metrics."""
    
    metrics = {}
    
    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    metrics['per_class'] = {}
    for idx, class_name in enumerate(class_names):
        mask = y_true == idx
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_true[mask]).mean()
            
            # Binary metrics for this class
            y_true_binary = (y_true == idx).astype(int)
            y_pred_binary = (y_pred == idx).astype(int)
            
            metrics['per_class'][class_name] = {
                'accuracy': float(class_acc),
                'precision': float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
                'recall': float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
                'f1': float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
                'support': int(mask.sum()),
            }
            
            # ROC AUC for this class (one-vs-rest)
            if len(np.unique(y_true)) > 1:
                try:
                    fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, idx])
                    metrics['per_class'][class_name]['auc'] = float(auc(fpr, tpr))
                except:
                    metrics['per_class'][class_name]['auc'] = None
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          class_names: list, save_path: Path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion Matrix (Counts)', fontweight='bold')
    
    # Normalized
    ax2 = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('Confusion Matrix (Normalized)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to: {save_path}")


def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, 
                    class_names: list, save_path: Path):
    """Plot ROC curves for each class."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for idx, (class_name, color) in enumerate(zip(class_names, colors)):
        y_true_binary = (y_true == idx).astype(int)
        
        if y_true_binary.sum() > 0 and y_true_binary.sum() < len(y_true_binary):
            fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curves saved to: {save_path}")


def plot_precision_recall_curves(y_true: np.ndarray, y_probs: np.ndarray,
                                  class_names: list, save_path: Path):
    """Plot precision-recall curves for each class."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for idx, (class_name, color) in enumerate(zip(class_names, colors)):
        y_true_binary = (y_true == idx).astype(int)
        
        if y_true_binary.sum() > 0:
            precision, recall, _ = precision_recall_curve(y_true_binary, y_probs[:, idx])
            ap = average_precision_score(y_true_binary, y_probs[:, idx])
            ax.plot(recall, precision, color=color, linewidth=2,
                    label=f'{class_name} (AP = {ap:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Precision-recall curves saved to: {save_path}")


def plot_per_class_metrics(metrics: Dict, class_names: list, save_path: Path):
    """Plot per-class metrics comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_names = ['precision', 'recall', 'f1', 'accuracy']
    x = np.arange(len(class_names))
    width = 0.2
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']
    
    for i, metric_name in enumerate(metric_names):
        values = [metrics['per_class'][c][metric_name] for c in class_names]
        bars = ax.bar(x + i * width, values, width, label=metric_name.capitalize(), color=colors[i])
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Per-class metrics saved to: {save_path}")


def analyze_misclassifications(dataset: Dataset, y_true: np.ndarray, 
                                y_pred: np.ndarray, y_probs: np.ndarray,
                                class_names: list, save_dir: Path, 
                                max_samples: int = 20):
    """Analyze and visualize misclassified samples."""
    
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    if len(misclassified_idx) == 0:
        logger.info("No misclassifications found!")
        return []
    
    # Sort by confidence (most confident mistakes first)
    confidences = [y_probs[i, y_pred[i]] for i in misclassified_idx]
    sorted_indices = np.argsort(confidences)[::-1]
    misclassified_idx = misclassified_idx[sorted_indices]
    
    # Create visualization
    n_samples = min(len(misclassified_idx), max_samples)
    cols = 4
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Misclassified Samples ({len(misclassified_idx)} total)', 
                 fontsize=14, fontweight='bold')
    
    misclassification_details = []
    
    for i, idx in enumerate(misclassified_idx[:n_samples]):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        # Get image
        img_path = dataset.get_image_path(idx)
        img = Image.open(img_path)
        
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        confidence = y_probs[idx, y_pred[idx]]
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1%})',
                     color='red', fontsize=10)
        
        misclassification_details.append({
            'image_path': str(img_path),
            'true_label': true_label,
            'predicted_label': pred_label,
            'confidence': float(confidence),
            'probabilities': {class_names[j]: float(y_probs[idx, j]) 
                            for j in range(len(class_names))}
        })
    
    # Hide unused subplots
    for i in range(n_samples, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'misclassifications.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Misclassification visualization saved")
    
    return misclassification_details


def generate_report(metrics: Dict, checkpoint: Dict, 
                    misclassifications: list, save_path: Path):
    """Generate detailed evaluation report."""
    
    report = {
        'evaluation_date': datetime.now().isoformat(),
        'model_info': {
            'best_val_acc': checkpoint.get('best_val_acc'),
            'training_config': checkpoint.get('training_config', {}),
        },
        'test_results': {
            'accuracy': metrics['accuracy'],
            'precision_macro': metrics['precision_macro'],
            'recall_macro': metrics['recall_macro'],
            'f1_macro': metrics['f1_macro'],
            'precision_weighted': metrics['precision_weighted'],
            'recall_weighted': metrics['recall_weighted'],
            'f1_weighted': metrics['f1_weighted'],
        },
        'per_class_metrics': metrics['per_class'],
        'confusion_matrix': metrics['confusion_matrix'],
        'misclassification_count': len(misclassifications),
        'misclassification_details': misclassifications[:10],  # Top 10
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation report saved to: {save_path}")


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate(
    model_path: Optional[Path] = None,
    batch_size: int = BATCH_SIZE,
    save_visualizations: bool = True,
):
    """
    Main evaluation function.
    
    Args:
        model_path: Path to the trained model checkpoint
        batch_size: Batch size for evaluation
        save_visualizations: Whether to save visualization plots
    """
    
    # Setup
    log_section(logger, "ANKLE ALIGNMENT MODEL EVALUATION")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find model path
    if model_path is None:
        model_path = MODEL_DIR / "final_model.pth"
        if not model_path.exists():
            model_path = MODEL_DIR / "best_model.pth"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    # Create output directory
    eval_dir = MODEL_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, checkpoint = load_model(model_path, DEVICE)
    
    # Create test dataset and loader
    logger.info("\nLoading test data...")
    eval_transform = transforms.Compose([
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    
    test_dataset = AnkleAlignmentDataset(
        TEST_DIR, LABEL_TO_IDX, transform=eval_transform, load_to_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Get predictions
    logger.info("\nRunning evaluation...")
    results = get_predictions(model, test_loader, DEVICE)
    
    y_true = results['labels']
    y_pred = results['predictions']
    y_probs = results['probabilities']
    class_names = list(LABEL_TO_IDX.keys())
    
    # Compute metrics
    logger.info("\nComputing metrics...")
    metrics = compute_metrics(y_true, y_pred, y_probs, class_names)
    
    # Build metrics dict for logging
    cm = np.array(metrics['confusion_matrix'])
    report_str = classification_report(y_true, y_pred, target_names=class_names)
    
    # Create comprehensive metrics dict
    eval_metrics = {
        'Accuracy': metrics['accuracy'],
        'Precision (macro)': metrics['precision_macro'],
        'Recall (macro)': metrics['recall_macro'],
        'F1-Score (macro)': metrics['f1_macro'],
        'Precision (weighted)': metrics['precision_weighted'],
        'Recall (weighted)': metrics['recall_weighted'],
        'F1-Score (weighted)': metrics['f1_weighted'],
    }
    
    # Add per-class metrics
    for class_name, class_metrics in metrics['per_class'].items():
        eval_metrics[f'Accuracy ({class_name})'] = class_metrics['accuracy']
        eval_metrics[f'F1-Score ({class_name})'] = class_metrics['f1']
        if class_metrics.get('auc'):
            eval_metrics[f'AUC ({class_name})'] = class_metrics['auc']
    
    # Log evaluation results using utility
    log_evaluation_results(logger, eval_metrics, cm, report_str)
    
    # Also log per-class details
    logger.info("Per-Class Details:")
    for class_name, class_metrics in metrics['per_class'].items():
        logger.info(f"  {class_name}:")
        logger.info(f"    Precision: {class_metrics['precision']:.4f}")
        logger.info(f"    Recall:    {class_metrics['recall']:.4f}")
        logger.info(f"    F1 Score:  {class_metrics['f1']:.4f}")
        logger.info(f"    Support:   {class_metrics['support']}")
        if class_metrics.get('auc'):
            logger.info(f"    AUC:       {class_metrics['auc']:.4f}")
    
    # Generate visualizations
    if save_visualizations:
        logger.info("\nGenerating visualizations...")
        
        plot_confusion_matrix(y_true, y_pred, class_names, 
                              eval_dir / 'confusion_matrix.png')
        
        plot_roc_curves(y_true, y_probs, class_names,
                        eval_dir / 'roc_curves.png')
        
        plot_precision_recall_curves(y_true, y_probs, class_names,
                                      eval_dir / 'precision_recall_curves.png')
        
        plot_per_class_metrics(metrics, class_names,
                               eval_dir / 'per_class_metrics.png')
        
        # Analyze misclassifications
        misclassifications = analyze_misclassifications(
            test_dataset, y_true, y_pred, y_probs, class_names, eval_dir
        )
    else:
        misclassifications = []
    
    # Generate report
    generate_report(metrics, checkpoint, misclassifications, 
                    eval_dir / 'evaluation_report.json')
    
    # Summary
    log_section(logger, "EVALUATION COMPLETE")
    misclassified_count = len([i for i in range(len(y_true)) if y_true[i] != y_pred[i]])
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"")
    logger.info(f"Summary:")
    logger.info(f"  Test Accuracy:      {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score (macro):   {metrics['f1_macro']:.4f}")
    logger.info(f"  F1 Score (weighted):{metrics['f1_weighted']:.4f}")
    logger.info(f"  Misclassified:      {misclassified_count}/{len(y_true)} ({misclassified_count/len(y_true)*100:.1f}%)")
    logger.info(f"")
    logger.info(f"Results saved to: {eval_dir}")
    
    return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate ankle alignment classification model'
    )
    
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default=None,
        help='Path to model checkpoint (default: models/final_model.pth)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=BATCH_SIZE,
        help=f'Batch size for evaluation (default: {BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Disable saving visualization plots'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    model_path = Path(args.model_path) if args.model_path else None
    
    evaluate(
        model_path=model_path,
        batch_size=args.batch_size,
        save_visualizations=not args.no_visualizations,
    )
