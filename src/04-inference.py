"""
Inference Script for Ankle Alignment Classification

This script runs the trained model on new images to predict ankle alignment:
- Pronation
- Neutral
- Supination

Supports:
- Single image prediction
- Batch prediction from directory
- Output to console, JSON, or CSV
- Optional visualization

Usage:
    # Single image
    python src/04-inference.py --image path/to/image.jpg

    # Directory of images
    python src/04-inference.py --input-dir path/to/images/

    # With visualization
    python src/04-inference.py --image path/to/image.jpg --visualize

    # Save results to file
    python src/04-inference.py --input-dir path/to/images/ --output results.json
"""

import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

from config import (
    MODEL_DIR,
    IMAGE_SIZE,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    NUM_CLASSES,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    DEVICE,
)
from utils import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Prediction:
    """Prediction result for a single image."""
    image_path: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    
    def to_dict(self) -> dict:
        return asdict(self)


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
# Inference Engine
# ============================================================================

class AnkleAlignmentPredictor:
    """
    Predictor class for ankle alignment classification.
    
    Handles model loading, image preprocessing, and inference.
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    def __init__(self, model_path: Optional[Path] = None, device: str = DEVICE):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to model checkpoint. If None, uses default.
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = None
        self.transform = None
        self.label_to_idx = LABEL_TO_IDX
        self.idx_to_label = IDX_TO_LABEL
        self.image_size = IMAGE_SIZE
        
        # Load model
        self._load_model(model_path)
        
        # Setup transform
        self._setup_transform()
    
    def _load_model(self, model_path: Optional[Path] = None):
        """Load the trained model."""
        if model_path is None:
            # Try default paths
            for path in [MODEL_DIR / "final_model.pth", MODEL_DIR / "best_model.pth"]:
                if path.exists():
                    model_path = path
                    break
        
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(
                f"Model not found. Please provide a valid model path or "
                f"ensure a model exists in {MODEL_DIR}"
            )
        
        logger.info(f"Loading model from: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get configuration from checkpoint
        self.label_to_idx = checkpoint.get('label_to_idx', LABEL_TO_IDX)
        self.idx_to_label = checkpoint.get('idx_to_label', IDX_TO_LABEL)
        self.image_size = checkpoint.get('image_size', IMAGE_SIZE)
        num_classes = checkpoint.get('num_classes', NUM_CLASSES)
        
        # Handle idx_to_label keys (may be strings from JSON)
        if isinstance(list(self.idx_to_label.keys())[0], str):
            self.idx_to_label = {int(k): v for k, v in self.idx_to_label.items()}
        
        # Create and load model
        self.model = ResNetTransfer(num_classes=num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully")
        logger.info(f"  Classes: {list(self.label_to_idx.keys())}")
        logger.info(f"  Image size: {self.image_size}x{self.image_size}")
    
    def _setup_transform(self):
        """Setup image preprocessing transform."""
        # Handle image_size as either int or tuple
        if isinstance(self.image_size, int):
            size = (self.image_size, self.image_size)
        else:
            size = self.image_size
        
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
    
    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: Path to image file or PIL Image
            
        Returns:
            Preprocessed tensor ready for model input
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, image: Union[str, Path, Image.Image]) -> Prediction:
        """
        Run inference on a single image.
        
        Args:
            image: Path to image file or PIL Image
            
        Returns:
            Prediction object with class, confidence, and probabilities
        """
        # Get image path for result
        if isinstance(image, (str, Path)):
            image_path = str(image)
        else:
            image_path = "PIL Image"
        
        # Preprocess
        tensor = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Convert to Python types
        predicted_idx = predicted_idx.item()
        confidence = confidence.item()
        probs = probabilities[0].cpu().numpy()
        
        # Create probability dict
        prob_dict = {
            self.idx_to_label[i]: float(probs[i]) 
            for i in range(len(probs))
        }
        
        return Prediction(
            image_path=image_path,
            predicted_class=self.idx_to_label[predicted_idx],
            confidence=confidence,
            probabilities=prob_dict
        )
    
    def predict_batch(self, images: List[Union[str, Path]]) -> List[Prediction]:
        """
        Run inference on multiple images.
        
        Args:
            images: List of image paths
            
        Returns:
            List of Prediction objects
        """
        predictions = []
        
        for image_path in images:
            try:
                pred = self.predict(image_path)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue
        
        return predictions
    
    def predict_directory(self, directory: Union[str, Path], 
                          recursive: bool = False) -> List[Prediction]:
        """
        Run inference on all images in a directory.
        
        Args:
            directory: Path to directory containing images
            recursive: Whether to search subdirectories
            
        Returns:
            List of Prediction objects
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all image files
        image_files = []
        pattern = '**/*' if recursive else '*'
        
        for ext in self.SUPPORTED_FORMATS:
            image_files.extend(directory.glob(f'{pattern}{ext}'))
            image_files.extend(directory.glob(f'{pattern}{ext.upper()}'))
        
        if not image_files:
            logger.warning(f"No images found in {directory}")
            return []
        
        logger.info(f"Found {len(image_files)} images in {directory}")
        
        return self.predict_batch(sorted(image_files))


# ============================================================================
# Visualization
# ============================================================================

def visualize_prediction(image_path: Union[str, Path], prediction: Prediction,
                         save_path: Optional[Path] = None):
    """Visualize a prediction with the image and probability bars."""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show image
    ax1 = axes[0]
    img = Image.open(image_path)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title(f'Predicted: {prediction.predicted_class}\n'
                  f'Confidence: {prediction.confidence:.1%}',
                  fontsize=14, fontweight='bold')
    
    # Show probabilities
    ax2 = axes[1]
    classes = list(prediction.probabilities.keys())
    probs = list(prediction.probabilities.values())
    colors = ['#e74c3c' if c == 'Pronation' else 
              '#3498db' if c == 'Neutral' else '#2ecc71' 
              for c in classes]
    
    bars = ax2.barh(classes, probs, color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
    
    # Add probability labels
    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{prob:.1%}', va='center', fontsize=11)
    
    # Highlight predicted class
    pred_idx = classes.index(prediction.predicted_class)
    bars[pred_idx].set_edgecolor('black')
    bars[pred_idx].set_linewidth(2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_batch(predictions: List[Prediction], save_path: Optional[Path] = None,
                    max_images: int = 12):
    """Visualize multiple predictions in a grid."""
    
    n_images = min(len(predictions), max_images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Batch Predictions', fontsize=16, fontweight='bold')
    
    for i, pred in enumerate(predictions[:n_images]):
        row, col = i // cols, i % cols
        ax = axes[row, col]
        
        try:
            img = Image.open(pred.image_path)
            ax.imshow(img)
        except:
            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
        
        ax.axis('off')
        
        # Color based on confidence
        color = 'green' if pred.confidence > 0.8 else 'orange' if pred.confidence > 0.5 else 'red'
        ax.set_title(f'{pred.predicted_class}\n{pred.confidence:.1%}',
                     color=color, fontsize=10)
    
    # Hide unused subplots
    for i in range(n_images, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Batch visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Output Functions
# ============================================================================

def save_results_json(predictions: List[Prediction], output_path: Path):
    """Save predictions to JSON file."""
    results = [pred.to_dict() for pred in predictions]
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_path}")


def save_results_csv(predictions: List[Prediction], output_path: Path):
    """Save predictions to CSV file."""
    if not predictions:
        return
    
    fieldnames = ['image_path', 'predicted_class', 'confidence', 
                  'prob_Pronation', 'prob_Neutral', 'prob_Supination']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for pred in predictions:
            row = {
                'image_path': pred.image_path,
                'predicted_class': pred.predicted_class,
                'confidence': f'{pred.confidence:.4f}',
            }
            for class_name, prob in pred.probabilities.items():
                row[f'prob_{class_name}'] = f'{prob:.4f}'
            writer.writerow(row)
    
    logger.info(f"Results saved to: {output_path}")


def print_prediction(pred: Prediction, verbose: bool = False):
    """Print a single prediction to console."""
    print(f"\n{'='*50}")
    print(f"Image: {pred.image_path}")
    print(f"Prediction: {pred.predicted_class}")
    print(f"Confidence: {pred.confidence:.1%}")
    
    if verbose:
        print(f"\nAll probabilities:")
        for class_name, prob in sorted(pred.probabilities.items(), 
                                        key=lambda x: x[1], reverse=True):
            bar = '█' * int(prob * 20)
            print(f"  {class_name:12s}: {bar:20s} {prob:.1%}")


def print_summary(predictions: List[Prediction]):
    """Print summary of batch predictions."""
    if not predictions:
        print("No predictions to summarize.")
        return
    
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(predictions)}")
    
    # Class distribution
    class_counts = {}
    for pred in predictions:
        class_counts[pred.predicted_class] = class_counts.get(pred.predicted_class, 0) + 1
    
    print(f"\nClass distribution:")
    for class_name in ['Pronation', 'Neutral', 'Supination']:
        count = class_counts.get(class_name, 0)
        pct = count / len(predictions) * 100
        print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    # Confidence statistics
    confidences = [p.confidence for p in predictions]
    print(f"\nConfidence statistics:")
    print(f"  Mean: {np.mean(confidences):.1%}")
    print(f"  Min:  {np.min(confidences):.1%}")
    print(f"  Max:  {np.max(confidences):.1%}")
    
    # Low confidence warnings
    low_conf = [p for p in predictions if p.confidence < 0.5]
    if low_conf:
        print(f"\n⚠️  {len(low_conf)} predictions with low confidence (<50%):")
        for p in low_conf[:5]:
            print(f"    {Path(p.image_path).name}: {p.predicted_class} ({p.confidence:.1%})")
        if len(low_conf) > 5:
            print(f"    ... and {len(low_conf) - 5} more")


# ============================================================================
# Main Inference Function
# ============================================================================

def predict(
    image: Optional[str] = None,
    input_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    output: Optional[str] = None,
    visualize: bool = False,
    recursive: bool = False,
    verbose: bool = False,
):
    """
    Main inference function.
    
    Args:
        image: Path to single image file
        input_dir: Path to directory containing images
        model_path: Path to model checkpoint
        output: Path to save results (JSON or CSV based on extension)
        visualize: Whether to show/save visualizations
        recursive: Whether to search subdirectories
        verbose: Whether to print detailed output
    """
    
    logger.info("=" * 60)
    logger.info("ANKLE ALIGNMENT INFERENCE")
    logger.info("=" * 60)
    
    # Initialize predictor
    predictor = AnkleAlignmentPredictor(
        model_path=Path(model_path) if model_path else None,
        device=DEVICE
    )
    
    predictions = []
    
    # Single image prediction
    if image:
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"\nProcessing image: {image_path}")
        pred = predictor.predict(image_path)
        predictions.append(pred)
        
        print_prediction(pred, verbose=verbose)
        
        if visualize:
            visualize_prediction(image_path, pred)
    
    # Directory prediction
    elif input_dir:
        input_path = Path(input_dir)
        logger.info(f"\nProcessing directory: {input_path}")
        
        predictions = predictor.predict_directory(input_path, recursive=recursive)
        
        if verbose:
            for pred in predictions:
                print_prediction(pred, verbose=True)
        
        print_summary(predictions)
        
        if visualize and predictions:
            visualize_batch(predictions)
    
    else:
        raise ValueError("Please provide either --image or --input-dir")
    
    # Save results
    if output and predictions:
        output_path = Path(output)
        
        if output_path.suffix.lower() == '.csv':
            save_results_csv(predictions, output_path)
        else:
            save_results_json(predictions, output_path)
    
    logger.info("\nInference complete!")
    
    return predictions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference on ankle alignment images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python src/04-inference.py --image data/test_image.jpg

  # Directory of images
  python src/04-inference.py --input-dir data/new_images/

  # With visualization and output
  python src/04-inference.py --image img.jpg --visualize --output result.json

  # Batch with CSV output
  python src/04-inference.py --input-dir images/ --output predictions.csv
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--image', '-i',
        type=str,
        help='Path to a single image file'
    )
    input_group.add_argument(
        '--input-dir', '-d',
        type=str,
        help='Path to directory containing images'
    )
    
    # Model options
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default=None,
        help='Path to model checkpoint (default: models/final_model.pth)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save results (.json or .csv)'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Show/save visualization of predictions'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Search subdirectories when using --input-dir'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output for each prediction'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    predict(
        image=args.image,
        input_dir=args.input_dir,
        model_path=args.model_path,
        output=args.output,
        visualize=args.visualize,
        recursive=args.recursive,
        verbose=args.verbose,
    )
