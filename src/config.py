# Configuration settings for the ankle alignment classification project

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# Input paths - Raw labeler data
ANKLEALIGN_DIR = PROJECT_ROOT / "anklealign"
LABELS_DIR = DATA_DIR / "labels"

# Intermediate paths - Prepared images (after extracting from labeler folders)
PREPARED_IMAGES_DIR = DATA_DIR / "prepared_images"
RAW_IMAGES_DIR = PREPARED_IMAGES_DIR  # Alias for backwards compatibility
IMAGE_PROPERTIES_FILE = DATA_DIR / "image_properties.csv"
CLEANED_IMAGE_PROPERTIES_FILE = DATA_DIR / "cleaned_image_properties.csv"
MERGED_LABELS_FILE = LABELS_DIR / "merged_labels.csv"

# Output paths
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"

# Model paths
MODEL_SAVE_PATH = MODEL_DIR / "best_model.pth"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
# Target image size (width, height) - 224x224 is standard for many CNN models
# like ResNet, VGG, EfficientNet, etc.
IMAGE_SIZE = (224, 224)

# Alternative sizes for different model architectures:
# IMAGE_SIZE = (256, 256)  # Slightly larger
# IMAGE_SIZE = (299, 299)  # InceptionV3
# IMAGE_SIZE = (384, 384)  # ViT-Base
# IMAGE_SIZE = (512, 512)  # High resolution

# Image normalization (ImageNet statistics - commonly used for transfer learning)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Resizing interpolation method
# Options: 'bilinear', 'bicubic', 'lanczos', 'nearest'
RESIZE_INTERPOLATION = 'bilinear'

# Whether to maintain aspect ratio and pad (True) or stretch (False)
MAINTAIN_ASPECT_RATIO = True

# Padding color when maintaining aspect ratio (RGB values 0-255)
PAD_COLOR = (0, 0, 0)  # Black padding

# ============================================================================
# DATA SPLITTING
# ============================================================================
# Train/Validation/Test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Stratify split by label to maintain class distribution
STRATIFY_SPLIT = True

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Learning rate scheduler
LR_SCHEDULER = 'cosine'  # Options: 'cosine', 'step', 'plateau'
LR_STEP_SIZE = 10  # For step scheduler
LR_GAMMA = 0.1  # For step scheduler

# ============================================================================
# DATA AUGMENTATION
# ============================================================================
# Augmentations for training
AUGMENTATION_ENABLED = True
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.0  # Usually not useful for ankle images
ROTATION_DEGREES = 15
COLOR_JITTER = {
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': 0.1
}

# ============================================================================
# CLASSES
# ============================================================================
# Label mapping
LABEL_TO_IDX = {
    'Pronation': 0,
    'Neutral': 1,
    'Supination': 2
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}

NUM_CLASSES = len(LABEL_TO_IDX)

# ============================================================================
# DEVICE
# ============================================================================
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
