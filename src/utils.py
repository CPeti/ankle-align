# Utility functions
# Common helper functions used across the project.
import logging
import sys
from pathlib import Path

# Default log file path
LOG_DIR = Path(__file__).parent.parent / "log"
LOG_FILE = LOG_DIR / "run.log"

# Global flag to track if root logger is configured
_logger_configured = False


def setup_logger(name: str = __name__, log_file: Path = None) -> logging.Logger:
    """
    Sets up a logger that outputs to both console (stdout) and a log file.
    
    This ensures all training, evaluation, and preprocessing logs are captured
    for grading and debugging purposes.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        log_file: Optional custom log file path. Defaults to log/run.log
        
    Returns:
        Configured logger instance
    """
    global _logger_configured
    
    # Use default log file if not specified
    if log_file is None:
        log_file = LOG_FILE
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get the logger
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (stdout for Docker capture)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (persistent log file)
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    return logger


def get_log_file_path() -> Path:
    """
    Get the path to the log file.
    
    Returns:
        Path to log/run.log
    """
    return LOG_FILE


def clear_log_file():
    """
    Clear the log file contents (useful for fresh runs).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("")


def log_separator(logger: logging.Logger, char: str = "=", length: int = 70):
    """
    Log a separator line for visual clarity.
    
    Args:
        logger: Logger instance
        char: Character to use for separator
        length: Length of separator line
    """
    logger.info(char * length)


def log_section(logger: logging.Logger, title: str, char: str = "="):
    """
    Log a section header with title.
    
    Args:
        logger: Logger instance
        title: Section title
        char: Character to use for separator
    """
    logger.info("")
    logger.info(char * 70)
    logger.info(title)
    logger.info(char * 70)


def log_dict(logger: logging.Logger, data: dict, title: str = None, indent: int = 2):
    """
    Log a dictionary in a readable format.
    
    Args:
        logger: Logger instance
        data: Dictionary to log
        title: Optional title
        indent: Indentation spaces
    """
    if title:
        logger.info(f"{title}:")
    
    prefix = " " * indent
    for key, value in data.items():
        logger.info(f"{prefix}{key}: {value}")


def log_model_summary(logger: logging.Logger, model, title: str = "MODEL ARCHITECTURE"):
    """
    Log model architecture summary with parameter counts.
    
    Args:
        logger: Logger instance
        model: PyTorch model
        title: Section title
    """
    import torch.nn as nn
    
    log_section(logger, title)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    logger.info(f"Model Class: {model.__class__.__name__}")
    logger.info("")
    logger.info("Parameter Summary:")
    logger.info(f"  Total parameters:         {total_params:>12,}")
    logger.info(f"  Trainable parameters:     {trainable_params:>12,}")
    logger.info(f"  Non-trainable parameters: {non_trainable_params:>12,}")
    
    # Log layer-wise parameter counts (top-level modules only)
    logger.info("")
    logger.info("Layer Summary:")
    for name, module in model.named_children():
        layer_params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info(f"  {name}: {layer_params:,} params ({trainable:,} trainable)")
    
    logger.info("")


def log_training_config(logger: logging.Logger, config: dict):
    """
    Log training configuration/hyperparameters.
    
    Args:
        logger: Logger instance
        config: Dictionary of hyperparameters
    """
    log_section(logger, "TRAINING CONFIGURATION")
    
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("")


def log_data_summary(logger: logging.Logger, train_size: int, val_size: int, 
                     test_size: int, class_distribution: dict = None):
    """
    Log data loading and preprocessing summary.
    
    Args:
        logger: Logger instance
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        class_distribution: Optional dict of class counts
    """
    log_section(logger, "DATA SUMMARY")
    
    total = train_size + val_size + test_size
    logger.info(f"  Total samples: {total}")
    logger.info(f"  Training samples:   {train_size:>6} ({train_size/total*100:.1f}%)")
    logger.info(f"  Validation samples: {val_size:>6} ({val_size/total*100:.1f}%)")
    logger.info(f"  Test samples:       {test_size:>6} ({test_size/total*100:.1f}%)")
    
    if class_distribution:
        logger.info("")
        logger.info("  Class Distribution (Training):")
        for class_name, count in class_distribution.items():
            pct = count / train_size * 100 if train_size > 0 else 0
            logger.info(f"    {class_name}: {count} ({pct:.1f}%)")
    
    logger.info("")


def log_epoch(logger: logging.Logger, epoch: int, total_epochs: int,
              train_loss: float, train_acc: float, 
              val_loss: float, val_acc: float,
              lr: float, is_best: bool = False):
    """
    Log training progress for an epoch.
    
    Args:
        logger: Logger instance
        epoch: Current epoch number
        total_epochs: Total number of epochs
        train_loss: Training loss
        train_acc: Training accuracy
        val_loss: Validation loss
        val_acc: Validation accuracy
        lr: Current learning rate
        is_best: Whether this is the best model so far
    """
    best_marker = " *BEST*" if is_best else ""
    logger.info(
        f"Epoch [{epoch:03d}/{total_epochs:03d}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
        f"LR: {lr:.2e}{best_marker}"
    )


def log_evaluation_results(logger: logging.Logger, metrics: dict, 
                           confusion_matrix=None, classification_report: str = None):
    """
    Log final evaluation results on test set.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of evaluation metrics
        confusion_matrix: Optional confusion matrix (numpy array)
        classification_report: Optional sklearn classification report string
    """
    log_section(logger, "FINAL EVALUATION RESULTS")
    
    logger.info("Test Set Metrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")
    
    if confusion_matrix is not None:
        logger.info("")
        logger.info("Confusion Matrix:")
        # Format confusion matrix nicely
        for row in confusion_matrix:
            logger.info("  " + "  ".join(f"{x:4d}" for x in row))
    
    if classification_report:
        logger.info("")
        logger.info("Classification Report:")
        for line in classification_report.split('\n'):
            if line.strip():
                logger.info(f"  {line}")
    
    logger.info("")


def load_config():
    """
    Load configuration (placeholder for future use).
    """
    pass
