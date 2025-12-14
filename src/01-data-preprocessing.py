"""
Data Preprocessing Script for Ankle Alignment Classification

This script performs the complete data pipeline:
1. DATA FETCHING: Scans labeler folders and extracts images + labels from Label Studio JSON
2. DATA PREPARATION: Converts all images to PNG format, creates labels CSV
3. DATA CLEANSING: Extracts image properties, filters invalid/unlabeled images
4. DATA SPLITTING: Splits data into train/validation/test sets
5. IMAGE PROCESSING: Resizes images to consistent size for model training

Usage:
    python src/01-data-preprocessing.py [--size 224] [--no-aspect-ratio] [--skip-fetch]
"""

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Set

import requests
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    PROJECT_ROOT,
    DATA_DIR,
    ANKLEALIGN_DIR,
    LABELS_DIR,
    PREPARED_IMAGES_DIR,
    RAW_IMAGES_DIR,
    IMAGE_PROPERTIES_FILE,
    CLEANED_IMAGE_PROPERTIES_FILE,
    MERGED_LABELS_FILE,
    PROCESSED_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    RESIZE_INTERPOLATION,
    MAINTAIN_ASPECT_RATIO,
    PAD_COLOR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    STRATIFY_SPLIT,
    LABEL_TO_IDX,
)
from utils import setup_logger, log_section

logger = setup_logger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# URL for downloading the anklealign dataset
DATASET_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQB8kDcLEuTqQphHx7pv4Cw5AW7XMJp5MUbwortTASU223A?e=Uu6CTj&download=1"

# Label mapping from Hungarian to English
LABEL_MAP = {
    '1_Pronacio': 'Pronation',
    '1_Pronáló': 'Pronation',
    'pronation': 'Pronation',
    'Pronation': 'Pronation',
    '2_Neutralis': 'Neutral',
    'neutral': 'Neutral',
    '2_Neutral': 'Neutral',
    'Neutral': 'Neutral',
    '3_Szupinacio': 'Supination',
    '3_Szupináló': 'Supination',
    'supination': 'Supination',
    'Supination': 'Supination',
}

# Image extensions to process
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff'}

# Pattern for 6 alphanumeric character labeler IDs
LABELER_ID_PATTERN = re.compile(r'^[A-Za-z0-9]{6}$')

# Interpolation method mapping
INTERPOLATION_METHODS = {
    'nearest': Image.Resampling.NEAREST,
    'bilinear': Image.Resampling.BILINEAR,
    'bicubic': Image.Resampling.BICUBIC,
    'lanczos': Image.Resampling.LANCZOS,
}


# ============================================================================
# PHASE 0: DOWNLOAD AND EXTRACT DATASET
# ============================================================================

def download_and_extract_dataset(
    url: str = DATASET_URL,
    output_dir: Path = PROJECT_ROOT,
    zip_filename: str = "anklealign.zip",
    force_download: bool = False
) -> bool:
    """
    Download and extract the anklealign dataset from URL.
    
    Args:
        url: URL to download the zip file from
        output_dir: Directory to extract files to
        zip_filename: Name of the zip file
        force_download: Re-download even if already exists
        
    Returns:
        True if successful, False otherwise
    """
    zip_path = output_dir / zip_filename
    extract_marker = output_dir / ".anklealign_extracted"
    
    # Check if already extracted
    if not force_download and extract_marker.exists():
        logger.info("Dataset already downloaded and extracted (found marker file)")
        return True
    
    # Check if anklealign directory already exists
    if not force_download and ANKLEALIGN_DIR.exists():
        labeler_folders = find_labeler_folders(ANKLEALIGN_DIR)
        if labeler_folders:
            logger.info(f"Dataset already exists with {len(labeler_folders)} labeler folders")
            return True
    
    logger.info("=" * 60)
    logger.info("PHASE 0: DOWNLOAD AND EXTRACT DATASET")
    logger.info("=" * 60)
    
    try:
        # Download the zip file
        logger.info(f"\nDownloading dataset from URL...")
        logger.info(f"This may take a few minutes depending on your connection...")
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        
        # Save with progress bar
        with open(zip_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # No content-length header, download without progress
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        logger.info(f"Downloaded to: {zip_path}")
        
        # Extract the zip file
        logger.info(f"\nExtracting to: {output_dir}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files for progress
            file_list = zip_ref.namelist()
            
            for file in tqdm(file_list, desc="Extracting"):
                zip_ref.extract(file, output_dir)
        
        logger.info("Extraction complete!")
        
        # Create marker file
        extract_marker.touch()
        
        # Optionally remove the zip file to save space
        # zip_path.unlink()
        # logger.info(f"Removed zip file: {zip_path}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid zip file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error during download/extract: {e}")
        return False


# ============================================================================
# PHASE 1: DATA FETCHING (From Labeler Folders)
# ============================================================================

def find_labeler_folders(anklealign_path: Path) -> List[Path]:
    """
    Find all labeler folders with 6 alphanumeric character names.
    
    Args:
        anklealign_path: Path to anklealign directory
        
    Returns:
        List of labeler folder paths
    """
    labeler_folders = []
    
    if not anklealign_path.exists():
        logger.warning(f"Anklealign directory not found: {anklealign_path}")
        return labeler_folders
    
    for folder in anklealign_path.iterdir():
        if folder.is_dir() and LABELER_ID_PATTERN.match(folder.name):
            # Exclude special folders
            if folder.name.lower() not in {'consensus', 'sample'}:
                labeler_folders.append(folder)
    
    return sorted(labeler_folders, key=lambda x: x.name.upper())


def extract_image_name_from_upload(file_upload: str) -> str:
    """
    Extract original image name from Label Studio file_upload path.
    Format: "uuid-original_name.ext" -> "original_name.ext"
    """
    filename = Path(file_upload).name
    # Remove UUID prefix (8 hex chars + hyphen)
    uuid_pattern = r'^[a-f0-9]{8}-(.+)$'
    match = re.match(uuid_pattern, filename, re.IGNORECASE)
    if match:
        return match.group(1)
    return filename


def parse_labeler_json(json_path: Path) -> Dict[str, str]:
    """
    Parse Label Studio JSON and return {image_name: label} mapping.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Dictionary mapping image filenames to labels
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Error reading JSON {json_path}: {e}")
        return {}
    
    labels = {}
    
    for task in data:
        file_upload = task.get('file_upload', '')
        image_name = extract_image_name_from_upload(file_upload)
        
        # Extract label from annotations
        annotations = task.get('annotations', [])
        if annotations:
            results = annotations[0].get('result', [])
            if results:
                choices = results[0].get('value', {}).get('choices', [])
                if choices:
                    raw_label = choices[0]
                    labels[image_name] = LABEL_MAP.get(raw_label, raw_label)
    
    return labels


def find_image_label(
    image_name: str,
    labels_dict: Dict[str, str],
    labeler_id: str
) -> Optional[str]:
    """
    Try to find a label for an image, handling various naming issues.
    
    Args:
        image_name: Name of the image file
        labels_dict: Dictionary of labels from JSON
        labeler_id: Labeler ID for prefix handling
        
    Returns:
        Label if found, None otherwise
    """
    # Direct match
    if image_name in labels_dict:
        return labels_dict[image_name]
    
    # Case-insensitive match
    image_lower = image_name.lower()
    for name, label in labels_dict.items():
        if name.lower() == image_lower:
            return label
    
    # Try without labeler prefix if it was added
    if image_name.lower().startswith(labeler_id.lower() + '_'):
        stripped = image_name[len(labeler_id) + 1:]
        if stripped in labels_dict:
            return labels_dict[stripped]
        for name, label in labels_dict.items():
            if name.lower() == stripped.lower():
                return label
    
    # Try matching by finding the image name in labels keys
    stem = Path(image_name).stem
    for name, label in labels_dict.items():
        if stem.lower() in name.lower() or name.lower() in stem.lower():
            return label
    
    return None


def convert_and_save_as_png(source_path: Path, dest_path: Path) -> bool:
    """
    Open image and save as PNG.
    
    Args:
        source_path: Source image path
        dest_path: Destination PNG path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(source_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            img.save(dest_path, 'PNG')
        return True
    except Exception as e:
        logger.warning(f"Error converting {source_path}: {e}")
        return False


def process_standard_labeler_folder(
    folder: Path,
    labels_dict: Dict[str, str],
    output_path: Path
) -> List[Dict]:
    """
    Process a standard labeler folder with images at root level.
    
    Returns:
        List of label records for CSV
    """
    labeler_id = folder.name
    label_rows = []
    
    for image_file in folder.iterdir():
        if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        
        # Find label for this image
        label = find_image_label(image_file.name, labels_dict, labeler_id)
        
        if label is None:
            label = 'Unlabeled'
        
        # Determine output filename
        original_stem = image_file.stem
        new_filename = f"{labeler_id}_{original_stem}.png"
        dest_path = output_path / new_filename
        
        # Convert and save as PNG
        if convert_and_save_as_png(image_file, dest_path):
            label_rows.append({
                'filename': new_filename,
                'original_filename': image_file.name,
                'label': label,
                'labeler_id': labeler_id
            })
    
    return label_rows


def process_subdirectory_labeler(
    folder: Path,
    output_path: Path,
    folder_label_map: Dict[str, str]
) -> List[Dict]:
    """
    Process a labeler folder with images organized in subdirectories.
    Used for special cases like NC1O2T where labels are folder names.
    
    Args:
        folder: Labeler folder path
        output_path: Output directory for images
        folder_label_map: Mapping of folder names to labels
        
    Returns:
        List of label records for CSV
    """
    labeler_id = folder.name
    label_rows = []
    
    for folder_name, label in folder_label_map.items():
        subfolder = folder / folder_name
        if not subfolder.exists():
            continue
        
        for image_file in subfolder.iterdir():
            if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            
            original_stem = image_file.stem
            new_filename = f"{labeler_id}_{original_stem}.png"
            dest_path = output_path / new_filename
            
            if convert_and_save_as_png(image_file, dest_path):
                label_rows.append({
                    'filename': new_filename,
                    'original_filename': image_file.name,
                    'label': label,
                    'labeler_id': labeler_id
                })
    
    return label_rows


def fetch_data(
    anklealign_path: Path = ANKLEALIGN_DIR,
    output_path: Path = PREPARED_IMAGES_DIR,
    labels_path: Path = LABELS_DIR,
    clean_output: bool = False
) -> pd.DataFrame:
    """
    Phase 1: Fetch data from labeler folders.
    
    Scans labeler directories, extracts labels from JSON files,
    converts images to PNG, and creates labels CSV.
    
    Args:
        anklealign_path: Path to anklealign directory with labeler folders
        output_path: Path to save prepared images
        labels_path: Path to save label CSV files
        clean_output: Whether to clean existing output directories
        
    Returns:
        DataFrame with merged labels
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA FETCHING")
    logger.info("=" * 60)
    
    # Find labeler folders
    labeler_folders = find_labeler_folders(anklealign_path)
    
    if not labeler_folders:
        logger.warning("No labeler folders found. Checking if data already prepared...")
        if MERGED_LABELS_FILE.exists():
            logger.info(f"Loading existing labels from: {MERGED_LABELS_FILE}")
            return pd.read_csv(MERGED_LABELS_FILE)
        else:
            raise FileNotFoundError(
                f"No labeler folders found in {anklealign_path} and no existing labels file."
            )
    
    logger.info(f"Found {len(labeler_folders)} labeler folders")
    
    # Clean output if requested
    if clean_output:
        if output_path.exists():
            shutil.rmtree(output_path)
        if labels_path.exists():
            shutil.rmtree(labels_path)
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    # Statistics tracking
    stats = {
        'total_images': 0,
        'by_label': {'Pronation': 0, 'Neutral': 0, 'Supination': 0, 'Unlabeled': 0},
        'no_json': 0,
        'errors': 0
    }
    
    all_label_rows = []
    
    # Process each labeler folder
    for folder in tqdm(labeler_folders, desc="Processing labelers"):
        labeler_id = folder.name
        
        # Special handling for NC1O2T (images in subdirectories)
        if labeler_id == 'NC1O2T':
            folder_label_map = {
                'normal': 'Neutral',
                'supination': 'Supination',
                'pronation': 'Pronation'
            }
            
            label_rows = process_subdirectory_labeler(
                folder, output_path, folder_label_map
            )
        else:
            # Standard labeler: find JSON and process
            json_files = list(folder.glob('*.json'))
            
            if not json_files:
                logger.warning(f"No JSON file found for {labeler_id}")
                stats['no_json'] += 1
                continue
            
            # Load labels from JSON
            labels_dict = parse_labeler_json(json_files[0])
            
            if not labels_dict:
                logger.warning(f"No labels extracted from {json_files[0].name}")
                continue
            
            label_rows = process_standard_labeler_folder(
                folder, labels_dict, output_path
            )
        
        # Update statistics
        for row in label_rows:
            label = row['label']
            if label in stats['by_label']:
                stats['by_label'][label] += 1
            else:
                stats['by_label']['Unlabeled'] += 1
        
        stats['total_images'] += len(label_rows)
        all_label_rows.extend(label_rows)
        
        # Save per-labeler CSV
        if label_rows:
            df_labeler = pd.DataFrame(label_rows)
            csv_path = labels_path / f"{labeler_id}_labels.csv"
            df_labeler.to_csv(csv_path, index=False)
    
    # Create merged labels DataFrame
    df_merged = pd.DataFrame(all_label_rows)
    
    # Save merged labels
    df_merged.to_csv(MERGED_LABELS_FILE, index=False)
    logger.info(f"Saved merged labels to: {MERGED_LABELS_FILE}")
    
    # Print statistics
    logger.info("\nFetching Statistics:")
    logger.info(f"  Total images processed: {stats['total_images']}")
    logger.info(f"  Labelers without JSON: {stats['no_json']}")
    logger.info("\n  By Label:")
    for label, count in stats['by_label'].items():
        pct = count / stats['total_images'] * 100 if stats['total_images'] > 0 else 0
        logger.info(f"    {label}: {count} ({pct:.1f}%)")
    
    return df_merged


# ============================================================================
# PHASE 2: DATA CLEANSING (Extract Properties, Filter, Remove Duplicates)
# ============================================================================

def compute_file_hash(file_path: Path) -> Optional[str]:
    """
    Compute MD5 hash of a file for duplicate detection.
    
    Args:
        file_path: Path to the file
        
    Returns:
        MD5 hash as hex string, or None if error
    """
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"Error computing hash for {file_path}: {e}")
        return None


def get_image_properties(image_path: Path) -> Optional[Dict]:
    """
    Extract properties from an image file.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary of properties or None if error
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            file_size = image_path.stat().st_size / 1024  # KB
            
            return {
                'width': width,
                'height': height,
                'aspect_ratio': round(width / height, 4),
                'pixels': width * height,
                'mode': mode,
                'file_size_kb': round(file_size, 2)
            }
    except Exception as e:
        logger.warning(f"Error reading properties from {image_path}: {e}")
        return None


def cleanse_data(
    df_labels: pd.DataFrame,
    images_path: Path = PREPARED_IMAGES_DIR,
    min_dimension: int = 100,
    max_dimension: int = 10000,
    min_file_size_kb: float = 5.0,
    remove_duplicates: bool = True
) -> pd.DataFrame:
    """
    Phase 2: Cleanse and validate the data.
    
    Extracts image properties, detects and removes duplicates,
    validates dimensions, and filters out problematic images.
    
    Args:
        df_labels: DataFrame with labels
        images_path: Path to prepared images
        min_dimension: Minimum allowed width/height
        max_dimension: Maximum allowed width/height
        min_file_size_kb: Minimum file size in KB
        remove_duplicates: Whether to remove duplicate images
        
    Returns:
        Cleaned DataFrame with image properties
    """
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: DATA CLEANSING")
    logger.info("=" * 60)
    
    initial_count = len(df_labels)
    logger.info(f"Starting with {initial_count} images")
    
    # Extract image properties and compute hashes
    logger.info("\nExtracting image properties and computing hashes...")
    properties_list = []
    
    for _, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="Analyzing"):
        image_path = images_path / row['filename']
        
        if not image_path.exists():
            logger.warning(f"Image not found: {row['filename']}")
            continue
        
        props = get_image_properties(image_path)
        
        if props:
            props['filename'] = row['filename']
            props['label'] = row['label']
            props['labeler_id'] = row.get('labeler_id', row['filename'].split('_')[0])
            
            # Compute file hash for duplicate detection
            if remove_duplicates:
                props['file_hash'] = compute_file_hash(image_path)
            
            properties_list.append(props)
    
    df_props = pd.DataFrame(properties_list)
    logger.info(f"Successfully analyzed {len(df_props)} images")
    
    # Save raw image properties
    df_props.to_csv(IMAGE_PROPERTIES_FILE, index=False)
    logger.info(f"Saved image properties to: {IMAGE_PROPERTIES_FILE}")
    
    # Apply filters
    logger.info("\nApplying data quality filters...")
    
    # Track removal reasons
    removal_stats = {}
    
    # 1. Filter unlabeled images
    valid_labels = list(LABEL_TO_IDX.keys())
    mask_labeled = df_props['label'].isin(valid_labels)
    n_unlabeled = (~mask_labeled).sum()
    removal_stats['Unlabeled'] = n_unlabeled
    logger.info(f"  Unlabeled images: {n_unlabeled}")
    
    df_clean = df_props[mask_labeled].copy()
    
    # 2. Remove exact duplicates (by file hash)
    if remove_duplicates and 'file_hash' in df_clean.columns:
        before_dedup = len(df_clean)
        
        # Find duplicates
        hash_counts = df_clean['file_hash'].value_counts()
        duplicate_hashes = hash_counts[hash_counts > 1]
        
        if len(duplicate_hashes) > 0:
            logger.info(f"\n  Duplicate Detection:")
            logger.info(f"    Found {len(duplicate_hashes)} sets of exact duplicate images")
            
            # Log duplicate sets (limit output for readability)
            logged_count = 0
            for file_hash in duplicate_hashes.index:
                if logged_count >= 5:
                    remaining = len(duplicate_hashes) - logged_count
                    logger.info(f"    ... and {remaining} more duplicate sets")
                    break
                
                dup_files = df_clean[df_clean['file_hash'] == file_hash][['filename', 'label']]
                count = len(dup_files)
                
                # Check if duplicates have conflicting labels
                unique_labels = dup_files['label'].unique()
                if len(unique_labels) > 1:
                    logger.warning(f"    ⚠️ Duplicate set with CONFLICTING labels ({count} files):")
                else:
                    logger.info(f"    Duplicate set ({count} files):")
                
                for _, dup_row in dup_files.iterrows():
                    logger.info(f"      - {dup_row['filename']} [{dup_row['label']}]")
                
                logged_count += 1
            
            # Remove duplicates, keeping first occurrence
            df_clean = df_clean.drop_duplicates(subset=['file_hash'], keep='first')
            n_removed = before_dedup - len(df_clean)
            removal_stats['Exact Duplicates'] = n_removed
            logger.info(f"\n  Removed {n_removed} duplicate images (kept first occurrence)")
        else:
            logger.info(f"  No exact duplicates found")
            removal_stats['Exact Duplicates'] = 0
    
    # 3. Filter by dimension constraints
    before_dims = len(df_clean)
    mask_dimensions = (
        (df_clean['width'] >= min_dimension) &
        (df_clean['width'] <= max_dimension) &
        (df_clean['height'] >= min_dimension) &
        (df_clean['height'] <= max_dimension)
    )
    n_bad_dims = (~mask_dimensions).sum()
    removal_stats['Invalid Dimensions'] = n_bad_dims
    logger.info(f"  Invalid dimensions (<{min_dimension}px or >{max_dimension}px): {n_bad_dims}")
    
    df_clean = df_clean[mask_dimensions]
    
    # 4. Filter by file size
    before_size = len(df_clean)
    mask_size = df_clean['file_size_kb'] >= min_file_size_kb
    n_small = (~mask_size).sum()
    removal_stats['Too Small Files'] = n_small
    logger.info(f"  Too small files (<{min_file_size_kb}KB): {n_small}")
    
    df_clean = df_clean[mask_size].copy()
    
    # Summary
    final_count = len(df_clean)
    total_removed = initial_count - final_count
    
    logger.info("\n" + "-" * 40)
    logger.info("CLEANSING SUMMARY")
    logger.info("-" * 40)
    logger.info(f"  Original images: {initial_count}")
    logger.info(f"  Removed:")
    for reason, count in removal_stats.items():
        if count > 0:
            logger.info(f"    - {reason}: {count}")
    logger.info(f"  Total removed: {total_removed} ({total_removed/initial_count*100:.1f}%)")
    logger.info(f"  Remaining: {final_count} images")
    
    # Save cleaned properties
    # Remove hash column before saving (not needed for training)
    export_columns = [col for col in df_clean.columns if col != 'file_hash']
    df_clean[export_columns].to_csv(CLEANED_IMAGE_PROPERTIES_FILE, index=False)
    logger.info(f"\nSaved cleaned properties to: {CLEANED_IMAGE_PROPERTIES_FILE}")
    
    # Print label distribution
    logger.info("\nCleaned label distribution:")
    for label in sorted(df_clean['label'].unique()):
        count = len(df_clean[df_clean['label'] == label])
        pct = count / len(df_clean) * 100
        logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    return df_clean


# ============================================================================
# PHASE 3: IMAGE RESIZING AND SPLITTING
# ============================================================================

def resize_image(
    image: Image.Image,
    target_size: Tuple[int, int],
    maintain_aspect_ratio: bool = True,
    interpolation: str = 'bilinear',
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Resize an image to the target size.
    
    Args:
        image: PIL Image to resize
        target_size: Target (width, height)
        maintain_aspect_ratio: If True, resize maintaining aspect ratio and pad
        interpolation: Interpolation method
        pad_color: RGB color for padding
        
    Returns:
        Resized PIL Image
    """
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    target_w, target_h = target_size
    resample = INTERPOLATION_METHODS.get(interpolation, Image.Resampling.BILINEAR)
    
    if maintain_aspect_ratio:
        # Calculate scaling factor
        orig_w, orig_h = image.size
        scale = min(target_w / orig_w, target_h / orig_h)
        
        # New dimensions
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize image
        resized = image.resize((new_w, new_h), resample=resample)
        
        # Create padded image
        padded = Image.new('RGB', target_size, pad_color)
        
        # Paste in center
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        padded.paste(resized, (paste_x, paste_y))
        
        return padded
    else:
        return image.resize(target_size, resample=resample)


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    stratify_col = df['label'] if stratify else None
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=stratify_col
    )
    
    # Second split: separate train and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    stratify_col = train_val_df['label'] if stratify else None
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
        stratify=stratify_col
    )
    
    return train_df, val_df, test_df


def process_and_save_images(
    df: pd.DataFrame,
    source_dir: Path,
    output_dir: Path,
    target_size: Tuple[int, int],
    maintain_aspect_ratio: bool = True,
    interpolation: str = 'bilinear',
    pad_color: Tuple[int, int, int] = (0, 0, 0)
) -> List[Dict]:
    """
    Process images and save to output directory organized by label.
    
    Returns:
        List of processed image records
    """
    processed_records = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {output_dir.name}"):
        filename = row['filename']
        label = row['label']
        
        if label not in LABEL_TO_IDX:
            continue
        
        source_path = source_dir / filename
        
        if not source_path.exists():
            logger.warning(f"Source file not found: {source_path}")
            continue
        
        try:
            image = Image.open(source_path)
            
            processed = resize_image(
                image,
                target_size,
                maintain_aspect_ratio=maintain_aspect_ratio,
                interpolation=interpolation,
                pad_color=pad_color
            )
            
            # Create output directory for this label
            label_dir = output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            
            # Save processed image
            output_path = label_dir / filename
            processed.save(output_path, 'PNG', optimize=True)
            
            processed_records.append({
                'filename': filename,
                'label': label,
                'label_idx': LABEL_TO_IDX[label],
                'relative_path': f"{label}/{filename}",
                'original_width': row.get('width', image.size[0]),
                'original_height': row.get('height', image.size[1]),
                'processed_width': target_size[0],
                'processed_height': target_size[1]
            })
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
    
    return processed_records


def create_manifest(records: List[Dict], output_path: Path) -> None:
    """Save a manifest CSV file for a split."""
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved manifest: {output_path} ({len(records)} images)")


def print_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """Print statistics about the data splits."""
    total = len(train_df) + len(val_df) + len(test_df)
    
    logger.info(f"\nTotal images: {total}")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df)} ({len(val_df)/total*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df)} ({len(test_df)/total*100:.1f}%)")
    
    logger.info("\nLabel distribution by split:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        logger.info(f"\n  {split_name}:")
        for label in sorted(split_df['label'].unique()):
            count = len(split_df[split_df['label'] == label])
            pct = count / len(split_df) * 100
            logger.info(f"    {label}: {count} ({pct:.1f}%)")


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess(
    image_size: Tuple[int, int] = IMAGE_SIZE,
    maintain_aspect_ratio: bool = MAINTAIN_ASPECT_RATIO,
    interpolation: str = RESIZE_INTERPOLATION,
    pad_color: Tuple[int, int, int] = PAD_COLOR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = RANDOM_SEED,
    stratify: bool = STRATIFY_SPLIT,
    skip_download: bool = False,
    force_download: bool = False,
    skip_fetch: bool = False,
    remove_duplicates: bool = True,
    clean_output: bool = True
) -> None:
    """
    Main preprocessing pipeline.
    
    Runs all phases:
    0. Download and Extract Dataset (from URL)
    1. Data Fetching (from labeler folders)
    2. Data Cleansing (extract properties, remove duplicates, filter)
    3. Splitting and Processing (resize, organize)
    
    Args:
        image_size: Target image size (width, height)
        maintain_aspect_ratio: Whether to maintain aspect ratio when resizing
        interpolation: Interpolation method for resizing
        pad_color: Padding color when maintaining aspect ratio
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
        stratify: Whether to stratify split by label
        skip_download: Skip Phase 0 (download) if dataset already exists
        force_download: Force re-download even if dataset exists
        skip_fetch: Skip Phase 1 if data already prepared
        remove_duplicates: Whether to remove duplicate images
        clean_output: Whether to clean existing output directories
    """
    logger.info("=" * 60)
    logger.info("ANKLE ALIGNMENT DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    # ========================================
    # PHASE 0: DOWNLOAD AND EXTRACT
    # ========================================
    if not skip_download:
        success = download_and_extract_dataset(
            url=DATASET_URL,
            output_dir=PROJECT_ROOT,
            force_download=force_download
        )
        
        if not success and not ANKLEALIGN_DIR.exists():
            raise RuntimeError(
                "Failed to download dataset and no existing data found. "
                "Please check your internet connection or download manually."
            )
    else:
        logger.info("\nSkipping Phase 0 (Download)")
    
    # ========================================
    # PHASE 1: DATA FETCHING
    # ========================================
    if skip_fetch:
        logger.info("\nSkipping Phase 1 (Data Fetching)")
        
        # Load existing labels
        if MERGED_LABELS_FILE.exists():
            df_labels = pd.read_csv(MERGED_LABELS_FILE)
            logger.info(f"Loaded {len(df_labels)} records from {MERGED_LABELS_FILE}")
        elif CLEANED_IMAGE_PROPERTIES_FILE.exists():
            df_labels = pd.read_csv(CLEANED_IMAGE_PROPERTIES_FILE)
            logger.info(f"Loaded {len(df_labels)} records from {CLEANED_IMAGE_PROPERTIES_FILE}")
        elif IMAGE_PROPERTIES_FILE.exists():
            df_labels = pd.read_csv(IMAGE_PROPERTIES_FILE)
            logger.info(f"Loaded {len(df_labels)} records from {IMAGE_PROPERTIES_FILE}")
        else:
            raise FileNotFoundError(
                "No existing labels found. Run without --skip-fetch to fetch data."
            )
    else:
        df_labels = fetch_data(
            anklealign_path=ANKLEALIGN_DIR,
            output_path=PREPARED_IMAGES_DIR,
            labels_path=LABELS_DIR,
            clean_output=clean_output
        )
    
    # ========================================
    # PHASE 2: DATA CLEANSING
    # ========================================
    if skip_fetch and CLEANED_IMAGE_PROPERTIES_FILE.exists():
        logger.info("\nSkipping Phase 2 (Data Cleansing) - using existing cleaned data")
        df_clean = pd.read_csv(CLEANED_IMAGE_PROPERTIES_FILE)
    else:
        df_clean = cleanse_data(
            df_labels,
            images_path=PREPARED_IMAGES_DIR,
            remove_duplicates=remove_duplicates
        )
    
    # ========================================
    # PHASE 3: SPLITTING AND PROCESSING
    # ========================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: SPLITTING AND PROCESSING")
    logger.info("=" * 60)
    
    # Split data
    logger.info(f"\nSplitting data (train={train_ratio}, val={val_ratio}, test={test_ratio})")
    train_df, val_df, test_df = split_data(
        df_clean,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify=stratify,
        random_seed=random_seed
    )
    
    print_split_statistics(train_df, val_df, test_df)
    
    # Clean processed directory if requested
    if clean_output and PROCESSED_DIR.exists():
        logger.info(f"\nCleaning output directory: {PROCESSED_DIR}")
        shutil.rmtree(PROCESSED_DIR)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    logger.info(f"\nProcessing images to size {image_size[0]}x{image_size[1]}")
    logger.info(f"Maintain aspect ratio: {maintain_aspect_ratio}")
    logger.info(f"Interpolation: {interpolation}")
    
    all_records = {}
    
    for split_name, split_df, split_dir in [
        ('train', train_df, TRAIN_DIR),
        ('val', val_df, VAL_DIR),
        ('test', test_df, TEST_DIR)
    ]:
        logger.info(f"\nProcessing {split_name} split...")
        
        records = process_and_save_images(
            split_df,
            PREPARED_IMAGES_DIR,
            split_dir,
            target_size=image_size,
            maintain_aspect_ratio=maintain_aspect_ratio,
            interpolation=interpolation,
            pad_color=pad_color
        )
        
        all_records[split_name] = records
        
        # Create manifest
        manifest_path = PROCESSED_DIR / f"{split_name}_manifest.csv"
        create_manifest(records, manifest_path)
    
    # Create combined manifest
    all_data = []
    for split_name, records in all_records.items():
        for record in records:
            record['split'] = split_name
            all_data.append(record)
    
    combined_manifest = PROCESSED_DIR / "all_manifest.csv"
    create_manifest(all_data, combined_manifest)
    
    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nOutput directory: {PROCESSED_DIR}")
    logger.info(f"Image size: {image_size[0]}x{image_size[1]}")
    logger.info(f"\nProcessed images:")
    for split_name, records in all_records.items():
        logger.info(f"  {split_name}: {len(records)} images")
    logger.info(f"\nTotal: {sum(len(r) for r in all_records.values())} images")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Complete data preprocessing pipeline for ankle alignment classification'
    )
    
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=IMAGE_SIZE[0],
        help=f'Target image size (square, default: {IMAGE_SIZE[0]})'
    )
    
    parser.add_argument(
        '--no-aspect-ratio',
        action='store_true',
        help='Disable aspect ratio preservation (stretch images)'
    )
    
    parser.add_argument(
        '--interpolation', '-i',
        type=str,
        choices=['nearest', 'bilinear', 'bicubic', 'lanczos'],
        default=RESIZE_INTERPOLATION,
        help=f'Interpolation method (default: {RESIZE_INTERPOLATION})'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=TRAIN_RATIO,
        help=f'Training set ratio (default: {TRAIN_RATIO})'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=VAL_RATIO,
        help=f'Validation set ratio (default: {VAL_RATIO})'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=TEST_RATIO,
        help=f'Test set ratio (default: {TEST_RATIO})'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed (default: {RANDOM_SEED})'
    )
    
    parser.add_argument(
        '--no-stratify',
        action='store_true',
        help='Disable stratified splitting'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip dataset download (use existing anklealign folder)'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download even if dataset already exists'
    )
    
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip data fetching (use existing prepared images)'
    )
    
    parser.add_argument(
        '--keep-duplicates',
        action='store_true',
        help='Keep duplicate images (do not remove based on file hash)'
    )
    
    parser.add_argument(
        '--keep-existing',
        action='store_true',
        help='Keep existing output directories (do not clean)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    preprocess(
        image_size=(args.size, args.size),
        maintain_aspect_ratio=not args.no_aspect_ratio,
        interpolation=args.interpolation,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        stratify=not args.no_stratify,
        skip_download=args.skip_download,
        force_download=args.force_download,
        skip_fetch=args.skip_fetch,
        remove_duplicates=not args.keep_duplicates,
        clean_output=not args.keep_existing
    )
