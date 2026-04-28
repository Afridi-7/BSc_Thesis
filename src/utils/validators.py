"""
Validation utilities for Hybrid Multimodal Lab Assistant system.

Provides validation functions for inputs, configurations, and data.
"""

import os
from pathlib import Path
from typing import Union, List, Optional
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def validate_image(image_path: Union[str, Path]) -> bool:
    """
    Validate that image file exists and is readable.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> validate_image('smear.jpg')
        True
    """
    try:
        image_path = Path(image_path)
        
        # Check existence
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return False
        
        # Check if file
        if not image_path.is_file():
            logger.warning(f"Not a file: {image_path}")
            return False
        
        # Try to open with PIL
        with Image.open(image_path) as img:
            img.verify()
        
        return True
        
    except Exception as e:
        logger.warning(f"Invalid image {image_path}: {e}")
        return False


def validate_image_batch(image_paths: List[Union[str, Path]]) -> List[Union[str, Path]]:
    """
    Validate batch of images and return only valid ones.
    
    Args:
        image_paths: List of image paths
        
    Returns:
        List of valid image paths
        
    Examples:
        >>> valid = validate_image_batch(['img1.jpg', 'img2.jpg'])
        >>> print(f"Valid: {len(valid)} images")
    """
    valid_paths = []
    
    for path in image_paths:
        if validate_image(path):
            valid_paths.append(path)
    
    if len(valid_paths) < len(image_paths):
        skipped = len(image_paths) - len(valid_paths)
        logger.warning(f"Skipped {skipped} invalid images")
    
    return valid_paths


def validate_model_file(model_path: Union[str, Path]) -> bool:
    """
    Validate that model file exists.
    
    Args:
        model_path: Path to model file (.pt, .pth, etc.)
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        FileNotFoundError: If model file not found
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.is_file():
        raise ValueError(f"Not a file: {model_path}")
    
    # Check file extension
    valid_extensions = {'.pt', '.pth', '.ckpt', '.weights'}
    if model_path.suffix not in valid_extensions:
        logger.warning(f"Unusual model file extension: {model_path.suffix}")
    
    return True


def validate_directory(dir_path: Union[str, Path], create: bool = False) -> bool:
    """
    Validate directory exists, optionally create if missing.
    
    Args:
        dir_path: Path to directory
        create: Whether to create directory if it doesn't exist
        
    Returns:
        True if valid/created, False otherwise
    """
    dir_path = Path(dir_path)
    
    if dir_path.exists():
        if not dir_path.is_dir():
            logger.error(f"Path exists but is not a directory: {dir_path}")
            return False
        return True
    
    if create:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            return False
    
    logger.error(f"Directory not found: {dir_path}")
    return False


def validate_array_shape(
    array: np.ndarray,
    expected_shape: Optional[tuple] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None
) -> bool:
    """
    Validate numpy array shape.
    
    Args:
        array: Numpy array to validate
        expected_shape: Expected shape tuple (None for any dimension)
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> arr = np.random.rand(224, 224, 3)
        >>> validate_array_shape(arr, expected_shape=(224, 224, 3))
        True
        >>> validate_array_shape(arr, min_dims=2, max_dims=3)
        True
    """
    if not isinstance(array, np.ndarray):
        logger.error(f"Not a numpy array: {type(array)}")
        return False
    
    # Check dimensions
    if min_dims is not None and array.ndim < min_dims:
        logger.error(f"Array has {array.ndim} dims, expected >= {min_dims}")
        return False
    
    if max_dims is not None and array.ndim > max_dims:
        logger.error(f"Array has {array.ndim} dims, expected <= {max_dims}")
        return False
    
    # Check exact shape (with None wildcards)
    if expected_shape is not None:
        if len(expected_shape) != array.ndim:
            logger.error(f"Shape mismatch: {array.shape} vs {expected_shape}")
            return False
        
        for actual, expected in zip(array.shape, expected_shape):
            if expected is not None and actual != expected:
                logger.error(f"Shape mismatch: {array.shape} vs {expected_shape}")
                return False
    
    return True


def validate_confidence(confidence: float) -> bool:
    """
    Validate confidence value is in [0, 1].
    
    Args:
        confidence: Confidence value
        
    Returns:
        True if valid, False otherwise
    """
    if not (0.0 <= confidence <= 1.0):
        logger.error(f"Invalid confidence: {confidence} (must be in [0, 1])")
        return False
    return True


def validate_class_label(label: str, valid_classes: List[str]) -> bool:
    """
    Validate class label is in valid set.
    
    Args:
        label: Class label
        valid_classes: List of valid class names
        
    Returns:
        True if valid, False otherwise
    """
    if label not in valid_classes:
        logger.error(f"Invalid class '{label}'. Valid: {valid_classes}")
        return False
    return True


def validate_config(config: dict) -> bool:
    """
    Validate configuration dictionary has required keys.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['models', 'detection', 'classification', 'rag', 'llm', 'pipeline']
    
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Configuration missing required keys: {missing}")
    
    return True
