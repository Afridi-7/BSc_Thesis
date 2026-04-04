"""
Metrics calculation utilities for Blood Smear Domain Expert system.

Provides functions for computing uncertainty metrics, statistics, and evaluation metrics.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def calculate_uncertainty_metrics(
    probabilities: np.ndarray,
    epsilon: float = 1e-8
) -> Dict[str, float]:
    """
    Calculate uncertainty metrics from prediction probabilities.
    
    Args:
        probabilities: Array of shape (n_samples, n_classes) with class probabilities
        epsilon: Small constant for numerical stability
        
    Returns:
        Dictionary with uncertainty metrics:
            - confidence: Maximum probability (prediction confidence)
            - entropy: Shannon entropy
            - variance: Variance across classes
            
    Examples:
        >>> probs = np.array([[0.8, 0.1, 0.1]])
        >>> metrics = calculate_uncertainty_metrics(probs)
        >>> print(f"Confidence: {metrics['confidence']:.3f}")
    """
    # Mean probabilities across MC samples (if multiple)
    if probabilities.ndim == 3:
        # Shape: (n_passes, n_samples, n_classes) -> (n_samples, n_classes)
        mean_probs = np.mean(probabilities, axis=0)
    else:
        mean_probs = probabilities
    
    # Confidence: max probability
    confidence = float(np.max(mean_probs, axis=-1).mean())
    
    # Entropy: -sum(p * log(p))
    entropy = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=-1)
    entropy = float(entropy.mean())
    
    # Variance: mean variance across classes
    if probabilities.ndim == 3:
        variance = np.var(probabilities, axis=0).mean()
    else:
        variance = 0.0
    variance = float(variance)
    
    return {
        'confidence': confidence,
        'entropy': entropy,
        'variance': variance
    }


def classify_uncertainty_level(
    confidence: float,
    entropy: float,
    low_conf_threshold: float = 0.85,
    low_entropy_threshold: float = 0.3,
    med_conf_threshold: float = 0.65,
    med_entropy_threshold: float = 0.6
) -> Tuple[str, bool]:
    """
    Classify uncertainty level based on confidence and entropy thresholds.
    
    Args:
        confidence: Prediction confidence (0-1)
        entropy: Shannon entropy
        low_conf_threshold: Minimum confidence for LOW uncertainty
        low_entropy_threshold: Maximum entropy for LOW uncertainty
        med_conf_threshold: Minimum confidence for MEDIUM uncertainty
        med_entropy_threshold: Maximum entropy for MEDIUM uncertainty
        
    Returns:
        Tuple of (uncertainty_level, flagged):
            - uncertainty_level: 'LOW', 'MEDIUM', or 'HIGH'
            - flagged: True if requires expert review (HIGH uncertainty)
            
    Examples:
        >>> level, flagged = classify_uncertainty_level(0.9, 0.2)
        >>> print(f"Level: {level}, Flagged: {flagged}")
        Level: LOW, Flagged: False
    """
    if confidence >= low_conf_threshold and entropy < low_entropy_threshold:
        return 'LOW', False
    elif confidence >= med_conf_threshold and entropy < med_entropy_threshold:
        return 'MEDIUM', False
    else:
        return 'HIGH', True


def calculate_batch_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate statistical summary for batch of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with statistics:
            - mean: Mean value
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - median: Median value
            
    Examples:
        >>> stats = calculate_batch_statistics([0.8, 0.9, 0.7, 0.85])
        >>> print(f"Mean: {stats['mean']:.3f}")
    """
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }
    
    values_array = np.array(values)
    
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'median': float(np.median(values_array))
    }


def calculate_class_distribution(labels: List[str]) -> Dict[str, int]:
    """
    Calculate class distribution from list of labels.
    
    Args:
        labels: List of class labels
        
    Returns:
        Dictionary mapping class name to count
        
    Examples:
        >>> dist = calculate_class_distribution(['WBC', 'RBC', 'WBC', 'Platelet'])
        >>> print(dist)
        {'WBC': 2, 'RBC': 1, 'Platelet': 1}
    """
    return dict(Counter(labels))


def calculate_cell_count_statistics(
    counts_per_image: List[Dict[str, int]]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for cell counts across multiple images.
    
    Args:
        counts_per_image: List of count dictionaries, e.g.,
                         [{'WBC': 10, 'RBC': 50}, {'WBC': 12, 'RBC': 48}, ...]
                         
    Returns:
        Dictionary with mean and variance for each class:
            {
                'mean': {'WBC': 11.0, 'RBC': 49.0, ...},
                'variance': {'WBC': 1.0, 'RBC': 1.0, ...}
            }
            
    Examples:
        >>> counts = [{'WBC': 10, 'RBC': 50}, {'WBC': 12, 'RBC': 48}]
        >>> stats = calculate_cell_count_statistics(counts)
        >>> print(stats['mean']['WBC'])
        11.0
    """
    if not counts_per_image:
        return {'mean': {}, 'variance': {}}
    
    # Get all class names
    all_classes = set()
    for counts in counts_per_image:
        all_classes.update(counts.keys())
    
    # Calculate mean and variance per class
    mean_counts = {}
    variance_counts = {}
    
    for cls in all_classes:
        values = [counts.get(cls, 0) for counts in counts_per_image]
        mean_counts[cls] = float(np.mean(values))
        variance_counts[cls] = float(np.var(values))
    
    return {
        'mean': mean_counts,
        'variance': variance_counts
    }


def calculate_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of true labels
        
    Returns:
        Accuracy as float (0-1)
        
    Raises:
        ValueError: If lists have different lengths
        
    Examples:
        >>> preds = ['WBC', 'RBC', 'WBC']
        >>> truth = ['WBC', 'WBC', 'WBC']
        >>> acc = calculate_accuracy(preds, truth)
        >>> print(f"Accuracy: {acc:.2%}")
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) "
            f"vs ground_truth ({len(ground_truth)})"
        )
    
    if not predictions:
        return 0.0
    
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def create_summary_statistics(
    predictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create comprehensive summary statistics from batch predictions.
    
    Args:
        predictions: List of prediction dictionaries with keys:
                    'predicted_class', 'confidence', 'entropy', 'variance',
                    'uncertainty_level', 'flagged'
                    
    Returns:
        Dictionary with aggregated statistics
        
    Examples:
        >>> preds = [
        ...     {'predicted_class': 'WBC', 'confidence': 0.9, 'flagged': False},
        ...     {'predicted_class': 'RBC', 'confidence': 0.7, 'flagged': True}
        ... ]
        >>> summary = create_summary_statistics(preds)
        >>> print(f"Flagged: {summary['flagged_count']}")
    """
    if not predictions:
        return {
            'sample_count': 0,
            'flagged_count': 0,
            'requires_expert_review': False
        }
    
    # Extract metrics
    confidences = [p.get('confidence', 0.0) for p in predictions]
    entropies = [p.get('entropy', 0.0) for p in predictions]
    variances = [p.get('variance', 0.0) for p in predictions]
    classes = [p.get('predicted_class', 'unknown') for p in predictions]
    flagged = [p.get('flagged', False) for p in predictions]
    
    return {
        'sample_count': len(predictions),
        'class_distribution': calculate_class_distribution(classes),
        'confidence_stats': calculate_batch_statistics(confidences),
        'entropy_stats': calculate_batch_statistics(entropies),
        'variance_stats': calculate_batch_statistics(variances),
        'flagged_count': sum(flagged),
        'requires_expert_review': any(flagged)
    }
