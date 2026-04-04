"""
Cell detection module using YOLOv8.

Provides blood cell detection functionality for WBC, RBC, and Platelet classes.
"""

import logging
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from src.config.config_loader import Config, get_model_path
from src.utils.validators import validate_image

logger = logging.getLogger(__name__)


class CellDetector:
    """YOLOv8-based cell detector for blood smear images."""
    
    def __init__(self, config: Config):
        """
        Initialize cell detector with configuration.
        
        Args:
            config: Configuration object with detection parameters
            
        Raises:
            FileNotFoundError: If model file not found
        """
        self.config = config
        
        # Load configuration parameters
        self.confidence_threshold = config.get('detection.confidence_threshold', 0.25)
        self.canonical_classes = config.get('detection.canonical_classes', ['WBC', 'RBC', 'Platelet'])
        self.max_images_per_batch = config.get('detection.max_images_per_batch')
        self.fail_fast = config.get('detection.fail_fast', False)
        
        # Set device
        device_config = config.get('detection.device', 'auto')
        self.device = self._setup_device(device_config)
        
        # Load YOLO model
        model_path = get_model_path(config, 'yolo_detection')
        
        logger.info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        
        logger.info(f"CellDetector initialized (device={self.device}, conf={self.confidence_threshold})")
    
    def _setup_device(self, device_config: str) -> str:
        """
        Setup computation device.
        
        Args:
            device_config: Device configuration ('auto', 'cpu', 'cuda', or device ID)
            
        Returns:
            Device string for YOLO
        """
        if device_config == 'auto':
            return '0' if torch.cuda.is_available() else 'cpu'
        return str(device_config)
    
    def detect(
        self,
        image_input: Union[str, Path, List[Union[str, Path]]],
        conf: Optional[float] = None,
        max_images: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run cell detection on single image or batch of images.
        
        Args:
            image_input: Single image path or list of image paths
            conf: Confidence threshold override (None to use config default)
            max_images: Maximum images to process (None for unlimited)
            
        Returns:
            Dictionary with detection results:
                {
                    'image_count': int,
                    'per_image': [
                        {
                            'image_path': str,
                            'image': numpy.ndarray (RGB),
                            'boxes': [{'xyxy': [x1,y1,x2,y2], 'class': str, 'confidence': float}],
                            'counts': {'WBC': int, 'RBC': int, 'Platelet': int}
                        },
                        ...
                    ],
                    'total_counts': {'WBC': int, 'RBC': int, 'Platelet': int},
                    'cell_count_stats': {
                        'mean': {'WBC': float, 'RBC': float, 'Platelet': float},
                        'variance': {'WBC': float, 'RBC': float, 'Platelet': float}
                    },
                    'skipped_paths': [str],
                    'failed_count': int
                }
                
        Examples:
            >>> detector = CellDetector(config)
            >>> results = detector.detect('blood_smear.jpg')
            >>> print(f"Found {results['total_counts']['WBC']} WBCs")
        """
        # Normalize input to list
        image_paths = self._to_image_list(image_input)
        
        # Limit batch size if specified
        max_limit = max_images or self.max_images_per_batch
        if max_limit:
            image_paths = image_paths[:max_limit]
        
        logger.info(f"Processing {len(image_paths)} image(s)")
        
        # Use provided confidence or config default
        confidence = conf if conf is not None else self.confidence_threshold
        
        # Process images
        per_image_results = []
        skipped_paths = []
        failed_count = 0
        
        for img_path in image_paths:
            try:
                result = self._detect_single(img_path, confidence)
                per_image_results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                skipped_paths.append(str(img_path))
                failed_count += 1
                
                if self.fail_fast:
                    raise
        
        # Calculate aggregate statistics
        total_counts = self._calculate_total_counts(per_image_results)
        cell_count_stats = self._calculate_statistics(per_image_results)
        
        return {
            'image_count': len(per_image_results),
            'per_image': per_image_results,
            'total_counts': total_counts,
            'cell_count_stats': cell_count_stats,
            'skipped_paths': skipped_paths,
            'failed_count': failed_count
        }
    
    def _detect_single(self, image_path: Union[str, Path], confidence: float) -> Dict[str, Any]:
        """
        Run detection on a single image.
        
        Args:
            image_path: Path to image file
            confidence: Confidence threshold
            
        Returns:
            Dictionary with single-image results
        """
        image_path = Path(image_path)
        
        # Validate image
        if not validate_image(image_path):
            raise ValueError(f"Invalid image: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Run YOLO inference
        results = self.model.predict(
            source=str(image_path),
            conf=confidence,
            verbose=False,
            device=self.device
        )
        
        # Extract boxes
        boxes = []
        counts = self._empty_counts()
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                # Extract box data
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf_val = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = results[0].names[class_id]
                
                # Normalize class name
                normalized_name = self._normalize_class_name(class_name)
                
                boxes.append({
                    'xyxy': xyxy,
                    'class': normalized_name,
                    'confidence': conf_val,
                    'raw_class': class_name
                })
                
                # Update counts
                if normalized_name in counts:
                    counts[normalized_name] += 1
        
        return {
            'image_path': str(image_path),
            'image': image_np,
            'boxes': boxes,
            'counts': counts
        }
    
    def _to_image_list(self, image_input: Union[str, Path, List]) -> List[Path]:
        """Convert input to list of Path objects."""
        if isinstance(image_input, (str, Path)):
            return [Path(image_input)]
        return [Path(p) for p in image_input]
    
    def _empty_counts(self) -> Dict[str, int]:
        """Return dictionary with all classes initialized to 0."""
        return {cls: 0 for cls in self.canonical_classes}
    
    def _normalize_class_name(self, name: str) -> str:
        """
        Normalize class name to canonical form.
        
        Handles variations like 'Platelets' -> 'Platelet'.
        """
        # Common variations
        if name.lower() == 'platelets':
            return 'Platelet'
        
        # Default: return as-is if in canonical classes
        if name in self.canonical_classes:
            return name
        
        # Try case-insensitive match
        for canonical in self.canonical_classes:
            if name.lower() == canonical.lower():
                return canonical
        
        logger.warning(f"Unknown class name: {name}")
        return name
    
    def _calculate_total_counts(self, results: List[Dict]) -> Dict[str, int]:
        """Calculate total counts across all images."""
        total = self._empty_counts()
        
        for result in results:
            counts = result.get('counts', {})
            for cls, count in counts.items():
                if cls in total:
                    total[cls] += count
        
        return total
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate mean and variance for cell counts."""
        if not results:
            return {'mean': self._empty_counts(), 'variance': self._empty_counts()}
        
        # Collect counts per class across images
        counts_by_class = {cls: [] for cls in self.canonical_classes}
        
        for result in results:
            counts = result.get('counts', {})
            for cls in self.canonical_classes:
                counts_by_class[cls].append(counts.get(cls, 0))
        
        # Calculate statistics
        mean_counts = {}
        variance_counts = {}
        
        for cls, values in counts_by_class.items():
            mean_counts[cls] = float(np.mean(values))
            variance_counts[cls] = float(np.var(values))
        
        return {
            'mean': mean_counts,
            'variance': variance_counts
        }
    
    def extract_wbc_crops(
        self,
        detection_result: Dict[str, Any],
        padding: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Extract WBC crops from detection results.
        
        Args:
            detection_result: Single-image detection result
            padding: Pixels to add around bounding box
            
        Returns:
            List of dictionaries with WBC crops:
                [{'image': np.ndarray, 'box': dict, 'index': int}, ...]
                
        Examples:
            >>> results = detector.detect('smear.jpg')
            >>> wbc_crops = detector.extract_wbc_crops(results['per_image'][0])
            >>> print(f"Extracted {len(wbc_crops)} WBC crops")
        """
        crops = []
        image = detection_result['image']
        boxes = detection_result['boxes']
        
        h, w = image.shape[:2]
        
        for idx, box in enumerate(boxes):
            if box['class'] == 'WBC':
                # Get coordinates
                x1, y1, x2, y2 = box['xyxy']
                
                # Add padding
                x1 = max(0, int(x1) - padding)
                y1 = max(0, int(y1) - padding)
                x2 = min(w, int(x2) + padding)
                y2 = min(h, int(y2) + padding)
                
                # Crop image
                crop = image[y1:y2, x1:x2]
                
                crops.append({
                    'image': crop,
                    'box': box,
                    'index': idx
                })
        
        logger.info(f"Extracted {len(crops)} WBC crops")
        return crops
