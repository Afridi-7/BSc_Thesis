"""Utility helpers for pipeline orchestration."""

from typing import List, Dict, Any


def collect_wbc_crops(detector, per_image_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract WBC crops from all images and attach provenance metadata."""
    all_crops: List[Dict[str, Any]] = []
    for image_index, image_result in enumerate(per_image_results):
        image_crops = detector.extract_wbc_crops(image_result)
        for crop in image_crops:
            crop["source_image_path"] = image_result.get("image_path", "")
            crop["source_image_index"] = image_index
            all_crops.append(crop)
    return all_crops
