"""Pydantic response schemas for the public API.

The pipeline emits dicts shaped as in ``src/pipeline.py``; these schemas
describe the *cleaned* shape served to the frontend (image arrays stripped,
counts coerced to ints, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    pipeline_ready: bool
    version: str


class BoundingBox(BaseModel):
    xyxy: List[float] = Field(..., description="[x1, y1, x2, y2] in pixels")
    cls: str = Field(..., alias="class")
    confidence: float

    model_config = {"populate_by_name": True}


class DetectionImageResult(BaseModel):
    image_path: str
    boxes: List[BoundingBox]
    counts: Dict[str, int]


class DetectionResult(BaseModel):
    image_count: int
    per_image: List[DetectionImageResult]
    total_counts: Dict[str, int]
    cell_count_stats: Dict[str, Any] = {}


class ClassificationPrediction(BaseModel):
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    entropy: Optional[float] = None
    variance: Optional[float] = None
    uncertainty_level: Optional[str] = None
    flagged: Optional[bool] = None

    model_config = {"extra": "allow"}


class ClassificationResult(BaseModel):
    predictions: List[ClassificationPrediction] = []
    summary: Dict[str, Any] = {}
    uncertainty_summary: Dict[str, Any] = {}
    total_wbc_crops: int = 0
    images_processed: int = 0
    error: Optional[str] = None


class ReasoningResult(BaseModel):
    clinical_interpretation: Optional[str] = None
    key_findings: List[str] = []
    differential_diagnoses: List[Any] = []
    recommendations: List[str] = []
    safety_flags: List[str] = []
    requires_expert_review: bool = False
    citations: List[Any] = []
    retrieval_quality: Dict[str, Any] = {}
    retrieved_references: List[Dict[str, Any]] = []

    model_config = {"extra": "allow"}


class AnalyzeResponse(BaseModel):
    """Top-level response for POST /api/analyze."""

    metadata: Dict[str, Any]
    stage1_detection: Optional[DetectionResult] = None
    stage2_classification: Optional[ClassificationResult] = None
    stage3_reasoning: Optional[ReasoningResult] = None
    cbc_report: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Tabular CBC modality findings (only present when CBC was supplied). "
            "Shape: {findings: [...], abnormal_count: int, has_abnormalities: bool, sex: str|None}."
        ),
    )
    annotated_image_base64: Optional[str] = Field(
        default=None,
        description="PNG bytes (base64) with detection boxes drawn over the input.",
    )


class SampleImage(BaseModel):
    name: str
    size_bytes: int


class SamplesResponse(BaseModel):
    samples: List[SampleImage]
