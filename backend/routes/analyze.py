"""POST /api/analyze — accept an image upload, run the full pipeline."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from config import get_settings
from dependencies import get_pipeline
from schemas import AnalyzeResponse
from services.visualization import render_overlay_base64

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analyze"])


def _strip_image_arrays(results: dict) -> dict:
    """Reuse the pipeline's own JSON sanitiser so responses stay small + safe."""

    pipeline = get_pipeline()
    return pipeline._to_json_safe(results)  # noqa: SLF001 — intentional reuse


def _save_upload_to_temp(upload: UploadFile, dest_dir: Path) -> Path:
    """Persist uploaded bytes to disk so YOLO can ingest a real path."""

    settings = get_settings()
    suffix = Path(upload.filename or "upload.png").suffix.lower() or ".png"
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {suffix}")

    dest = dest_dir / f"upload{suffix}"
    written = 0
    with dest.open("wb") as fp:
        while chunk := upload.file.read(64 * 1024):
            written += len(chunk)
            if written > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Upload exceeds {settings.max_upload_bytes} bytes",
                )
            fp.write(chunk)

    # Pillow round-trip ensures the file is actually a decodable image (defence
    # against polyglot uploads that pass MIME checks but aren't real images).
    try:
        with Image.open(dest) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    return dest


@router.post("/analyze", response_model=AnalyzeResponse, response_model_exclude_none=True)
async def analyze(
    image: UploadFile = File(..., description="Blood smear image (jpg/png/bmp/tif)"),
    save_results: bool = Form(False, description="Persist stage outputs under results/"),
    include_overlay: bool = Form(True, description="Return base64 PNG with detection boxes"),
) -> AnalyzeResponse:
    """Run the full Detection → Classification → RAG-Reasoning pipeline."""

    settings = get_settings()

    if image.content_type and image.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {image.content_type}",
        )

    pipeline = get_pipeline()

    with tempfile.TemporaryDirectory(prefix="bsde_upload_") as tmpdir:
        tmp_path = _save_upload_to_temp(image, Path(tmpdir))
        logger.info("Analyzing upload (%d bytes) -> %s", tmp_path.stat().st_size, tmp_path.name)

        results = pipeline.analyze(str(tmp_path), save_results=save_results)

        overlay_b64: Optional[str] = None
        if include_overlay:
            try:
                stage1 = results.get("stage1_detection") or {}
                per_image = stage1.get("per_image") or []
                if per_image:
                    boxes = per_image[0].get("boxes") or []
                    with Image.open(tmp_path) as src:
                        overlay_b64 = render_overlay_base64(src, boxes)
            except Exception:  # noqa: BLE001
                logger.exception("Overlay rendering failed; returning result without it")
                overlay_b64 = None

    safe_results = _strip_image_arrays(results)

    return AnalyzeResponse(
        metadata=safe_results.get("metadata", {}),
        stage1_detection=safe_results.get("stage1_detection"),
        stage2_classification=safe_results.get("stage2_classification"),
        stage3_reasoning=safe_results.get("stage3_reasoning"),
        annotated_image_base64=overlay_b64,
    )
