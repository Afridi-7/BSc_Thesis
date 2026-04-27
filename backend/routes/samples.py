"""List bundled sample blood-smear images so the UI can offer one-click demos."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config import get_settings
from schemas import SampleImage, SamplesResponse

router = APIRouter(prefix="/samples", tags=["samples"])

_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@router.get("", response_model=SamplesResponse)
def list_samples() -> SamplesResponse:
    """Enumerate ``examples/sample_images/*`` for the demo gallery."""

    settings = get_settings()
    samples = []
    if settings.samples_dir.is_dir():
        for path in sorted(settings.samples_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in _ALLOWED_EXTS:
                # Skip diagnostic probe artefacts (named like _probe_*.jpg).
                if path.name.startswith("_"):
                    continue
                samples.append(SampleImage(name=path.name, size_bytes=path.stat().st_size))
    return SamplesResponse(samples=samples)


@router.get("/{name}")
def get_sample(name: str) -> FileResponse:
    """Stream a sample image so the UI can preview it before analysis."""

    settings = get_settings()
    # Defence against path traversal: only allow basenames inside samples_dir.
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid sample name")

    path = (settings.samples_dir / name).resolve()
    if not path.is_file() or settings.samples_dir.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail=f"Sample not found: {name}")

    return FileResponse(path)
