"""Backend settings.

We deliberately keep this tiny and deterministic — the pipeline already owns
its own configuration via ``src/config/config_loader.py``. This module only
holds web-server concerns (CORS, upload limits, sample paths).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGES_DIR = REPO_ROOT / "examples" / "sample_images"
RESULTS_DIR = REPO_ROOT / "results"


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the FastAPI app."""

    # Allowed origins for CORS — Vite dev server defaults to 5173.
    cors_origins: List[str] = field(
        default_factory=lambda: _split_csv(
            os.environ.get(
                "BACKEND_CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            )
        )
    )

    # Maximum upload size (bytes). Default 15 MB — typical smear photos are <5 MB.
    max_upload_bytes: int = int(os.environ.get("BACKEND_MAX_UPLOAD_BYTES", 15 * 1024 * 1024))

    # Allowed image content types (defence-in-depth — Pillow re-validates).
    allowed_content_types: tuple = (
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/bmp",
        "image/tiff",
    )

    # Where to look for bundled demo images served by /api/samples.
    samples_dir: Path = SAMPLE_IMAGES_DIR


def get_settings() -> Settings:
    """Return process-wide settings (cheap to call repeatedly)."""

    return Settings()
