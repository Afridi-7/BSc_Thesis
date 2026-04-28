"""FastAPI application entrypoint.

Run locally from the ``backend/`` directory:

    uvicorn main:app --reload --port 8767

The frontend (Vite, port 5173) reaches this server via CORS-allowed
``http://localhost:8767``. Override the port via the ``BACKEND_PORT`` env var
if 8767 is taken; remember to set ``VITE_BACKEND_URL`` on the frontend to match.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the repo root importable so ``from src.pipeline import ...`` works
# when uvicorn is launched from inside the ``backend/`` directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from routes import analyze, health, samples
from version import __version__

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Application factory â€” easy to import in tests."""

    settings = get_settings()

    app = FastAPI(
        title="Hybrid Multimodal Lab Assistant API",
        version=__version__,
        description=(
            "HTTP interface for the multimodal pipeline: YOLOv8 detection â†’ "
            "EfficientNet WBC classification (with MC-Dropout uncertainty) â†’ "
            "RAG-grounded GPT-4o clinical reasoning. Outputs are advisory only."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(samples.router, prefix="/api")
    app.include_router(analyze.router, prefix="/api")

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "name": "Hybrid Multimodal Lab Assistant API",
            "version": __version__,
            "docs": "/docs",
            "openapi": "/openapi.json",
        }

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import os
    import uvicorn

    port = int(os.environ.get("BACKEND_PORT", "8767"))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)
