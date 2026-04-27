"""Liveness / readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from dependencies import _pipeline_instance  # type: ignore[attr-defined]
from schemas import HealthResponse
from version import __version__

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Cheap liveness probe; reports whether the pipeline has been warmed up."""

    return HealthResponse(
        status="ok",
        pipeline_ready=_pipeline_instance is not None,
        version=__version__,
    )
