"""Pipeline singleton + small wrappers shared across routes.

The pipeline is heavy to construct (loads two PyTorch models, an embedding
model, builds / opens the Chroma collection, etc.). We build it lazily on the
first request and reuse it for the lifetime of the process.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from src.pipeline import BloodSmearPipeline, create_pipeline

logger = logging.getLogger(__name__)

_pipeline_lock = threading.Lock()
_pipeline_instance: Optional[BloodSmearPipeline] = None


def get_pipeline() -> BloodSmearPipeline:
    """Return the lazily-initialised, process-wide pipeline instance."""

    global _pipeline_instance
    if _pipeline_instance is not None:
        return _pipeline_instance

    with _pipeline_lock:
        if _pipeline_instance is None:
            logger.info("Initialising BloodSmearPipeline (first request)...")
            _pipeline_instance = create_pipeline()
            logger.info("BloodSmearPipeline ready.")
    return _pipeline_instance


def reset_pipeline() -> None:
    """Tear down the cached pipeline (used by tests)."""

    global _pipeline_instance
    with _pipeline_lock:
        _pipeline_instance = None
