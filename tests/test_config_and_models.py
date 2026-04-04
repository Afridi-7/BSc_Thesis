from pathlib import Path

import pytest
import yaml

from src.config.config_loader import Config, get_model_path, load_config, validate_config


def _write_config(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _minimal_config(model_a: str, model_b: str) -> dict:
    return {
        "models": {
            "yolo_detection": model_a,
            "efficientnet_classification": model_b,
        },
        "detection": {"confidence_threshold": 0.25},
        "classification": {
            "mc_dropout_passes": 5,
            "uncertainty": {
                "low": {"min_confidence": 0.85, "max_entropy": 0.3},
                "medium": {"min_confidence": 0.65, "max_entropy": 0.6},
            },
        },
        "rag": {"retrieval": {"top_k": 3}},
        "llm": {"temperature": 0.1, "api_key_env_var": "OPENAI_API_KEY"},
        "pipeline": {"enable_stage1": True, "enable_stage2": True, "enable_stage3": False},
    }


def test_load_and_validate_config(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    data = _minimal_config("models/yolo.pt", "models/eff.pt")
    _write_config(cfg_path, data)

    cfg = load_config(str(cfg_path))
    validate_config(cfg)

    assert cfg.get("detection.confidence_threshold") == 0.25


def test_get_model_path_prefers_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env_models = tmp_path / "env_models"
    env_models.mkdir()
    (env_models / "yolo.pt").write_bytes(b"ok")

    cfg = Config(_minimal_config("models/yolo.pt", "models/eff.pt"))
    monkeypatch.setenv("THESIS_MODELS_DIR", str(env_models))

    resolved = get_model_path(cfg, "yolo_detection")
    assert resolved == env_models / "yolo.pt"


def test_get_model_path_raises_actionable_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = Config(_minimal_config("models/missing.pt", "models/missing2.pt"))
    monkeypatch.delenv("THESIS_MODELS_DIR", raising=False)

    with pytest.raises(FileNotFoundError) as exc:
        get_model_path(cfg, "yolo_detection")

    assert "THESIS_MODELS_DIR" in str(exc.value)
