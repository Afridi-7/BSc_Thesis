from src.config.config_loader import Config
from src.rag.llm_reasoner import ClinicalReasoner


def test_reasoner_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    config = Config(
        {
            "llm": {
                "model_name": "gpt-4o",
                "temperature": 0.1,
                "max_tokens": 64,
                "api_key_env_var": "OPENAI_API_KEY",
                "safety": {
                    "enable_abstention": True,
                    "propagate_vision_uncertainty": True,
                    "require_citations": True,
                },
            }
        }
    )

    reasoner = ClinicalReasoner(config)
    assert reasoner.model_name == "gpt-4o"


def test_reasoner_missing_api_key_has_setup_instructions(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = Config(
        {
            "llm": {
                "model_name": "gpt-4o",
                "temperature": 0.1,
                "max_tokens": 64,
                "api_key_env_var": "OPENAI_API_KEY",
                "safety": {
                    "enable_abstention": True,
                    "propagate_vision_uncertainty": True,
                    "require_citations": True,
                },
            }
        }
    )

    try:
        ClinicalReasoner(config)
        assert False, "Expected missing key error"
    except ValueError as exc:
        msg = str(exc)
        assert ".env file in project root" in msg
        assert "Windows PowerShell" in msg
