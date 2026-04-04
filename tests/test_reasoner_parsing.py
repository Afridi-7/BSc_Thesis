import json

from src.rag.llm_reasoner import ClinicalReasoner


def _reasoner_stub() -> ClinicalReasoner:
    reasoner = ClinicalReasoner.__new__(ClinicalReasoner)
    reasoner.model_name = "gpt-4o"
    reasoner.require_citations = True
    return reasoner


def test_parse_response_normalizes_and_propagates_flags():
    reasoner = _reasoner_stub()
    payload = {
        "clinical_interpretation": "Findings suggest reactive leukocytosis [Reference 1].",
        "key_findings": "single-string",  # intentionally wrong shape
        "differential_diagnoses": ["Reactive process [Reference 1]"],
        "recommendations": ["Peripheral smear correlation"],
        "safety_flags": [],
        "citations_used": [1, 99],
    }

    parsed = reasoner._parse_response(
        json.dumps(payload),
        {"flagged_count": 2},
        [{"source": "a.pdf"}, {"source": "b.pdf"}],
    )

    assert isinstance(parsed["key_findings"], list)
    assert "HIGH_UNCERTAINTY" in parsed["safety_flags"]
    assert parsed["requires_expert_review"] is True
    assert parsed["citations_used"] == [1]


def test_parse_response_non_json_fallback():
    reasoner = _reasoner_stub()
    parsed = reasoner._parse_response("not-json", None, None)

    assert "NON_JSON_RESPONSE" in parsed["safety_flags"]
    assert parsed["requires_expert_review"] is True


def test_create_fallback_response_includes_uncertainty_flag():
    reasoner = _reasoner_stub()
    fallback = reasoner._create_fallback_response("boom", {"flagged_samples": 1})

    assert "LLM_ERROR" in fallback["safety_flags"]
    assert "HIGH_UNCERTAINTY" in fallback["safety_flags"]
