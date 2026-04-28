"""Unit tests for the CBC multimodal analyser."""

from __future__ import annotations

import pytest

from src.multimodal.cbc_analyzer import (
    CBCInput,
    analyze_cbc,
    format_cbc_report_for_prompt,
)


def test_normal_cbc_has_no_abnormalities():
    report = analyze_cbc(
        {
            "wbc": 7.0,
            "hemoglobin": 14.5,
            "platelets": 250,
            "mcv": 90,
            "sex": "M",
        }
    )
    assert report.has_abnormalities is False
    assert report.abnormal_count == 0
    assert all(f.direction == "normal" for f in report.findings)
    assert {f.analyte for f in report.findings} == {"wbc", "hemoglobin", "platelets", "mcv"}


def test_leukocytosis_and_anaemia_detected():
    report = analyze_cbc(
        {"wbc": 18.0, "hemoglobin": 8.5, "platelets": 500, "sex": "F"}
    )
    labels = {f.label for f in report.findings if f.direction != "normal"}
    assert "leukocytosis" in labels
    assert "anaemia" in labels
    assert "thrombocytosis" in labels
    assert report.abnormal_count == 3
    assert report.has_abnormalities is True


def test_severity_buckets():
    # 4.0–11.0 normal range, span = 7. mild < 0.25*span (=1.75) above high.
    mild = analyze_cbc({"wbc": 12.0})
    moderate = analyze_cbc({"wbc": 14.0})
    severe = analyze_cbc({"wbc": 25.0})
    assert mild.findings[0].severity == "mild"
    assert moderate.findings[0].severity == "moderate"
    assert severe.findings[0].severity == "severe"


def test_microcytosis_and_macrocytosis():
    micro = analyze_cbc({"mcv": 70})
    macro = analyze_cbc({"mcv": 110})
    assert micro.findings[0].label == "microcytosis"
    assert macro.findings[0].label == "macrocytosis"


def test_neutropenia_label():
    report = analyze_cbc({"neutrophils_abs": 0.5})
    assert report.findings[0].label == "neutropenia"


def test_hematocrit_percentage_normalised():
    # 45 (%) should become 0.45 (fraction).
    report = analyze_cbc({"hematocrit": 45, "sex": "M"})
    f = next(x for x in report.findings if x.analyte == "hematocrit")
    assert f.direction == "normal"
    assert f.value == 0.45


def test_unknown_sex_uses_envelope():
    # Hb 13.0 is below male low (13.5) but above female low (12.0).
    # Envelope low = min(13.5, 12.0) = 12.0 → should be normal.
    report = analyze_cbc({"hemoglobin": 13.0})
    f = next(x for x in report.findings if x.analyte == "hemoglobin")
    assert f.direction == "normal"


def test_alias_keys_accepted():
    report = analyze_cbc({"hgb": 9.0, "plt": 100, "sex": "F"})
    labels = {f.label for f in report.findings}
    assert "anaemia" in labels
    assert "thrombocytopenia" in labels


def test_invalid_value_silently_skipped():
    # Non-numeric value should not crash.
    report = analyze_cbc({"wbc": "not-a-number", "hemoglobin": 14.0, "sex": "M"})
    assert {f.analyte for f in report.findings} == {"hemoglobin"}


def test_report_to_dict_shape():
    report = analyze_cbc({"wbc": 18.0})
    d = report.to_dict()
    assert set(d.keys()) == {"findings", "abnormal_count", "has_abnormalities", "sex"}
    assert d["findings"][0]["reference_range"] == [4.0, 11.0]


def test_format_for_prompt_contains_marker():
    report = analyze_cbc({"wbc": 18.0, "hemoglobin": 8.0, "sex": "F"})
    text = format_cbc_report_for_prompt(report)
    assert "CBC Laboratory Findings" in text
    assert "leukocytosis" in text
    assert "anaemia" in text
    assert "⚠️" in text


def test_format_empty_report_returns_blank():
    report = analyze_cbc({})  # nothing supplied
    assert format_cbc_report_for_prompt(report) == ""


def test_cbc_input_from_dict_filters_unknown_keys():
    cbc = CBCInput.from_dict({"wbc": 5.0, "garbage_field": 99, "SEX": "f"})
    assert cbc.wbc == 5.0
    assert cbc.sex == "F"
    assert not hasattr(cbc, "garbage_field")


def test_custom_ranges_override():
    # Tighten WBC range so that 7.0 is flagged.
    custom = {"wbc": {"low": 8.0, "high": 9.0, "unit": "×10⁹/L", "name": "WBC"}}
    report = analyze_cbc({"wbc": 7.0}, custom_ranges=custom)
    f = report.findings[0]
    assert f.direction == "low"
    assert f.label == "leukopenia"
