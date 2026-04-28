"""CBC (Complete Blood Count) analyser — second input modality.

This module turns a structured CBC dictionary into a list of clinical findings
(e.g. ``leukocytosis``, ``anaemia``, ``thrombocytopenia``) by comparing each
analyte against published adult reference ranges. The findings are fused with
the image-derived WBC differential in ``BloodSmearPipeline`` so that Stage 3
reasons over **image + tabular numeric data** — i.e. genuinely multimodal
input.

Reference ranges are deliberately conservative adult (≥18 y) ranges and are
sourced from the textbook corpus already shipped with the project. They are
configurable: callers may inject their own ranges through ``custom_ranges``.

Notes
-----
* Pediatric / pregnancy ranges differ and are out of scope.
* Units are SI: counts in ``×10⁹/L`` (WBC, neutrophils, etc.) or ``×10¹²/L``
  (RBC), hemoglobin in ``g/dL``, hematocrit as a fraction (0.0–1.0) **or**
  percentage (auto-detected), MCV in ``fL``, platelets in ``×10⁹/L``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


# ---------------------------------------------------------------------------
# Reference ranges (adults, conservative consensus from textbook corpus)
# ---------------------------------------------------------------------------

# Each entry: (low, high, unit, friendly_name)
ADULT_REFERENCE_RANGES: Dict[str, Dict[str, Any]] = {
    "wbc":          {"low": 4.0,  "high": 11.0, "unit": "×10⁹/L", "name": "WBC count"},
    "rbc_male":     {"low": 4.5,  "high": 5.9,  "unit": "×10¹²/L", "name": "RBC count (male)"},
    "rbc_female":   {"low": 4.1,  "high": 5.1,  "unit": "×10¹²/L", "name": "RBC count (female)"},
    "hemoglobin_male":   {"low": 13.5, "high": 17.5, "unit": "g/dL",    "name": "Hemoglobin (male)"},
    "hemoglobin_female": {"low": 12.0, "high": 15.5, "unit": "g/dL",    "name": "Hemoglobin (female)"},
    "hematocrit_male":   {"low": 0.41, "high": 0.53, "unit": "fraction", "name": "Hematocrit (male)"},
    "hematocrit_female": {"low": 0.36, "high": 0.46, "unit": "fraction", "name": "Hematocrit (female)"},
    "mcv":          {"low": 80.0, "high": 100.0, "unit": "fL",     "name": "MCV"},
    "platelets":    {"low": 150.0, "high": 450.0, "unit": "×10⁹/L", "name": "Platelet count"},
    # Absolute differential counts
    "neutrophils_abs":  {"low": 1.8, "high": 7.7, "unit": "×10⁹/L", "name": "Absolute neutrophils"},
    "lymphocytes_abs":  {"low": 1.0, "high": 4.8, "unit": "×10⁹/L", "name": "Absolute lymphocytes"},
    "monocytes_abs":    {"low": 0.2, "high": 1.0, "unit": "×10⁹/L", "name": "Absolute monocytes"},
    "eosinophils_abs":  {"low": 0.0, "high": 0.5, "unit": "×10⁹/L", "name": "Absolute eosinophils"},
    "basophils_abs":    {"low": 0.0, "high": 0.2, "unit": "×10⁹/L", "name": "Absolute basophils"},
}


@dataclass
class CBCInput:
    """Validated CBC input. All fields optional — only what's provided is checked.

    Sex is required to disambiguate sex-specific ranges (RBC, hgb, hct). If
    omitted, those analytes are scored against the *union* of male+female
    ranges (i.e. flagged only if outside the wider envelope).
    """

    wbc: Optional[float] = None             # ×10⁹/L
    rbc: Optional[float] = None             # ×10¹²/L
    hemoglobin: Optional[float] = None      # g/dL
    hematocrit: Optional[float] = None      # fraction (0.0–1.0) or % (auto-detected)
    mcv: Optional[float] = None             # fL
    platelets: Optional[float] = None       # ×10⁹/L

    neutrophils_abs: Optional[float] = None
    lymphocytes_abs: Optional[float] = None
    monocytes_abs: Optional[float] = None
    eosinophils_abs: Optional[float] = None
    basophils_abs: Optional[float] = None

    sex: Optional[str] = None               # 'M' / 'F' / None
    age_years: Optional[int] = None         # informational; ranges are adult only

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CBCInput":
        # Permissive parsing: accept either snake_case or common short forms.
        aliases = {
            "hgb": "hemoglobin",
            "hb": "hemoglobin",
            "hct": "hematocrit",
            "plt": "platelets",
            "platelet": "platelets",
            "neut": "neutrophils_abs",
            "neutrophils": "neutrophils_abs",
            "lymph": "lymphocytes_abs",
            "lymphocytes": "lymphocytes_abs",
            "mono": "monocytes_abs",
            "monocytes": "monocytes_abs",
            "eos": "eosinophils_abs",
            "eosinophils": "eosinophils_abs",
            "baso": "basophils_abs",
            "basophils": "basophils_abs",
        }
        canonical: Dict[str, Any] = {}
        for raw_key, value in data.items():
            key = raw_key.lower().strip()
            key = aliases.get(key, key)
            if key in cls.__dataclass_fields__:
                canonical[key] = value

        # Hematocrit auto-normalisation: if value > 1, assume percentage.
        if (hct := canonical.get("hematocrit")) is not None:
            try:
                if float(hct) > 1.0:
                    canonical["hematocrit"] = float(hct) / 100.0
            except (TypeError, ValueError):
                canonical["hematocrit"] = None

        # Sex normalisation
        if (sex := canonical.get("sex")) is not None:
            s = str(sex).strip().upper()
            canonical["sex"] = s[:1] if s and s[0] in ("M", "F") else None

        return cls(**canonical)


@dataclass
class CBCFinding:
    """One finding produced by the analyser."""

    analyte: str
    value: float
    direction: str          # 'low' | 'high' | 'normal'
    severity: str           # 'mild' | 'moderate' | 'severe' | 'normal'
    label: str              # e.g. 'leukocytosis', 'anaemia'
    reference_low: float
    reference_high: float
    unit: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyte": self.analyte,
            "value": self.value,
            "direction": self.direction,
            "severity": self.severity,
            "label": self.label,
            "reference_range": [self.reference_low, self.reference_high],
            "unit": self.unit,
            "explanation": self.explanation,
        }


@dataclass
class CBCReport:
    """Aggregate result of analysing a CBC."""

    findings: List[CBCFinding] = field(default_factory=list)
    abnormal_count: int = 0
    has_abnormalities: bool = False
    sex: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "abnormal_count": self.abnormal_count,
            "has_abnormalities": self.has_abnormalities,
            "sex": self.sex,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Maps direction -> (analyte canonical key) -> finding label
_FINDING_LABELS: Dict[str, Dict[str, str]] = {
    "low": {
        "wbc": "leukopenia",
        "hemoglobin": "anaemia",
        "platelets": "thrombocytopenia",
        "neutrophils_abs": "neutropenia",
        "lymphocytes_abs": "lymphopenia",
        "mcv": "microcytosis",
    },
    "high": {
        "wbc": "leukocytosis",
        "hemoglobin": "polycythaemia",
        "platelets": "thrombocytosis",
        "neutrophils_abs": "neutrophilia",
        "lymphocytes_abs": "lymphocytosis",
        "monocytes_abs": "monocytosis",
        "eosinophils_abs": "eosinophilia",
        "basophils_abs": "basophilia",
        "mcv": "macrocytosis",
    },
}


def _classify_severity(value: float, low: float, high: float) -> str:
    """Bucket the deviation magnitude into mild/moderate/severe."""
    if low <= value <= high:
        return "normal"
    span = max(high - low, 1e-9)
    if value < low:
        deficit_ratio = (low - value) / span
    else:
        deficit_ratio = (value - high) / span
    if deficit_ratio < 0.25:
        return "mild"
    if deficit_ratio < 0.75:
        return "moderate"
    return "severe"


def _get_range(
    analyte: str,
    sex: Optional[str],
    ranges: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve sex-aware range for an analyte. Returns None if no range exists."""
    table = ranges if ranges is not None else ADULT_REFERENCE_RANGES

    if analyte in table:
        return table[analyte]

    sexed = f"{analyte}_{'male' if sex == 'M' else 'female'}" if sex in ("M", "F") else None
    if sexed and sexed in table:
        return table[sexed]

    if sex is None:
        # Build the union envelope when sex is unknown.
        male_key = f"{analyte}_male"
        female_key = f"{analyte}_female"
        if male_key in table and female_key in table:
            m = table[male_key]
            f_ = table[female_key]
            return {
                "low":  min(m["low"], f_["low"]),
                "high": max(m["high"], f_["high"]),
                "unit": m["unit"],
                "name": m["name"].split("(")[0].strip() + " (sex unknown)",
            }
    return None


def _build_finding(
    analyte: str,
    value: float,
    rng: Dict[str, Any],
) -> CBCFinding:
    low = float(rng["low"])
    high = float(rng["high"])
    severity = _classify_severity(value, low, high)
    if severity == "normal":
        direction = "normal"
        label = "normal"
        explanation = f"{rng['name']} {value} {rng['unit']} is within reference range ({low}–{high})."
    else:
        direction = "low" if value < low else "high"
        label = _FINDING_LABELS[direction].get(analyte, f"{direction}_{analyte}")
        explanation = (
            f"{rng['name']} {value} {rng['unit']} is {severity} "
            f"{'below' if direction == 'low' else 'above'} the reference range "
            f"({low}–{high}); pattern: {label}."
        )

    return CBCFinding(
        analyte=analyte,
        value=value,
        direction=direction,
        severity=severity,
        label=label,
        reference_low=low,
        reference_high=high,
        unit=rng["unit"],
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Analytes the analyser will scan (in display order). Sex resolution is
# handled inside ``_get_range``, so callers pass the canonical short keys.
_ANALYTE_ORDER = (
    "wbc",
    "hemoglobin",
    "hematocrit",
    "rbc",
    "mcv",
    "platelets",
    "neutrophils_abs",
    "lymphocytes_abs",
    "monocytes_abs",
    "eosinophils_abs",
    "basophils_abs",
)


def analyze_cbc(
    cbc: CBCInput | Mapping[str, Any],
    custom_ranges: Optional[Dict[str, Dict[str, Any]]] = None,
) -> CBCReport:
    """Compute a structured CBC report from an input dictionary or dataclass.

    Args:
        cbc: Either a ``CBCInput`` instance or a mapping accepted by
            :py:meth:`CBCInput.from_dict`.
        custom_ranges: Optional override mapping with the same shape as
            :data:`ADULT_REFERENCE_RANGES` (e.g. for paediatric ranges).

    Returns:
        ``CBCReport`` listing each analyte that was provided, whether it was
        normal or abnormal, and a human-readable explanation.
    """

    if not isinstance(cbc, CBCInput):
        cbc = CBCInput.from_dict(cbc)

    # Allow caller-supplied ranges by shadowing the module-level table.
    if custom_ranges:
        # Local copy: never mutate the module global.
        ranges = {**ADULT_REFERENCE_RANGES, **custom_ranges}
    else:
        ranges = ADULT_REFERENCE_RANGES

    findings: List[CBCFinding] = []
    for analyte in _ANALYTE_ORDER:
        value = getattr(cbc, analyte, None)
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue

        # Resolve range (sex-aware) using the active table.
        rng = _get_range(analyte, cbc.sex, ranges=ranges)

        if rng is None:
            continue
        findings.append(_build_finding(analyte, value_f, rng))

    abnormal = [f for f in findings if f.direction != "normal"]
    return CBCReport(
        findings=findings,
        abnormal_count=len(abnormal),
        has_abnormalities=bool(abnormal),
        sex=cbc.sex,
    )


def format_cbc_report_for_prompt(report: CBCReport) -> str:
    """Render the CBC report as Markdown for inclusion in the LLM prompt."""
    if not report.findings:
        return ""

    lines: List[str] = ["**CBC Laboratory Findings (tabular modality):**"]
    if report.sex:
        lines.append(f"- Sex: {report.sex}")
    lines.append(f"- Abnormal analytes: {report.abnormal_count} of {len(report.findings)}")
    lines.append("")

    for f in report.findings:
        marker = "⚠️ " if f.direction != "normal" else "  "
        lines.append(
            f"{marker}**{f.analyte}** = {f.value} {f.unit} "
            f"(ref {f.reference_low}–{f.reference_high}) → {f.label} ({f.severity})"
        )
    return "\n".join(lines)
