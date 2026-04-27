import type { ReasoningResult } from "../types";

interface Props {
  reasoning?: ReasoningResult;
}

function diagnosisLabel(d: unknown): string {
  if (typeof d === "string") return d;
  if (d && typeof d === "object") {
    const obj = d as { name?: string; rationale?: string; condition?: string };
    return obj.name ?? obj.condition ?? JSON.stringify(d);
  }
  return String(d);
}

export default function ReasoningPanel({ reasoning }: Props) {
  if (!reasoning) {
    return (
      <div className="card">
        <h2>Stage 3 — Clinical Reasoning</h2>
        <p className="empty">No reasoning yet.</p>
      </div>
    );
  }

  const interpretation = reasoning.clinical_interpretation;
  const findings = reasoning.key_findings ?? [];
  const ddx = reasoning.differential_diagnoses ?? [];
  const recs = reasoning.recommendations ?? [];
  const flags = reasoning.safety_flags ?? [];
  const refs = reasoning.retrieved_references ?? [];

  return (
    <div className="card">
      <h2>Stage 3 — Clinical Reasoning (RAG + GPT-4o)</h2>

      {reasoning.requires_expert_review && (
        <div className="error-banner" style={{ marginBottom: 16 }}>
          🚨 Expert review required — outputs flagged as uncertain or implausible.
        </div>
      )}

      {interpretation && (
        <p style={{ lineHeight: 1.6, fontSize: 14 }}>{interpretation}</p>
      )}

      {findings.length > 0 && (
        <>
          <h3>Key findings</h3>
          <ul className="bullet">
            {findings.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
        </>
      )}

      {ddx.length > 0 && (
        <>
          <h3>Differential diagnoses</h3>
          <ul className="bullet">
            {ddx.map((d, i) => (
              <li key={i}>{diagnosisLabel(d)}</li>
            ))}
          </ul>
        </>
      )}

      {recs.length > 0 && (
        <>
          <h3>Recommendations</h3>
          <ul className="bullet">
            {recs.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </>
      )}

      {flags.length > 0 && (
        <>
          <h3>Safety flags</h3>
          <ul className="flag-list">
            {flags.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
        </>
      )}

      {refs.length > 0 && (
        <>
          <h3>Retrieved references</h3>
          <div className="kv">
            {refs.map((r) => (
              <span key={r.reference_id} className="k" style={{ gridColumn: "1 / -1" }}>
                [{r.reference_id}] {r.source}
                {typeof r.score === "number" ? ` (score=${r.score.toFixed(3)})` : ""}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
