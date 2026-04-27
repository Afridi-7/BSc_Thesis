import type { ClassificationResult } from "../types";

interface Props {
  classification?: ClassificationResult;
}

export default function ClassificationPanel({ classification }: Props) {
  if (!classification) {
    return (
      <div className="card">
        <h2>Stage 2 — Classification</h2>
        <p className="empty">No classification yet.</p>
      </div>
    );
  }

  if (classification.error || !classification.predictions?.length) {
    return (
      <div className="card">
        <h2>Stage 2 — WBC Classification</h2>
        <p className="empty">{classification.error ?? "No WBC crops were classified."}</p>
      </div>
    );
  }

  const distribution = classification.summary?.class_distribution ?? {};
  const total = Object.values(distribution).reduce((a, b) => a + b, 0) || 1;
  const sorted = Object.entries(distribution).sort((a, b) => b[1] - a[1]);

  const u = classification.uncertainty_summary ?? {};
  const flagged = u.flagged_count ?? u.flagged_samples ?? 0;
  const totalSamples = u.total_samples ?? u.sample_count ?? classification.predictions.length;
  const flaggedPct =
    typeof u.flagged_percentage === "number"
      ? u.flagged_percentage
      : totalSamples
        ? (Number(flagged) / Number(totalSamples)) * 100
        : 0;

  return (
    <div className="card">
      <h2>Stage 2 — WBC Classification (EfficientNet + MC-Dropout)</h2>

      <h3>Differential</h3>
      <div className="bar-list">
        {sorted.map(([cls, count]) => {
          const pct = (count / total) * 100;
          return (
            <div className="bar-row" key={cls}>
              <span>{cls}</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${pct}%` }} />
              </div>
              <span>{pct.toFixed(1)}%</span>
            </div>
          );
        })}
      </div>

      <div style={{ marginTop: 16 }}>
        <h3>Uncertainty</h3>
        <div className="kv">
          <span className="k">Mean confidence</span>
          <span className="v">
            {typeof u.mean_confidence === "number" ? u.mean_confidence.toFixed(3) : "—"}
          </span>
          <span className="k">Mean entropy</span>
          <span className="v">
            {typeof u.mean_entropy === "number" ? u.mean_entropy.toFixed(3) : "—"}
          </span>
          <span className="k">Mean variance</span>
          <span className="v">
            {typeof u.mean_variance === "number" ? u.mean_variance.toFixed(4) : "—"}
          </span>
          <span className="k">Flagged for review</span>
          <span className="v">
            {String(flagged)} / {String(totalSamples)} ({flaggedPct.toFixed(1)}%)
          </span>
        </div>
      </div>
    </div>
  );
}
