import type { ClassificationPrediction } from "../types";

interface Props {
  predictions?: ClassificationPrediction[];
}

const LEVEL_CLASS: Record<string, string> = {
  LOW: "uncert-low",
  MEDIUM: "uncert-medium",
  HIGH: "uncert-high",
};

export default function GradCamPanel({ predictions }: Props) {
  // No analysis run yet.
  if (predictions === undefined) {
    return (
      <div className="card">
        <h2>Grad-CAM Saliency</h2>
        <p className="empty">Run an analysis to view saliency overlays.</p>
      </div>
    );
  }

  const withCam = predictions.filter(
    (p) => typeof p.gradcam_base64 === "string" && p.gradcam_base64.length > 0,
  );

  if (withCam.length === 0) {
    return (
      <div className="card">
        <h2>Grad-CAM Saliency</h2>
        <p className="empty">
          No saliency overlays available. Enable
          <code> classification.gradcam.enabled: true </code>
          in <code>config.yaml</code> to generate them.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>Grad-CAM Saliency ({withCam.length})</h2>
      <p className="card-subtitle">
        Per-crop class-activation maps highlight the regions the WBC classifier
        relied on. Diffuse or off-cell heatmaps are clues that a prediction
        deserves human review.
      </p>
      <div className="gradcam-grid">
        {withCam.map((p, i) => {
          const level = String(p.uncertainty_level ?? "").toUpperCase();
          const conf =
            typeof p.confidence === "number"
              ? (p.confidence * 100).toFixed(1) + "%"
              : "—";
          return (
            <figure key={i} className="gradcam-card">
              <img
                src={p.gradcam_base64 as string}
                alt={`Grad-CAM overlay #${i + 1}`}
                loading="lazy"
              />
              <figcaption>
                <strong>{p.predicted_class ?? "?"}</strong>
                <span className="muted"> · {conf}</span>
                {level && (
                  <span className={`badge ${LEVEL_CLASS[level] ?? ""}`}>
                    {level}
                  </span>
                )}
                {p.flagged && <span className="badge badge-warn">flagged</span>}
              </figcaption>
            </figure>
          );
        })}
      </div>
    </div>
  );
}
