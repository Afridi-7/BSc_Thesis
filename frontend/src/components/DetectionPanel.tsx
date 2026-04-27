import type { DetectionResult } from "../types";

interface Props {
  detection?: DetectionResult;
  annotatedImage?: string;
}

const ORDER = ["WBC", "RBC", "Platelet"];

export default function DetectionPanel({ detection, annotatedImage }: Props) {
  if (!detection) {
    return (
      <div className="card">
        <h2>Stage 1 — Detection</h2>
        <p className="empty">No detection results yet.</p>
      </div>
    );
  }

  const total = detection.total_counts ?? {};
  const orderedKeys = [
    ...ORDER.filter((k) => k in total),
    ...Object.keys(total).filter((k) => !ORDER.includes(k)),
  ];
  const boxCount = detection.per_image?.[0]?.boxes?.length ?? 0;

  return (
    <div className="card">
      <h2>Stage 1 — Cell Detection (YOLOv8)</h2>
      <div className="kv">
        <span className="k">Images analysed</span>
        <span className="v">{detection.image_count}</span>
        <span className="k">Total boxes</span>
        <span className="v">{boxCount}</span>
      </div>
      <div className="count-grid">
        {orderedKeys.map((key) => (
          <div key={key} className={`count-card ${key}`}>
            <span className="label">{key}</span>
            <span className="value">{total[key]}</span>
          </div>
        ))}
      </div>
      {annotatedImage && (
        <div className="preview" style={{ marginTop: 16 }}>
          <h3>Annotated overlay</h3>
          <img src={annotatedImage} alt="detections overlay" />
        </div>
      )}
    </div>
  );
}
