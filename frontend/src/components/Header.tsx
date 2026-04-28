import type { HealthResponse } from "../types";

interface Props {
  health: HealthResponse | null;
  healthError: string | null;
}

export default function Header({ health, healthError }: Props) {
  const ok = !!health && !healthError;
  const label = healthError
    ? "Backend unreachable"
    : health
      ? `API v${health.version} • pipeline ${health.pipeline_ready ? "warm" : "cold"}`
      : "Connecting…";

  return (
    <header className="app-header">
      <div>
        <h1>Hybrid Multimodal Lab Assistant</h1>
        <div className="tagline">
          YOLOv8 detection → EfficientNet (MC-Dropout) → RAG-grounded clinical reasoning
        </div>
      </div>
      <span className={`health-pill ${ok ? "ok" : ""}`}>
        <span className="dot" />
        {label}
      </span>
    </header>
  );
}
