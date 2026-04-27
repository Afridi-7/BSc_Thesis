import { useEffect, useState } from "react";
import {
  ApiError,
  analyzeImage,
  analyzeSample,
  fetchHealth,
  fetchSamples,
} from "./api";
import type { AnalyzeResponse, HealthResponse, SampleImage } from "./types";
import Header from "./components/Header";
import UploadCard from "./components/UploadCard";
import DetectionPanel from "./components/DetectionPanel";
import ClassificationPanel from "./components/ClassificationPanel";
import GradCamPanel from "./components/GradCamPanel";
import ReasoningPanel from "./components/ReasoningPanel";
import AgentTracePanel from "./components/AgentTracePanel";

// Replacer for JSON.stringify: hides huge base64 image blobs in the debug view
// so the <pre> doesn't render as a giant unreadable block.
function redactBase64(key: string, value: unknown): unknown {
  if (typeof value === "string" && /(_base64|gradcam|heatmap|overlay)/i.test(key) && value.length > 256) {
    return `<base64 omitted, ${value.length} chars>`;
  }
  return value;
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleImage[]>([]);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((e: unknown) =>
        setHealthError(e instanceof Error ? e.message : "Backend unreachable"),
      );

    fetchSamples()
      .then(setSamples)
      .catch(() => setSamples([]));
  }, []);

  async function run(action: () => Promise<AnalyzeResponse>) {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const response = await action();
      setResult(response);
    } catch (e) {
      const message =
        e instanceof ApiError
          ? `${e.message}: ${e.details ?? ""}`
          : e instanceof Error
            ? e.message
            : "Unknown error";
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app-shell">
      <Header health={health} healthError={healthError} />

      <div className="disclaimer">
        ⚠️ Educational research prototype. <strong>Not for clinical diagnosis.</strong>{" "}
        Outputs are advisory only and must be reviewed by a qualified hematologist.
      </div>

      <div className="layout">
        <UploadCard
          busy={busy}
          samples={samples}
          onAnalyzeFile={(file) => run(() => analyzeImage(file))}
          onAnalyzeSample={(name) => run(() => analyzeSample(name))}
        />

        <div className="results">
          {error && <div className="error-banner">{error}</div>}
          <DetectionPanel
            detection={result?.stage1_detection}
            annotatedImage={result?.annotated_image_base64}
          />
          <ClassificationPanel classification={result?.stage2_classification} />
          <GradCamPanel predictions={result?.stage2_classification?.predictions} />
          <ReasoningPanel reasoning={result?.stage3_reasoning} />
          <AgentTracePanel
            trace={result?.stage3_reasoning?.agent_trace}
            mode={result?.stage3_reasoning?.reasoning_mode as string | undefined}
          />
          {result && (
            <details className="card">
              <summary>Raw response (debug)</summary>
              <pre>{JSON.stringify(result, redactBase64, 2)}</pre>
            </details>
          )}
        </div>
      </div>
    </div>
  );
}
