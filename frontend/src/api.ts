import type { AnalyzeResponse, HealthResponse, SampleImage } from "./types";

const API_BASE = "/api";

export class ApiError extends Error {
  status: number;
  details?: string;

  constructor(status: number, message: string, details?: string) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

async function unwrap<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body?.detail ?? JSON.stringify(body);
    } catch {
      // ignore — fall back to statusText
    }
    throw new ApiError(response.status, `HTTP ${response.status}`, detail);
  }
  return (await response.json()) as T;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`);
  return unwrap<HealthResponse>(response);
}

export async function fetchSamples(): Promise<SampleImage[]> {
  const response = await fetch(`${API_BASE}/samples`);
  const body = await unwrap<{ samples: SampleImage[] }>(response);
  return body.samples;
}

export function sampleImageUrl(name: string): string {
  return `${API_BASE}/samples/${encodeURIComponent(name)}`;
}

async function blobFromSample(name: string): Promise<File> {
  const response = await fetch(sampleImageUrl(name));
  if (!response.ok) {
    throw new ApiError(response.status, `Could not load sample ${name}`);
  }
  const blob = await response.blob();
  return new File([blob], name, { type: blob.type || "image/jpeg" });
}

export async function analyzeImage(
  file: File,
  options: { saveResults?: boolean; includeOverlay?: boolean } = {},
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("image", file);
  form.append("save_results", String(options.saveResults ?? false));
  form.append("include_overlay", String(options.includeOverlay ?? true));

  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
  });
  return unwrap<AnalyzeResponse>(response);
}

export async function analyzeSample(name: string): Promise<AnalyzeResponse> {
  const file = await blobFromSample(name);
  return analyzeImage(file);
}
