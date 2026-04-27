// Mirrors the backend Pydantic schemas in `backend/schemas.py`.
// Keep this file in sync if you change the API surface.

export interface BoundingBox {
  xyxy: [number, number, number, number];
  class: string;
  confidence: number;
}

export interface DetectionImageResult {
  image_path: string;
  boxes: BoundingBox[];
  counts: Record<string, number>;
}

export interface DetectionResult {
  image_count: number;
  per_image: DetectionImageResult[];
  total_counts: Record<string, number>;
  cell_count_stats?: Record<string, unknown>;
}

export interface ClassificationPrediction {
  predicted_class?: string;
  confidence?: number;
  entropy?: number;
  variance?: number;
  uncertainty_level?: "LOW" | "MEDIUM" | "HIGH" | string;
  flagged?: boolean;
  gradcam_base64?: string;
  [key: string]: unknown;
}

export interface UncertaintySummary {
  total_samples?: number;
  sample_count?: number;
  flagged_count?: number;
  flagged_samples?: number;
  flagged_percentage?: number;
  mean_confidence?: number;
  mean_entropy?: number;
  mean_variance?: number;
  uncertainty_distribution?: Record<string, number>;
  [key: string]: unknown;
}

export interface ClassificationSummary {
  class_distribution?: Record<string, number>;
  sample_count?: number;
  flagged_count?: number;
  requires_expert_review?: boolean;
  [key: string]: unknown;
}

export interface ClassificationResult {
  predictions?: ClassificationPrediction[];
  summary?: ClassificationSummary;
  uncertainty_summary?: UncertaintySummary;
  total_wbc_crops?: number;
  images_processed?: number;
  error?: string;
}

export interface RetrievedReference {
  reference_id: number;
  source: string;
  chunk_id?: string | number;
  score?: number;
}

export interface AgentTraceStep {
  type: "tool_call" | "tool_result" | "thought" | string;
  name?: string;
  args?: Record<string, unknown>;
  content?: string;
}

export interface ReasoningResult {
  clinical_interpretation?: string;
  key_findings?: string[];
  differential_diagnoses?: Array<string | { name?: string; rationale?: string }>;
  recommendations?: string[];
  safety_flags?: string[];
  requires_expert_review?: boolean;
  citations?: unknown[];
  retrieval_quality?: Record<string, unknown>;
  retrieved_references?: RetrievedReference[];
  reasoning_mode?: "linear" | "agent" | string;
  agent_trace?: AgentTraceStep[];
  [key: string]: unknown;
}

export interface AnalyzeResponse {
  metadata: Record<string, unknown>;
  stage1_detection?: DetectionResult;
  stage2_classification?: ClassificationResult;
  stage3_reasoning?: ReasoningResult;
  annotated_image_base64?: string;
}

export interface SampleImage {
  name: string;
  size_bytes: number;
}

export interface HealthResponse {
  status: string;
  pipeline_ready: boolean;
  version: string;
}
