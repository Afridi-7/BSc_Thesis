# Blood Smear Domain Expert

End-to-end BSc thesis repository for a hematology decision-support pipeline that combines computer vision and retrieval-augmented clinical reasoning.

The system is implemented as three linked stages:

1. Stage 1: YOLOv8 blood cell detection.
2. Stage 2: EfficientNet WBC subtype classification with uncertainty quantification.
3. Stage 3: RAG + LLM clinical reasoning with safety and evidence constraints.

This README is written as a paper-ready project log so methods, experiments, and outputs are easy to transfer into thesis chapters.

## Research Objective

Build a clinically safer AI assistant for peripheral blood smear interpretation by:

1. Detecting cell categories (WBC, RBC, Platelet).
2. Classifying WBC subtypes with calibrated uncertainty signals.
3. Producing evidence-grounded clinical reasoning with citations and abstention when evidence is weak.

## Repository Structure

```
BSc_Thesis/
├── README.md
├── YOLOv8_detection/
│   ├── YOLOv8_on_TBL_PBC_dataset.ipynb
│   └── results/
├── Efficientnet_classification/
│   ├── Efficientnet_classification.ipynb
│   ├── README.md
│   └── results/
└── LLM_RAG_Pipline/
    ├── LLM_Rag_pipline.ipynb
    ├── pdfs/
    └── results/
```

## End-to-End Method

### Stage 1: Detection (YOLOv8)

1. Detects WBC, RBC, Platelet instances from smear images.
2. Supports both single-image and batch-image inference.
3. Produces counts and aggregate batch statistics used by downstream stages.

Expected Stage 1 outputs:

1. Cell count summary by class.
2. Optional visual overlays for thesis figures.
3. Run metadata for reproducibility.

### Stage 2: WBC Classification + Uncertainty (EfficientNet-B0)

1. Receives WBC crops and predicts subtype probabilities.
2. Uses Monte Carlo Dropout at inference time.
3. Computes confidence, entropy, variance, and uncertainty flags.

Uncertainty behavior:

1. High confidence + low entropy: low uncertainty.
2. Intermediate values: medium uncertainty.
3. Low confidence or high entropy: high uncertainty and review flag.

### Stage 3: RAG + LLM Clinical Reasoning

1. Builds a retrieval query from structured vision summary.
2. Retrieves hematology evidence chunks from PDF-derived knowledge base.
3. Calls OpenAI `gpt-4o` for structured JSON reasoning.
4. Applies safety guardrails and abstention logic.

Important note:

1. Some function names still contain `gemini` from earlier iterations, but active generation is now OpenAI (`gpt-4o`).

## Unified Data Contract (Cross-Notebook)

Required fields used across stages:

1. `cell_counts`
2. `total_cells`
3. `wbc_differential`
4. `uncertainty_summary`

Additive batch-analysis fields:

1. `image_paths`
2. `image_count`
3. `cell_count_stats.mean`
4. `cell_count_stats.variance`
5. `batch_mode`
6. `skipped_paths`

Class compatibility rule:

1. Canonical label: `Platelet`
2. Compatibility alias accepted in older outputs: `Platelets`

## Safety Design

The reasoning layer is explicitly conservative:

1. Output must be evidence-grounded and citation-aware.
2. Uncertainty from vision stage propagates to reasoning stage.
3. If evidence is weak, model should avoid definitive diagnosis.
4. Safety flags are emitted for downstream review.

Core flags used in reports:

1. `INSUFFICIENT_EVIDENCE`
2. `HIGH_UNCERTAINTY`
3. `NON_JSON_RESPONSE` (fallback handling)

## Experimental and Evaluation Components

Implemented evaluation utilities include:

1. Retrieval proxy metrics:
   1. Recall@k proxy by expected-term matching.
   2. Citation coverage.
2. Generation quality checks:
   1. JSON schema validity rate.
   2. Citation-attached-claim rate.
   3. Uncertainty-trigger rate.
3. Mode comparison:
   1. Baseline LLM-only.
   2. RAG.
   3. RAG + uncertainty-aware prompting.

These are saved as JSON/CSV artifacts for tables and appendix material.

## Runtime and Environment Notes

### Recommended runtime

1. Google Colab with GPU for smooth end-to-end execution.
2. VS Code Jupyter is supported with proper local setup.

### Local VS Code requirements

1. Place model checkpoints in `./models` or set `THESIS_MODELS_DIR`.
2. Ensure `OPENAI_API_KEY` is set in environment (or Colab Secrets in Colab).
3. Keep notebook execution order consistent.

Required model files for Stage 3 notebook:

1. `yolov8s_blood.pt`
2. `efficientnet_wbc_finetuned.pt`

## Execution Order (Full Thesis Pipeline)

1. `YOLOv8_detection/YOLOv8_on_TBL_PBC_dataset.ipynb`
2. `Efficientnet_classification/Efficientnet_classification.ipynb`
3. `LLM_RAG_Pipline/LLM_Rag_pipline.ipynb`

Within the LLM-RAG notebook, run top-to-bottom to ensure:

1. Models are loaded.
2. Retrieval index is built.
3. `summary` is created before uncertainty visualization and evaluation cells.

## Generated Artifacts (Paper Evidence Map)

### Stage 1 (YOLO)

Common outputs:

1. Detection figures (`predictions`, curves, confusion outputs depending on execution path).
2. `results/run_metadata_yolo.json`.

### Stage 2 (EfficientNet)

Common outputs:

1. `figures/confusion_matrix.png`
2. `figures/training_curves.png`
3. `figures/predictions.png`
4. `results/stage2_uncertainty_summary.json`
5. `results/run_metadata_efficientnet.json`

### Stage 3 (LLM-RAG)

Common outputs:

1. `figures/pipeline_output.png`
2. `figures/uncertainty_analysis.png`
3. `results/eval_summary_<timestamp>.json`
4. `results/eval_modes_<timestamp>.csv`
5. `results/run_metadata_llm_rag.json`

## Thesis Writing Guide (Directly Reusable)

Use this section to build your paper structure quickly.

### Methods chapter should include

1. Three-stage architecture diagram (Detection -> Classification+Uncertainty -> RAG Reasoning).
2. Stage 2 uncertainty equations/concepts (entropy, variance, MC dropout passes).
3. Safety protocol and abstention policy.
4. Output schema and interoperability contract.

### Results chapter should include

1. Detection performance visuals and counts.
2. Classification performance and uncertainty distributions.
3. Qualitative examples of reasoning with citations.
4. Comparative table: baseline vs RAG vs RAG+uncertainty-aware.
5. Cases that trigger `INSUFFICIENT_EVIDENCE` and why this improves safety.

### Appendix should include

1. Run metadata JSON files (all stages).
2. Example output JSON schemas.
3. Full prompt template used for reasoning.
4. Error handling and fallback behavior (retrieval fallback, JSON fallback).

## Known Limitations

1. Performance and runtime depend on available GPU/CPU and memory.
2. Some environments may block native extensions used by vector stores.
3. Retrieval quality depends on PDF quality and extraction success.
4. LLM output structure is guarded but still probabilistic.
5. This is research software and not clinically validated for deployment.

## Compliance and Ethics Note

This repository is an academic prototype for decision support research.

1. It is not a certified medical device.
2. It must not be used as a standalone diagnostic tool.
3. Clinical interpretation requires qualified human oversight.
