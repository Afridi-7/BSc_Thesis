# Blood Smear Domain Expert

End-to-end BSc thesis project for a domain-specialized hematology assistant that integrates:

1. Stage 1: YOLOv8 blood cell detection
2. Stage 2: EfficientNet WBC subtype classification with uncertainty
3. Stage 3: RAG plus Gemini clinical reasoning

The codebase is organized as three executable notebooks that are now aligned on schema, batch handling, safety behavior, and reproducibility outputs.

## Repository Layout

```
BSc_Thesis/
├── README.md
├── YOLOv8_detection/
│   ├── YOLOv8_on_TBL_PBC_dataset.ipynb
│   └── results/
├── Efficientnet_classification/
│   ├── Efficientnet_classification.ipynb
│   └── results/
└── LLM_RAG_Pipline/
    ├── LLM_Rag_pipline.ipynb
    └── results/
```

## Pipeline Summary

### Stage 1: Detection (YOLOv8)

1. Detects WBC, RBC, and Platelet objects from smear images.
2. Supports single-image and multi-image input.
3. Produces aggregate statistics for batch mode.

### Stage 2: Classification (EfficientNet)

1. Classifies WBC subtype from crops.
2. Uses Monte Carlo Dropout during inference for uncertainty estimation.
3. Produces confidence, entropy, variance, and flagged uncertainty markers.

### Stage 3: Clinical Reasoning (RAG + Gemini)

1. Builds retrieval query from structured vision summary.
2. Retrieves evidence chunks from hematology knowledge base.
3. Generates structured, safety-aware reasoning with citations.
4. Abstains and raises safety flags when evidence is insufficient.

## Cross-Notebook Output Contract

The following required keys are preserved for backward compatibility:

1. cell_counts
2. total_cells
3. wbc_differential
4. uncertainty_summary

Additive keys used for batch mode and analysis:

1. image_paths
2. image_count
3. cell_count_stats:
   1. mean
   2. variance
4. batch_mode
5. skipped_paths

Class compatibility rule:

1. Canonical internal class: Platelet
2. Compatibility mapping accepts Platelets

## Notebook Responsibilities

### YOLO Notebook

Path: YOLOv8_detection/YOLOv8_on_TBL_PBC_dataset.ipynb

Main responsibilities:

1. Stage 1 model training/evaluation and prediction visualization.
2. Reusable Stage 1 single plus batch inference helpers.
3. Aggregated detection stats (total, mean, variance).
4. Reproducibility metadata export.
5. Stage 1 smoke checks.

Generated artifacts include:

1. figures/predictions.png
2. results/run_metadata_yolo.json

### EfficientNet Notebook

Path: Efficientnet_classification/Efficientnet_classification.ipynb

Main responsibilities:

1. Stage 2 model training and fine-tuning.
2. Stage 2 uncertainty-aware inference helpers.
3. Single plus batch Stage 2 inference summaries.
4. Reproducibility metadata export.
5. Stage 2 smoke checks.

Generated artifacts include:

1. figures/predictions.png
2. figures/confusion_matrix.png
3. figures/training_curves.png
4. results/stage2_uncertainty_summary.json
5. results/run_metadata_efficientnet.json

### LLM RAG Notebook

Path: LLM_RAG_Pipline/LLM_Rag_pipline.ipynb

Main responsibilities:

1. Combined Stage 1 and Stage 2 summary generation.
2. RAG query construction and retrieval.
3. Safety-aware structured LLM reasoning function.
4. Evaluation utilities for retrieval and generation quality.
5. Reproducibility metadata export.
6. End-to-end smoke checks.

Generated artifacts include:

1. figures/pipeline_output.png
2. figures/uncertainty_analysis.png
3. results/eval_summary_<timestamp>.json
4. results/eval_modes_<timestamp>.csv
5. results/run_metadata_llm_rag.json

## Safety and Clinical Guardrails

The reasoning stage enforces:

1. Evidence-grounded output with citations.
2. Uncertainty-aware wording.
3. Non-definitive behavior when evidence is insufficient.
4. Safety flag propagation, including:
   1. INSUFFICIENT_EVIDENCE
   2. HIGH_UNCERTAINTY

This project is decision-support research software, not a diagnostic medical device.

## How To Run

Recommended execution context:

1. Google Colab runtime with GPU enabled.
2. Run each notebook from top to bottom.
3. Keep notebook-specific results directories for outputs.

Suggested run order for full thesis pipeline:

1. YOLOv8_detection/YOLOv8_on_TBL_PBC_dataset.ipynb
2. Efficientnet_classification/Efficientnet_classification.ipynb
3. LLM_RAG_Pipline/LLM_Rag_pipline.ipynb

## Smoke Test Checklist

Run the smoke-check cells included in each notebook to verify:

1. Stage 1 single-image inference.
2. Stage 1 batch inference.
3. Stage 2 single-image uncertainty output.
4. Stage 2 batch uncertainty output.
5. RAG reasoning response generation.
6. Evaluation utility output and result file creation.
7. Required key presence in summary schema.

## Reproducibility

Each notebook now writes run metadata JSON containing:

1. UTC timestamp
2. Python version
3. Platform info
4. Device info
5. Key package versions

Use these metadata files in thesis appendix for reproducibility evidence.

## Known Limitations

1. Results are dependent on notebook runtime environment and available GPU memory.
2. Some editor diagnostics in local VS Code may show unresolved imports for Colab-only packages.
3. LLM JSON compliance is strongly encouraged by prompt and guarded by fallback logic but still model-dependent.
4. Clinical outputs must be reviewed by qualified professionals.

## Thesis-Ready Evidence To Include

For submission, include:

1. Detection and classification figures from figures directories.
2. stage2_uncertainty_summary.json.
3. eval_summary_<timestamp>.json and eval_modes_<timestamp>.csv.
4. run_metadata JSON files from all three notebooks.
5. Short explanation of safety flags and abstention behavior.

## Disclaimer

This repository is an academic research prototype for a BSc thesis. It is not a certified medical device and must not be used as a standalone clinical decision system.
