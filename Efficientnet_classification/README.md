# Stage 2 - EfficientNet WBC Classification

This folder contains the Stage 2 notebook for white blood cell subtype classification and uncertainty-aware inference.

Notebook:

1. Efficientnet_classification.ipynb

## Scope

Stage 2 is responsible for:

1. Training and fine-tuning EfficientNet-B0 on peripheral blood cell classes.
2. Producing subtype predictions with confidence scores.
3. Running Monte Carlo Dropout uncertainty estimation for inference.
4. Exporting summary artifacts for thesis evaluation and reproducibility.

## Input and Output Role in Full Pipeline

Input (from Stage 1 in integrated pipeline):

1. WBC crop images.

Output (consumed by Stage 3 in integrated pipeline):

1. predicted_class
2. confidence
3. variance
4. entropy
5. uncertainty_level
6. flagged

Batch inference summary fields:

1. sample_count
2. class_distribution
3. confidence_stats
4. entropy_stats
5. variance_stats
6. flagged_count
7. requires_expert_review

## Training Notes

The notebook keeps the original training flow:

1. Frozen-backbone training phase.
2. Fine-tuning phase with smaller learning rate.

Checkpoint and load logic are preserved in notebook code.

## Uncertainty Inference

The notebook includes uncertainty-aware helpers that align with the LLM pipeline behavior:

1. enable_dropout
2. classify_wbc_with_uncertainty
3. run_stage2_inference

These provide single and batch inference with additive summary outputs.

## Generated Artifacts

Expected files from successful runs:

1. figures/confusion_matrix.png
2. figures/training_curves.png
3. figures/predictions.png
4. results/stage2_uncertainty_summary.json
5. results/run_metadata_efficientnet.json

## Smoke Checks in Notebook

Included smoke checks validate:

1. Single-sample uncertainty output.
2. Batch-sample uncertainty output.
3. Flag propagation and summary statistics.

## Reproducibility

A dedicated metadata cell exports:

1. UTC timestamp
2. Python version
3. Platform
4. Device
5. Package versions

Saved to:

1. results/run_metadata_efficientnet.json

## Dependencies

Typical notebook dependencies:

1. torch
2. timm
3. torchvision
4. numpy
5. opencv-python
6. matplotlib
7. seaborn
7. scikit-learn

## Clinical and Research Disclaimer

This stage is part of an academic research prototype and not a certified medical device. Predictions and uncertainty outputs are decision-support signals and must be reviewed by qualified clinicians.
