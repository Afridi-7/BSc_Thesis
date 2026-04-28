# Stage 1 - YOLOv8 Detection Results

This folder stores Stage 1 result artifacts from:

1. YOLOv8_on_TBL_PBC_dataset.ipynb

## Stage 1 Responsibilities

1. Detect WBC, RBC, and Platelet from blood smear images.
2. Support both single-image and batch-image inference.
3. Provide aggregate statistics for batch mode.
4. Produce visualization output for thesis reporting.

## Output Compatibility

Stage 1 class compatibility rule:

1. Internal canonical class: Platelet
2. Compatibility mapping accepts Platelets

Batch-capable Stage 1 outputs include:

1. image_paths
2. image_count
3. total_counts
4. cell_count_stats
5. skipped_paths

## Expected Artifacts

Depending on run path and execution order, expected outputs include:

1. predictions.png
2. training_curves.png
3. confusion_matrix.png
4. PR_curve.png
5. F1_curve.png
6. run_metadata_yolo.json

Note:

1. Some artifacts are generated in figures/ and some in results/ depending on notebook save calls.
2. Keep both folders when exporting thesis evidence.

## Validation Notes

The notebook includes smoke checks for:

1. Single-image Stage 1 inference.
2. Batch-image Stage 1 inference.
3. Aggregate mean and variance calculations.

## Reproducibility

A metadata cell exports runtime metadata to:

1. results/run_metadata_yolo.json

It includes:

1. UTC timestamp
2. Python version
3. Platform
4. Device
5. Package versions

## Disclaimer

This is a research-stage artifact set for thesis evaluation only. It is not a clinical diagnostic output package.
