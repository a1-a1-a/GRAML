# Multi-Task Ablation Data

This folder contains training sets for multi-task ablations.

## Files

- `no_assessment.json`: 23,180 records; Detection 21,537, Description 823, Localization 820.
- `no_description.json`: 23,177 records; Detection 21,537, Localization 820, Assessment 820.
- `no_location.json`: 23,180 records; Detection 21,537, Description 823, Assessment 820.
- `only_detection.json`: 21,537 records; Detection only.

All files use the same record schema as the main training set and can be passed
directly to the training scripts in `training_inference/`.
