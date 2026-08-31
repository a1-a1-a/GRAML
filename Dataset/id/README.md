# In-Distribution Dataset

This folder contains the main in-distribution dataset used by GRAML.

## Files

- `Ultimate_train.json`: training split (21,600 samples)
- `Ultimate_valid.json`: validation split (2,310 samples)
- `Ultimate_test.json`: test split (2,311 samples)
- `raw_cve/`: raw CVE samples collected from 2019 to 2026 used to construct the training set

## Task Distribution (Training)

| Task | Count |
|---|---|
| Detection | 19,137 |
| Description | 823 |
| Localization | 820 |
| Assessment | 820 |

## What This Folder Is For

This is the default dataset folder for the main GRAML pipeline:

- fine-tuning in `training_inference/train_unsloth.py`
- threshold selection in `training_inference/infer_unsloth.py`
- baseline comparisons in `baseline/`

## Recommended Workflow

1. Train on `Ultimate_train.json`
2. Select hyperparameters and thresholds on `Ultimate_valid.json`
3. Report final results on `Ultimate_test.json`

## Notes

- The files are already formatted for the scripts in this repository.
- The task field is typically `detection`, but the repository also supports richer task variants elsewhere.
