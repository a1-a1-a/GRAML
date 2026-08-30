# In-Distribution Dataset

This folder contains the main in-distribution dataset used by GRAML.

## Files

- `Ultimate_train.json`: training split
- `Ultimate_valid.json`: validation split
- `Ultimate_test.json`: test split

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
