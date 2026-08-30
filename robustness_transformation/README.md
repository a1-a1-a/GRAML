# Robustness Transformation Guide

This folder contains the code used to generate perturbed robustness datasets.

## Files

- `apply_transformations.py`: command-line entry point for applying perturbations to a dataset
- `transformations.py`: transformation function library

## What This Folder Is For

Use this folder when you want to:

- create new robustness variants from an existing GRAML-formatted dataset
- reproduce the perturbation workflow behind `Dataset/robustness/`
- test additional transformation combinations not already released in the repository

## Example

```bash
python robustness_transformation/apply_transformations.py \
  --input_path Dataset/id/Ultimate_test.json \
  --output_path outputs/robustness/Ultimate_test_tf1_tf6.json \
  --transforms tf_1 tf_6
```

## Notes

- The script reads `input` as the code field and rewrites that field after transformation.
- The transformation names correspond to functions such as `tf_1`, `tf_2`, and so on in `transformations.py`.
- Some transformations are intended for more specialized usage and are skipped by the simple application path.
