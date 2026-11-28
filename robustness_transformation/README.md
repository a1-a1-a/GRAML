# Robustness Transformation Guide

This folder contains the code used to generate the perturbed robustness datasets.

## Files

- `apply_transformations.py`: command-line entry point for applying perturbations to a dataset
- `transformations.py`: transformation function library

## Transformation Categories

The robustness files use three combined categories:

| Category | Transformations |
|---|---|
| Semantic obfuscation | `tf_1`, `tf_2`, `tf_3` |
| Long-context noise | `tf_4`, `tf_5` |
| Structural restructuring | `tf_6`, `tf_7` |

The transformations in a category are applied together to each test function.

## Example

```bash
# Semantic obfuscation category
python robustness_transformation/apply_transformations.py \
  --input_path Dataset/eval/ID_test.json \
  --output_path outputs/robustness/Ultimate_test_obfuscate.json \
  --transforms tf_1 tf_2 tf_3

# Long-context noise category
python robustness_transformation/apply_transformations.py \
  --input_path Dataset/eval/ID_test.json \
  --output_path outputs/robustness/Ultimate_test_noise.json \
  --transforms tf_4 tf_5

# Structural restructuring category
python robustness_transformation/apply_transformations.py \
  --input_path Dataset/eval/ID_test.json \
  --output_path outputs/robustness/Ultimate_test_structure.json \
  --transforms tf_6 tf_7
```

## Notes

- The implementation provides exactly the seven transformations `tf_1`–`tf_7`.
