# GraphCodeBERT Baseline

This folder contains the GraphCodeBERT-based baseline.

## Files

- `run.py`: training, validation, and testing entry point
- `model.py`: model wrapper used by the script

## What This Folder Is For

Use this folder to evaluate GraphCodeBERT as a baseline on the GRAML data splits.

## Example

```bash
python baseline/graphcodebert/run.py \
  --train_data_file Dataset/id/Ultimate_train.json \
  --eval_data_file Dataset/id/Ultimate_valid.json \
  --test_data_file Dataset/id/Ultimate_test.json \
  --output_dir outputs/graphcodebert \
  --model_type roberta \
  --model_name_or_path microsoft/graphcodebert-base \
  --tokenizer_name microsoft/graphcodebert-base \
  --do_train \
  --do_eval \
  --do_test \
  --epoch 10 \
  --block_size 512
```

## Notes

- This directory follows the same script structure as several other transformer baselines in the repository.
- Metrics can be exported via `--csv_path`.
