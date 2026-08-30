# CodeBERT Baseline

This folder contains the CodeBERT-based vulnerability detection baseline.

## Files

- `run.py`: training, validation, and testing entry point
- `model.py`: model wrapper used by the training script

## What This Folder Is For

Use this folder to run a CodeBERT baseline on the GRAML JSON datasets.

## Example

```bash
python baseline/codebert/run.py \
  --train_data_file Dataset/id/Ultimate_train.json \
  --eval_data_file Dataset/id/Ultimate_valid.json \
  --test_data_file Dataset/id/Ultimate_test.json \
  --output_dir outputs/codebert \
  --model_type roberta \
  --model_name_or_path microsoft/codebert-base \
  --tokenizer_name microsoft/codebert-base \
  --do_train \
  --do_eval \
  --do_test \
  --epoch 10 \
  --block_size 512
```

## Notes

- The script converts `Yes` and `No` labels into binary classification targets.
- Results can be appended to a CSV file with `--csv_path`.
