# UniXcoder Baseline

This folder contains the UniXcoder-based baseline.

## Files

- `run.py`: training, validation, testing, and metric aggregation script
- `model.py`: model wrapper and classification head

## What This Folder Is For

Use this folder when you want to benchmark UniXcoder on the GRAML datasets.

## Example

```bash
python baseline/Unixcoder/run.py \
  --train_data_file Dataset/ID_dataset/Ultimate_train.json \
  --eval_data_file Dataset/ID_dataset/Ultimate_valid.json \
  --test_data_file Dataset/ID_dataset/Ultimate_test.json \
  --output_dir outputs/unixcoder \
  --result_dir outputs/unixcoder/results \
  --model_name_or_path microsoft/unixcoder-base \
  --do_train \
  --do_eval \
  --do_test \
  --num_train_epochs 10 \
  --block_size 512
```

## Notes

- The script exposes additional dataset-key arguments such as `--code_key` and `--label_key`.
- It also supports metric export through `--csv_path`, `--metrics_path`, and `--metrics_format`.
