# RoBERTa Baseline

This folder contains the RoBERTa-based baseline.

## Files

- `run.py`: training, validation, and testing entry point
- `model.py`: model wrapper used by the script

## What This Folder Is For

Use this folder to benchmark a standard RoBERTa-style classifier on the GRAML datasets.

## Example

```bash
python baseline/roberta/run.py \
  --train_data_file Dataset/id/Ultimate_train.json \
  --eval_data_file Dataset/id/Ultimate_valid.json \
  --test_data_file Dataset/id/Ultimate_test.json \
  --output_dir outputs/roberta \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --tokenizer_name roberta-base \
  --do_train \
  --do_eval \
  --do_test \
  --epoch 10 \
  --block_size 512
```

## Notes

- This baseline uses the same GRAML-style `input` and `output` fields as the rest of the repository.
- You can change dropout behavior with `--dropout_probability`.
