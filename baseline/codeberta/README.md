# CodeBERTa Baseline

This folder contains the CodeBERTa-style baseline used in GRAML.

## Files

- `run.py`: training, validation, and testing entry point
- `model.py`: model wrapper used during training and evaluation

## What This Folder Is For

Use this folder when you want a baseline based on the CodeBERTa family while keeping the same GRAML data interface.

## Example

```bash
python baseline/codeberta/run.py \
  --train_data_file Dataset/id/Ultimate_train.json \
  --eval_data_file Dataset/id/Ultimate_valid.json \
  --test_data_file Dataset/id/Ultimate_test.json \
  --output_dir outputs/codeberta \
  --model_type roberta \
  --model_name_or_path path/to/codeberta-checkpoint \
  --tokenizer_name path/to/codeberta-checkpoint \
  --do_train \
  --do_eval \
  --do_test \
  --epoch 10 \
  --block_size 512
```

## Notes

- The folder keeps the experiment naming used in the repository.
- Replace `path/to/codeberta-checkpoint` with the specific checkpoint you want to evaluate.
