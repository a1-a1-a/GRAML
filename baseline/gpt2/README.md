# GPT-2 Baseline

This folder contains the GPT-2-based baseline.

## Files

- `run_gpt2.py`: training and evaluation entry point
- `model.py`: model wrapper used by the script

## What This Folder Is For

Use this folder to benchmark a GPT-2-style classifier on the GRAML data format.

## Example

```bash
python baseline/gpt2/run_gpt2.py \
  --train_data_file Dataset/ID_dataset/Ultimate_train.json \
  --eval_data_file Dataset/ID_dataset/Ultimate_valid.json \
  --test_data_file Dataset/ID_dataset/Ultimate_test.json \
  --output_dir outputs/gpt2 \
  --model_type gpt2 \
  --model_name_or_path gpt2 \
  --tokenizer_name gpt2 \
  --do_train \
  --do_eval \
  --do_test \
  --epoch 10 \
  --block_size 512
```

## Notes

- The script includes GPT-2-specific handling for missing padding tokens.
- As with the other baselines, labels are normalized into binary targets.
