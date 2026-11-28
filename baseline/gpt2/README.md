# GPT-2 Baseline

## Files

- `run_gpt2.py`: training and evaluation entry point
- `model.py`: model wrapper used by the script

## Example

```bash
python baseline/gpt2/run_gpt2.py \
  --train_data_file Dataset/ablations/multi_task/only_detection.json \
  --eval_data_file Dataset/eval/ID_valid.json \
  --test_data_file Dataset/eval/ID_test.json \
  --output_dir outputs/gpt2 \
  --model_type gpt2 \
  --model_name_or_path gpt2 \
  --tokenizer_name gpt2 \
  --do_train \
  --do_eval \
  --do_test \
  --learning_rate 2e-4 \
  --seed 42 \
  --epoch 10 \
  --block_size 512
```
