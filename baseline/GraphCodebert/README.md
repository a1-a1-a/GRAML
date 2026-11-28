# GraphCodeBERT Baseline

## Files

- `run.py`: training, validation, and testing entry point
- `model.py`: model wrapper used by the script

## Example

```bash
python baseline/GraphCodebert/run.py \
  --train_data_file Dataset/ablations/multi_task/only_detection.json \
  --eval_data_file Dataset/eval/ID_valid.json \
  --test_data_file Dataset/eval/ID_test.json \
  --output_dir outputs/graphcodebert \
  --model_type roberta \
  --model_name_or_path microsoft/graphcodebert-base \
  --tokenizer_name microsoft/graphcodebert-base \
  --do_train \
  --do_eval \
  --do_test \
  --learning_rate 2e-4 \
  --seed 42 \
  --epoch 5 \
  --block_size 512
```
