# UniXcoder Baseline

## Files

- `run.py`: training, validation, testing, and metric aggregation script
- `model.py`: model wrapper and classification head

## Example

```bash
python baseline/Unixcoder/run.py \
  --train_data_file Dataset/ablations/multi_task/only_detection.json \
  --eval_data_file Dataset/eval/ID_valid.json \
  --test_data_file Dataset/eval/ID_test.json \
  --output_dir outputs/unixcoder \
  --result_dir outputs/unixcoder/results \
  --model_name_or_path microsoft/unixcoder-base \
  --do_train \
  --do_eval \
  --do_test \
  --learning_rate 2e-4 \
  --seed 42 \
  --num_train_epochs 10 \
  --block_size 512
```
