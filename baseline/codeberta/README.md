# CodeBERTa Baseline

## Files

- `run.py`: training, validation, and testing entry point
- `model.py`: model wrapper used during training and evaluation

## Example

```bash
python baseline/codeberta/run.py \
  --train_data_file Dataset/ablations/multi_task/only_detection.json \
  --eval_data_file Dataset/eval/ID_valid.json \
  --test_data_file Dataset/eval/ID_test.json \
  --output_dir outputs/codeberta \
  --model_type roberta \
  --model_name_or_path path/to/codeberta-checkpoint \
  --tokenizer_name path/to/codeberta-checkpoint \
  --do_train \
  --do_eval \
  --do_test \
  --learning_rate 2e-4 \
  --seed 42 \
  --epoch 10 \
  --block_size 512
```
