# VulBERTa Baseline

## Files

- `run_vulberta.py`: GRAML-friendly training and testing script
- `models.py`, `custom.py`: supporting model and utility code
- `Pretraining_VulBERTa.ipynb`: upstream pretraining notebook
- `Finetuning_VulBERTa-MLP.ipynb`: upstream fine-tuning notebook
- `Evaluation_VulBERTa-MLP.ipynb`: upstream evaluation notebook
- `Finetuning+evaluation_VulBERTa-CNN.ipynb`: upstream CNN variant notebook
- `tokenizer/`: local tokenizer assets
- `data/`: upstream data note and download instructions

## Example

```bash
python baseline/VulBERTa/run_vulberta.py \
  --train_file Dataset/ablations/multi_task/only_detection.json \
  --valid_file Dataset/eval/ID_valid.json \
  --test_file Dataset/eval/ID_test.json \
  --output_dir outputs/vulberta \
  --model_name_or_path claudios/VulBERTa-MLP-Devign \
  --learning_rate 2e-5 \
  --seed 123456 \
  --epochs 5 \
  --batch_size 16 \
  --block_size 512 \
  --do_train \
  --do_test
```

## Subfolder Guides

- [data](data/README.md)
- [tokenizer](tokenizer/README.md)
