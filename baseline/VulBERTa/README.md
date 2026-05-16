# VulBERTa Baseline

This folder contains the VulBERTa-based baseline and related upstream assets.

## Files

- `run_vulberta.py`: GRAML-friendly training and testing script
- `models.py`, `custom.py`: supporting model and utility code
- `Pretraining_VulBERTa.ipynb`: upstream pretraining notebook
- `Finetuning_VulBERTa-MLP.ipynb`: upstream fine-tuning notebook
- `Evaluation_VulBERTa-MLP.ipynb`: upstream evaluation notebook
- `Finetuning+evaluation_VulBERTa-CNN.ipynb`: upstream CNN variant notebook
- `tokenizer/`: local tokenizer assets
- `data/`: upstream data note and download instructions

## What This Folder Is For

This folder serves two roles:

- it preserves upstream VulBERTa materials
- it provides a simplified `run_vulberta.py` entry point that fits the GRAML repository structure

## Example

```bash
python baseline/VulBERTa/run_vulberta.py \
  --train_file Dataset/ID_dataset/Ultimate_train.json \
  --valid_file Dataset/ID_dataset/Ultimate_valid.json \
  --test_file Dataset/ID_dataset/Ultimate_test.json \
  --output_dir outputs/vulberta \
  --model_name_or_path claudios/VulBERTa-MLP-Devign \
  --epochs 5 \
  --batch_size 16 \
  --block_size 512 \
  --do_train \
  --do_test
```

## Notes

- `run_vulberta.py` tries to use the local tokenizer files in `tokenizer/` when they are present.
- If the custom tokenizer files are missing or incomplete, the script falls back to `roberta-base`.
- The upstream notebooks are kept for users who want the original VulBERTa workflow.

## Subfolder Guides

- [data](data/README.md)
- [tokenizer](tokenizer/README.md)
