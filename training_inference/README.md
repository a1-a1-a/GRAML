# Training and Inference Guide

This folder contains the main GRAML pipeline built on top of Unsloth.

## Files

- `train_unsloth.py`: LoRA fine-tuning script
- `infer_unsloth.py`: inference and threshold-selection script

## What This Folder Is For

This is the primary folder to use if you want to reproduce the main GRAML experiments.

## Training Example

```bash
python training_inference/train_unsloth.py \
  --model_name path/to/base-model \
  --train_path Dataset/id/Ultimate_train.json \
  --valid_path Dataset/id/Ultimate_valid.json \
  --output_dir checkpoints/graml_main_run \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --max_seq_length 16384 \
  --lora_r 64 \
  --lora_alpha 16 \
  --prompt_style deepseek
```

## Inference Example

```bash
python training_inference/infer_unsloth.py \
  --adapter_path checkpoints/graml_main_run/checkpoint-1731 \
  --test_path Dataset/id/Ultimate_test.json \
  --valid_path Dataset/id/Ultimate_valid.json \
  --output_path outputs/predictions \
  --curve_path outputs/curves \
  --summary_csv_path outputs/all_results_summary.csv \
  --prompt_style deepseek \
  --find_optimal \
  --metric f1 \
  --load_in_4bit \
  --max_seq_length 16384 \
  --threshold_min 0.01 \
  --threshold_max 0.9 \
  --threshold_step 0.01
```

## Practical Notes

- Replace placeholder model and checkpoint paths with your own local paths.
- The scripts support several prompt templates: `mistral`, `deepseek`, `alpaca`, `qwen`.
- `infer_unsloth.py` is designed around binary `Yes` or `No` vulnerability detection and supports threshold search on a validation set.
- If LoRA adapter loading fails during inference, provide `--base_model`.
