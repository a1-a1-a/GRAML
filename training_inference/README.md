# Training and Inference Guide

This folder contains the main GRAML pipeline built on top of Unsloth.

## Files

- `train_unsloth.py`: LoRA fine-tuning script
- `infer_unsloth.py`: inference and threshold-selection script
- `zero_shot_eval.py`: direct inference with a base model and an ID-validation-selected threshold
- `task_prompts.py`: canonical task instructions and the shared system prompt

Direct CPG targets keep their target-format instruction, which asks for raw Joern
CPG edge records instead of a natural-language description.

## Training Example

```bash
python training_inference/train_unsloth.py \
  --model_name path/to/base-model \
  --train_path Dataset/id/expansion/Ultimate_train_expanded.json \
  --valid_path Dataset/eval/ID_valid.json \
  --output_dir checkpoints/graml_main_run \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --max_seq_length 16384 \
  --lora_r 32 \
  --lora_alpha 16 \
  --prompt_style deepseek
```

### Prompt style

Select `--prompt_style` for the backbone:

| Backbone | `--prompt_style` |
|---|---|
| CodeLlama-7B | `alpaca` |
| Mistral-7B-Instruct-v0.2 | `mistral` |
| DeepSeek-Coder-6.7B | `deepseek` |
| Qwen2.5-Coder-7B | `qwen` |

### Sequence length

Use `--max_seq_length 8192` for Mistral-7B-Instruct-v0.2 and
`--max_seq_length 16384` for CodeLlama-7B, DeepSeek-Coder-6.7B, and
Qwen2.5-Coder-7B.

## Inference Example

```bash
python training_inference/infer_unsloth.py \
  --adapter_path checkpoints/graml_main_run \
  --test_path Dataset/eval/ID_test.json \
  --valid_path Dataset/eval/ID_valid.json \
  --output_path outputs/predictions \
  --curve_path outputs/curves \
  --summary_csv_path outputs/all_results_summary.csv \
  --prompt_style deepseek \
  --find_optimal \
  --metric f1 \
  --max_seq_length 16384 \
  --threshold_min 0.01 \
  --threshold_max 0.9 \
  --threshold_step 0.01
```

The selected threshold is fixed before evaluating ID, OOD, and robustness
test sets.

## Direct Inference (No Fine-Tuning)

Use `zero_shot_eval.py` to evaluate a base model. A ready-to-use dataset list is
provided in `training_inference/direct_eval_datasets.json`:

```bash
python training_inference/zero_shot_eval.py \
  --model_path path/to/base-model \
  --model_name CodeLlama-7B \
  --prompt_style alpaca \
  --valid_path Dataset/eval/ID_valid.json \
  --datasets_json training_inference/direct_eval_datasets.json \
  --out_dir outputs/direct \
  --find_optimal \
  --metric f1 \
  --threshold_min 0.01 \
  --threshold_max 0.90 \
  --threshold_step 0.01 \
  --curve_path outputs/direct/curves \
  --max_seq_length 16384
```

The selected threshold and validation search metadata are written to
`outputs/direct/<model_name>/validation_threshold.json`. Each prediction row
records both `threshold` and `threshold_source`.

Prompt styles for direct inference:

| Backbone | `--prompt_style` |
|---|---|
| CodeLlama-7B | `alpaca` |
| Mistral-7B-Instruct-v0.2 | `mistral` |
| DeepSeek-Coder-6.7B | `deepseek` |
| Qwen2.5-Coder-7B | `qwen` |

`zero_shot_eval.py` defaults to 16,384 input tokens and automatically uses
8,192 when `--model_name` contains Mistral. The threshold is selected on
`ID_valid.json` over 0.01-0.90 in steps of 0.01 and applied unchanged to
every test set.
