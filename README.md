# GRAML

GRAML is a research repository for LLM-based vulnerability analysis. It releases the datasets, training and inference scripts, baseline implementations, robustness transformations, and ToT-guided vulnerability reasoning code used in our experiments.

The repository is organized so that users can reproduce the main pipeline with their own local paths. All example commands below use repository-relative paths or placeholder directories instead of author-specific absolute server paths.

## Highlights

- `Dataset/` contains in-distribution, out-of-distribution, robustness, and ablation datasets.
- `traing+inference/` contains the Unsloth-based fine-tuning and inference scripts.
- `baseline/` contains seven baseline model implementations.
- `robustness_transformation/` contains code for generating perturbed robustness test sets.
- `ToT/` contains the ToT-guided vulnerability reasoning script and generated vulnerability descriptions.

## Repository Structure

```text
GRAML/
|-- Dataset/
|   |-- Ablation/
|   |-- ID_dataset/
|   |-- OOD_dataset/
|   `-- robustness/
|-- ToT/
|-- baseline/
|-- robustness_transformation/
`-- traing+inference/
```

## Dataset Overview

### 1. `Dataset/ID_dataset/`

This folder contains the in-distribution dataset used in the main pipeline:

- `Ultimate_train.json`: training set
- `Ultimate_valid.json`: validation set
- `Ultimate_test.json`: test set

### 2. `Dataset/OOD_dataset/`

This folder contains out-of-distribution evaluation sets from six external open-source sources:

- `CVEfixes`
- `Devign`
- `DiverseVul`
- `Juliet`
- `PrimeVul`
- `ReVeal`

Each subfolder contains an external `test.json`. The `Juliet` folder also includes `valid.json`.

### 3. `Dataset/robustness/`

This folder contains robustness test sets produced by perturbing the original test data. It includes perturbed versions for:

- `Ultimate_test`
- `CVEfixes`
- `Devign`
- `DiverseVul`
- `Juliet`
- `PrimeVul`
- `ReVeal`

For each dataset, three perturbation types are provided:

- `*_noise.json`
- `*_obfuscate.json`
- `*_structure.json`

### 4. `Dataset/Ablation/`

This folder contains the ablation datasets:

- `no_assessment.json`
- `no_description.json`
- `no_location.json`
- `only_detection.json`

These files are used to study the contribution of different task components.

## Data Format

The training and inference scripts accept either:

- a JSON list of samples, or
- a JSON object with a top-level `data` field

Each sample typically contains the following fields:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "Task": "detection"
}
```

The scripts also tolerate several alternative task keys such as `task`, `category`, and `type`.

## Environment Setup

We recommend using Python 3.10+ with a CUDA-enabled GPU for the Unsloth training and inference pipeline.

1. Install a PyTorch build that matches your CUDA environment.
2. Install the main dependencies:

```bash
pip install unsloth transformers datasets trl accelerate peft bitsandbytes tqdm openai
```

If your environment requires a CUDA-specific Unsloth installation, please follow the official Unsloth installation guide:

- [Unsloth installation](https://docs.unsloth.ai/get-started/install-and-update)

## Main Training Pipeline

The main training script is:

```text
traing+inference/train_unsloth.py
```

Example:

```bash
python traing+inference/train_unsloth.py \
  --model_name path/to/base-model \
  --train_path Dataset/ID_dataset/Ultimate_train.json \
  --valid_path Dataset/ID_dataset/Ultimate_valid.json \
  --output_dir checkpoints/deepseek6.7b_r64a16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --max_seq_length 16384 \
  --lora_r 64 \
  --lora_alpha 16 \
  --prompt_style deepseek
```

### Important Arguments

- `--model_name`: path or model identifier of the base model
- `--train_path`: training dataset path
- `--valid_path`: validation dataset path
- `--output_dir`: directory to save checkpoints and tokenizer files
- `--max_seq_length`: maximum sequence length used by Unsloth
- `--lora_r`, `--lora_alpha`: LoRA hyperparameters
- `--prompt_style`: one of `mistral`, `deepseek`, `alpaca`, or `qwen`

## Main Inference Pipeline

The main inference script is:

```text
traing+inference/infer_unsloth.py
```

Example:

```bash
python traing+inference/infer_unsloth.py \
  --adapter_path checkpoints/your_run/checkpoint-1731 \
  --test_path Dataset/ID_dataset/Ultimate_test.json \
  --valid_path Dataset/ID_dataset/Ultimate_valid.json \
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

### Notes

- `--adapter_path` should point to a saved LoRA checkpoint directory or the final saved adapter directory.
- `--output_path` can be either a file path or a directory. If it is a directory, the script automatically generates an output filename.
- `--curve_path` can also be a directory. The threshold-search curve will be saved automatically.
- `--find_optimal` searches the best threshold on the validation set.
- `--metric` controls which metric is optimized during threshold search.
- If direct adapter loading fails, provide `--base_model path/to/base-model`.
- Use `--only_detection` if you want to evaluate detection-only samples.

## Robustness Transformation

The robustness transformation code is located in:

```text
robustness_transformation/
```

The main script is:

```bash
python robustness_transformation/apply_transformations.py \
  --input_path Dataset/ID_dataset/Ultimate_test.json \
  --output_path outputs/robustness/Ultimate_test_tf1_tf6.json \
  --transforms tf_1 tf_6
```

Transformation functions are defined in `robustness_transformation/transformations.py`.

## ToT-Guided Vulnerability Reasoning

The `ToT/` folder contains a ToT-guided vulnerability reasoning implementation:

- `ToT/tot_vr.py`: reference implementation of the ToT-VR pipeline
- `ToT/ToT_description.json`: generated vulnerability descriptions

`tot_vr.py` uses the OpenAI API. Set your API key before running:

```bash
export OPENAI_API_KEY=your_api_key
python ToT/tot_vr.py
```

The current script includes a built-in example. You can adapt it for your own code snippets or integrate the `ToTVRGenerator` class into a larger evaluation pipeline.

## Baselines

The `baseline/` folder contains implementations for seven baseline models:

- CodeBERT
- CodeBERTa
- GPT-2
- GraphCodeBERT
- RoBERTa
- UniXcoder
- VulBERTa

Each baseline has its own subdirectory and runnable code. In particular, `baseline/VulBERTa/README.md` provides additional model-specific details.

## Reproducibility Notes

- Replace all placeholder model and checkpoint paths with paths in your own environment.
- Keep the repository directory names unchanged when using the commands above, especially `Dataset/` and `traing+inference/`.
- The inference script is designed for Yes/No vulnerability detection and supports validation-based threshold tuning.
- Output directories such as `checkpoints/` and `outputs/` are only examples; you can organize them however you prefer.

## Citation

If you find this repository useful in your research, please cite the corresponding paper. The BibTeX entry can be added here after publication.
