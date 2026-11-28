# GRAML

GRAML is a repository for LLM-based vulnerability analysis. It contains
the datasets, the Unsloth training and inference pipeline, encoder and LLM
baselines, robustness transformations, and the ToT-VR description-generation
code.

## Repository layout

```text
GRAML/
|-- Dataset/                    # datasets and construction scripts
|-- training_inference/         # main training and inference
|-- baseline/                   # encoder, LLM, and recent-method baselines
|-- robustness_transformation/  # perturbation code
`-- ToT/                        # ToT-VR, CoT, and Simple ToT code
```

Each folder has a README with its files and commands.

## Final data

- train: `Dataset/id/expansion/Ultimate_train_expanded.json` (24,000)
- validation: `Dataset/eval/ID_valid.json` (3,000)
- test: `Dataset/eval/ID_test.json` (3,000)
- OOD: `Dataset/eval/{CVEfixes,Devign,DiverseVul,Juliet,PrimeVul,ReVeal}_test.json`

The training split contains Detection 21,537, Description 823, Localization 820,
and Assessment 820 records.

Additional datasets are under `Dataset/robustness/`,
`Dataset/ablations/`, and `Dataset/graph_context/`.

## Data format

Records use `instruction`, `input`, `output`, and a task key (`Task`, `task`,
`category`, or `type`). Scripts accept a JSON list or an object with a top-level
`data` field.

## Setup

```bash
bash install.sh cu124
```

The script creates a `graml` conda environment and installs the dependencies.
Available CUDA tags are `cu121`, `cu124`, and `cu128`. For a manual install, use
`pip install -r requirements.txt`; see the
[Unsloth installation guide](https://docs.unsloth.ai/get-started/install-and-update)
if a CUDA-specific build is needed.

## Training

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

Use `--max_seq_length 8192` for Mistral-7B-Instruct-v0.2 and 16384 for the other
backbones.

## Inference

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

The threshold is selected on `ID_valid.json` by F1 over 0.01-0.90 and
fixed for test.

## Robustness

```bash
python robustness_transformation/apply_transformations.py \
  --input_path Dataset/eval/ID_test.json \
  --output_path outputs/robustness/Ultimate_test_obfuscate.json \
  --transforms tf_1 tf_2 tf_3
```

The three categories and all commands are in
`robustness_transformation/README.md`.

## ToT-VR

`ToT/` contains `tot_vr.py`, `cot_baseline.py`, `simple_tot.py`, and the
related preparation and integration scripts.

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="<provider-endpoint>"
python ToT/tot_vr.py
```

## Baselines

`baseline/` contains CodeBERT, CodeBERTa, GPT-2, GraphCodeBERT, RoBERTa,
UniXcoder, VulBERTa, LLM API baselines, and configurations for SAGE, LOSVER, and
VulLLM. The encoder baselines use the Detection-only subset and the same
ID-validation threshold protocol.
