# ToT Guide

ToT-VR, CoT, and Simple ToT code for generating vulnerability descriptions.

## Files

- `tot_vr.py`: ToT-VR pipeline
- `run_tot_411.py`: resumable runner for the 411-sample generation
- `cot_baseline.py`: source-only chain-of-thought baseline
- `simple_tot.py`: source-only Simple ToT baseline
- `integrate_simple_tot.py`: replaces the 411 positive Description outputs
- `prepare_tot_inputs.py`: builds inputs from critical lines and external CVE descriptions
- `tot_inputs_train.jsonl`: prepared inputs

## ToT-VR

`tot_vr.py` runs five stages: branch initialization, reflection and pruning,
refinement with structural evidence, verification with backtracking, and final
synthesis. Branch count is dynamic between `--min_branches` and
`--max_branches` (default 2 and 6). Without a CVE description, verification uses
structural evidence.

Set the provider credentials and run:

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="<provider-endpoint>"
python ToT/run_tot_411.py
```

`TOT_MODEL` sets the model name (default `gpt-5`).

## Building the inputs

Inputs join the 411 positive Description samples with
`Dataset/graph_context/critical_lines/output/train_crit.json` and external CVE
descriptions.

```bash
python ToT/prepare_tot_inputs.py \
  --description_source Dataset/id/expansion/Ultimate_train_expanded.json \
  --cve_jsonl /path/to/train.jsonl \
  --diversevul /path/to/diversevul_20230702.json \
  --output ToT/tot_inputs_train.jsonl \
```

## CoT baseline

```bash
python ToT/cot_baseline.py \
  --input ToT/tot_inputs_train.jsonl \
  --output ToT/experiments/cot_411.jsonl \
  --summary ToT/experiments/cot_411_summary.json \
  --training_output ToT/experiments/cot_411_training.json \
  --model gpt-5 \
  --workers 4 \
  --resume
```

## Simple ToT baseline

```bash
python ToT/simple_tot.py \
  --input ToT/tot_inputs_train.jsonl \
  --output ToT/experiments/simple_tot_411.jsonl \
  --summary ToT/experiments/simple_tot_411_summary.json \
  --training_output ToT/experiments/simple_tot_411_training.json \
  --model gpt-5 \
  --workers 4 \
  --resume
```

## Integrating Simple ToT descriptions

```bash
python ToT/integrate_simple_tot.py \
  --descriptions ToT/experiments/simple_tot_411_training.json \
  --train Dataset/id/expansion/Ultimate_train_expanded.json \
  --output Dataset/id/expansion/Ultimate_train_expanded_simple_tot.json \
  --write
```
