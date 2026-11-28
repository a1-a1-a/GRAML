#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT="$SCRIPT_DIR/infer_openai_baseline.py"
EXAMPLES="$SCRIPT_DIR/few_shot_examples.json"
DATASET_DIR="$ROOT_DIR/Dataset/eval"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running}"
: "${BASE_URL:?Set BASE_URL before running}"
MODEL="${MODEL:-gpt-5}"
NUM_SHOTS="${NUM_SHOTS:-4}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MODEL_TAG="$(printf '%s' "$MODEL" | tr -c 'A-Za-z0-9._-' '_')"
OUTPUT_DIR="$ROOT_DIR/predictions/few_shot/$MODEL_TAG"
SUMMARY_CSV="$OUTPUT_DIR/summary.csv"
mkdir -p "$OUTPUT_DIR"

datasets=(
  "CVEfixes:$DATASET_DIR/CVEfixes_test.json"
  "Devign:$DATASET_DIR/Devign_test.json"
  "DiverseVul:$DATASET_DIR/DiverseVul_test.json"
  "Juliet:$DATASET_DIR/Juliet_test.json"
  "PrimeVul:$DATASET_DIR/PrimeVul_test.json"
  "ReVeal:$DATASET_DIR/ReVeal_test.json"
  "ID_Test:$DATASET_DIR/ID_test.json"
)

for entry in "${datasets[@]}"; do
  name="${entry%%:*}"
  path="${entry#*:}"
  output="$OUTPUT_DIR/$name.jsonl"
  log="$OUTPUT_DIR/${name}_log.txt"

  echo "========================================"
  echo "Running: $name (${NUM_SHOTS}-shot)"
  echo "========================================"
  python "$SCRIPT" \
    --test_path "$path" \
    --output_path "$output" \
    --summary_csv_path "$SUMMARY_CSV" \
    --model "$MODEL" \
    --base_url "$BASE_URL" \
    --api_key_env "OPENAI_API_KEY" \
    --examples_path "$EXAMPLES" \
    --num_shots "$NUM_SHOTS" \
    --max_example_code_chars 2000 \
    --num_workers "$NUM_WORKERS" \
    --resume \
    --disable_thinking \
    --max_output_tokens 4 \
    --temperature 0.0 \
    --progress_every 20 \
    2>&1 | tee "$log"

  echo "DONE: $name"
done

echo ""
echo "ALL DATASETS COMPLETED"
