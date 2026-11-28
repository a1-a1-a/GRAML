# LLM Baselines

OpenAI-compatible API for code-only vulnerability detection. The model returns
a discrete `Yes`/`No` label, so no score threshold is applied.

## Files

- `infer_openai_baseline.py`: zero-shot and few-shot inference.
- `few_shot_examples.json`: fixed 8-shot pool (4 vulnerable and 4
  non-vulnerable); the first 4 form a balanced 4-shot prompt.
- `run_zero_shot.ps1`, `run_zero_shot.sh`, `run_few_shot.ps1`,
  `run_few_shot.sh`: Windows and Linux runners.

## Prompting

Zero-shot uses the code and the fixed detection instruction. Few-shot adds
labeled examples before the target code. The default few-shot configuration is
4-shot, using two vulnerable and two non-vulnerable examples from the fixed
pool.

The example pool is drawn from the training split and is disjoint from all
evaluation sets. The same examples are used for every test set and model.

## Usage

Set the provider configuration in the environment before running:

```text
OPENAI_API_KEY=<your-api-key>
BASE_URL=<provider-endpoint>
MODEL=gpt-5
NUM_SHOTS=4
```

### Runners

```bash
bash baseline/llm_baselines/run_few_shot.sh
bash baseline/llm_baselines/run_zero_shot.sh
```

On Windows, use `./run_few_shot.ps1` and `./run_zero_shot.ps1`.

Outputs go to `predictions/few_shot/<model>/` or
`predictions/zero_shot/<model>/`, with logs and `summary.csv` in the same
directory.

Direct inference command:

```bash
python baseline/llm_baselines/infer_openai_baseline.py \
  --test_path Dataset/eval/CVEfixes_test.json \
  --output_path predictions/zero_shot/CVEfixes.jsonl \
  --model "$MODEL" --base_url "$BASE_URL"
```

To inspect a prompt locally before an API call:

```bash
python baseline/llm_baselines/infer_openai_baseline.py \
  --test_path Dataset/eval/CVEfixes_test.json \
  --output_path outputs/prompt_preview.jsonl \
  --model "$MODEL" \
  --examples_path baseline/llm_baselines/few_shot_examples.json \
  --num_shots 4 --limit 1 --dry_run
```

## Datasets

The runners use the final evaluation files in `Dataset/eval/`:
CVEfixes, Devign, DiverseVul, Juliet, PrimeVul, ReVeal, and the ID test set.
