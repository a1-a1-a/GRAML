# LLM Baselines (Few-shot / Zero-shot)

This folder contains baseline scripts that call large language models through an OpenAI-compatible API.

## Files

- `infer_openai_baseline.py`: generic LLM inference for vulnerability detection (Yes/No), supporting zero-shot and few-shot prompting.
.\run_zero_shot.ps1

## Usage

.\run_zero_shot.ps1

```powershell
$env:OPENAI_API_KEY = "your-key"
.\run_zero_shot.ps1
```

To switch to a different LLM (e.g., GLM, DeepSeek, Qwen), change `$MODEL` and `$BASE_URL`.

## What These Baselines Cover

These scripts reproduce the few-shot / zero-shot LLM baselines reported in the paper (e.g., GPT-5).
