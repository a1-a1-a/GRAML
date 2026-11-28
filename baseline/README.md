# Baseline Guide

Seven encoder baselines, LLM API baselines, and configurations for SAGE, LOSVER,
and VulLLM.

## Folders

- [codebert](codebert/README.md)
- [codeberta](codeberta/README.md)
- [gpt2](gpt2/README.md)
- [GraphCodebert](GraphCodebert/README.md)
- [RoBerta](RoBerta/README.md)
- [Unixcoder](Unixcoder/README.md)
- [VulBERTa](VulBERTa/README.md)
- [llm_baselines](llm_baselines/README.md): few-shot and zero-shot LLM baselines
- [recent_methods](recent_methods/README.md): configurations and shared evaluation protocol

## Shared protocol

The encoder baselines use:

- `Dataset/ablations/multi_task/only_detection.json`
- `Dataset/eval/ID_valid.json`
- `Dataset/eval/ID_test.json`

They discard non-Detection and non-`Yes`/`No` records when a multi-task file is
supplied, select the decision threshold by F1 on `ID_valid.json` over
0.01-0.90 in steps of 0.01, and keep it fixed for test.
