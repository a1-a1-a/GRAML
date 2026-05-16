# Baseline Guide

This folder contains the baseline models used for comparison against the main GRAML pipeline.

## Included Baselines

- [codebert](codebert/README.md)
- [codeberta](codeberta/README.md)
- [gpt2](gpt2/README.md)
- [GraphCodebert](GraphCodebert/README.md)
- [RoBerta](RoBerta/README.md)
- [Unixcoder](Unixcoder/README.md)
- [VulBERTa](VulBERTa/README.md)

## What This Folder Is For

Use these subfolders to reproduce baseline results under the same GRAML-style JSON data format.

## Common Pattern

Most baseline folders contain:

- `run.py` or `run_gpt2.py`: training and evaluation entry point
- `model.py`: model wrapper or classification head definition

Most baselines consume:

- `Dataset/ID_dataset/Ultimate_train.json`
- `Dataset/ID_dataset/Ultimate_valid.json`
- `Dataset/ID_dataset/Ultimate_test.json`

## Shared Data Assumption

The scripts generally expect:

- `input`: code snippet
- `output`: binary label in `Yes` or `No`, or values convertible to `0` or `1`

## Practical Note

The folder names are kept as they appear in the experimental repository, even when capitalization is unconventional, so that the file paths remain consistent with the code.
