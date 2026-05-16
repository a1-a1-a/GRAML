# VulBERTa Tokenizer Assets

This folder contains the tokenizer files used by the local VulBERTa baseline wrapper.

## Files

- `drapgh-vocab.json`: BPE vocabulary
- `drapgh-merges.txt`: BPE merge rules

## What This Folder Is For

`run_vulberta.py` checks this folder first when loading a tokenizer. If both files are present, it builds a local BPE tokenizer compatible with the VulBERTa setup.

## Practical Note

If these files are unavailable or cannot be loaded, `run_vulberta.py` falls back to `roberta-base` so that the baseline remains runnable.
