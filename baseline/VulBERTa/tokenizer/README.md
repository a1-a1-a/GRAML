# VulBERTa Tokenizer Assets

## Files

- `drapgh-vocab.json`: BPE vocabulary
- `drapgh-merges.txt`: BPE merge rules

`run_vulberta.py` loads these files to build the VulBERTa tokenizer. Runs that
select `--model_name_or_path roberta-base` use the corresponding Hugging Face
tokenizer instead.
