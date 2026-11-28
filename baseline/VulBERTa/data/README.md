# VulBERTa Data Layout

This folder documents the upstream VulBERTa data layout and download workflow.

## Upstream Layout

The VulBERTa project uses this directory for:

- tokenizer training data
- pretraining data
- fine-tuning data

Download the corresponding assets from the upstream VulBERTa release and place
them here to run the upstream notebooks and tokenizer-training workflow.

## Repository Use

The folder documents the paths expected by the upstream VulBERTa notebooks.
GRAML runs use the JSON datasets under `Dataset/`; populate this
directory for tokenizer training, pretraining, or upstream fine-tuning runs.
