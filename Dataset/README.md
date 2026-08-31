# Dataset Guide

This folder contains all datasets released with GRAML. It is the central entry point for reproduction, evaluation, ablation, and robustness experiments.

## What This Folder Contains

- `id/`: the main in-distribution split used for training, validation, and testing
- `ood/`: external out-of-distribution evaluation sets from multiple open-source sources
- `robustness/`: perturbed versions of test sets used for robustness analysis
- `ablations/`: ablation dataset variants organized by experiment
- `graph_context/`: critical-line selection outputs and Joern CPG extraction examples
- `scripts/`: dataset construction scripts

## How to Use It

- Use `id/` for the main training and validation pipeline.
- Use `ood/` to test cross-dataset generalization.
- Use `robustness/` to test perturbation robustness.
- Use `ablations/` to measure the contribution of different components (multi-task supervision, graph evidence, direct CPG, random lines).

## Data Format

Across the repository, the JSON files in this folder are used in a consistent way:

- `instruction`: task instruction given to the model
- `input`: source code snippet or code-centered input
- `output`: target label or target text
- `Task` or related keys such as `task`, `category`, `type`: task type

The scripts in `training_inference/` accept either a plain JSON list or a JSON object with a top-level `data` field.

## Subfolder Guides

- [ablations](ablations/multi_task/README.md)
- [graph_context](graph_context/critical_lines/README.md)
- [id](id/README.md)
- [ood](ood/README.md)
- [robustness](robustness/README.md)
