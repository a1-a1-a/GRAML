# Out-of-Distribution Dataset

This folder contains the out-of-distribution evaluation datasets used to test generalization beyond the in-distribution training data.

## Included Sources

- [CVEfixes](CVEfixes/README.md)
- [Devign](Devign/README.md)
- [DiverseVul](DiverseVul/README.md)
- [Juliet](Juliet/README.md)
- [PrimeVul](PrimeVul/README.md)
- [ReVeal](ReVeal/README.md)

## What This Folder Is For

Use these datasets to evaluate whether a model trained on the GRAML in-distribution data can still perform well on code drawn from different sources and distributions.

## Structure

Most subfolders contain:

- `test.json`: the OOD evaluation split used in experiments

The `Juliet/` folder additionally contains:

- `valid.json`: an extra validation split

## Recommended Usage

- Train on `Dataset/id/Ultimate_train.json`
- Tune thresholds on `Dataset/id/Ultimate_valid.json` or an external validation split when appropriate
- Evaluate on one OOD folder at a time
