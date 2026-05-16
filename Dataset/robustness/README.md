# Robustness Dataset

This folder contains perturbed evaluation sets used for robustness testing.

## What This Folder Contains

Each subfolder corresponds to one original dataset source:

- [Ultimate_test](Ultimate_test/README.md)
- [CVEfixes](CVEfixes/README.md)
- [Devign](Devign/README.md)
- [DiverseVul](DiverseVul/README.md)
- [Juliet](Juliet/README.md)
- [PrimeVul](PrimeVul/README.md)
- [ReVeal](ReVeal/README.md)

## Perturbation Types

Each source folder contains three robustness variants:

- `*_noise.json`: noise-based perturbations
- `*_obfuscate.json`: obfuscation-style perturbations
- `*_structure.json`: structure-preserving code transformations

## What This Folder Is For

Use these files to evaluate whether performance remains stable under benign but distribution-shifting changes to the code surface form.

## How These Files Were Produced

The perturbation pipeline is documented in [robustness_transformation](../../robustness_transformation/README.md).
