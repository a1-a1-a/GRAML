# Robustness Dataset

This folder contains the perturbed evaluation sets used for robustness testing.

## Datasets

Each subfolder corresponds to one original dataset source:

- [Ultimate_test](Ultimate_test/README.md)
- [CVEfixes](CVEfixes/README.md)
- [Devign](Devign/README.md)
- [DiverseVul](DiverseVul/README.md)
- [Juliet](Juliet/README.md)
- [PrimeVul](PrimeVul/README.md)
- [ReVeal](ReVeal/README.md)

## Transformation Categories

Each source folder contains three combined perturbation categories:

| File suffix | Category | Transformations |
|---|---|---|
| `*_obfuscate.json` | Semantic obfuscation | `tf_1` parameter renaming, `tf_2` function-name renaming, `tf_3` comment removal |
| `*_noise.json` | Long-context noise | `tf_4` dead-code insertion, `tf_5` irrelevant-comment insertion |
| `*_structure.json` | Structural restructuring | `tf_6` function extraction, `tf_7` void-function call insertion |

The transformations in each category are applied together.

## Record Counts

| Dataset | Each category |
|---|---:|
| `Ultimate_test` | 3,000 |
| `CVEfixes` | 2,108 |
| `Devign` | 936 |
| `DiverseVul` | 766 |
| `Juliet` | 1,576 |
| `PrimeVul` | 548 |
| `ReVeal` | 1,014 |

The transformation code and commands are documented in
[robustness_transformation](../../robustness_transformation/README.md).
