# Ablation Dataset

This folder contains the ablation datasets used to analyze how different task components contribute to GRAML's performance.

## Files

- `no_assessment.json`: removes the vulnerability assessment component
- `no_description.json`: removes the vulnerability explanation or description component
- `no_location.json`: removes the vulnerability localization component
- `only_detection.json`: keeps only the binary vulnerability detection task

## What This Folder Is For

Use these files when you want to answer questions such as:

- How much does vulnerability assessment help overall performance?
- How important is localization information?
- How much performance comes from detection alone versus richer multi-part supervision?

## Typical Usage

These files can be substituted for the standard training or evaluation data when running the scripts in `traing+inference/` or adapted baseline experiments.
