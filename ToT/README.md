# ToT Guide

This folder contains the Tree-of-Thought-style vulnerability reasoning component used in GRAML.

## Files

- `tot_vr.py`: the main ToT-guided vulnerability reasoning implementation
- `ToT_description.json`: generated vulnerability descriptions

## What `tot_vr.py` Does

The script implements a five-stage reasoning pipeline:

1. branch initialization
2. branch reflection and pruning
3. branch refinement with structural evidence
4. branch verification against the CVE description
5. final vulnerability description synthesis

## API Requirement

The script uses the OpenAI API and expects:

- `OPENAI_API_KEY` to be set in the environment

Example:

```bash
export OPENAI_API_KEY=your_api_key
python ToT/tot_vr.py
```

## Typical Usage

- use the built-in example in `tot_vr.py` for a quick sanity check
- replace the example code, vulnerable lines, and CVE description with your own data
- integrate `ToTVRGenerator` into a larger evaluation script if needed
