# IDS Uncertainty Estimation — Phase 0 Replication Package

## Overview

This archive contains the complete replication materials for Phase 0 (Foundation
Validation) of the study *"Prompt-Based Uncertainty Quantification for LLM-Assisted
Intrusion Detection"*, submitted to the Journal of Information Security and
Applications (JISA).

**Contents:** 51 experiment result files, 2 evaluation datasets,
14 Python source files, and pre-computed aggregated metrics.

## Directory Structure

```
ids-uncertainty-replication-phase0/
├── README.md                 # This file
├── .zenodo.json              # Zenodo deposition metadata
├── LICENSE                   # CC-BY-4.0
├── config.yaml               # Experiment configuration (source of truth)
├── pyproject.toml            # Python dependencies
├── aggregated_results.json   # Pre-computed metrics for all 50 experiments
│
├── datasets/                 # Evaluation datasets (50 stratified samples each)
│   ├── cicids2017_test_50.jsonl
│   └── cicids2018_test_50.jsonl
│
├── results/                  # Raw experiment outputs (JSONL)
│   ├── baseline/{model}_{dataset}/baseline_results.jsonl
│   ├── ensemble/…
│   ├── monte_carlo/…
│   ├── hybrid/…
│   └── composite/…
│
└── code/                     # Experiment software
    ├── runners/              # 5 uncertainty method implementations
    ├── models/               # Pydantic schemas & config loading
    ├── analysis/             # Evaluation & metrics
    ├── data_processing/      # Dataset preparation
    └── utilities/            # Retry logic
```

## Models

| Display Name | Internal Key |
|:-------------|:-------------|
| GPT-OSS-20B | `openai-gpt-oss-20b` |
| Kimi | `groq-kimi` |
| Llama4S | `nvidia-llama4-scout` |
| Llama70B | `nvidia-llama-70b` |
| Qwen3-80B | `qwen3-next-80b` |

## Uncertainty Methods

| Manuscript Name | Code Name |
|:----------------|:----------|
| Baseline | `baseline` |
| Composite | `composite` |
| Ensemble Method | `hybrid` |
| Monte Carlo Prompt | `monte_carlo` |
| Multi-Prompt Voting | `ensemble` |

## Dataset Schema (JSONL)

Each line in the dataset files is a JSON object:

```json
{
  "id": "cicids_1",
  "description": "Network flow to destination port 443 (HTTPS): ...",
  "label": "benign",
  "attack_type": "benign"
}
```

## Result Schema (JSONL)

Each line in the result files is a JSON object:

```json
{
  "sample_id": "cicids_1",
  "model_name": "nvidia-llama4-scout",
  "timestamp": "2026-01-22T01:15:31.351402",
  "ground_truth": {"is_malicious": false, "attack_type": "benign"},
  "prediction": {"is_malicious": false, "confidence_score": 0.9, "reasoning": "..."},
  "is_correct": true,
  "uncertainty_metrics": {"confidence": 0.9, "entropy": null, "variance": null},
  "routing_decision": "accept",
  "error": null
}
```

## Experimental Protocol

**Phase 0 — Foundation Validation:**
- **Sample size:** 50 samples per dataset (stratified by attack type)
- **Datasets:** CIC-IDS2017, CIC-IDS2018
- **Models:** 5 LLMs (2 general-purpose, 2 reasoning, 1 instruction-following)
- **Methods:** 5 uncertainty quantification approaches
- **Success criteria:** Baseline accuracy > 70%, ECE < 0.35

**Routing thresholds:**
- Accept (auto-classify): confidence >= 0.85
- Escalate (human review): 0.60 <= confidence < 0.85
- Reject (block): confidence < 0.60

## Reproducing Results

### Prerequisites

- Python >= 3.9
- [uv](https://docs.astral.sh/uv/) package manager
- API keys for LLM providers (NVIDIA NIM, Groq, OpenAI)

### Steps

```bash
# Install dependencies
uv sync

# Run a baseline experiment (example)
uv run python code/runners/baseline_runner.py \
    --model nvidia-llama4-scout --dataset cicids --samples 50

# Calculate metrics
uv run python code/analysis/metrics_calculator.py \
    results/baseline/nvidia-llama4-scout_cicids/baseline_results.jsonl

# Evaluate Phase 0 criteria
uv run python code/analysis/evaluate_phase0.py --calculate-metrics
```

## Citation

If you use this data, please cite:

> Prompt-Based Uncertainty Quantification for LLM-Assisted Intrusion Detection.
> Journal of Information Security and Applications (JISA), Elsevier, 2026.

## License

This work is licensed under the [Creative Commons Attribution 4.0 International
License](https://creativecommons.org/licenses/by/4.0/).
