---
title: >-
  [Paper Note] WildChat-50m: A Deep Dive Into the Role of Synthetic Data in Post-Training
description: >-
  [ICML 2025][Model Compression][Synthetic Data] Constructs the largest public chat dataset to date, WildChat-50m (50+ open-source models $\times$ 1M+ conversations = 125 million transcripts), systematically investigates the synthetic data quality of different data generation models (DGMs), and designs the Re-Wild SFT mixing scheme, which outperforms Tulu-3 using only 40% of its SFT data volume.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Synthetic Data"
  - "SFT"
  - "Data Generation Models"
  - "LLM Post-Training"
  - "Data Mixing"
date: 2026-05-08
content_hash: fb17a66b56c7f913
---

# WildChat-50m: A Deep Dive Into the Role of Synthetic Data in Post-Training

**Conference**: ICML 2025  
**arXiv**: [2501.18511](https://arxiv.org/abs/2501.18511)  
**Code**: [https://github.com/penfever/wildchat-50m](https://github.com/penfever/wildchat-50m)  
**Area**: Model Compression/LLM Post-Training  
**Keywords**: Synthetic Data, SFT, Data Generation Models, LLM Post-Training, Data Mixing

## TL;DR

Constructs the largest public chat dataset to date, WildChat-50m (50+ open-source models $\times$ 1M+ conversations = 125 million transcripts), systematically investigates the synthetic data quality of different data generation models (DGMs), and designs the Re-Wild SFT mixing scheme, which outperforms Tulu-3 using only 40% of its SFT data volume.

## Background & Motivation

### 1. Importance of LLM Post-Training

LLM post-training (SFT, DPO, distillation) is a critical step to unlock model capabilities. OpenAI's test-time scaling and DeepSeek's reasoning models both heavily rely on high-quality synthetic data. However, the open-source community lags significantly behind industrial labs in data curation.

### 2. Key Challenge

- Scale public synthetic datasets are scarce, hindering systematic comparison of data generation model quality.
- How large is the response quality gap among different DGMs? How should the optimal DGM be selected?
- Existing SFT data mixing schemes (e.g., Tulu-3) are complex and massive; can a more efficient alternative be found?

### 3. Key Insight

By generating responses using 50+ open-source models on over 1M prompts from WildChat-1M, this work builds an ultra-large-scale dataset of 125M transcripts to support systematic DGM comparisons and SFT experiments.

## Method

### Overall Architecture

1. **Data Collection**: Run vLLM inference on 50+ models using a $12 \times 8$ H100 cluster to generate multi-turn responses for WildChat prompts.
2. **Data Analysis**: Compare the throughput efficiency, response similarity, and quality metrics of each model.
3. **SFT Experiments**: Design the Re-Wild data mixing scheme, perform SFT on Llama-3.1-8B-Base, and evaluate across multiple benchmarks.

### Key Designs

#### 1. Large-Scale Multi-Model Data Collection

- **Scale**: 54 DGMs (19 pretrained models + 35 fine-tuned variants) with parameter sizes from 0.5B to 104B.
- **Unified Environment**: Same hardware (H100) + same inference framework (vLLM) to ensure a fair comparison.
- **Total Cost**: Approximately 10,000 H100-hours.
- The largest models use FP8 quantized inference, while the rest use bfloat16.

#### 2. DGM Quality Analysis

- **Throughput Efficiency**: The fastest model (Llama-2-7B: 37,357 tok/s) is over 10$\times$ faster than the slowest (Qwen2.5-72B: 3,163 tok/s).
- **Response Similarity**: Responses from different LLMs are surprisingly similar—even when pre-training/post-training data do not fully overlap.
- **Quality Ranking**: The quality of synthetic data from DGMs is indirectly measured via downstream SFT performance.

#### 3. Re-Wild Data Mixture

| Data Source | Quantity |
|--------|------|
| WildChat-Q72 (Qwen2.5-72B responses) | 246,750 |
| MMLU Auxiliary Train | 99,800 |
| Tulu 3 Persona Hub Algebra | 20,000 |
| **Total** | **~366K** |

This accounts for only ~40% of the Tulu-3 SFT data volume. Design principle: complementary skills (chat + knowledge + math). Training: AdamW, lr=2e-5, 1 epoch, cosine scheduler, 4$\times$H100, ~5.5 hours.

## Key Experimental Results

### Main Results: Re-Wild vs Baselines

| Method | Data Volume | Benchmark Performance Trends | Description |
|------|--------|------------|------|
| Tulu-3 SFT (Allen AI) | ~900K | Baseline | Complex multi-source mixture |
| **Re-Wild (Ours)** | **~366K** | **Outperforms Tulu-3** | 40% data volume |
| L8B:Q72 Single-Source | 250K | Medium-High | Chat-only data |
| L8B:L8I Self-Distillation | 250K | Medium | Limited distillation effectiveness for small models |

Figure 1 in the paper indicates that Re-Wild outperforms all baselines on the weighted average across 9 benchmarks.

### Ablation Study: DGM Selection

| DGM | Avg Score Trend Post-SFT | Parameters | Inference Speed |
|-----|---------------|--------|---------|
| Qwen2.5-72B-Instruct | Highest | 72B | 3,163 tok/s |
| Llama-3.3-70B | Second Highest | 70B | Medium |
| Cohere-CRP-104B | Medium-High | 104B | Medium-Low |
| Qwen2-7B-Instruct | Medium | 7B | High |
| Llama-2-7B-Chat | Low | 7B | 37,357 tok/s |

### Key Findings

- Larger models ($72\text{B}+$) as DGMs provide significantly higher synthetic data quality than smaller models.
- Responses from different LLMs are surprisingly similar, yet subtle differences are amplified during downstream SFT.
- A simple three-source mixture can outperform complex multi-source schemes—more data is not always better.
- Re-Wild outperforms Tulu-3 using only 40% of the data volume.

## Highlights & Insights

- **Unprecedented Dataset Scale**: Over 50$\times$ larger than WildChat-1M, representing the first dataset that supports systematic DGM comparisons.
- **"Less is More" Data Curation**: Selecting quality DGMs and complementary data sources is far more critical than simply stacking data volume.
- **Practical DGM Selection Guide**: Provides a quantitative reference for academic labs regarding "which model to use for generating training data."
- **Fully Open-Source**: Datasets, code, and SFT schemes are all fully open-sourced.

## Limitations & Future Work

- Experiments are limited to SFT; preference alignment stages like DPO/RLHF remain to be explored.
- The sources of differences in the surprisingly similar LLM responses are not analyzed in depth.
- The largest models utilize FP8 quantized inference; the exact impact of quantization on response quality is not isolated.
- SFT evaluations are only validated using Llama-3.1-8B-Base.

## Related Work & Insights

- **vs Tulu-3**: Outperforms Tulu-3 with only 40% of the data volume despite Tulu-3 using a more complex, multi-source mixture.
- **vs DeepSeek Distillation**: Bottom-up distillation processes do not release their data in industrial labs, whereas this work provides an open-source alternative.
- **vs Data Curation Research**: First to systematically compare synthetic data quality at a scale of 50+ DGMs.

## Rating

- Novelty: ⭐⭐⭐⭐ Dataset scale and systematic comparison are the core contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 50+ models $\times$ 9 benchmarks $\times$ multiple ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and deep analysis.
- Value: ⭐⭐⭐⭐⭐ Highly practical value for the open-source LLM post-training community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] StolenLoRA: Exploring LoRA Extraction Attacks via Synthetic Data](../../ICCV2025/model_compression/stolenlora_exploring_lora_extraction_attacks_via_synthetic_data.md)
- [\[ICML 2025\] BoA: Attention-aware Post-training Quantization without Backpropagation](boa_attention-aware_post-training_quantization_without_backpropagation.md)
- [\[ICML 2025\] Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation](merge-friendly_post-training_quantization_for_multi-target_domain_adaptation.md)
- [\[ECCV 2024\] MetaAug: Meta-Data Augmentation for Post-Training Quantization](../../ECCV2024/model_compression/metaaug_meta-data_augmentation_for_post-training_quantization.md)
- [\[ICLR 2026\] Asymmetric Synthetic Data Update for Domain Incremental Dataset Distillation](../../ICLR2026/model_compression/asymmetric_synthetic_data_update_for_domain_incremental_dataset_distillation.md)

</div>

<!-- RELATED:END -->
