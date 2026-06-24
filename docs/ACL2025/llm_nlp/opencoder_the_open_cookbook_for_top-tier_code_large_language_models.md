---
title: >-
  [Paper Note] OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models
description: >-
  [ACL2025][LLM (Other)][Code LLM] Proposes OpenCoder, a fully open-source code large language model (including 1.5B and 8B versions) that not only achieves top-tier performance but also serves as an "open cookbook" by releasing reproducible data processing pipelines, pretraining datasets, ablation studies, and training protocols, providing foundational infrastructure for research in code intelligence.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Code LLM"
  - "Data Curation"
  - "Pretraining Pipeline"
  - "Reproducible"
  - "RefineCode"
date: 2026-05-08
content_hash: 39415147674fa5d3
---

# OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models

**Conference**: ACL2025  
**arXiv**: [2411.04905](https://arxiv.org/abs/2411.04905)  
**Code**: [OpenCoder](https://opencoder-llm.github.io)  
**Area**: Code Large Language Models / LLM Pretraining  
**Keywords**: Code LLM, Data Curation, Pretraining Pipeline, Reproducible, RefineCode  

## TL;DR

Proposes OpenCoder, a fully open-source code large language model (including 1.5B and 8B versions) that not only achieves top-tier performance but also serves as an "open cookbook" by releasing reproducible data processing pipelines, pretraining datasets, ablation studies, and training protocols, providing foundational infrastructure for research in code intelligence.

## Background & Motivation

- **Gap in Open-Source Code LLMs**: Although commercial tools like Copilot and Cursor have transformed developer workflows, open-source code LLMs still lag behind closed-source SOTA models in performance.
- **Lack of Transparency**: Leading code LLMs generally do not release their training data or processing pipelines, which limits community baseline building and deep research.
- **Scarcity of Reproducibility**: Existing open-source code LLMs rarely release training data, data processing pipelines, SFT corpora, and intermediate checkpoints simultaneously.
- **Three Main Goals**: (1) Provide a transparent and strong baseline for mechanistic interpretability research; (2) Conduct in-depth research on the curation pipeline of pretraining and instruction data; (3) Unlock diverse customization solutions based on transparent code LLMs.

## Method

### Overall Architecture

The training of OpenCoder consists of three stages: Pretraining → Annealing → Two-Stage Instruction Fine-Tuning (SFT).

### Pretraining Data: RefineCode

A high-quality code pretraining dataset, RefineCode, containing approximately 960B tokens and covering 607 programming languages, is constructed. It mainly consists of raw source code and code-related web data.

**Raw Code Processing Pipeline**:

1. **Preprocessing**: Exclude files >8MB and filter 607 programming languages based on file extensions.
2. **Deduplication (Prioritized)**:
    - Exact Deduplication: SHA256 hash matching, keeping files with the highest stars and latest commits (removing approximately 75% of completely duplicate files).
    - Near-deduplication: 5-gram MinHash + LSH (16 bands, 128 rows), removing approximately 6% of files.
    - Key Finding: File-level deduplication outperforms repository-level deduplication.
3. **Transformation**:
    - Copyright Notice Removal: Over 15% of code files contain repetitive copyright information.
    - PII Redaction: Regular expressions are used to replace sensitive information like passwords, emails, and IPs with placeholders.
4. **Filtering (Three Categories of Heuristic Rules)**:
    - Natural language filtering rules: Common text features such as file size and line counts.
    - Common code filtering rules: Code features such as variable counts and average function length.
    - **Language-Specific Filtering Rules (First-of-its-kind)**: Specially designed rules for 8 languages: Python, C, C++, C#, Java, JavaScript, Go, and HTML.
5. **Data Sampling**: Downsampled Java (449GB → 200GB) and HTML (474GB → 64GB), resulting in approximately 730B tokens.

**Code-Related Web Data** (Approx. 330GB):
- Recalled from CommonCrawl using a FastText classifier through three iterative rounds.
- Code-related domain discovery + manual URL annotation.
- Additional recall from FineWeb, SkyPile, AutoMathText, etc.
- Trained a classifier to identify code-related content from GitHub Markdown text (an additional 178GB).

### Annealing Stage Data

Annealing serves as a bridge from pretraining to SFT, totaling approximately 100B tokens:
- **Raw Distribution Data (84%)**: Maintains the same data distribution as the pretraining stage to prevent catastrophic forgetting.
- **Algorithmic Corpus**: Code files containing keywords like "leetcode" and "def solution" sampled from the raw data.
- **Synthetic Data**:
    - High-quality code snippets: Using the algorithmic corpus as seeds, the LLM generates self-contained functions and test cases, keeping only the data that passes the test.
    - Code textbooks: Interactive code analyses generated using Qwen2-72B based on the hqcode dataset.

### Model Architecture

| Parameter | 1.5B | 8B |
|------|------|------|
| Layers | 24 | 32 |
| Hidden Dimension | 2240 | 4096 |
| Attention Heads | 14 | 32 |
| KV Heads | 14 | 8 |
| Activation Function | SwiGLU | SwiGLU |
| Vocabulary Size | 96,640 | 96,640 |
| Context Length | 4096 | 8192 |

### Two-Stage Instruction Fine-Tuning

**Data Sources**:
- Open-source data: Evol-Instruct, Infinity-Instruct (filtered for code-related content by LLM), McEval, and WildChat user queries.
- Educational-Instruct: High-quality seed data → score filtering → LLM-generated Q&A + test case validation.
- Package-Instruct: Using the latest library documentation to generate accurate tool-calling instructions (addressing the outdated API issue).
- Large-scale Diverse-Instruct: Web cleaning → task specification definition → LLM-generated QA → code execution validation.

**Stage 1 (Theoretical Knowledge)**: RealUser 0.7M + Diverse 2.3M + Infinity 1.0M, batch=4096, LR=2e-5
**Stage 2 (Practical Coding)**: McEval 36K + Evol 111K + Educational 110K + Package 110K, batch=512, LR=5e-5

### Decontamination

Strictly execute SFT data deduplication: remove data containing entry points of the HumanEval/MBPP test sets, and conduct 10-gram overlap removal.

## Experiments

### Base Model Evaluation

| Model | Size | HE | HE+ | MBPP | MBPP+ | BigCodeBench |
|------|------|------|------|------|------|------|
| DS-Coder-6.7B | 6.7B | 47.6 | 39.6 | 70.2 | 56.6 | 41.1 |
| Yi-Coder-9B | 9B | 53.7 | 46.3 | 48.4 | 40.7 | 42.9 |
| Qwen2.5-Coder-7B | 7B | 61.6 | 53.0 | 76.9 | 62.9 | 45.8 |
| StarCoder2-15B | 15B | 46.3 | 37.8 | 66.2 | 53.1 | 38.4 |
| **OpenCoder-8B** | **8B** | **66.5** | **63.4** | **79.9** | **70.4** | 40.5 |
| **OpenCoder-1.5B** | **1.5B** | **54.3** | **49.4** | **70.6** | **58.7** | 24.5 |

OpenCoder-8B outperforms all models of the same class on HumanEval and MBPP, with HE+ reaching 63.4, significantly exceeding the second-best model Qwen2.5-Coder-7B at 53.0.

### Instruct Model Evaluation

| Model | HE | HE+ | MBPP | LiveCodeBench |
|------|------|------|------|------|
| DS-Coder-V2-Lite-Instruct | 81.1 | 75.0 | 82.3 | 24.3 |
| Yi-Coder-9B-Chat | 85.4 | 79.9 | - | 20.5 |
| Qwen2.5-Coder-7B-Instruct | 88.4 | 84.1 | - | 26.8 |
| **OpenCoder-8B-Instruct** | **83.5** | **78.7** | **80.2** | 21.0 |

OpenCoder-8B-Instruct achieves a HumanEval Pass@1 of 83.5, outperforming all fully open-source models (including those with reproducible datasets).

### Key Ablation Study Findings

1. **Deduplication Strategy**: File-level deduplication outperforms repository-level deduplication, improving downstream performance by preserving data diversity.
2. **GitHub Stars Filtering**: Filtering by star counts may reduce data diversity, leading to suboptimal results.
3. **Code-Related Web Data**: Significantly enhances the model's capabilities.
4. **Annealing Data Quality**: The quality of data during the annealing stage is far more important than its quantity.
5. **Two-Stage SFT**: The staged strategy allows the model to acquire broad capabilities before refining specific code tasks, which outperforms the single-stage strategy.

## Highlights & Insights

1. **Transparency Benchmark**: First top-tier code LLM to simultaneously release its data processing pipeline, a reproducible pretraining dataset, large-scale SFT corpora, and intermediate checkpoints.
2. **First-of-its-kind Language-Specific Filtering Rules**: Devises filtering rules based on the unique characteristics of different programming languages, such as the frequency of `pass` statements in Python and the usage of `goto` in C.
3. **PCA Visualization Validation**: Demonstrates via CodeBERT embedding visualization that RefineCode is more compactly distributed and has fewer outliers compared to Stack v2.
4. **Package-Instruct Addressing Real Pain Points**: Generates SFT data using the latest documentation to address the issue of LLMs using outdated API versions.
5. **Training Efficiency**: The 1.5B model trained on RefineCode outperforms the same-sized model trained on Stack v2 at just 600B tokens.
6. **Strict Decontamination**: 10-gram granularity deduplication ensures fair evaluation.

## Limitations & Future Work

1. **Overall Performance Lags Behind the Strongest Closed-Source Models**: Still falls behind models using larger training tokens like Qwen2.5-Coder on complex tasks such as BigCodeBench.
2. **Limited Computational Budget**: The data mixture ratio during the annealing stage for the 1.5B model might be suboptimal.
3. **Insufficient Non-English Coverage**: Relatively less processing of Chinese code-related data.
4. **Average Performance on LiveCodeBench**: There remains a gap with top-tier methods on complex algorithmic reasoning tasks.
5. **Synthetic Data Quality Ceiling**: Structural reliance on the quality of the teacher model limits the upper bound of synthetic data.

## Related Work & Insights

- **Code LLMs**: CodeLlama, StarCoder series, DeepSeek-Coder, Qwen-Coder, etc.
- **Pretraining Data**: Open-source code datasets like The Stack v1/v2, FineWeb, etc.
- **Data Quality**: Studies on data cleaning, deduplication, quality filtering, etc.
- **Instruction Fine-Tuning**: Methods for constructing code SFT data, such as Code-Alpaca, Evol-Instruct, etc.
- **Evaluation Benchmarks**: HumanEval, MBPP, BigCodeBench, LiveCodeBench, etc.

## Rating ⭐⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ The methods themselves are not completely new, but the philosophy of releasing an open cookbook has paradigm-shifting signficance.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Numerous ablation studies validate each design decision.
- **Value**: ⭐⭐⭐⭐⭐ Fully open-source, top-tier code LLM + reproducible data pipelines.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The motivation and impact of each decision steps are clearly discussed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models](warriorcoder_learning_from_expert_battles_to_augment_code_large_language_models.md)
- [\[ACL 2025\] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization](to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)
- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [\[ACL 2025\] Open-Set Living Need Prediction with Large Language Models](open-set_living_need_prediction_with_large_language_models.md)
- [\[ACL 2025\] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](toolcoder_code_empowered_tool_learning.md)

</div>

<!-- RELATED:END -->
