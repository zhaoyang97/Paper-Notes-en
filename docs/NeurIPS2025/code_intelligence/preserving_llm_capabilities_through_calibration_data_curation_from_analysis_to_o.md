---
title: >-
  [Paper Note] Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization
description: >-
  [NeurIPS 2025][Code Intelligence][LLM compression] This paper systematically investigates how compositional properties of calibration data (sequence length, sample size, source…
tags:
  - "NeurIPS 2025"
  - "Code Intelligence"
  - "LLM compression"
  - "calibration data"
  - "post-training quantization"
  - "post-training pruning"
  - "capability preservation"
  - "activation space"
date: 2026-05-08
content_hash: 9ed626c90a272cf0
---

# Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2510.10618](https://arxiv.org/abs/2510.10618)
**Authors**: Bowei He, Lihao Yin, Huiling Zhen, Shuqi Liu, Han Wu, Xiaokun Zhang, Mingxuan Yuan, Chen Ma (City University of Hong Kong, Huawei)
**Code**: Publicly available
**Area**: Code Intelligence
**Keywords**: LLM compression, calibration data, post-training quantization, post-training pruning, capability preservation, activation space

## TL;DR

This paper systematically investigates how compositional properties of calibration data (sequence length, sample size, source, format) and domain correspondence affect capability preservation after LLM compression. It finds that representativeness and diversity in the activation space are the fundamental determinants of calibration data quality, and proposes a three-stage calibration data curation framework, COLA.

## Background & Motivation

### State of the Field
Post-training compression (pruning and quantization) is the mainstream approach for deploying LLMs. During compression, **calibration data** is used to assess weight importance and activation dynamic ranges, making it critical to compression quality. However, existing compression methods generally assume robustness to calibration data distribution, and systematic investigation remains lacking.

### Limitations of Prior Work
- Earlier work examined calibration data effects only from **isolated perspectives** (sample size, data source, sequence length)
- Evaluation was limited to basic **language modeling perplexity** and **commonsense reasoning**, without covering advanced capabilities such as mathematical reasoning and code generation
- The underlying mechanisms were unexplored, leaving unclear what constitutes **optimal calibration data**
- No systematic calibration data curation strategy had been established

### Core Problem
To answer four key questions: (Q1) How do compositional properties of calibration data affect capability preservation? (Q2) How does domain correspondence affect capability preservation? (Q3) What constitutes optimal calibration data? (Q4) How can optimal calibration data be curated from available data?

## Method

### Experimental Setup
- **Models**: LLaMA3-8B-Instruct, Qwen2.5-7B-Instruct
- **Pruning methods**: SparseGPT (50% unstructured), Wanda (4:8 semi-structured)
- **Quantization methods**: GPTQ (4-bit), AWQ (4-bit)
- **Calibration data sources**: C4, WikiText, SlimPajama (pre-training data); CommonsenseQA, MathQA, CodeQA (downstream data)
- **Evaluation dimensions**: language modeling (PPL), commonsense reasoning (CS), mathematical reasoning (Math), code generation (Code), multilingual understanding

### Q1: Effect of Compositional Properties

**Sequence length**: Mathematical reasoning is most sensitive to length—SparseGPT exhibits a sharp 25.5% performance drop under short sequences (128); code generation shows non-monotonic variation (AWQ fluctuates between 38.71% and 47.53%). Pruning methods are more sensitive to length than quantization methods, and AWQ's per-channel scaling provides notable robustness.

**Sample size**: Diminishing returns appear beyond 64–128 samples. However, code generation exhibits anomalous patterns—on LLaMA3-8B+AWQ, 16 samples (46.40%) outperform 128 samples (38.71%); on Qwen2.5-7B+GPTQ, performance drops sharply from 57.67% at 16 samples to 34.03% at 128 samples. Additional samples may introduce suboptimal examples.

**Data source**: The choice of data source can have an impact **exceeding that of the compression method itself**. C4 demonstrates a clear advantage for code generation (19.4% higher than Wikipedia on LLaMA3-8B), while SlimPajama performs better on mathematical reasoning (8.7% higher than Wikipedia on Qwen2.5-7B).

**Data format**: Formats incorporating explicit reasoning chains (Q&A w/ ERC) yield the best results—a 9.5% improvement on mathematical tasks for Qwen2.5-7B+AWQ (47.34%→51.84%), as reasoning chains activate and preserve the model's internal reasoning mechanisms.

### Q2: Effect of Domain Correspondence

**Language alignment**: English calibration data performs best for English-dominant models; on mathematical tasks, English (31.22%) outperforms Japanese (16.72%) by 46.4% on LLaMA3-8B+GPTQ. On multilingual benchmarks, matching the evaluation language outperforms the default English approach.

**Domain matching**: Domain matching substantially enhances the corresponding capability while degrading others. MathQA calibration improves quantized mathematical performance by 5.92 percentage points; CodeQA improves quantized code generation by 7.49 percentage points, but both increase perplexity by 2–3 points.

### Q3: Defining Optimal Calibration Data

Core finding: **Representativeness and diversity in the activation space** fundamentally determine calibration data quality.
- **Representativeness**: Whether a sample triggers activation patterns typical of the target domain
- **Diversity**: The breadth of coverage over unique activation patterns triggered

### Q4: The COLA Framework (Three-Stage Curation)

**Stage 1 — Dataset Selection (Domain Correspondence)**:
Source datasets are selected according to the deployment scenario (general/specialized), considering language alignment, domain coverage, and reasoning difficulty. Formalized as a coverage optimization problem:

$$S = \arg\max_{S \subseteq \mathcal{D}} \sum_{c \in C} w_c \cdot \text{coverage}(S, c)$$

**Stage 2 — Data Processing (Compositional Properties)**:
Sequence length is optimized (typically 2048 tokens; shorter sequences are admissible for AWQ); format augmentation converts data into a Q&A format with explicit reasoning chains.

**Stage 3 — Sample Selection (Representativeness and Diversity in Activation Space)**:
1. Forward pass over candidate samples to extract layer-wise activation vectors $\mathbf{a}_i = [\mathbf{h}_i^1, \ldots, \mathbf{h}_i^L]$
2. Dimensionality reduction via random projection: $\mathbf{a}_i' = \frac{1}{\sqrt{d}} \mathbf{R} \mathbf{a}_i$
3. K-means clustering to partition the activation space
4. Selection of the sample closest to each cluster centroid: $x_j^* = \arg\min_{x_i \in C_j} \|\mathbf{a}_i' - \mu_j\|_2$

The number of clusters $k$ directly controls the final sample count; AWQ uses fewer samples while pruning methods use more.

## Key Experimental Results

### Experiment 1: Performance Comparison in General Deployment Scenarios

| Compression Method | Calibration Data | LLaMA3-8B PPL | CS | Math | Code | Qwen2.5-7B PPL | CS | Math | Code |
|---|---|---|---|---|---|---|---|---|---|
| AWQ (4-bit) | WikiText (random) | 15.86 | 65.26 | 36.46 | 38.71 | 17.36 | 66.42 | 47.34 | 62.10 |
| AWQ (4-bit) | C4 (random) | 15.48 | 66.21 | 37.19 | 39.87 | 17.00 | 67.42 | 48.29 | 63.72 |
| AWQ (4-bit) | Self-Gen | 15.59 | 67.08 | 37.51 | 39.75 | 17.12 | 68.04 | 48.66 | 63.67 |
| AWQ (4-bit) | **COLA** | **15.41** | **67.42** | **37.85** | **40.17** | **16.95** | **68.47** | **49.02** | **64.15** |
| SparseGPT (50%) | WikiText (random) | 20.15 | 41.85 | 19.18 | 15.34 | 21.54 | 42.23 | 17.85 | 13.45 |
| SparseGPT (50%) | **COLA** | **19.31** | **44.23** | **20.12** | **16.14** | **20.72** | **44.47** | **18.65** | **14.10** |
| GPTQ (4-bit) | WikiText (random) | 16.29 | 65.23 | 31.22 | 34.83 | 17.22 | 65.84 | 35.85 | 34.03 |
| GPTQ (4-bit) | **COLA** | **15.83** | **67.52** | **32.56** | **36.18** | **16.79** | **68.15** | **37.23** | **35.22** |

COLA consistently outperforms random sampling and Self-Gen baselines across all compression methods and models. Improvements are more pronounced for pruning methods (SparseGPT commonsense reasoning +2.38pp), consistent with the observation that pruning is more sensitive to calibration data.

### Experiment 2: Effect of Domain-Matched Calibration (AWQ 4-bit)

| Calibration Data | LLaMA3-8B CS | Math | Code | Qwen2.5-7B CS | Math | Code |
|---|---|---|---|---|---|---|
| WikiText | 65.26 | 36.46 | 38.71 | 66.42 | 47.34 | 62.10 |
| CommonsenseQA | 69.37 | 34.21 | 36.22 | 72.86 | 45.21 | 58.64 |
| MathQA | 64.15 | **41.85** | 35.89 | 65.38 | **54.42** | 57.21 |
| CodeQA | 63.92 | 33.42 | **44.62** | 64.97 | 43.85 | **68.73** |

Domain-matched calibration yields substantial gains on the target capability (MathQA→Math: +5.39pp; CodeQA→Code: +5.91pp), at the cost of degradation on other capabilities. This reveals that calibration data selection is inherently a **capability trade-off** problem.

## Highlights & Insights

- **Systematic investigation**: The first work to comprehensively study the effect of calibration data on LLM compression across both compositional properties and domain correspondence, covering advanced capabilities such as mathematical reasoning and code generation
- **Mechanistic insight**: Identifies representativeness and diversity in the activation space as the fundamental determinants of calibration data quality, transcending surface-level features such as data source
- **Practical framework**: The proposed three-stage COLA framework (selection → processing → sampling) is orthogonal to existing compression methods, plug-and-play compatible, and effective in both general and specialized deployment scenarios
- **Key finding**: The choice of calibration data source can have a larger impact than the choice of compression method itself; explicit reasoning chain format is critical for preserving reasoning capabilities

## Limitations & Future Work

- **Additional computational overhead**: Stage 3 requires a forward pass and clustering over candidate samples, increasing preprocessing cost
- **Limited model scale**: Validation is restricted to 7–8B parameter models; larger-scale models (e.g., 70B+) remain untested
- **Modest absolute gains**: In general deployment scenarios, COLA's improvement over random sampling is typically 1–2 percentage points, representing marginal gains
- **Compression-method-agnostic design**: The current framework applies a unified pipeline to all compression methods, without tailoring to the algorithmic characteristics of pruning versus quantization
- **Sensitivity of clustering parameters**: The selection of the number of clusters $k$ and the reduced dimensionality $d$ for K-means lacks theoretical guidance

## Related Work & Insights

- **Williams & Aletras (2024)**: Identified only the surface phenomenon that calibration data variation leads to performance differences; the present work investigates the underlying activation space mechanisms
- **Bandari et al. (2024)**: Studied whether C4 is the optimal calibration data for pruning, but was limited to pruning and offered no systematic curation strategy
- **Ji et al.**: Proposed Self-Gen, using model-generated data for calibration; this approach is consistently outperformed by COLA in the present experiments
- **Lee et al. (2023)**: Focused on sequence length alignment; the present work further demonstrates that length effects are capability-dependent and method-dependent
- **Zhang et al. (2024), Jaiswal et al. (2024)**: Explored the effect of sample size; the present work additionally reveals that increasing sample size may degrade performance on specific capabilities (e.g., code generation)
- **SparseGPT, Wanda, GPTQ, AWQ**: All mainstream compression methods benefit from the COLA framework, validating its orthogonality

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic study of calibration data effects on advanced reasoning capabilities, with an activation-space mechanistic explanation
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 2 models × 4 compression methods × multiple variable dimensions with comprehensive ablations, though model scale is limited
- Writing Quality: ⭐⭐⭐⭐ — Problem-driven (Q1–Q4) structure is clear; the logical progression from analysis to optimization is coherent
- Value: ⭐⭐⭐⭐ — Offers direct practical guidance for LLM compression; the COLA framework is simple, effective, and plug-and-play

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CoRe: Benchmarking LLMs' Code Reasoning Capabilities through Static Analysis Tasks](core_benchmarking_llms_code_reasoning_capabilities_through_static_analysis_tasks.md)
- [\[NeurIPS 2025\] FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis](fractalbench_diagnosing_visual-mathematical_reasoning_through_recursive_program_.md)
- [\[NeurIPS 2025\] A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions](a_stochastic_differential_equation_framework_for_multi-objective_llm_interaction.md)
- [\[NeurIPS 2025\] Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Learning From Design Procedure To Generate CAD Programs for Data Augmentation](learning_from_design_procedure_to_generate_cad_programs_for_data_augmentation.md)

</div>

<!-- RELATED:END -->
