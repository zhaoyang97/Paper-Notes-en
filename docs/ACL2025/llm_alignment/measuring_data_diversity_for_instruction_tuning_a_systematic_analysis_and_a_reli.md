---
title: >-
  [Paper Note] Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric
description: >-
  [ACL 2025][LLM Alignment][Data Diversity] Systematically analyzes the limitations of 11 existing diversity measurement methods and proposes NovelSum—a data diversity metric that simultaneously considers sample uniqueness and information density, achieving a 0.97 correlation with instruction tuning performance.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Data Diversity"
  - "instruction tuning"
  - "NovelSum"
  - "Data Selection"
  - "Metric Design"
date: 2026-05-08
content_hash: e431ecb35eaaac8a
---

# Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric

**Conference**: ACL 2025  
**arXiv**: [2502.17184](https://arxiv.org/abs/2502.17184)  
**Code**: [https://github.com/UmeanNever/NovelSum](https://github.com/UmeanNever/NovelSum)  
**Area**: LLM Alignment / Data Engineering  
**Keywords**: Data Diversity, instruction tuning, NovelSum, Data Selection, Metric Design

## TL;DR
Systematically analyzes the limitations of 11 existing diversity measurement methods and proposes NovelSum—a data diversity metric that simultaneously considers sample uniqueness and information density, achieving a 0.97 correlation with instruction tuning performance.

## Background & Motivation
**Background**: Data diversity is crucial for instruction tuning (IT), and various diversity-aware data selection methods continue to emerge.

**Limitations of Prior Work**: The fundamental problem of accurately defining and measuring data diversity has not been sufficiently explored, leaving data engineering as a black-box process.

**Key Challenge**: Existing diversity metrics have their own biases but fail to simultaneously capture sample distinctiveness and informational space density.

**Goal**: To provide a reliable diversity metric that is strongly correlated with the performance of fine-tuned models.

**Key Insight**: Conducting large-scale experiments to validate the correlation of 11 metrics with model performance, identifying causes of failure, and designing a new metric.

**Core Idea**: Dataset diversity = the sum of the "novelty" of each sample, where novelty is defined by proximity-weighted, density-aware distance.

## Method

### Overall Architecture
(1) Construct 53 IT datasets using various data selection strategies → (2) Measure diversity with 11 metrics → (3) Fine-tune and evaluate → (4) Analyze correlation → (5) Propose NovelSum.

### Key Designs
1. **Proximity-Weighted Sum**:

    - **Function**: Computes the uniqueness score of each sample.
    - **Mechanism**: For each sample, neighbors sorted by distance are assigned decreasing weights $w(x_i, x_j) = 1/\pi_i(j)$, meaning near neighbors have a greater impact than distant ones.
    - **Design Motivation**: DistSum is dominated by distant points, while KNN only considers the nearest neighbors. The proximity-weighted sum strikes a balance.

2. **Density-Aware Distance**:

    - **Function**: Introduces a local density factor on top of semantic distance.
    - **Mechanism**: $\Delta(x_i, x_j) = \sigma(x_j)^\beta \cdot d(x_i, x_j)$, where distances in high-density regions (such as math/code) are amplified.
    - **Design Motivation**: Semantically similar mathematical samples may contain highly diverse information. Pure semantic distance underestimates the diversity in high-density regions.

3. **NovelSelect Data Selection Strategy**:

    - **Function**: Greedy data selection based on NovelSum.
    - **Mechanism**: Iteratively selects samples that maximize the incremental gain of NovelSum.
    - **Design Motivation**: To directly translate the metric into an actionable selection strategy.

### Loss & Training
The full pool = WizardLM + ShareGPT + UltraChat, from which 10,000 samples are fixedly selected. Standard SFT is used to train LLaMA-3-8B and Qwen-2.5-7B.

## Key Experimental Results

### Main Results (LLaMA-3-8B, Correlation Comparison)

| Metric | Pearson r | Spearman r | Average r |
|------|----------|-----------|-------|
| NovelSum | **0.98** | **0.95** | **0.97** |
| Vendi Score | 0.61 | 0.64 | 0.63 |
| DistSum_cosine | 0.74 | 0.69 | 0.72 |
| Facility Location | 0.52 | 0.48 | 0.50 |
| KNN Distance | 0.71 | 0.66 | 0.69 |
| Partition Entropy | 0.55 | 0.51 | 0.53 |

### Ablation Study

| Component | Effect |
|------|------|
| Remove Proximity Weight | Similar to DistSum, dominated by distant points, r decreases by ~0.25 |
| Remove Density-Aware | Diversity in high-density areas is underestimated, r decreases by ~0.15 |
| $\alpha$ Hyperparameter | $\alpha=1$ performs best |
| $\beta$ Hyperparameter | $\beta=0.5$ is optimal |

### Key Findings
- Lexical diversity metrics (TTR, vocd-D) are almost uncorrelated with IT performance.
- Distance-based metrics ignore information density, overestimating outlier datasets.
- Distribution-based metrics ignore sample uniqueness, underestimating high-distance datasets.
- NovelSum maintains a strong correlation across two different backbone models.

## Highlights & Insights
- The analogy of "paper novelty" is highly intuitive—the novelty of a sample depends on its difference from related work in the same field.
- Simulation experiments cleverly visualize the differences in behavior among different metrics.
- NovelSelect demonstrates the feasibility of directly converting a metric into an actionable strategy.

## Limitations & Future Work
- Validated only in general-purpose IT scenarios; domain-specific (e.g., medical/code) scenarios have not been tested.
- The choice of embedding models may affect the results.
- Computational complexity scales with data volume, necessitating approximation algorithms for large-scale data.

## Related Work & Insights
- **vs QDIT (Bukharin et al. 2024)**: QDIT optimizes Facility Location, whereas NovelSum demonstrates that FL ignores sample uniqueness.
- **vs Repr Filter (Liu et al. 2023)**: Repr Filter is based on KNN thresholds, while NovelSum's density-aware approach is more precise.


## Additional Details
- Data source: WizardLM + ShareGPT + UltraChat, with 10,000 samples fixedly selected.
- Embedding model: BERT is used to extract semantic representations of samples.
- Evaluation benchmarks: MT-bench and AlpacaEval, aggregated into a Z-score.
- NovelSum hyperparameters $\alpha$ and $\beta$ control proximity weight decay and density impact, respectively.
- NovelSelect greedy algorithm: Iteratively selects samples that maximize the incremental gain of NovelSum.
- The high correlation of NovelSum was also validated on Qwen-2.5-7B.
- The code is open-source and includes construction scripts for all 53 datasets.
- K-Center-Greedy and Repr Filter datasets score high on distance metrics but do not necessarily perform well in practice.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Conducts the first systematic analysis of the correlation between diversity metrics and IT performance, proposing a novel methodology in NovelSum.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 53 datasets $\times$ 2 models $\times$ 11 metrics, extremely thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical and clear with a complete narrative arc of discovery-design-validation.
- Value: ⭐⭐⭐⭐⭐ Provides quantifiable diversity guidance for instruction tuning data engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Call for Rigor in Reporting Quality of Instruction Tuning Data](call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)
- [\[ACL 2025\] Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)
- [\[NeurIPS 2025\] T-SHIRT: Token-Selective Hierarchical Data Selection for Instruction Tuning](../../NeurIPS2025/llm_alignment/t-shirt_token-selective_hierarchical_data_selection_for_instruction_tuning.md)
- [\[ACL 2025\] Federated Data-Efficient Instruction Tuning for Large Language Models](federated_data-efficient_instruction_tuning_for_large_language_models.md)
- [\[ACL 2025\] JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning](jsontuning_towards_generalizable_robust_and_controllable_instruction_tuning.md)

</div>

<!-- RELATED:END -->
