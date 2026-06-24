---
title: >-
  [Paper Note] Redundancy Principles for MLLMs Benchmarks
description: >-
  [Multimodal VLM] This paper systematically quantifies the redundancy in current MLLM benchmarks across three levels: dimension redundancy, instance redundancy, and cross-benchmark redundancy. It proposes a redundancy analysis framework based on performance ranking correlations, providing principled guidance for future benchmark design.
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: d653acdd2ec39ba1
---

# Redundancy Principles for MLLMs Benchmarks

| Property | Content |
|------|------|
| Title | Redundancy Principles for MLLMs Benchmarks |
| Conference | ACL2025 |
| arXiv | [2501.13953](https://arxiv.org/abs/2501.13953) |
| Code | - |
| Area | Multimodal VLM / Benchmark Evaluation |
| Keywords | MLLM, Benchmark Redundancy, Evaluation, Correlation Analysis, VLMEvalKit |

## TL;DR

This paper systematically quantifies the redundancy in current MLLM benchmarks across three levels: dimension redundancy, instance redundancy, and cross-benchmark redundancy. It proposes a redundancy analysis framework based on performance ranking correlations, providing principled guidance for future benchmark design.

## Background & Motivation

- With the rapid iteration of Multimodal Large Language Models (MLLMs), hundreds of evaluation benchmarks emerge every year, yet substantial overlap and redundancy exist among them.
- Redundancy leads to inefficient evaluation, repeatedly testing similar capabilities without providing valuable new insights.
- Overemphasizing certain task types may distort research priorities.
- A systematic framework is needed to quantify the degree of redundancy and to provide guiding principles for benchmark design.

**Core Problem**: Exactly how much redundancy exists in current MLLM benchmarks? How can new benchmarks be scientifically designed to reduce unnecessary redundancy?

## Method

### Overall Architecture: Performance Correlation Redundancy Framework

Core Idea: When evaluating similar capabilities, performance rankings of MLLMs should exhibit a strong correlation; conversely, a large discrepancy in rankings indicates that the evaluated capabilities are relatively independent.

Based on this hypothesis, redundancy is quantified by measuring the correlation of MLLM performance rankings. The data is sourced from VLMEvalKit, encompassing evaluation results of 100+ MLLMs across 20+ benchmarks.

### 1. Dimension Redundancy

For a benchmark containing $m$ dimensions, the redundancy of each dimension $X_i$ is defined as:

$$\rho(X_i) = \frac{1}{m-1}\sum_{j \neq i} \text{CORR}(R_i, R_j)$$

where $R_i$ is the ranking of $N$ MLLMs on dimension $X_i$. The overall internal redundancy of the benchmark is the average of all dimension redundancies:

$$\rho_{BI} = \frac{1}{m}\sum_{i=1}^{m}\rho(X_i)$$

### 2. Instance Redundancy

By randomly sampling $A\%$ from a total of $M$ instances, the correlation between the post-sampling MLLM ranking and the ground truth ranking is calculated:

$$\rho(A\%) = \frac{1}{T}\sum_{t=1}^{T}\text{CORR}(R_{A\%_t}, R_{GT})$$

Sampling is repeated $T=100$ times and averaged to mitigate the effect of randomness. High correlation implies that only a small number of instances are required to represent the entire benchmark.

### 3. Cross-Benchmark Redundancy

For $l$ benchmarks within the same domain, the redundancy of the $i$-th benchmark is:

$$\rho(Y_i) = \frac{1}{l-1}\sum_{j \neq i}\text{CORR}(K_i, K_j)$$

High redundancy indicates that this benchmark can serve as an "anchor benchmark" within the domain, representing multiple other benchmarks.

### Correlation Metrics

Three metrics are adopted: SRCC (Spearman's Rank Correlation Coefficient) to measure ranking consistency, PLCC (Pearson Linear Correlation Coefficient) to measure linear relationship, and $R^2$ (Coefficient of Determination) to measure goodness of fit.

### Top-K Analysis

The analysis focuses on Top-K high-performance MLLMs, as the performance of top-tier models receives more attention from the research community.

## Key Experimental Results

### 1. Dimension Redundancy Analysis (Taking MMBench as an Example)

**Top-50 MLLMs**:
- Image Emotion and Social Relation exhibit strong redundancy, indicating highly overlapping evaluated skills.
- Structuralized Image-Text Understanding is redundant with multiple dimensions such as Spatial Relationship, Physical Property Reasoning, and OCR.
- Celebrity Recognition remains relatively independent, as it is a knowledge-based rather than perception-based task.
- Nature Relation and Spatial Relationship exhibit the highest redundancy, as they are foundational skills for many other tasks.

**Bottom-50 MLLMs**:
- Almost all dimensions show significantly higher redundancy ($SRCC/PLCC > 0.6$).
- Reason: Weak models generally lack fundamental capabilities; improvements in one dimension often lead to simultaneous gains in other dimensions.
- Key Insight: Redundancy analysis should avoid including universally poor-performing models.

### 2. Instance Redundancy Analysis

| Finding | Description |
|------|------|
| Most benchmarks have 50%+ instance redundancy | With 0.95 SRCC/PLCC as the threshold, at least half of the instances are redundant. |
| Ranking vs. Absolute Performance Prediction | Far fewer instances are needed for a reliable ranking than for precise performance prediction (the latter requires 90%+ instances to achieve $R^2 > 0.95$). |
| Model capability impacts redundancy | Top-50 require more instances, while Bottom-50 can be ranked with fewer instances. |
| Significant variation across benchmarks | RealWorldQA has the lowest redundancy (requiring 80% of instances to saturate), whereas other benchmarks require far less. |

### 3. Cross-Benchmark Redundancy Analysis (Math Domain)

Analyzing MathVista, MathVision, MathVerse, and DynaMath:
- MathVista shows the lowest redundancy because it contains 30-40% non-mathematical tasks ("noise" such as General VQA and chart understanding).
- MathVerse and MathVision show the highest redundancy, focusing on standard mathematical tasks.
- After removing non-mathematical tasks from MathVista, its redundancy increases significantly, aligning better with other math benchmarks.

## Highlights & Insights

1. **Systematic Quantification of MLLM Benchmark Redundancy for the First Time**: Establishes a complete redundancy analysis methodology across three levels (dimension, instance, cross-benchmark).
2. **Relationship Between Model Capability and Redundancy**: Strong models reduce benchmark redundancy, while weak models increase redundancy—this finding provides crucial guidance for benchmark design.
3. **Pragmatic Principles**: Proposes concrete principles for benchmark design—benchmarks designed for domain-wide assessment should have high redundancy with other benchmarks in the same domain, while specialized benchmarks should maintain low redundancy.
4. **Clear Efficiency Benefits**: Most benchmarks can cut their instance counts in half without affecting MLLM rankings, which is vital for saving computational resources.
5. **Concept of Anchor Benchmarks**: High-redundancy benchmarks can act as domain representatives, allowing domain performance estimation without running all benchmarks.

## Limitations & Future Work

1. The core assumption (that rankings for similar capabilities are highly correlated) might fail in certain cases, as similar tasks may lead to performance divergence due to subtle variations.
2. Correlation metrics ($SRCC/PLCC/R^2$) may not fully capture the complete complexity of models across different tasks and conditions.
3. The redundancy analysis results depend on the selection of MLLMs participating in the calculation; different choices might lead to different conclusions.
4. The analysis is based on statistical methods; niche or task-specific benchmarks require case-by-case analysis.

## Related Work & Insights

- **Traditional VQA Benchmarks**: GQA, VQA-V2, VizWiz, TextVQA, etc., contain simple questions and are unsuitable for evaluating modern MLLMs.
- **New-Generation VQA Benchmarks**: MMBench, MMVet, MMMU, etc., are more flexible but suffer from redundancy due to rapid iteration.
- **Domain-Specific Benchmarks**: Mathematics (MathVista, MathVerse), OCR, medical, remote sensing, etc.
- **Evaluation Toolkits**: VLMEvalKit provides a unified evaluation framework and open-source data.

## Rating ⭐⭐⭐⭐

**Pros**: Proposes a highly practical and systematic framework to analyze benchmark redundancy. The experiments comprehensively cover 20+ benchmarks and 100+ models, and the conclusions offer direct guiding value to the community.

**Cons**: The methodology itself is relatively straightforward (based on ranking correlations) and lacks deeper capability modeling; the analysis relies on existing evaluation data and cannot predict the redundancy of newly emerging benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](teaching_vlm_ask_ambiguity.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)

</div>

<!-- RELATED:END -->
