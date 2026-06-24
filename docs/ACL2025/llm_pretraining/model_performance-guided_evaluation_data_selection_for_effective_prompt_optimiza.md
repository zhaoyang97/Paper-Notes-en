---
title: >-
  [Paper Note] Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization
description: >-
  [ACL 2025][LLM Pretraining][Prompt Optimization] Proposed IPOMP, a two-stage evaluation data selection method. The first stage selects diverse samples through semantic clustering and boundary analysis, and the second stage iteratively replaces redundant samples using real-time model performance during the prompt optimization process. It improves prompt optimization performance by 1.6%-3.1% and stability by 50%+ on BIG-bench and LIAR, with less than 1% extra overhead.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Prompt Optimization"
  - "Coreset Selection"
  - "Evaluation Data Selection"
  - "Real-Time Model Performance"
  - "Semantic Clustering"
date: 2026-05-08
content_hash: fb592c1eee3281c6
---

# Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization

**Conference**: ACL 2025  
**arXiv**: [2505.10736](https://arxiv.org/abs/2505.10736)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: Prompt Optimization, Coreset Selection, Evaluation Data Selection, Real-Time Model Performance, Semantic Clustering

## TL;DR

Proposed IPOMP, a two-stage evaluation data selection method. The first stage selects diverse samples through semantic clustering and boundary analysis, and the second stage iteratively replaces redundant samples using real-time model performance during the prompt optimization process. It improves prompt optimization performance by 1.6%-3.1% and stability by 50%+ on BIG-bench and LIAR, with less than 1% extra overhead.

## Background & Motivation

Prompt Optimization is a key step in improving LLM task performance, but automated prompt optimization faces a neglected core problem: **the selection of evaluation data**.

Limitations of Prior Work:

**Full evaluation is impractical**: Evaluating each candidate prompt on the entire training set is too costly.

**Random sampling is unreliable**: Randomly selected subsets are likely unrepresentative, leading to unstable evaluations and sub-optimal prompts.

**Existing coreset methods are inapplicable**:
   - Semantic clustering performs poorly on highly similar task samples.
   - Model performance-based methods require pre-collecting evaluation data, which is expensive and has poor transferability.
   - No historical performance data is available for new or private datasets.

This is the first work to propose an evaluation data selection method specifically for prompt optimization scenarios.

## Method

### Overall Architecture

IPOMP is a two-stage method embedded into the iterative process of prompt optimization:

- **Stage 1 (Diverse Sample Selection)**: Semantic clustering + boundary sample selection
- **Stage 2 (Real-time Performance-Guided Refinement)**: Dynamically replacing redundant samples during each iteration.

### Key Designs

1. **Stage 1: Diverse Sample Selection**:

    - All training samples are encoded by Sentence-BERT, followed by K-means clustering ($k=5$).
    - Proportional sampling of $\alpha N$ samples from each cluster (semantic representativeness).
    - Find the furthest distance sample pairs in the semantic space to select $(1-\alpha)N$ samples (boundary diversity).
    - Use HNSW to accelerate boundary point detection, avoiding the full $O(dN^2)$ distance calculation.
    - Design Motivation: Clustering ensures representativeness, while boundary samples compensate for coverage in extreme cases.

2. **Stage 2: Real-time Performance-Guided Refinement**:

    - Core Observation: Around 20% of the samples show highly correlated performance (>0.9) across different candidate prompts, indicating significant redundancy.
    - Record a performance matrix during each iteration: samples $\times$ (number of output labels $\times$ number of candidate prompts), using logits.
    - Hierarchical clustering groups highly correlated samples, randomly selecting a $\beta$ ratio of redundant samples.
    - Replace these with the most dissimilar training samples in the semantic space.
    - Design Motivation: Leverage "free" model feedback during optimization to identify redundancy with zero extra inference overhead.

3. **Key Differences from Existing Performance-Guided Methods**:

    - Anchor-Point requires an extra warmup phase (~200 seconds + API cost).
    - Prediction-based requires training an evaluator on other LLMs.
    - IPOMP directly utilizes the byproducts of the optimization process, incurring zero extra inference cost.

### Key Hyperparameters

- Evaluation set size $N=20$, number of clusters $K=5$
- Cluster vs. boundary ratio $\alpha=0.5$
- Correlation threshold $CT=0.9$, replacement rate $\beta=0.5$

## Key Experimental Results

### Main Results (Accuracy ± SD, averaged across three optimization methods)

| Method | GPT-3.5 BIG-bench | GPT-4o-mini BIG-bench | GPT-3.5 LIAR | GPT-4o-mini LIAR |
|------|:-:|:-:|:-:|:-:|
| Random | 0.719±0.035 | 0.704±0.041 | 0.742±0.041 | 0.748±0.048 |
| Clustering | 0.725±0.029 | 0.725±0.029 | 0.752±0.036 | 0.788±0.038 |
| Boundary | 0.727±0.040 | 0.723±0.038 | 0.770±0.035 | 0.797±0.047 |
| Anchor-Point | 0.745±0.028 | 0.756±0.027 | 0.801±0.027 | 0.807±0.024 |
| Prediction-based | 0.725±0.038 | 0.705±0.044 | 0.746±0.039 | 0.750±0.043 |
| **IPOMP** | **0.757±0.012** | **0.778±0.011** | **0.820±0.012** | **0.833±0.012** |

### Ablation Study

| Variant | GPT-3.5 BB Acc | GPT-4o-mini BB Acc |
|------|:-:|:-:|
| IPOMP (Full) | 0.757±0.012 | 0.778±0.011 |
| Stage 1 only | 0.733±0.034 | 0.743±0.012 |
| Random + Stage 2 | 0.737±0.021 | 0.738±0.012 |

Removing Stage 2: Acc drops by 2.4%, SD increases by 2.83x. Replacing Stage 1 with Random: Acc drops by 2%.

### Time Overhead (BIG-bench, GPT-3.5, seconds)

| Component | APO | APE | EVOPROMPT | Average |
|------|-----|-----|-----------|------|
| Stage 1 | 0.45 | 0.37 | 0.51 | 0.45 |
| Stage 2 | 2.74 | 2.31 | 3.43 | 2.83 |
| Anchor-Point Warmup | 200.36 | 205.32 | 207.85 | 204.51 |

IPOMP extra overhead < 1%, while Anchor-Point warmup increases time by ~50%.

### Key Findings

1. All coreset methods outperform Random, demonstrating that evaluation data selection is crucial for prompt optimization.
2. Compared to the second-best Anchor-Point, IPOMP achieves +1.6%~3.1% Acc gain and reducing SD by 50%+.
3. Stage 2 can serve as a plug-and-play plugin: bringing 2.3%/1.1%/1.5% gains to Random/Boundary/Clustering, respectively.
4. A sample size of 20 is the optimal trade-off point.
5. A single round of refinement can reduce highly correlated redundancy from 19% to 10%.

## Highlights & Insights

1. **Novel Problem Formulation**: First to investigate evaluation data selection in prompt optimization.
2. **Elegant Design of Real-Time Performance Utilization**: Masterfully reuses existing model feedback from the optimization process.
3. **Generality**: Stage 2 can be applied plug-and-play to enhance any data selection method.
4. **Improved Stability**: Reducing SD by 50%+ signifies a substantial boost in reliability in production environments.

## Limitations & Future Work

1. **Limited Model Coverage**: Only evaluated on GPT-3.5 and GPT-4o-mini.
2. **Task Bias towards Classification**: Performance on generation tasks remains unverified.
3. **Logits Dependency**: Limited applicability to APIs where logits cannot be obtained.
4. **Prompt Optimization Techniques**: Covers only APE, APO, and EVOPROMPT.

## Related Work & Insights

- Three major categories of prompt optimization: non-directional (APE/EvoPrompt), directional (APO).
- Evolution of coreset selection from geometric methods to performance-based methods.
- Core insight: In prompt optimization, "what to evaluate" is as important as "how to optimize".

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to define and address the evaluation data selection problem in prompt optimization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated across multiple datasets, models, and optimization techniques with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation of the problem, systematic description of the method.
- **Value**: ⭐⭐⭐⭐ — High practical value for Stage 2 as a general plug-and-play plugin.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](data_whisperer_data_selection.md)
- [\[ACL 2025\] DavIR: Data Selection via Implicit Reward for Large Language Models](davir_data_selection_via_implicit_reward_for_large_language_models.md)
- [\[ICLR 2026\] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](../../ICLR2026/llm_pretraining/predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)
- [\[ICML 2025\] LLM Data Selection and Utilization via Dynamic Bi-level Optimization](../../ICML2025/llm_pretraining/llm_data_selection_and_utilization_via_dynamic_bi-level_optimization.md)
- [\[ICLR 2026\] OptimSyn: Influence-Guided Rubrics Optimization for Synthetic Data Generation](../../ICLR2026/llm_pretraining/optimsyn_influence-guided_rubrics_optimization_for_synthetic_data_generation.md)

</div>

<!-- RELATED:END -->
