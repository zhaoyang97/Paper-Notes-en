---
title: >-
  [Paper Note] Learning Uncertainty from Sequential Internal Dispersion in Large Language Models
description: >-
  [ACL 2026][Time Series][uncertainty estimation] This paper proposes the SIVR framework, which computes internal variance statistics (generalised variance, circular variance, and token entropy) across layers of LLM hidden states as token-level features, and aggregates full-sequence patterns via a lightweight Transformer encoder to estimate uncertainty and detect hallucinations, achieving significant improvements over baselines with stronger generalization.
tags:
  - ACL 2026
  - Time Series
  - uncertainty estimation
  - hallucination detection
  - hidden state variance
  - sequential aggregation
  - internal representation dispersion
date: 2026-05-08
content_hash: e2b4835852309ff0
---

# Learning Uncertainty from Sequential Internal Dispersion in Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.15741](https://arxiv.org/abs/2604.15741)
**Code**: [GitHub](https://github.com/ponhvoan/internal-variance)
**Area**: Uncertainty Estimation / Hallucination Detection
**Keywords**: uncertainty estimation, hallucination detection, hidden state variance, sequential aggregation, internal representation dispersion

## TL;DR

This paper proposes the SIVR framework, which computes internal variance statistics (generalised variance, circular variance, and token entropy) across layers of LLM hidden states as token-level features, and aggregates full-sequence patterns via a lightweight Transformer encoder to estimate uncertainty and detect hallucinations, achieving significant improvements over baselines with stronger generalization.

## Background & Motivation

**State of the Field**: Uncertainty estimation is a critical approach for detecting hallucinations in LLMs. Existing methods include sampling-based consistency (e.g., Semantic Entropy), output probability methods (e.g., Entropy), and internal state probe methods.

**Limitations of Prior Work**: (1) Sampling-based methods incur substantial computational overhead; (2) methods such as CoE impose overly strict assumptions about layer-wise evolution that do not hold across models or tasks; (3) using only the last or average token discards temporal patterns in the sequence.

**Root Cause**: CoE compresses information into a single scalar, ignoring variance patterns across different token positions. For example, in "Praia is in Portugal," a variance spike at "Portugal" can flag an error, but mean aggregation masks such signals.

**Paper Goals**: To design internal state features based on more relaxed assumptions while preserving complete sequential information.

**Starting Point**: Uncertainty is reflected in the degree of "dispersion" of hidden states across layers — representations are more concentrated when correct and more dispersed when incorrect.

**Core Idea**: Three dispersion statistics (generalised variance, circular variance, and token entropy) characterize the cross-layer dispersion of each token, and a Transformer encoder learns full-sequence patterns to predict hallucinations.

## Method

### Overall Architecture

For each generated token, hidden states from all layers are extracted, and three internal variance features $\bm{v}_t = [v_t, c_t, e_t]$ are computed, forming a sequence that is fed into a lightweight Transformer encoder for binary classification.

### Key Designs

1. **Generalised Variance**:

    - Function: Measures the volumetric dispersion across layers.
    - Mechanism: Computes the log-determinant of the regularised covariance matrix $v_t = \log\det(\Sigma') = \sum_i \log \lambda_i$, aggregating the full feature spectrum.
    - Design Motivation: Unlike CoE, which only examines differences between adjacent layers, generalised variance is directly related to differential entropy and provides a more comprehensive measure of dispersion.

2. **Circular Variance**:

    - Function: Measures the directional dispersion across layers.
    - Mechanism: Normalizes hidden states at each layer and computes the magnitude of the mean vector, $c_t = 1 - \|\frac{1}{L+1}\sum_l \hat{\bm{h}}_t^l\|$.
    - Design Motivation: Complementary to generalised variance — the former captures magnitude while the latter captures direction, implicitly encoding all pairwise layer relationships.

3. **Sequential Aggregation Transformer Classifier**:

    - Function: Learns hallucination detection from the full sequence of dispersion patterns.
    - Mechanism: An embedding layer (128-dimensional) followed by a single-layer Transformer encoder and a linear classification head, trained with binary cross-entropy and $l_2$ regularization.
    - Design Motivation: Preserving sequential order captures temporal patterns such as variance spikes, which is more effective than mean or last-token aggregation.

### Loss & Training

Binary cross-entropy with $l_2$ regularization; requires only hundreds to thousands of labeled samples.

## Key Experimental Results

### Main Results

AUC comparison across 7 datasets on Llama-3.1-8B:

| Method | TriviaQA | SciQ | MedMCQA | MATH | Avg. AUC | Rank |
|--------|---------|------|---------|------|---------|------|
| Entropy | 80.46 | 72.85 | 62.76 | 62.77 | 67.63 | 7.96 |
| SE | 84.44 | 79.44 | 66.88 | 67.27 | 68.87 | 7.13 |
| CoE-C | 66.97 | 75.06 | 62.14 | 58.67 | 61.25 | 11.08 |
| **SIVR** | **90.75** | **83.64** | **68.37** | **71.22** | **75.35** | **1.88** |

### Ablation Study

| Configuration | Avg. AUC | Notes |
|---------------|---------|-------|
| Token Entropy only | 71.2 | Effective but insufficient alone |
| Generalised Variance only | 72.8 | Complementary signal |
| All three combined (SIVR) | **75.35** | Best performance |
| Mean aggregation instead of sequence | 72.5 | Loss of temporal patterns |

### Key Findings

- SIVR achieves an average rank of 1.88, substantially outperforming the second-best method; the three features exhibit strong complementarity.
- Sequential aggregation improves AUC by 2–3 points over mean/last-token aggregation, demonstrating the value of temporal patterns.
- Out-of-distribution generalization is significantly better than CoE, requiring only a small amount of training data.

## Highlights & Insights

- **The "dispersion" assumption is more robust than the "step size" assumption** — CoE's assumptions are inconsistent across models, whereas SIVR's assumptions are more fundamental and general.
- **The paradigm of preserving sequential structure is transferable** — any task that requires inferring sequence-level properties from token-level signals can benefit from this approach.
- **Lightweight yet effective** — three statistics combined with a single-layer Transformer result in negligible inference overhead.

## Limitations & Future Work

- Labeled data are required; although the quantity is small, new domains necessitate additional annotation.
- Evaluation is limited to greedy decoding; performance under sampling-based decoding remains to be assessed.
- Validation on large-scale models (70B+) is insufficient.
- The use of SIVR for proactive hallucination mitigation has not been explored.

## Related Work & Insights

- **vs. CoE**: CoE's overly strong assumptions fail across tasks; SIVR adopts more relaxed assumptions.
- **vs. Semantic Entropy**: SE requires multiple sampling passes and is computationally expensive; SIVR requires only a single forward pass.
- **vs. Lookback Lens**: Lookback Lens focuses on specific layers and attention patterns, whereas SIVR provides a more global perspective.

## Rating

- Novelty: ⭐⭐⭐⭐ The internal variance feature idea is conceptually clear; individual components are simple but effective in combination.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablations across 7 datasets and multiple models.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly argued with effective visualizations.
- Value: ⭐⭐⭐⭐⭐ Highly practical with direct applicability to hallucination detection.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] PlanU: Large Language Model Reasoning through Planning under Uncertainty](../../NeurIPS2025/time_series/planu_large_language_model_reasoning_through_planning_under_uncertainty.md)
- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](../../ICLR2026/time_series/timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [\[NeurIPS 2025\] Causal Masking on Spatial Data: An Information-Theoretic Case for Learning Spatial Datasets with Unimodal Language Models](../../NeurIPS2025/time_series/causal_masking_on_spatial_data_an_information-theoretic_case_for_learning_spatia.md)
- [\[ICCV 2025\] VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models](../../ICCV2025/time_series/vlrmbench_a_comprehensive_and_challenging_benchmark_for_vision-language_reward_m.md)
- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](../../NeurIPS2025/time_series/causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)

<!-- RELATED:END -->
