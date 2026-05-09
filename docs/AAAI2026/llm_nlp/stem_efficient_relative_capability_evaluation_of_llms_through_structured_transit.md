---
title: >-
  [Paper Note] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model
description: >-
  [LLM/NLP] This paper proposes STEM, a framework that identifies "Significant Transition Samples" (STS) across models of the same architecture but varying scales to construct a lightweight evaluation subset, enabling efficient relative capability localization of unknown LLMs. STEM achieves 100% localization accuracy with only 100 samples, substantially outperforming random sampling and Bayesian methods.
tags:
  - LLM/NLP
date: 2026-05-08
content_hash: 56c69a5ae72909fc
---

# STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model

- **Conference**: AAAI 2026
- **arXiv**: [2508.12096](https://arxiv.org/abs/2508.12096)
- **Code**: Not released
- **Area**: LLM Evaluation / Model Capability Localization
- **Keywords**: LLM evaluation, significant transition samples, transition index, benchmark bias, data contamination, scaling laws

## TL;DR

This paper proposes STEM, a framework that identifies "Significant Transition Samples" (STS) across models of the same architecture but varying scales to construct a lightweight evaluation subset, enabling efficient relative capability localization of unknown LLMs. STEM achieves 100% localization accuracy with only 100 samples, substantially outperforming random sampling and Bayesian methods.

## Background & Motivation

- **Background**: LLMs frequently set new SOTA records on standard benchmarks such as MMLU, GPQA, and GSM8K, yet a notable gap persists between benchmark scores and real-world user experience. Data contamination—where benchmark samples are memorized during training—inflates scores and fails to reflect genuine reasoning ability.
- **Limitations of Prior Work**: Anomalous scaling behavior has been observed in the Qwen3 series on GPQA, where the 8B model scores 44.44, higher than both the 14B (39.90) and 30B-A3B (43.94) models, indicating that larger model size does not consistently yield improved capability.
- **Key Challenge**: Existing benchmarks exhibit severely polarized difficulty distributions. Simple samples account for 52.81% of MMLU and 59.59% of GSM8K, while hard samples constitute 52.53% of GPQA and 55.94% of SuperGPQA. The proportion of intermediate-difficulty samples is low (only 20.07% for GPQA and 34.80% for GSM8K), resulting in insufficient discriminability across models.
- **Goal**: Full-benchmark evaluation incurs high computational cost, and random sampling suffers from high variance and instability. A lightweight yet reliable evaluation approach is therefore urgently needed.

## Method

### Mechanism

The core observation underlying STEM is that as model parameter count increases, most samples exhibit predictable capability transitions—smaller models answer incorrectly while larger models answer correctly. By filtering for samples satisfying a monotonic transition condition, STEM constructs a difficulty-balanced evaluation subset that can be used to infer the capability position of an unknown model within a known model family.

### Key Design 1: Inference Result Vector (IRV) and Significant Transition Samples (STS)

For each sample, an Inference Result Vector is defined as $\text{IRV} = \{v_1, v_2, \dots, v_n\}$, where $v_i \in \{-1, 0, 1\}$ denotes inference failure, incorrect answer, and correct answer for model $M_i$, respectively. The model sequence is strictly ordered by parameter count: $M_1 \prec M_2 \prec \cdots \prec M_n$.

A **Significant Transition Sample (STS)** must satisfy two conditions:

1. **Monotonicity**: There exists a unique transition point $k$ such that $\forall i < k, v_i = 0$ (all smaller models answer incorrectly) and $\forall i' > k, v_{i'} = 1$ (all larger models answer correctly).
2. **Uniqueness**: The IRV contains exactly one $0 \to 1$ transition with no oscillation.

For example, $\text{IRV} = (0,0,0,1,1,1,1,1)$ indicates a transition point at index 3 (the fourth model is the first to answer correctly). Anomalous IRVs such as $(0,0,1,0,1,1,0,1)$ suggest data contamination and are filtered out.

Each STS is assigned a **Transition Index (TI)** $k$, representing the minimum model scale that can reliably answer the sample correctly, effectively encoding the sample's difficulty level.

### Key Design 2: Benchmark Discriminability Weighting and Capability Reference Score

To establish a unified capability reference ranking for LLMs, the paper introduces a benchmark discriminability metric. The discriminability $D_j$ of the $j$-th benchmark is defined as:

$$D_j = \sigma_{S_j} \times \rho_{S_j, \log(P)}$$

where $\sigma_{S_j}$ is the standard deviation of model scores on that benchmark (reflecting discriminative power), and $\rho_{S_j, \log(P)}$ is the Pearson correlation between scores and the logarithm of model parameter count (measuring consistency with scaling laws). Weights are computed from discriminability as:

$$w_j = \frac{D_j}{\sum_{j=1}^{m} D_j}$$

The final capability reference score for each model is a weighted aggregation of its scores across benchmarks. This design avoids the information loss caused by simple averaging, which ignores differences in benchmark informativeness.

### Key Design 3: Structured Evaluation Protocol

1. **STS Pool Construction**: All STS are extracted from complete benchmarks and categorized by TI value $k \in \{1, 2, \dots, n+1\}$ (where $k = n+1$ indicates that even the largest model answers incorrectly).
2. **Balanced Subset Sampling**: An equal number of STS are randomly sampled from each TI level, ensuring that the evaluation subset covers all difficulty thresholds while keeping the total sample count manageable.
3. **Capability Boundary Inference**: The unknown model is evaluated on the balanced subset; its capability boundary is defined as the lowest TI value at which accuracy begins to drop significantly.

## Experiments

### Experimental Setup

- **Reference Model Family**: Qwen3 series with 8 models (0.6B → 235B-A22B), covering the full parameter range from small to large.
- **External Test Models**: LLaMA3-8B and GLM4-9B (different architectures, similar capability levels).
- **Benchmarks**: MMLU, MMLU-Pro, SuperGPQA, GPQA, GSM8K, MATH (6 total).
- **Baselines**: Random sampling, Bayesian method (Xiao et al. 2025), STEM.
- **Settings**: 100 samples, repeated 100 times; zero-shot non-CoT inference; FP32 precision.

### Table 1: Benchmark Discriminability

| Benchmark | MMLU | MMLU-Pro | SuperGPQA | GPQA | GSM8K | MATH |
|-----------|------|----------|-----------|------|-------|------|
| Discriminability $D$ | 10.36 | 13.13 | 8.75 | 7.04 | 9.57 | 10.77 |

MMLU-Pro achieves the highest discriminability (13.13), while GPQA achieves the lowest (7.04), indicating that GPQA is relatively ineffective at distinguishing models of different capability levels.

### Table 2: Localization Accuracy of Three Evaluation Strategies

| Model | Random Sampling | Bayesian | STEM |
|-------|-----------------|----------|------|
| LLaMA3-8B | 100% | 0% | **100%** |
| GLM4-9B | 88% | 0% | **100%** |

- **Random Sampling**: Mean scores align with ground-truth rankings on average, but variance is high. In 12% of trials, GLM4-9B's score exceeds the reference score of Qwen3-4B, causing incorrect localization.
- **Bayesian Method**: Systematically overestimates the capability of both models. It places LLaMA3-8B between Qwen3-1.7B and Qwen3-4B with 99.9% probability (ground truth: Qwen3-0.6B to Qwen3-1.7B), and GLM4-9B between Qwen3-8B and Qwen3-14B with 75.1% probability (ground truth: Qwen3-1.7B to Qwen3-4B), yielding 0% accuracy across 100 trials.
- **STEM**: Achieves precise localization via the sharp accuracy drop along the TI dimension, with correct localization in all 100 trials.

### Table 3: Anomalous Sample Rate per Benchmark

| Benchmark | GPQA | SuperGPQA | MMLU-Pro | MATH | MMLU | GSM8K |
|-----------|------|-----------|----------|------|------|-------|
| Anomaly Rate | 65.85% | 53.20% | 47.93% | 41.94% | 37.72% | 13.16% |

GPQA has the highest anomaly rate—nearly two-thirds of its samples fail the monotonic transition condition—suggesting severe data contamination. GSM8K is the "cleanest" benchmark, with only 13.16% anomalous samples.

## Key Findings

1. **Existing benchmarks exhibit severe structural bias**: Polarized difficulty distributions, with an excess of easy and hard samples, reduce sensitivity to differences in model capability.
2. **Data contamination is widespread**: In GPQA, 65.85% of samples display non-monotonic IRVs, indicating that smaller models answer correctly through memorization rather than reasoning.
3. **STS transfer across architectures**: The STS pool constructed from Qwen3 accurately localizes LLaMA3-8B and GLM4-9B, which belong to different architectures.
4. **STEM achieves 100% localization accuracy with only 100 samples**, whereas random sampling incurs a 12% misclassification risk and the Bayesian method fails entirely.
5. **STEM can distinguish models with very similar capabilities**: LLaMA3-8B (53.90) and GLM4-9B (56.88) differ by only 3 points in reference score, yet STEM reliably places them in distinct capability intervals.

## Highlights & Insights

- **Novel Conceptualization**: The paper analyzes capability transition patterns at the sample level, introduces the STS and TI concepts, and reframes the evaluation problem as a structured difficulty-tier localization task.
- **High Efficiency**: Only 100 carefully selected samples are required. The STS pool is constructed offline once and reused indefinitely, substantially reducing evaluation cost.
- **Data Contamination Detection as a Byproduct**: IRV analysis naturally provides sample-level contamination detection capability, revealing structural deficiencies in existing benchmarks.
- **Simple and Interpretable**: The method does not rely on complex statistical models; transition indices directly correspond to model capability thresholds, making evaluation results intuitive and easy to interpret.

## Limitations & Future Work

1. **Dependency on a scale-controlled reference model family**: The framework requires model series of the same architecture with multiple parameter scales (e.g., Qwen3). Such model families are currently scarce, limiting broad applicability.
2. **Static STS pool**: As new models are released, the STS pool requires periodic recalibration, incurring non-negligible computational cost.
3. **Validated only on multiple-choice and binary-answer benchmarks**: The framework has not been extended to generative tasks (e.g., summarization, dialogue), limiting its scope of applicability.
4. **Capability definition is benchmark-bound**: Model capability is measured in strong coupling with the selected benchmarks rather than as an independent capability metric.
5. **Limited external model validation**: Only LLaMA3-8B and GLM4-9B were tested; the generalizability of the transferability conclusions warrants more extensive verification.

## Related Work & Insights

- **LLM Evaluation Paradigms**: Full-benchmark evaluation (stable but costly) vs. random sampling (low cost but high variance). STEM is positioned as an efficient and precise alternative between these two extremes.
- **Benchmark Structural Bias**: Existing data contamination detection methods include n-gram, permutation, and half-cut techniques, but these are typically tailored to specific benchmark types. The STS in STEM provides a general sample-level contamination analysis mechanism.
- **LLM Emergent Capabilities**: Prior work focuses on task-level emergence; STEM refines emergence analysis to the sample level.

## Rating

⭐⭐⭐⭐ — The method is novel and the experimental design is sound, achieving precise model localization with only 100 samples. The primary limitations are the dependency on a reference model family and the relatively small scale of empirical validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)
- [\[ICLR 2026\] Unsupervised Evaluation of Multi-Turn Objective-Driven Interactions](../../ICLR2026/llm_nlp/unsupervised_evaluation_of_multi-turn_objective-driven_interactions.md)
- [\[ICLR 2026\] KVComm: Enabling Efficient LLM Communication through Selective KV Sharing](../../ICLR2026/llm_nlp/kvcomm_enabling_efficient_llm_communication_through_selective_kv_sharing.md)
- [\[AAAI 2026\] Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging](do_not_merge_my_model_safeguarding_open-source_llms_against_unauthorized_model_m.md)
- [\[AAAI 2026\] Position on LLM-Assisted Peer Review: Addressing Reviewer Gap through Mentoring and Feedback](position_on_llm-assisted_peer_review_addressing_reviewer_gap_through_mentoring_a.md)

</div>

<!-- RELATED:END -->
