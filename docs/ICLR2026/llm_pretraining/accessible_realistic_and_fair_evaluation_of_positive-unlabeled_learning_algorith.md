---
title: >-
  [Paper Note] Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms
description: >-
  [ICLR 2026][LLM Pretraining][Positive-Unlabeled Learning] This paper proposes the first unified benchmark for PU learning, systematically addressing two key issues: (1) implementing model selection without negative samples using Proxy Accuracy and Proxy AUC; (2) identifying and resolving the Internal Label Shift problem in the one-sample setting through a simple calibration method that merges positive samples into the unlabeled set, enabling fair comparison of two-sample algo…
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "Positive-Unlabeled Learning"
  - "benchmark"
  - "model selection"
  - "label shift"
  - "fair evaluation"
date: 2026-05-08
content_hash: b4d39c2d0bc8143f
---

# Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms

**Conference**: ICLR 2026  
**arXiv**: [2509.24228](https://arxiv.org/abs/2509.24228)  
**Code**: Yes (benchmark)  
**Area**: Weakly Supervised Learning  
**Keywords**: Positive-Unlabeled Learning, benchmark, model selection, label shift, fair evaluation

## TL;DR
This paper proposes the first unified benchmark for PU learning, systematically addressing two key issues: (1) implementing model selection without negative samples using Proxy Accuracy and Proxy AUC; (2) identifying and resolving the Internal Label Shift problem in the one-sample setting through a simple calibration method that merges positive samples into the unlabeled set, enabling fair comparison of two-sample algorithms over one-sample evaluations.

## Background & Motivation

**Background**: Numerous algorithms for PU learning (binary classification with only positive and unlabeled samples) have emerged recently, but experimental setups are highly inconsistent, making it difficult to determine which algorithm performs better.

**Limitations of Prior Work**: (a) Many algorithms rely on validation sets containing negative samples for model selection, which contradicts the "no negative samples" premise of PU learning; (b) PU learning involves two settings: one-sample (OS) and two-sample (TS). Existing evaluations favor the OS setting but ignore the critical difference—under the OS setting, the class prior and marginal distribution of unlabeled data differ from the population.

**Key Challenge**: TS algorithms assume that unlabeled data comes from the marginal distribution $p(x)$, but in the OS setting, the density of unlabeled data is $\bar{p}(x) = \bar{\pi}p(x|y=+1) + (1-\bar{\pi})p(x|y=-1)$, where $\bar{\pi} \neq \pi$. This leads to the destruction of risk consistency in TS algorithms.

**Goal**: To provide an accessible, realistic, and fair evaluation framework for PU learning.

**Key Insight**: (a) Derive proxy metrics from information theory that can be calculated using only positive/unlabeled data; (b) Identify the Internal Label Shift problem and provide theoretical guarantees for its resolution.

**Core Idea**: Restore the unbiasedness of the marginal distribution by merging positive samples into the unlabeled set. A single-line code modification can eliminate fairness bias between OS and TS settings.

## Method

### Overall Architecture
Instead of proposing a new algorithm, this paper constructs an evaluation system capable of fairly judging existing PU algorithms. It provides solutions for three long-neglected issues: First, model selection "secretly" uses negative samples; second, an "Internal Label Shift" exists in the one-sample (OS) setting that distorts cross-setting comparisons; and third, inconsistent implementation details contaminate comparisons. The framework unifies data generation, training, and evaluation protocols.

> As this is an evaluation/benchmark paper (consisting of two independent methodological contributions plus a unified protocol without a serial data flow pipeline), a framework diagram is omitted; consistency is maintained by the sequential alignment of the "Key Designs" below with the overall framework.

### Key Designs

**1. Proxy Accuracy (PA) and Proxy AUC (PAUC): Model Selection Without Negative Samples**

The premise of PU learning is the absence of clean negative samples, yet existing works commonly rely on validation sets containing negative samples (i.e., oracle accuracy) for early stopping and hyperparameter selection. This paper derives two proxy metrics based on information theory that use only positive/unlabeled validation data. PA is built on the equality $\mathbb{E}[\text{PA}(f)] = \text{ACC}(f) - 1 + 2\pi$, linking the unobservable true accuracy to a computable proxy, though it requires knowing the class prior $\pi$. PAUC treats unlabeled data as "noisy negative samples" to calculate AUC, requiring no knowledge of $\pi$. Proposition 1-2 prove these proxy metrics are rank-preserving—meaning the model ordering they produce is consistent with that of the ground truth.

**2. Identification and Calibration of Internal Label Shift (ILS): Restoring TS Consistency in OS Settings with One Line of Code**

The risk consistency of TS algorithms relies on the assumption that "unlabeled data comes from the marginal distribution $p(x)$." However, in OS settings, positive samples are partitioned out before the unlabeled set is designated. The proportion of positive instances in the remaining unlabeled data drops from $\pi$ to $\bar{\pi} = (1-c)\pi/(1-c\pi)$, and the density becomes $\bar{p}(x)=\bar{\pi}p(x|y=+1)+(1-\bar{\pi})p(x|y=-1)$. This paper identifies this shift as Internal Label Shift. The calibration method is straightforward: when calculating the unlabeled loss, the positive samples are merged back into the unlabeled set, using $\mathcal{D}_k^U \cup \mathcal{D}_k^P$ instead of $\mathcal{D}_k^U$. This recovers the unbiasedness of the marginal distribution. Theorem 1 proves the resulting risk estimator is unbiased, and Theorem 2 provides convergence guarantees.

**3. Unified Benchmark Framework: Eliminating Noise from Implementation Details**

To ensure comparisons are not contaminated by differences in data augmentation, warm-up, or training protocols, this framework standardizes these elements. It covers three major algorithm families (17 algorithms total): cost-sensitive, sample selection, and biased PU learning. It evaluates them on CIFAR-10, ImageNette, USPS, and Letter datasets to ensure performance gaps reflect algorithmic differences rather than engineering details.

### Loss & Training
Logistic loss is used uniformly. The calibration only modifies how the unlabeled loss is calculated (merging positive samples into the unlabeled set) without altering the core of the algorithms, allowing it to be applied to any TS algorithm.

## Key Experimental Results

### Main Results

**CIFAR-10 PU Version (Different Positive Sample Sizes):**

| TS Algorithm | OS Setting (Uncalibrated) | OS Setting (Calibrated) | TS Setting |
|--------------|---------------------------|-------------------------|------------|
| uPU          | Significant Drop          | ≈ TS Performance        | Baseline   |
| nnPU         | Drop                      | Restored                | Baseline   |
| Dist-PU      | Drop                      | Restored                | Baseline   |
| VPU          | Drop                      | Restored                | Baseline   |

### Ablation Study

**Comparison of Model Selection Criteria:**

| Selection Criterion | Needs Negative Samples | ACC Correlation | AUC Correlation |
|---------------------|------------------------|-----------------|-----------------|
| Oracle Accuracy (OA)| ✓                      | Best            | Best            |
| Proxy Accuracy (PA) | ✗                      | Near OA         | Moderate        |
| Proxy AUC (PAUC)    | ✗                      | Moderate        | Near OA         |

### Key Findings
- No single algorithm dominates across all datasets and metrics; some early simple methods remain very strong.
- TS algorithms suffer significant performance degradation in OS settings without calibration, which is restored after calibration—validating the real-world impact of ILS.
- PA is suitable for accuracy-related model selection, while PAUC is better for AUC-related selection.
- Calibration is consistently effective across various settings and algorithms with near-zero computational overhead.

## Highlights & Insights
- **Impact of One Line of Code**: The calibration method essentially includes positive samples in the unlabeled loss calculation. This simple change is theoretically grounded and highly effective, showing that "understanding the problem is more important than designing the method."
- **First Identification of ILS**: Previous literature evaluated TS algorithms in OS settings without calibration, leading to unfair comparisons for years. This finding may change evaluation standards in the field.
- **Practical Value of Proxy Metrics**: The paper proves that effective model selection can be done without negative samples, making PU learning evaluation truly self-consistent.

## Limitations & Future Work
- Proxy Accuracy requires knowing or estimating the class prior $\pi$; inaccurate estimates may affect model selection.
- The benchmark currently covers image classification; expansion to text/tabular modalities is needed.
- The calibration method assumes the labeling probability $c$ is constant (SCAR assumption); non-uniform labeling scenarios are not yet covered.
- In deep learning, hyperparameter searches might make the theoretical guarantees of proxy metrics less tight under limited sample sizes.

## Related Work & Insights
- **vs. Existing PU Learning Papers**: Most papers use validation sets with negative samples for early stopping, which is identified as unrealistic.
- **vs. Weakly Supervised Benchmarks**: Similar to the contribution of CleanLab to noisy label learning, this work provides a much-needed standardized benchmark for PU learning.
- **vs. du Plessis et al. 2015**: The risk consistency of uPU is broken in the OS setting, and the calibration method restores it—acting as a critical supplement to classic methods.

## Rating
- Novelty: ⭐⭐⭐⭐ The discovery of ILS and the simplicity of the calibration method are elegant; proxy metrics have solid theoretical guarantees.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 algorithm families × multiple datasets × OS/TS settings × calibration comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition and rigorous theoretical derivation.
- Value: ⭐⭐⭐⭐⭐ As the first PU learning benchmark, it is likely to reshape evaluation practices in the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)
- [\[AAAI 2026\] Rectified Noise: A Generative Model Using Positive-incentive Noise](../../AAAI2026/llm_pretraining/rectified_noise_a_generative_model_using_positive-incentive_noise.md)
- [\[ICLR 2026\] DUET: Optimizing LLM Training Data Mixtures via Noisy Feedback from Unseen, Downstream Evaluation Tasks](duet_optimizing_llm_training_data_mixtures_via_noisy_feedback_from_unseen_downst.md)
- [\[ICLR 2026\] Learning Facts at Scale with Active Reading](learning_facts_at_scale_with_active_reading.md)
- [\[ACL 2025\] Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization](../../ACL2025/llm_pretraining/model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza.md)

</div>

<!-- RELATED:END -->
