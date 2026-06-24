---
title: >-
  [Paper Note] Enhancing Marker Scoring Accuracy through Ordinal Confidence Modelling in Educational Assessments
description: >-
  [ACL 2025 (Industry Track)][Automated Essay Scoring] This paper proposes a confidence modeling approach based on Kernel-Weighted Ordinal Classification Cross-Entropy (KWOCCE). By leveraging the ordinal structure of CEFR levels and a score binning strategy, this method achieves up to 47% of scores released under 100% CEFR consistency, and 99% released under $\ge 95\%$ consistency, which is a significant improvement over the approximately 92% consistency obtained without confid…
tags:
  - "ACL 2025 (Industry Track)"
  - "Automated Essay Scoring"
  - "confidence modeling"
  - "ordinal classification"
  - "KWOCCE loss"
  - "CEFR"
date: 2026-05-08
content_hash: 885bcd5d40163876
---

# Enhancing Marker Scoring Accuracy through Ordinal Confidence Modelling in Educational Assessments

**Conference**: ACL 2025 (Industry Track)  
**arXiv**: [2505.23315](https://arxiv.org/abs/2505.23315)  
**Code**: None (uses proprietary dataset)  
**Area**: Other  
**Keywords**: Automated Essay Scoring, confidence modeling, ordinal classification, KWOCCE loss, CEFR

## TL;DR

This paper proposes a confidence modeling approach based on Kernel-Weighted Ordinal Classification Cross-Entropy (KWOCCE). By leveraging the ordinal structure of CEFR levels and a score binning strategy, this method achieves up to 47% of scores released under 100% CEFR consistency, and 99% released under $\ge 95\%$ consistency, which is a significant improvement over the approximately 92% consistency obtained without confidence filtering.

## Background & Motivation

Automated Essay Scoring (AES) systems are becoming increasingly prevalent in large-scale examinations, but ensuring scoring reliability is crucial in high-stakes scenarios. The core problem lies in:

**When to trust automated scoring?** Existing AES systems release all predictions uniformly, but the reliability across different score intervals varies significantly.

**The ordinal nature of CEFR levels is ignored**: CEFR levels from A1 to C2 are ordered. Misclassifying B1 as B2 is far less severe than misclassifying it as C2, yet standard classification losses fail to distinguish between indeed different errors.

**Inadequate confidence modeling**: Most methods simply rely on softmax probabilities or prediction intervals, lacking modeling of the ordinal structure.

**Score boundary sensitivity**: Errors near level boundaries have the most significant impact on test outcomes (e.g., transitioning from B1 to B2 might alter job application or study abroad eligibility).

The motivation of this paper is to introduce ordinal classification concepts into AES confidence modeling, developing a Hybrid Marking System (HMS) that achieves a better balance between automated scoring and human review.

## Method

### Overall Architecture

The proposed Hybrid Marking System (HMS) consists of two components:
- **Automated Marker (AM)**: A regression model based on a Transformer encoder that outputs scores and LLM embeddings.
- **Confidence Model**: A downstream classifier that takes the AM embeddings, predicted scores, and CEFR cut-offs as input, and outputs a 0-1 confidence score.

Low-confidence scores are forwarded to human review, while high-confidence scores are released directly.

### Key Designs

1. **Binary Classification Baseline**:

    - The simplest framework: predicting whether the AM score correctly corresponds to the CEFR level.
    - Uses Cross-Entropy (CE) loss, with the final probability serving as the confidence score.
    - Limitation: Lacks fine-grained estimation of uncertainty.

2. **CEFR Level N-ary Classification**:

    - The model outputs a probability distribution over all CEFR levels.
    - Uses Categorical Cross-Entropy (CCE) loss.
    - Confidence = The probability value corresponding to the CEFR level predicted by the AM.
    - Advantage: Can capture scenarios with competing CEFR probabilities.

3. **Score-level Binned N-ary Classification (Core Innovation)**:

    - Discretizes continuous scores into independent bins, and then sums the score probabilities within the same level based on the CEFR cut-offs.
    - Models the reliability variations of the AM across different score intervals in a more fine-grained manner.
    - This step leads to a dramatic performance improvement (F1 from 0.733 → 0.954).

4. **KWOCCE Loss Function Family (Core Contribution)**:

    - Introduces a kernel-based penalty based on ordinal distance on top of standard CCE.
    - Four types of kernel functions:
        - **Linear Kernel**: $K_{linear}(x,N) = \max(0, 1 - |x|/N)$
        - **Logarithmic Kernel**: $K_{log}(x,N;\alpha) = \max(0, 1 - \alpha\log(1+|x|)/\log(N))$
        - **Exponential Kernel**: Based on a sigmoid shape, where small errors are barely penalized, while big errors are exponentially penalized.
        - **Gaussian Kernel**: $K_{gaussian}(x;\alpha) = \exp(-(x/\alpha)^2)$, producing a bell-shaped penalty.
    - Core formula: $\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{i=1}^{N} w_i \log \hat{y}_{ic_i}$

### Loss & Training

- Baseline Keras OCC loss: $\text{loss} = (w + 1) \cdot \text{CE}$ where the weight is $w = |argmax \mathbf{y} - argmax \hat{\mathbf{y}}| / (K-1)$
- KWOCCE replaces linear weights with different kernel functions to enable non-linear, distance-aware penalization.
- Mean reduction is used to ensure gradient stability.
- Training set consists of 230,000 responses, verification set 58,000, and evaluation set 644 responses (from 322 test-takers).

## Key Experimental Results

### Main Results: Architecture Comparison (Table)

| Classifier Type | Accuracy | Precision | Recall | F1 |
|-----------|----------|-----------|--------|-----|
| Binary | 0.578 | 0.579 | 0.997 | 0.733 |
| CEFR N-ary | 0.642 | 0.693 | 0.869 | 0.772 |
| Score Binned N-ary | 0.913 | 0.913 | 1.000 | **0.954** |

> The finer the granularity, the better the classification performance. Score Binned N-ary introduces a qualitative leap.

### Ablation Study: Performance of Different Loss Functions on Score Release (Table)

| Loss Function | 100% CEFR Consistency (% Released) | 95% CEFR Consistency (% Released) |
|---------|----------------------|---------------------|
| CCE Baseline | 29.80% | 91.83% |
| Keras OCC | 36.31% | 91.97% |
| KWOCCE Linear | **47.35%** | 98.16% |
| KWOCCE Log (α=3) | 19.86% | **98.89%** |
| KWOCCE Exp (α=1,β=3) | 41.01% | 99.12% |
| KWOCCE Gaussian (α=0.5) | 35.73% | 98.75% |

> KWOCCE Linear achieves the highest release rate at 100% consistency (47.35%), while KWOCCE Exp approaches 99% at 95% consistency.

### Key Findings

1. **Finer granularity yields better results**: Performance improves monotonically from Binary to CEFR N-ary, and then to Score Binning.
2. **Ordinal structure is crucial**: All KWOCCE variants significantly outperform standard CCE at 95% CEFR consistency.
3. **Different kernels fit different scenarios**: Linear is suitable when pursuing a high coverage rate, while Exp is optimal when aiming for high consistency.
4. **Highly significant practical impact**: Scaled up from approximately 92% consistency without confidence filtering to over 99% under conditional release.
5. **Comparison with original performance**: The RMSE of the AM alone is 1.095, while all confidence models exhibit reduced RMSE on the released subset.

## Highlights & Insights

- **Highly valuable for industrial deployment**: Directly addresses the core pain point of AES scoring trustworthiness in high-stakes testing.
- **Elegant KWOCCE design**: Encompasses linear, logarithmic, exponential, and Gaussian penalty schemes under a unified kernel framework, allowing flexible adaptation to various scenarios.
- **"Confidence as releasability"**: Translates confidence modeling into a metric directly linked to test fairness (CEFR consistency rate × release rate), rather than an abstract calibration error.
- **Granularity insight**: Demonstrates that a more fine-grained classification target leads to better confidence estimations, providing valuable insights for uncertainty modeling.

## Limitations & Future Work

1. **Single exam data**: Evaluated on only one specific exam; the generalizability remains unverified.
2. **Small evaluation set**: The 644 responses may lack sufficient statistical power.
3. **Proprietary data**: Dataset and CEFR thresholds are proprietary and cannot be shared publicly, making reproduction difficult.
4. **Lack of human evaluation**: Confidence-based decisions were not compared directly against human raters.
5. **Kernel hyperparameter tuning**: The hyperparameters α and β for different kernels require careful tuning, and the optimal setup may vary across exams.
6. **Only evaluated on Transformer encoder AM**: The effectiveness on LLM-based AMs has not been verified.

## Related Work & Insights

- The logarithmic ordinal loss from **Castagnos et al. (2022)** is a key inspiration for KWOCCE, but it is limited to a single logarithmic form.
- The class distance-weighted cross-entropy by **Polat et al. (2025)** used for medical severity classification shares a similar concept but does not generalize through kernel functions.
- Key takeaway: Ordinal-aware loss designs are widely applicable to any assessment or scoring task with level-based structures, such as sentiment intensity, disease severity, product quality grading, etc.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 3.5 | KWOCCE is a natural extension of existing methods, with an innovative kernel framework. |
| Experimental Thoroughness | 3.5 | Thorough ablation but limited to a single dataset with a relatively small evaluation set. |
| Writing Quality | 4 | Industry Track style; clear and highly practical. |
| Value | 4 | High industrial deployment value; directly addresses real-world deployment challenges. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DREsS: Dataset for Rubric-based Essay Scoring on EFL Writing](dress_dataset_rubric_based_essay_scoring_efl_writing.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)
- [\[ECCV 2024\] Enhancing Optimization Robustness in 1-bit Neural Networks through Stochastic Sign Descent](../../ECCV2024/others/enhancing_optimization_robustness_in_1-bit_neural_networks_through_stochastic_si.md)
- [\[CVPR 2025\] Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](../../CVPR2025/others/improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)
- [\[ACL 2025\] Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages](using_source-side_confidence_estimation_for_reliable_translation_into_unfamiliar.md)

</div>

<!-- RELATED:END -->
