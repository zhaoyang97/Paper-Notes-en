---
title: >-
  [Paper Note] ICYM2I: The Illusion of Multimodal Informativeness under Missingness
description: >-
  [ICLR 2026][Multimodal VLM][multimodal missingness] This paper identifies a largely overlooked problem in multimodal learning: distribution shift induced by modality missingness leads to severely biased modality value estimation. The proposed ICYM2I framework applies dual inverse probability weighting (IPW) to correct bias in both training and evaluation, achieving unbiased estimates of modality predictive utility and information-theoretic value under the MAR assumption.
tags:
  - ICLR 2026
  - Multimodal VLM
  - multimodal missingness
  - distribution shift
  - inverse probability weighting
  - information decomposition
  - modality value estimation
date: 2026-05-08
content_hash: 14877910a78f0e84
---

# ICYM2I: The Illusion of Multimodal Informativeness under Missingness

**Conference**: ICLR 2026
**arXiv**: [2505.16953](https://arxiv.org/abs/2505.16953)
**Code**: [https://github.com/reAIM-Lab/ICYM2I](https://github.com/reAIM-Lab/ICYM2I)
**Area**: Multimodal VLM / Machine Learning Theory
**Keywords**: multimodal missingness, distribution shift, inverse probability weighting, information decomposition, modality value estimation

## TL;DR

This paper identifies a largely overlooked problem in multimodal learning: distribution shift induced by modality missingness leads to severely biased modality value estimation. The proposed ICYM2I framework applies dual inverse probability weighting (IPW) to correct bias in both training and evaluation, achieving unbiased estimates of modality predictive utility and information-theoretic value under the MAR assumption.

## Background & Motivation

**State of the Field**: Multimodal learning is widely applied in healthcare, autonomous driving, recommendation systems, and other domains, with a core assumption that "multimodal > unimodal." In practice, modality value is commonly assessed via ablation studies—measuring performance degradation when a modality is removed.

**Limitations of Prior Work**: Real-world data collection is subject to substantial missingness due to sensor failures, cost constraints, privacy restrictions, and other factors, rendering certain modalities unavailable for certain samples. The prevailing approach is to discard incomplete samples and train and evaluate on the complete-case subset. However, if missingness is not completely at random (i.e., non-MCAR), the resulting subset distribution diverges from the true distribution, yielding biased modality value estimates.

**Root Cause**: The missingness mechanism is confounded with modality signals. When the missingness of a modality is associated with the label (MAR), the apparent "performance" of that modality on the complete-case subset is systematically over- or underestimated—a form of bias that has been almost entirely overlooked in the existing literature.

**Paper Goals**: Given non-random missingness, how can one unbiasedly estimate (a) the predictive utility of a modality (i.e., how much performance improves upon adding it) and (b) its information-theoretic value (i.e., the unique, shared, and complementary information it carries)?

**Starting Point**: The paper draws on inverse probability weighting (IPW) from causal inference, treating missingness-induced distribution shift as a correctable form of selection bias.

**Core Idea**: IPW is applied simultaneously to the training loss and evaluation metrics, enabling unbiased modality value estimation even when the observed data contains missingness.

## Method

### Overall Architecture

The ICYM2I framework consists of two complementary components:
- **ICYM2I-learn**: IPW-weighted training and evaluation of multimodal/unimodal models to correct missingness bias.
- **ICYM2I-PID**: IPW-corrected Partial Information Decomposition (PID) estimation, decomposing modality information into unique, shared, and complementary components.

**Assumptions**: Missing At Random (MAR) + Positivity (the probability of complete observation is strictly positive for any covariate configuration).

### Key Designs

1. **IPW-Weighted Training (ICYM2I-learn)**:

    - **Function**: Trains on the observed data $\Omega_{obs}$ while learning a mapping representative of the true distribution $\Omega$.
    - **Mechanism**: For each complete sample $(x_1, x_2, y)$, the loss is reweighted by the inverse probability weight $w = \frac{1}{1 - p(m_1, m_2, m_y | C)}$, where $p(m|C)$ is a missingness probability model (logistic regression) trained on observed covariates $C$.
    - **Design Motivation**: Among observed data, certain complete-case samples are over-represented because they are more likely to be fully observed. IPW downweights these over-represented samples to restore distributional balance.

2. **IPW-Corrected Evaluation**:

    - **Function**: Applies IPW correction when evaluating models on the complete-case subset.
    - **Mechanism**: Standard evaluation metrics (e.g., AUC) computed on $\Omega_{obs}$ are biased; IPW adjustment yields unbiased estimates with respect to the true distribution $\Omega$.
    - **Design Motivation**: Correcting training alone is insufficient—evaluating on a biased test set still precludes accurate assessment of modality value.

3. **ICYM2I-PID (Information-Theoretic Decomposition Correction)**:

    - **Function**: Provides unbiased estimation of the Partial Information Decomposition (PID) among modalities, including Unique, Shared, and Complementary information.
    - **Mechanism**: Integrates the PID framework of Bertschinger et al. with IPW. The key innovation lies in correcting the estimation of the trivariate mutual information $I(Y:(X_1,X_2))$: IPW-reweighted samples are used to reconstruct mutual information under the true distribution, with a modified Sinkhorn–Knopp procedure enforcing marginal distribution constraints.
    - **Design Motivation**: Predictive performance alone cannot distinguish whether a modality's information is unique or redundant. PID provides a finer-grained analysis of information structure, but standard PID estimates are equally susceptible to missingness-induced bias.

### Loss & Training

- Weighted cross-entropy loss: $l_{\Omega}(x_1,x_2,y) = \frac{1}{1-p(m_1,m_2,m_y|C)} \cdot l_{\Omega_{obs}}(x_1,x_2,y)$
- Missingness probability model: logistic regression trained on observed covariates $C$.
- PID optimization: parameterized neural network + Sinkhorn–Knopp iterations.

## Key Experimental Results

### Main Results

Bit-logical operator experiments (AND/OR/XOR, 50% MAR missingness):

| Operator | Method | X1 AUC | X2 AUC | Unique1 | Unique2 | Shared | Compl. |
|----------|--------|--------|--------|---------|---------|--------|--------|
| AND | Oracle | 0.83 | 0.84 | 0.05 | 0.03 | 0.26 | 0.47 |
| AND | Observed | 0.66 | 0.93 | **0.44** | 0.00 | 0.15 | 0.36 |
| AND | **ICYM2I** | **0.83** | **0.85** | **0.03** | **0.03** | **0.27** | **0.45** |
| XOR | Oracle | 0.51 | 0.49 | 0.00 | 0.00 | 0.00 | 0.99 |
| XOR | Observed | 0.52 | **0.80** | **0.34** | 0.07 | -0.07 | 0.62 |
| XOR | **ICYM2I** | **0.53** | **0.49** | **0.00** | **0.00** | **0.01** | **0.96** |

The Observed method severely overestimates X2's AUC (0.80 vs. ground truth 0.49) and Unique1 (0.34 vs. ground truth 0.00) in the XOR case; ICYM2I corrects both estimates precisely.

### Ablation Study

Analysis of training–evaluation correction combinations (AUC RMSE vs. Oracle):

| Training | Evaluation | AUC RMSE ↓ |
|----------|------------|------------|
| Standard | Standard | High (biased) |
| IPW | Standard | Medium (training corrected, evaluation still biased) |
| Standard | IPW | Medium |
| **IPW** | **IPW** | **Lowest (dual correction)** |

### Key Findings

- The direction of missingness bias depends on the missingness mechanism: for the OR operator, X1 is overestimated (because when X1=1, X2 is more likely to be missing, inflating X1's apparent predictive power in the complete-case subset); the opposite holds for the AND operator.
- **XOR is the most extreme case**: both modalities carry zero unique information (all information is complementary), yet without missingness correction, Unique1 is estimated at 0.34—a result that would severely mislead practitioners into concluding that X1 has standalone value.
- Both training and evaluation correction are necessary; neither alone suffices.
- The effectiveness of ICYM2I is further validated on real-world medical data (breast cancer screening).

## Highlights & Insights

- **A distinctive and far-reaching perspective**: All prior multimodal work implicitly assumes that the complete-case subset is representative of the full population. ICYM2I is the first to formally characterize the fragility of this assumption. This is not a marginal issue—in high-stakes domains such as healthcare and autonomous driving, missingness is frequently correlated with critical factors, and the resulting bias can have serious consequences.
- **Transfer of causal inference tools to multimodal learning**: IPW is a classical tool in causal inference; this paper applies it elegantly to address the missingness problem in multimodal learning, representing a productive cross-domain methodological transfer.
- **Distinguishing two fundamentally different modality missingness problems**: (1) *Target-environment missingness* (the traditional problem: how to be robust to sensor failures at deployment) and (2) *Source-environment missingness* (the focus of this paper: how missingness in training data biases modality value estimation).

## Limitations & Future Work

- **The MAR assumption may not hold**: If missingness depends on unobserved variables (MNAR), IPW cannot correct the bias. The paper discusses MNAR robustness in the appendix but acknowledges this limitation.
- **Accuracy of the missingness probability model is critical**: IPW weights are derived from logistic regression estimates of missingness probability; inaccuracies in this model propagate into the correction.
- **Extreme IPW weight instability**: When the observation probability for certain samples is very low, IPW weights become very large, leading to high variance. The paper does not discuss stabilization strategies such as weight truncation.
- **Limited experimental scale**: Validation is primarily conducted on synthetic/semi-synthetic data and small-scale medical datasets; applicability to large-scale multimodal benchmarks remains to be demonstrated.

## Related Work & Insights

- **vs. standard multimodal robustness methods** (e.g., imputation, knowledge distillation): These methods address the question of "how to maintain performance when modalities are missing at deployment," whereas ICYM2I targets the more fundamental question of "how source-data missingness biases our judgments about modality value."
- **vs. PID decomposition methods** (Liang et al. 2024a): PID decomposition implicitly assumes that the observed distribution equals the true distribution. ICYM2I demonstrates that this leads to severe bias under missingness and provides a principled correction.
- **Practical implications for system design**: When deciding whether to collect an expensive modality (e.g., biopsy in clinical settings), one cannot naively rely on ablation study conclusions drawn from retrospective data—missingness bias must first be corrected.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to formally characterize the multimodal missingness bias problem; distinctive perspective with broad implications.
- Experimental Thoroughness: ⭐⭐⭐ Experiments are relatively small-scale; validation on large multimodal benchmarks is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous formalization; motivating examples (bit-logical operators) are intuitive and compelling.
- Value: ⭐⭐⭐⭐ Offers important guidance for multimodal evaluation practice, though broader empirical support is needed.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates](../../CVPR2026/multimodal_vlm/balm_a_model-agnostic_framework_for_balanced_multimodal_learning_under_imbalance.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[CVPR 2026\] Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher](../../CVPR2026/multimodal_vlm/purify-then-align_towards_robust_human_sensing_under_modality_missing_with_knowl.md)
- [\[CVPR 2026\] FINER: MLLMs Hallucinate under Fine-grained Negative Queries](../../CVPR2026/multimodal_vlm/finer_mllms_hallucinate_under_fine-grained_negative_queries.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)

<!-- RELATED:END -->
