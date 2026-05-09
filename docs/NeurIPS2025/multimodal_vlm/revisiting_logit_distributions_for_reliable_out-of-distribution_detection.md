---
title: >-
  [Paper Note] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection
description: >-
  [NeurIPS 2025][Multimodal VLM][OOD detection] This paper proposes LogitGap, a novel post-hoc OOD detection scoring function that explicitly exploits the "gap" between the maximum logit and the remaining logits to distinguish in-distribution (ID) from out-of-distribution (OOD) samples. A top-N selection strategy is introduced to filter noisy logits. Theoretical analysis and experiments demonstrate that LogitGap outperforms MCM and MaxLogit across multiple scenarios.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - OOD detection
  - logit distribution
  - CLIP
  - post-hoc method
  - scoring function
date: 2026-05-08
content_hash: 61ab283b3333b698
---

# Revisiting Logit Distributions for Reliable Out-of-Distribution Detection

**Conference**: NeurIPS 2025
**arXiv**: [2510.20134](https://arxiv.org/abs/2510.20134)
**Code**: [GitHub](https://github.com/GIT-LJc/LogitGap)
**Area**: Multimodal VLM / OOD Detection
**Keywords**: OOD detection, logit distribution, CLIP, post-hoc method, scoring function

## TL;DR

This paper proposes LogitGap, a novel post-hoc OOD detection scoring function that explicitly exploits the "gap" between the maximum logit and the remaining logits to distinguish in-distribution (ID) from out-of-distribution (OOD) samples. A top-N selection strategy is introduced to filter noisy logits. Theoretical analysis and experiments demonstrate that LogitGap outperforms MCM and MaxLogit across multiple scenarios.

## Background & Motivation

OOD detection is a critical safety requirement for deploying deep learning models in open-world settings. Post-hoc methods have attracted broad interest due to their flexibility — they require no modification of model parameters. The central challenge is designing effective scoring functions that maximize the separability between ID and OOD samples.

**Limitations of two representative scoring functions**:

**MaxLogit**: $S(x) = \max_k z_k$, uses only the maximum logit value and completely ignores information from the remaining logits.

**MCM (Maximum Concept Matching)**: $S(x) = \max_k \frac{e^{z_k/\tau}}{\sum_j e^{z_j/\tau}}$, implicitly incorporates all logits via softmax, but softmax compresses absolute logit information, and different logit patterns may map to similar probability distributions.

**Key observation**: ID samples exhibit a peaked logit distribution (one dominant logit far exceeding the rest), while OOD samples exhibit a flatter distribution (the maximum logit is less prominent, non-maximum logits are relatively higher). This results in a noticeably larger "logit gap" for ID samples — a natural discriminative cue.

## Method

### Overall Architecture

LogitGap is a purely post-hoc method that takes the logit vector output of a pretrained model (e.g., CLIP) as input, requiring no training or model modification. The core pipeline is: compute logits → sort in descending order → compute the mean gap among top-N logits → use as the OOD score.

### Key Designs

1. **LogitGap Scoring Function**

    - The logit vector $\boldsymbol{z}$ is sorted in descending order as $\boldsymbol{z}'$; the average difference between the maximum logit and all remaining logits is computed:
    $S_{\text{LogitGap}}(x;f) = \frac{1}{K-1}\sum_{j=2}^{K}(z'_1 - z'_j)$
    - Equivalent form: $S = z'_1 - \bar{z}'_K$, i.e., the maximum logit minus the mean of the remaining logits.
    - ID samples receive higher scores (peaked distribution, large gap); OOD samples receive lower scores (flat distribution, small gap).
    - Key distinction from MCM: MCM implicitly incorporates non-maximum logits via the softmax denominator (losing absolute value information), whereas LogitGap explicitly quantifies the gap (preserving complete information).

2. **LogitGap-topN Refinement**

    - **Problem**: In $K$-way classification, a large number of tail classes are semantically irrelevant to the input; their logits contribute minimally to ID/OOD discrimination and instead introduce noise.
    - **Solution**: Compute the gap using only the top-N logits: $S_{\text{topN}} = \frac{1}{N-1}\sum_{j=2}^{N}(z'_1 - z'_j)$
    - **N selection**: The optimal $N$ is determined by maximizing the mean score difference between ID and OOD samples, which reduces to $\arg\max_{N}(\mathbb{E}_{OOD}[\bar{z}'_N] - \mathbb{E}_{ID}[\bar{z}'_N])$.
    - **Training-free strategy**: Only a small ID validation set (≤100 samples) is required; OOD data is approximated via interpolation transforms and noise injection.

3. **Theoretical Guarantee (Theorem 4.1)**

    - It is proved that when the temperature parameter satisfies $\tau > 2(K-1)$, the false positive rate (FPR) of LogitGap is strictly no greater than that of MCM.
    - Key insights:
        - MCM suffers severe information loss at high temperatures (probability mass is overly dispersed).
        - LogitGap operates on raw logit gaps and is insensitive to the temperature parameter.

### Composability with Other Methods

LogitGap can serve as a plug-in replacement for the scoring function in existing methods, and can be combined with few-shot approaches such as CoOp and ID-Like for additional gains.

## Key Experimental Results

### Main Results (CLIP ViT-B/16, ImageNet as ID, Zero-shot)

| Method | NINCO FPR95↓ | ImageNet-O FPR95↓ | ImageNetOOD FPR95↓ | Avg. FPR95↓ | Avg. AUROC↑ |
|------|-------------|-------------------|-------------------|------------|-----------|
| MCM | 79.67 | 75.85 | 80.98 | 78.83 | 77.15 |
| MaxLogit | 79.41 | 77.15 | 75.85 | 77.47 | 76.96 |
| GL-MCM | 74.38 | 72.35 | 79.16 | 75.30 | 74.74 |
| **LogitGap** | 76.83 | 72.35 | 76.37 | **75.18** | **79.23** |
| **LogitGap*** | 77.42 | **71.95** | **75.40** | **74.92** | **79.41** |

### Ablation Study (Few-shot Setting, Combined with Other Methods)

| Method | Avg. FPR95↓ | Avg. AUROC↑ |
|------|-----------|-----------|
| CoOp (1-shot) | 80.60 | 74.78 |
| CoOp + LogitGap* (1-shot) | **78.67** | **77.02** |
| ID-Like (1-shot) | 79.07 | 71.40 |
| ID-Like + LogitGap* (1-shot) | **71.68** | **78.48** |
| CoOp (4-shot) | 79.09 | 76.17 |
| CoOp + LogitGap* (4-shot) | **76.49** | **78.41** |

### Key Findings

- **LogitGap achieves state-of-the-art performance in both zero-shot and few-shot settings**: average FPR95 is reduced by 3.65% (ImageNet) and 5.78% (ImageNet-100) compared to MCM.
- **Strong complementarity with few-shot methods**: ID-Like + LogitGap* reduces FPR95 from 79.07 to 71.68 and improves AUROC from 71.40 to 78.48.
- **Applicable to conventionally trained models**: not limited to CLIP; effective on ResNet as well.
- **top-N selection consistently improves performance**: LogitGap* generally outperforms LogitGap with a fixed $N = 20\%K$.

## Highlights & Insights

- **Extreme simplicity**: the method requires no training, no additional data, and no model modification — the scoring function reduces to a single formula.
- **Solid theoretical foundation**: Theorem 4.1 establishes a rigorous FPR upper bound relationship between LogitGap and MCM, going beyond purely empirical justification.
- **Strong composability**: can serve as a drop-in replacement for any logit-based OOD scoring function.
- **Deep insight**: the difference in logit distribution shape between ID and OOD samples (peaked vs. flat) is an underexploited discriminative cue that LogitGap makes explicit.
- **Principled hyperparameter selection**: the choice of $N$ is formulated as a mean-gap maximization problem, avoiding blind search.

## Limitations & Future Work

- **Near-semantic OOD detection remains challenging**: when OOD data is semantically close to ID data (e.g., ImageNet-10 vs. ImageNet-20), the improvement from LogitGap diminishes.
- **Adaptive $N$ selection relies on OOD simulation**: the assumption of approximating OOD data via interpolation and noise injection may not generalize to all scenarios.
- **Restricted to logit-based models**: a complete logit vector output is required, rendering the method inapplicable to certain black-box APIs.
- **Combination with feature-based methods unexplored**: approaches such as Mahalanobis distance and KNN may be complementary to LogitGap.
- **Potential for nonlinear logit transformations**: the current approach relies on linear gaps; whether superior nonlinear exploitations exist remains an open question.

## Related Work & Insights

- **vs. MCM**: MCM implicitly leverages logit information via softmax but loses absolute values; LogitGap explicitly utilizes the gap, preserving complete information.
- **vs. MaxLogit**: MaxLogit considers only the maximum value; LogitGap exploits the shape of the entire logit distribution.
- **vs. GL-MCM**: GL-MCM introduces local feature cues; LogitGap operates purely at the logit level and is more lightweight.
- **vs. Energy**: the energy score is another global logit utilization strategy; LogitGap exploits logit gaps rather than log-sum-exp, yielding better performance.

## Rating

- Novelty: ⭐⭐⭐⭐ — Explicit exploitation of logit gaps offers a fresh perspective, though the core formula is remarkably simple (which is also a strength).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers zero-shot, few-shot, and conventionally trained settings with diverse ID/OOD combinations and comprehensive ablation experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical analysis is clear and rigorous; figures are intuitive (the score distribution comparison in Fig. 1 is particularly convincing).
- Value: ⭐⭐⭐⭐ — High practical value (zero-cost replacement), though the theoretical contribution is relatively incremental.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-distribution Detection](../../ICCV2025/multimodal_vlm/adaptive_prompt_learning_via_gaussian_outlier_synthesis_for_out_of_distribution_detection.md)
- [\[ICCV 2025\] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection](../../ICCV2025/multimodal_vlm/fa_forced_prompt_learning_of_vision-language_models_for_out-of-distribution_dete.md)
- [\[NeurIPS 2025\] AQuaMaM: An Autoregressive, Quaternion Manifold Model for Rapidly Estimating Complex SO(3) Distributions](aquamam_an_autoregressive_quaternion_manifold_model_for_rapidly_estimating_compl.md)
- [\[NeurIPS 2025\] Don't Just Chase "Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention](dont_just_chase_highlighted_tokens_in_mllms_revisiting_visual_holistic_context_r.md)
- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](../../ICCV2025/multimodal_vlm/negrefine_refining_negative_label-based_zero-shot_ood_detection.md)

</div>

<!-- RELATED:END -->
