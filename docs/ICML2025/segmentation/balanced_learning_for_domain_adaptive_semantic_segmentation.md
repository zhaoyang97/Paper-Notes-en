---
title: >-
  [Paper Note] Balanced Learning for Domain Adaptive Semantic Segmentation
description: >-
  [ICML 2025][Segmentation][domain adaptation] This paper proposes BLDA, which directly quantifies class bias by analyzing the logit distributions predicted by the network. It aligns the logit distributions of each class using a shared anchor distribution for post-processing calibration, while utilizing GMMs for online estimation and logit correction in self-training to generate unbiased pseudo-labels. This brings consistent improvements to various baseline methods on both GTA→…
tags:
  - "ICML 2025"
  - "Segmentation"
  - "domain adaptation"
  - "semantic segmentation"
  - "class imbalance"
  - "logit calibration"
  - "self-training"
date: 2026-05-08
content_hash: 79535ac09477492b
---

# Balanced Learning for Domain Adaptive Semantic Segmentation

**Conference**: ICML 2025  
**arXiv**: [2512.06886](https://arxiv.org/abs/2512.06886)  
**Code**: [https://github.com/Woof6/BLDA](https://github.com/Woof6/BLDA)  
**Area**: Semantic Segmentation  
**Keywords**: domain adaptation, semantic segmentation, class imbalance, logit calibration, self-training

## TL;DR

This paper proposes BLDA, which directly quantifies class bias by analyzing the logit distributions predicted by the network. It aligns the logit distributions of each class using a shared anchor distribution for post-processing calibration, while utilizing GMMs for online estimation and logit correction in self-training to generate unbiased pseudo-labels. This brings consistent improvements to various baseline methods on both GTA→CS and SYN→CS benchmarks.

## Background & Motivation

**Background**: Unsupervised Domain Adaptation (UDA) for semantic segmentation aims to transfer knowledge from a labeled source domain to an unlabeled target domain. Self-training techniques, which generate pseudo-labels via a teacher-student framework and optimize on the target domain, have become the mainstream paradigm in UDA segmentation.

**Limitations of Prior Work**: In practice, self-training methods struggle to balance the learning across different classes. Segmentation datasets themselves suffer from severe class imbalance (e.g., road pixels far outnumber rider pixels), and the dual shift of data and label distributions between domains in UDA settings further complicates the issue—some classes are easy to transfer (e.g., road) while others are difficult (e.g., rider), with the degree of bias not aligning strictly with the class frequencies of the training set. Confirmation bias in self-training HTML/text exacerbates this phenomenon, where incorrect pseudo-labels of biased classes are continuously reinforced during training.

**Key Challenge**: Existing reweighting and resampling strategies rely on the assumption that training and test distributions are identical. However, in UDA, the source and target domains differ in both data and label spaces, and the target domain class prior is unavailable. This leaves these heuristic methods without theoretical support and leads to unstable performance under UDA settings.

**Goal**: To evaluate and mitigate class bias directly from the model's own predictions without requiring the target domain distribution prior.

**Key Insight**: The authors observe that the network's predicted logit distributions exhibit systematic differences across classes—over-predicted classes have larger positive logit values, while under-predicted classes have smaller positive logit values. The ranking of this difference highly correlates with the ranking of class bias (correlation > 0.9), indicating that bias can be eliminated by balancing the logit distributions.

**Core Idea**: To achieve class-balanced domain adaptive segmentation by aligning the logit distributions of each class to a shared anchor distribution.

## Method

### Overall Architecture

BLDA is built upon the standard self-training UDA framework. The inputs are labeled source domain images and unlabeled target domain images, and the output is the semantic segmentation prediction on the target domain. The overall pipeline consists of four stages: (1) constructing a logit collection matrix $\mathcal{M}$ to collect positive/negative logit distributions for each class and quantify bias; (2) aligning each class's logits to an anchor distribution using CDF mapping for post-processing calibration to achieve balanced predictions; (3) employing GMMs during training to perform online estimation of logit distributions and embed correction terms into the loss function to generate unbiased pseudo-labels; (4) bridging the source and target domains by utilizing positive CDF values as domain-shared structural knowledge in an auxiliary regression task.

### Key Designs

1. **Based on logit distribution bias evaluation**:

    - **Function**: Directly quantifies the level of over- or under-prediction for each class from the network predictions, without requiring target domain label priors.
    - **Mechanism**: Replacing each element of the confusion matrix with its corresponding set of logit values yields a $C \times C$ logit collection matrix $\mathcal{M}$. The diagonal elements $\mathcal{M}_{ll}$ constitute the "positive logit distribution" of class $l$ (i.e., the logit values predicted as class $l$ when the actual class is indeed $l$), and the off-diagonal elements constitute the "negative logit distribution". The class bias is defined as $\text{Bias}(l) = \frac{1}{C}\sum_{c} \mathbb{P}(\arg\max f_\theta(x)[c'] = l | y=c) - 1/C$, which can be estimated by comparing the positive and negative logit distributions: over-predicted classes exhibit a larger discrepancy between positive and negative distributions (higher logit values), while under-predicted classes exhibit a smaller discrepancy (lower logit values).
    - **Design Motivation**: This bypasses the reliance on the target domain label distribution and diagnoses the issue directly from the model's own predictive behavior. The paper proves that a sufficient condition for an unbiased network is that the positive and negative distributions of each class are identical.

2. **Anchor Distribution Alignment (Post-processing Calibration)**:

    - **Function**: Rebalancing the logit predictions of different classes during inference.
    - **Mechanism**: Establishing shared positive anchor distribution $P_p$ and negative anchor distribution $P_n$ (taken from the global positive/negative logit distributions of the source domain), and correcting the logit values of each class point-by-point via inverse CDF mapping: $z' = F_p^{-1}(F_{cl}(z))$ (if $c=l$, i.e., positive logit) or $z' = F_n^{-1}(F_{cl}(z))$ (if $c \ne l$, i.e., negative logit). Defining the shift as $\Delta_{cl}(z) = z' - z$, the corrected prediction is given by $\tilde{y} = \arg\max_c \frac{\exp(f_\theta(x)[c] + \tau\Delta_{cc})}{\sum_{c'}\exp(f_\theta(x)[c'] + \tau\Delta_{cc'})}$.
    - **Design Motivation**: CDF mapping preserves the relative ranking within the distribution (structural information) while aligning the distribution to a uniform reference. Utilizing global distributions as anchors leverages Bernstein's inequality to reduce estimation errors, making them more stable than single-class distributions.

3. **Online Logit Correction (Calibration during Training)**:

    - **Function**: Generating unbiased pseudo-labels during the self-training process, thereby disrupting the positive feedback loop of confirmation bias.
    - **Mechanism**: Using Gaussian Mixture Models (GMMs) to online model the logit distributions of the source and target domains respectively. The source and target domains each maintain $C \times C \times K$ Gaussian mixtures. During each training iteration, the GMM parameters are updated using momentum EM: $\phi_{cl}^{(\mathcal{T})} \leftarrow (1-\tilde{\tau}^n)\phi_{cl}^{(\mathcal{T})} + \tilde{\tau}^n \hat{\phi}_{cl}$. The correction shift is calculated and embedded into the cross-entropy loss: $\tilde{\ell}_{ce}^s = -\log \frac{\exp(f_\theta(x)[y] - \tau\Delta_{y,y}^S)}{\sum_c \exp(f_\theta(x)[c] - \tau\Delta_{y,c}^S)}$. Essentially, this learns a balanced scorer $g(x)[y] = f(x)[y] - \tau\Delta_y(x)$, which is equivalent to an adaptive margin-based loss.
    - **Design Motivation**: While post-processing corrects baseline bias only during inference, the bias of pseudo-labels continuously accumulates and amplifies during training. Online correction halts this cycle at its source, and gradually aligns the target domain logit distribution toward the source domain through shared anchors.

### Loss & Training

The overall training objective is $\mathcal{L} = \tilde{\mathcal{L}}^s + \tilde{\mathcal{L}}^u + \lambda(\mathcal{L}_{reg}^S + \mathcal{L}_{reg}^T)$. Here, $\tilde{\mathcal{L}}^s$ and $\tilde{\mathcal{L}}^u$ represent the source-domain supervised loss and target-domain pseudo-label loss with embedded correction terms, respectively. $\mathcal{L}_{reg}$ is the CDF auxiliary regression loss: an auxiliary regression head is used to predict the positive CDF value of each pixel $d_{ij} = F_{y,y}(f_\theta(x)[y])$. This value measures the discriminative difficulty of a pixel within its class and serves as domain-invariant structural knowledge. The temperature is set to $\tau=0.1$, the loss weight is $\lambda=0.2$, and the model is trained for 40K iterations.

## Key Experimental Results

### Main Results (GTA5→Cityscapes mIoU %)

| Method | Architecture | mIoU ↑ | Sidewalk | Fence | Pole | Sign | Bus | Train | Bike |
|------|------|--------|----------|-------|------|------|-----|-------|------|
| DAFormer | T | 68.3 | 70.2 | 48.1 | 49.6 | 59.4 | 78.2 | 65.1 | 61.8 |
| **+BLDA** | T | **70.7** | **78.3** | **55.2** | **55.7** | **65.2** | **83.4** | **73.2** | **67.1** |
| HRDA | T | 73.8 | 74.4 | 51.5 | 57.1 | 69.3 | 85.7 | 75.9 | 67.5 |
| **+BLDA** | T | **75.6** | **77.6** | **57.9** | **62.1** | **72.5** | **87.7** | **79.9** | **68.9** |
| MIC | T | 75.9 | 80.1 | 56.9 | 59.7 | 71.3 | 90.3 | 80.4 | 68.5 |
| **+BLDA** | T | **77.1** | **82.6** | **61.0** | **64.9** | **74.8** | **88.8** | **82.6** | **72.0** |

### Ablation Study (DAFormer on GTA→CS)

| Configuration | mIoU | std ↓ | Description |
|------|------|-------|------|
| DAFormer Baseline | 68.3 | 16.8 | Severe inter-class bias |
| + Reweighting (Cui et al.) | 68.8 | - | Adjusting loss weights, limited effect |
| + Resampling (DAFormer RCS) | 69.2 | - | Adjusting sample distribution |
| + BLDA (Full) | **70.7** | **15.5** | Balanced logit distribution |
| DACS (CNN) | 52.1 | 27.1 | CNN Baseline |
| DACS + BLDA (CNN) | **54.7** | **25.6** | Equally effective on CNN |

### Key Findings

- The most significant improvements occur in under-predicted classes: on the DAFormer baseline, Sidewalk achieves +8.1, Fence +7.1, Train +8.1, and Bike +5.3, which are precisely the hard-to-transfer classes that traditional methods struggle with.
- BLDA consistently delivers improvements across four baseline methods (gains of +1.2 to +2.4 mIoU), while significantly reducing the inter-class standard deviation (DAFormer 16.8 $\dots$ 15.5), which validates the effectiveness of class balancing.
- Its effectiveness across both CNN (ResNet-101) and Transformer (MiT-B5) architectures indicates that the bias issue is architecture-agnostic.

## Highlights & Insights

- **Diagnosing class bias through logit distributions** is the core contribution of this work—instead of relying on the unattainable target domain ground truth label distribution, it extracts bias signals directly from the model's own predictive behavior, offering excellent generalizability.
- **CDF mapping as domain-bridging structural knowledge**: The positive CDF value measures the discriminative difficulty of a pixel within its class. This feature is inherently domain-invariant as it is unaffected by image style, making its utilization as an auxiliary task an elegant by-product.
- **Plug-and-play engineering friendliness**: It does not alter the network architecture or introduce inference overhead (online correction is only required during training) and can be seamlessly integrated into any self-training UDA framework.

## Limitations & Future Work

- The anchor distribution is fixed as the global source domain logit distribution, which may not be the optimal reference when the discrepancy between the source and target domains is extremely large.
- The Gaussian assumptions of GMMs may be inaccurate for certain classes (e.g., highly skewed or multi-modal distributions).
- Validation is restricted to urban driving scene segmentation (GTA/SYNTHIA→Cityscapes), and its performance in other UDA segmentation scenarios remains to be verified.

## Related Work & Insights

- **vs DAFormer + RCS**: RCS samples rare classes page-wise with higher frequency but remains essentially a resampling strategy requiring class frequency priors; BLDA corrects directly from the logit distributions, offering better adaptability.
- **vs CPSL (Li et al., 2022)**: CPSL uses class-level pseudo-label selection thresholds to balance classes, but the thresholds rely on prior knowledge; BLDA's GMM-based online estimation requires no manual configuration.
- **vs Traditional logit adjustment (Menon 2021)**: Traditional methods adjust logits based on label frequencies, assuming that the test and training distributions are consistent; BLDA bypasses this assumption via distribution alignment, making it more applicable to domain adaptation scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ Evaluates and corrects UDA class bias from the perspective of logit distributions, offering a fresh perspective with solid theoretical backing
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two benchmarks, four baselines, two architectures, and highly complete ablations
- Writing Quality: ⭐⭐⭐⭐ Thorough problem analysis with a complete chain of motivational derivation
- Value: ⭐⭐⭐⭐⭐ A plug-and-play UDA segmentation improvement module with exceptionally high engineering practicality

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Dual form Complementary Masking for Domain-Adaptive Image Segmentation](dual_form_complementary_masking_for_domain-adaptive_image_segmentation.md)
- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](../../CVPR2026/segmentation/heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[CVPR 2026\] Masked Representation Modeling for Domain-Adaptive Segmentation](../../CVPR2026/segmentation/mrm_masked_representation_modeling_domain_adaptive.md)
- [\[CVPR 2025\] Universal Domain Adaptation for Semantic Segmentation](../../CVPR2025/segmentation/universal_domain_adaptation_for_semantic_segmentation.md)
- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](../../ICCV2025/segmentation/exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)

</div>

<!-- RELATED:END -->
