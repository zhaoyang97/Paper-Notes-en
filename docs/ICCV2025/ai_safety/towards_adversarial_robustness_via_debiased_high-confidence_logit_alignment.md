---
title: >-
  [Paper Note] Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment
description: >-
  [ICCV 2025][AI Safety][adversarial training] This paper reveals that inverse adversarial attacks in adversarial training introduce spurious correlations by shifting model attention toward background features. The propose…
tags:
  - "ICCV 2025"
  - "AI Safety"
  - "adversarial training"
  - "adversarial robustness"
  - "inverse adversarial attack"
  - "spurious correlations"
  - "logit alignment"
  - "debiasing"
  - "attention shift"
date: 2026-05-08
content_hash: fad1205cbd9bdf8f
---

# Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment

**Conference**: ICCV 2025
**arXiv**: [2408.06079](https://arxiv.org/abs/2408.06079)  
**Code**: [KejiaZhang-Robust/DHAT](https://github.com/KejiaZhang-Robust/DHAT)  
**Authors**: Kejia Zhang, Juanjuan Weng, Shaozi Li, Zhiming Luo (Xiamen University, Jinan University)
**Area**: AI Safety
**Keywords**: adversarial training, adversarial robustness, inverse adversarial attack, spurious correlations, logit alignment, debiasing, attention shift

## TL;DR

This paper reveals that inverse adversarial attacks in adversarial training introduce spurious correlations by shifting model attention toward background features. The proposed DHAT method addresses this bias through two components—Debiased High-confidence Logit Regularization (DHLR) and Foreground Logit Orthogonal Enhancement (FLOE)—achieving state-of-the-art adversarial robustness on CIFAR-10/100 and ImageNet-1K.

## Background & Motivation

### Development of Adversarial Training

Adversarial Training (AT) is widely regarded as the most effective defense against adversarial attacks. It augments training with adversarial examples to improve model robustness, formulated as a min-max optimization problem: the outer minimization updates model parameters to reduce adversarial loss, while the inner maximization seeks the strongest perturbation.

### Introduction and Hidden Risks of Inverse Adversarial Attacks

Recent methods such as UIAT and ACR introduce **inverse adversarial attacks** to generate samples with higher confidence than natural ones, aiming to align adversarial examples toward high-confidence regions of the correct class. However, this paper identifies a critical issue:

**The high-confidence outputs of inverse adversarial samples stem from biased feature activations.** Grad-CAM visualizations reveal that inverse adversarial attacks improve prediction confidence by systematically shifting model attention from foreground objects (e.g., sheep) to irrelevant background regions (e.g., grass)—analogous to "recognizing grass to identify sheep," a form of **spurious correlation bias**.

### Quantitative Verification

Using IoU metrics on an ImageNet subset, the authors demonstrate that increasing the strength of inverse attacks in UIAT does not improve attention–foreground IoU; instead, attention–background IoU increases significantly. This bias causes the model to overfit background features, degrading robust generalization.

### Core Insight

Inverse adversarial training disproportionately biases model attention toward background features, inducing spurious correlations—the model excessively relies on contextual features that have no causal relationship with the target label. As shown in Table 5, this simultaneously degrades both robustness and generalization.

## Method

### Overall Architecture: DHAT

DHAT (Debiased High-Confidence Adversarial Training) consists of two core components:

1. **DHLR** (Debiased High-Confidence Logit Regularization): Quantifies and eliminates background feature bias by aligning adversarial logits with debiased high-confidence logits.
2. **FLOE** (Foreground Logit Orthogonal Enhancement): Reduces the correlation between high-confidence logits and background feature logits via orthogonal projection in affine space, restoring the model's focus on foreground features.

The total loss function is:

$$\mathcal{L}_{DHAT} = \mathcal{L}_{AT}(\hat{z}, y) + \lambda_1 \cdot \mathcal{L}_{DHLR}(\check{z}^*, \hat{z}) + \lambda_2 \cdot \mathcal{L}_{FLOE}(\check{z}, \check{z}_{(B)})$$

where $\lambda_1 = \lambda_2 = 1.0$.

### Key Design 1: Debiased High-Confidence Logit Regularization (DHLR)

**Step 1 — Background Feature Separation**: Using the Grad-CAM attention map $M$ of a natural sample $x$, background regions are extracted from the inverse adversarial sample $\check{x}$:

$$[\check{x}_{(B)}]_{(i,j)} = \mathbb{I}_{(M_{i,j} < \omega)} \cdot \check{x}_{(i,j)}$$

where $\omega$ is a predefined threshold and $\mathbb{I}$ is the indicator function. Low-attention regions are identified as background.

**Step 2 — Bias Quantification**: The background feature logit is computed as $\check{z}_{(B)} = f_\theta(\check{x}_{(B)})$, reflecting the degree of bias introduced by background activations during inference.

**Step 3 — Debiased Calibration**: The debiased high-confidence logit is obtained by subtracting the background logit from the inverse adversarial logit:

$$\check{z}^* = \check{z} - \check{z}_{(B)}$$

**Step 4 — KL Divergence Alignment**: The adversarial logit is aligned with the debiased logit:

$$\mathcal{L}_{DHLR}(\check{z}^*, \hat{z}) = \mathcal{L}_{KL}(\phi(\check{z}^*) \| \phi(\hat{z}))$$

### Key Design 2: Foreground Logit Orthogonal Enhancement (FLOE)

While DHLR calibrates the alignment target, it does not directly address the model's persistent background bias when processing inverse adversarial samples. FLOE reduces the projection of the high-confidence logit $\check{z}$ onto the background logit $\check{z}_{(B)}$, making $\check{z}$ less explainable by $\check{z}_{(B)}$:

$$\mathcal{L}_{FLOE}(\check{z}, \check{z}_{(B)}) = -\left|\check{z} - \frac{\check{z} \cdot \check{z}_{(B)}}{|\check{z}_{(B)}|^2} \cdot \check{z}_{(B)}\right|_p$$

Intuition: maximizing the orthogonal component of $\check{z}$ (i.e., the portion not explained by background) forces the model to rely more on foreground features for prediction.

### Training Procedure

1. Generate adversarial examples $\hat{x}$ via PGD.
2. Generate inverse adversarial examples $\check{x}$ via inverse PGD.
3. Compute the attention map $M$ via Grad-CAM and separate background features.
4. Compute the three loss terms and optimize jointly.

## Key Experimental Results

### Main Results: CIFAR-10 (WRN28-10, ε=8/255)

| Method | Clean↑ | PGD-10↑ | C&W↑ | AA↑ | Robust Gap↓ |
|--------|--------|---------|------|-----|-------------|
| MART | 82.99 | 56.25 | 52.26 | 50.67 | 9.52 |
| AWP | 82.67 | 57.80 | 54.82 | 51.90 | 6.90 |
| UIAT | 82.94 | 58.66 | 54.11 | 52.17 | 7.92 |
| SGLR | 85.76 | 57.53 | 54.28 | 52.07 | 9.38 |
| **DHAT** | 83.95 | **60.49** | 55.27 | 53.10 | **3.51** |
| **DHAT-CFA** | 84.49 | **62.67** | 55.95 | **54.05** | 6.33 |

### Main Results: CIFAR-100 (WRN28-10, ε=8/255)

| Method | Clean↑ | PGD-10↑ | AA↑ | Robust Gap↓ |
|--------|--------|---------|-----|-------------|
| AWP | 57.94 | 34.01 | 28.90 | 7.87 |
| UIAT | 57.65 | 34.27 | 29.03 | 11.70 |
| **DHAT** | 59.14 | **35.82** | **30.17** | **4.24** |
| **DHAT-CFA** | 61.54 | **37.67** | **30.93** | 5.93 |

### Main Results: ImageNet-1K (WRN28-10, ε=4/255)

| Method | Clean↑ | PGD-10↑ | AA↑ | Robust Gap↓ |
|--------|--------|---------|-----|-------------|
| AWP | 64.25 | 45.13 | 40.02 | 12.82 |
| UIAT | 62.64 | 45.29 | 40.18 | 14.68 |
| **DHAT** | 65.90 | **46.83** | **41.70** | **9.53** |
| **DHAT-CFA** | 66.26 | **48.27** | **42.45** | 11.64 |

### Cross-Architecture Evaluation (CIFAR-10, ε=8/255)

| Method | ResNet-50 AA↑ | VGG-16 AA↑ | Inception-V3 AA↑ |
|--------|--------------|------------|-------------------|
| UIAT | 51.00 | 45.27 | 51.23 |
| **DHAT-CFA** | **52.38** | **47.83** | **52.67** |

### Key Findings

1. **Comprehensive robustness gains**: DHAT outperforms all prior state-of-the-art methods across all datasets and attack types. On CIFAR-10 PGD-10, DHAT surpasses UIAT by 1.93%; on ImageNet-1K PGD-10, by 1.54%.
2. **Substantially reduced robust generalization gap**: The Robust Gap directly reflects spurious correlations. DHAT reduces the gap on CIFAR-10 from UIAT's 7.92% to 3.51%, a reduction of over 55%.
3. **Plug-and-play compatibility**: DHAT integrates seamlessly with advanced AT methods such as AWP and CFA for further gains. DHAT-CFA achieves the best performance across all experiments.
4. **Cross-architecture consistency**: Consistent improvements on ResNet-50, VGG-16, and Inception-V3 validate the generality of the method.
5. **Robustness across attack strengths**: DHAT maintains stable advantages under varying PGD and C&W perturbation budgets, with more gradual performance degradation.

## Highlights & Insights

1. **Revealing a fundamental flaw in inverse adversarial training**: The paper uncovers the critical insight that "high confidence does not imply correct attention"—inverse adversarial attacks boost confidence by amplifying background feature activations, a phenomenon entirely overlooked by prior work.
2. **Causal inference perspective**: Spurious correlations are analyzed through the lens of causal reasoning, linking attention shift to the robust generalization gap and providing a deeper understanding beyond mere accuracy improvement.
3. **Robust Gap as a proxy for spurious correlations**: The robustness gap between training and test sets directly reflects the degree of spurious association. DHAT's significant reduction of this gap demonstrates a fundamental decrease in reliance on spurious features.
4. **Simple yet effective design**: DHLR and FLOE each require only one additional loss term, with no modifications to the network architecture or attack pipeline, making them highly plug-and-play.
5. **Closed-loop argumentation**: Problem identification (attention shift) → quantitative measurement (IoU analysis) → solution (debiasing + orthogonalization) → validation (Gap reduction) forms a complete and rigorous narrative.

## Limitations & Future Work

1. **Additional computational overhead**: Generating inverse adversarial samples, computing Grad-CAM, and performing background feature forward passes introduce training costs higher than standard AT.
2. **Hard thresholding in attention maps**: A fixed threshold $\omega$ is used to binarize attention maps into foreground/background, which oversimplifies cases where the boundary is ambiguous.
3. **Incompatibility with spurious-feature-exploiting methods**: DHAT is not well-suited for combination with methods such as FSR and SGLR that leverage non-robust features, limiting compositional flexibility.
4. **Dependence on Grad-CAM quality**: The quality of background separation depends on the accuracy of Grad-CAM, whose reliability on adversarially trained models remains questionable.
5. **Validation limited to classification**: Robustness improvements have not been evaluated on tasks such as object detection or segmentation.

## Related Work & Insights

- **Adversarial attacks**: FGSM → PGD → C&W → AutoAttack, representing a continuous evolution of attack methodologies.
- **Adversarial training**: MART (aligning adversarial and natural sample logits), TRADES (robustness–accuracy trade-off), AWP (smoothing the loss landscape via weight perturbation), FSR (leveraging useful information in non-robust features), CFA (class-wise fair adversarial training), SGLR (self-distillation soft-label calibration).
- **Inverse adversarial training**: UIAT and ACR guide adversarial sample distributions using high-confidence samples; this paper identifies their hidden risks and proposes a correction.
- **Spurious correlations**: The model's reliance on non-causal features is analyzed from a causal inference perspective.

## Rating

| Dimension | Score (1–10) |
|-----------|-------------|
| Novelty | 9 — First to reveal the attention shift problem in inverse adversarial training |
| Theoretical Depth | 6 — Primarily empirically driven; lacks formal theoretical analysis |
| Experimental Thoroughness | 9 — Comprehensive validation across multiple datasets, architectures, and attacks |
| Practical Value | 8 — Plug-and-play; compatible with existing AT methods |
| Writing Quality | 8 — Clear narrative from problem identification to solution |
| **Overall** | **8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Fair Representation Learning with Controllable High Confidence Guarantees via Adversarial Inference](../../NeurIPS2025/ai_safety/fair_representation_learning_with_controllable_high_confidence_guarantees_via_ad.md)
- [\[NeurIPS 2025\] Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness](../../NeurIPS2025/ai_safety/bridging_symmetry_and_robustness_on_the_role_of_equivariance_in_enhancing_advers.md)
- [\[NeurIPS 2025\] Boosting Adversarial Transferability with Spatial Adversarial Alignment](../../NeurIPS2025/ai_safety/boosting_adversarial_transferability_with_spatial_adversarial_alignment.md)
- [\[NeurIPS 2025\] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits](../../NeurIPS2025/ai_safety/understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)
- [\[ICCV 2025\] Backdooring Self-Supervised Contrastive Learning by Noisy Alignment](backdooring_self-supervised_contrastive_learning_by_noisy_alignment.md)

</div>

<!-- RELATED:END -->
