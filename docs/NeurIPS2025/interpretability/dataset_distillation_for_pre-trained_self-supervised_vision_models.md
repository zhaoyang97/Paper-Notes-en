---
title: >-
  [Paper Note] Dataset Distillation for Pre-Trained Self-Supervised Vision Models
description: >-
  [NeurIPS 2025][Interpretability][dataset distillation] This paper proposes Linear Gradient Matching, a dataset distillation method for pre-trained self-supervised vision models. A single synthetic image per class suffice…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "dataset distillation"
  - "self-supervised learning"
  - "linear probing"
  - "gradient matching"
  - "CLIP/DINO"
date: 2026-05-08
content_hash: e07844134add40f6
---

# Dataset Distillation for Pre-Trained Self-Supervised Vision Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.16674](https://arxiv.org/abs/2511.16674)
**Code**: [https://georgecazenavette.github.io/linear-gm](https://georgecazenavette.github.io/linear-gm)
**Area**: Interpretability
**Keywords**: dataset distillation, self-supervised learning, linear probing, gradient matching, CLIP/DINO

## TL;DR
This paper proposes Linear Gradient Matching, a dataset distillation method for pre-trained self-supervised vision models. A single synthetic image per class suffices to train a linear classifier approaching full-dataset performance, and the distilled images transfer across model architectures.

## Background & Motivation

**Background**: Dataset distillation aims to synthesize a minimal set of images such that models trained from scratch on them can match the performance of training on the full dataset. Existing methods (DC, MTT, DM, etc.) are designed for randomly initialized models trained from scratch.

**Limitations of Prior Work**: The dominant vision paradigm has shifted from training from scratch to pre-trained large models with downstream fine-tuning or linear probing. Existing distillation methods are not designed for this paradigm and cannot exploit the advantages of pre-trained features.

**Key Challenge**: Traditional distillation requires backpropagation through the entire network (e.g., MTT), which causes severe memory and stability issues for large models. Moreover, distilled images tend to overfit severely to a single architecture and fail to transfer across models.

**Goal**: (1) Design an efficient distillation method tailored to pre-trained self-supervised models; (2) Enable distilled images to transfer across model architectures.

**Key Insight**: Since the downstream task only trains a linear classifier, distillation need only perform matching in the gradient space of the linear layer—substantially reducing optimization complexity. Inspired by the Platonic Representation Hypothesis, different pre-trained models learn similar representations, suggesting that distilled images may generalize across models.

**Core Idea**: Synthesize images by matching only the gradients of a linear classifier in the pre-trained feature space, combined with pyramid reparameterization and differentiable augmentation to enable cross-model transfer.

## Method

### Overall Architecture
Given a large-scale real dataset and a frozen pre-trained self-supervised feature extractor $\phi$, the method outputs one synthetic image per class. During distillation, both real and synthetic images are passed through the frozen $\phi$ to extract features, which are then fed into a randomly initialized linear classifier. The synthetic images are updated by matching the linear-layer gradients between the two.

### Key Designs

1. **Linear Gradient Matching**:

    - Function: At each distillation step, a random linear classifier $W \sim \mathcal{N}(0,1)^{c \times f}$ is sampled. The cross-entropy loss gradients with respect to $W$ are computed separately for real and synthetic data, and their cosine distance is minimized.
    - Core formulation: $\ell_{\text{real}} = \text{CE}(W\phi(X_{\text{real}}); Y_{\text{real}})$, $\ell_{\text{syn}} = \text{CE}(W\phi(X_{\text{syn}}); Y_{\text{syn}})$
    - Meta loss: $\mathcal{L}_{\text{meta}} = 1 - \cos\left(\text{vec}\left(\frac{\partial \ell_{\text{real}}}{\partial W}\right), \text{vec}\left(\frac{\partial \ell_{\text{syn}}}{\partial W}\right)\right)$
    - Design Motivation: Matching only the linear-layer gradients avoids the memory explosion of backpropagating through the full backbone. Sampling a fresh random $W$ at each step prevents overfitting.

2. **Pyramid Representation**:

    - Function: Synthetic images are not stored directly as pixels but as a multi-scale pyramid $\rho = \{1\times1, 2\times2, \ldots, 256\times256\}$.
    - Rendering formula: $X = \text{sigmoid}\left(\sum_{r \in \rho} \text{resize}_{256}(P_r)\right)$
    - Design Motivation: Direct pixel optimization at high resolution produces abundant high-frequency noise patterns that overfit severely to the distillation backbone. The coarse-to-fine pyramid synthesis introduces strong regularization, yielding more natural images with better cross-model generalization.

3. **Color Decorrelation**:

    - Function: Distilled images are learned in a decorrelated color space and then linearly transformed back to standard RGB at synthesis time.
    - Design Motivation: Removes color biases that a single backbone may introduce (e.g., a tendency toward blue tones), improving cross-model generality.

4. **Differentiable Augmentations**:

    - Function: Differentiable augmentations—horizontal flipping, random cropping, Gaussian noise, etc.—are applied to synthetic images, with multiple augmented views (10 by default) concatenated into each batch per step.
    - Design Motivation: Using a single augmentation forces one image to encode all information; multiple augmentations reframe the optimization objective as "all augmented views collectively form the optimal training set," substantially improving distillation effectiveness and cross-model performance.

### Loss & Training
- Distillation runs for 5,000 steps at $224\times224$ resolution using ViT-B backbones.
- ImageNet-1k uses 3 augmentation views per step due to computational constraints; all other datasets use 10.
- Pyramid optimization proceeds progressively, incorporating finer resolution levels incrementally.

## Key Experimental Results

### Main Results (ImageNet-1k, 1 image per class)

| Training Set | CLIP | DINO-v2 | EVA-02 | MoCo-v3 | Avg. |
|---|---|---|---|---|---|
| **Distilled (Ours)** | **63.0** | **75.0** | **70.3** | **63.2** | **67.9** |
| Centroids | 53.9 | 69.5 | 58.1 | 57.4 | 59.7 |
| Neighbors | 38.8 | 67.7 | 49.9 | 56.4 | 53.2 |
| Random | 31.7 | 50.3 | 37.7 | 38.8 | 39.6 |
| Full Dataset (1.3M) | 78.7 | 83.0 | 81.7 | 76.5 | 80.0 |

With only 1 synthetic image per class, DINO-v2 linear probing reaches 75% (vs. 83% on the full dataset, a gap of 8 points), substantially outperforming all real-image baselines.

### Ablation Study (ImageNet-100, same-model and cross-model evaluation)

| Configuration | Same-Model Avg. Acc | Cross-Model Avg. Acc |
|---|---|---|
| Full (complete method) | 87.2 | 77.8 |
| − Color Decorrelation | 86.5 | 76.4 |
| − Pyramid | 85.7 | 67.1 |
| − Augmentation | 68.6 | 33.3 |

### Key Findings
- **Differentiable augmentation** contributes most: removing it drops same-model accuracy by 18.6 points and collapses cross-model accuracy to 33.3%.
- **Pyramid representation** is critical for cross-model transfer: removing it causes a 10.7-point drop in cross-model accuracy, demonstrating that pixel-level optimization produces high-frequency patterns that severely overfit specific backbones.
- Datasets distilled with DINO-v2 generalize best across models (average 63.7%), confirming that higher-quality models yield more transferable distilled results.
- Distilled image embeddings tend to lie at the periphery or exterior of class clusters, encoding highly discriminative features.

## Highlights & Insights
- **Elegant yet effective**: Matching only the linear-layer gradients suffices. Leveraging the structure of the pre-trained feature space reduces distillation from a full network training problem to one at the level of linear classifier training.
- **Cross-model transfer**: Images distilled with DINO can directly train a CLIP linear probe with competitive performance, corroborating the hypothesis that large models converge toward similar representations.
- **Interpretability tool**: Distilled images reveal what a model attends to. On the Spawrious dataset, DINO's distilled images clearly depict dog breeds, while MoCo's images focus almost exclusively on background context—explaining why MoCo fails on spuriously correlated data.

## Limitations & Future Work
- The study is limited to the linear probing regime; distillation for fine-tuning or adapter-based tuning has not been explored.
- Cross-architecture transfer performs poorly between CLIP and MoCo, likely due to low alignment between their representation spaces.
- Experiments are conducted only on ViT-B; the effect of model scale (S/L/H) remains unverified.
- Distillation still requires 5,000 optimization steps, incurring non-trivial memory overhead at the ImageNet-1k scale.

## Related Work & Insights
- **vs. DC/MTT (traditional distillation)**: Traditional methods match gradients or trajectories over the full network; this work matches only linear-layer gradients, substantially reducing computational cost and adapting naturally to the pre-training paradigm.
- **vs. SRe2L (large-model distillation)**: Methods such as SRe2L extend distillation to large models but perform poorly in the extreme setting of one image per class; this work specifically targets this ultra-low-data regime.
- Framing distillation as an interpretability tool is a novel perspective that may inspire model understanding research in other directions.

## Rating
- Novelty: ⭐⭐⭐⭐ — Repositioning distillation for the linear probing regime of pre-trained models is a fresh angle, though the individual technical components (gradient matching, pyramid representation, augmentation) are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 4 backbones, multiple datasets, and cross-model, adversarial, fine-grained, and OOD settings.
- Writing Quality: ⭐⭐⭐⭐⭐ — Visually polished with a clear narrative; the interpretability analysis is particularly compelling.
- Value: ⭐⭐⭐⭐ — Provides a practical tool for few-shot linear probing in an era of increasingly prevalent pre-trained models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICCV 2025\] AIM: Amending Inherent Interpretability via Self-Supervised Masking](../../ICCV2025/interpretability/aim_amending_inherent_interpretability_via_self-supervised_masking.md)
- [\[NeurIPS 2025\] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning](learning_to_focus_causal_attention_distillation_via_gradient-guided_token_prunin.md)
- [\[NeurIPS 2025\] Latent Principle Discovery for Language Model Self-Improvement](latent_principle_discovery_for_language_model_self-improvement.md)
- [\[NeurIPS 2025\] Explaining Similarity in Vision-Language Encoders with Weighted Banzhaf Interactions](explaining_similarity_in_vision-language_encoders_with_weighted_banzhaf_interact.md)

</div>

<!-- RELATED:END -->
