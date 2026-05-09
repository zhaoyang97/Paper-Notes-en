---
title: >-
  [Paper Note] ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering
description: >-
  [CVPR 2026][AI Safety][Autoregressive image generation] This paper proposes ClusterMark, a watermarking method based on visual token clustering for autoregressive image generation models. By assigning semantically similar tokens to the same color set (red/green), ClusterMark substantially improves watermark robustness under image perturbations while preserving image quality and enabling fast verification.
tags:
  - CVPR 2026
  - AI Safety
  - Autoregressive image generation
  - watermarking
  - visual token clustering
  - robustness
  - KGW watermarking
date: 2026-05-08
content_hash: 855f21ae9ce56929
---

# ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering

**Conference**: CVPR 2026
**arXiv**: [2508.06656](https://arxiv.org/abs/2508.06656)
**Code**: [https://github.com/lukovnikov/ClusterMark](https://github.com/lukovnikov/ClusterMark)
**Area**: AI Security / Image Watermarking
**Keywords**: Autoregressive image generation, watermarking, visual token clustering, robustness, KGW watermarking

## TL;DR

This paper proposes ClusterMark, a watermarking method based on visual token clustering for autoregressive image generation models. By assigning semantically similar tokens to the same color set (red/green), ClusterMark substantially improves watermark robustness under image perturbations while preserving image quality and enabling fast verification.

## Background & Motivation

1. **Background**: Watermarking generated content is a critical tool for mitigating AI misuse. Watermark embedding in diffusion model generation has been extensively studied, whereas watermarking for autoregressive (AR) image models remains in an early stage.
2. **Limitations of Prior Work**: Directly transferring KGW watermarking from LLMs to AR image models is feasible but not robust — verification requires re-encoding images into tokens, and image perturbations cause inaccurate token reconstruction, significantly degrading watermark detection rates.
3. **Key Challenge**: The quantization process in VQ-VAE means that even minor perturbations can produce entirely different tokens, while in KGW schemes similar tokens are randomly assigned to red/green sets, resulting in unstable color assignments for reconstructed tokens.
4. **Goal**: Design a watermarking scheme for AR image models that is robust to image perturbations.
5. **Key Insight**: Cluster tokens with nearby embeddings in the codebook into the same group, performing green/red set partitioning at the cluster level rather than the token level.
6. **Core Idea**: Although tokens may change after perturbation, they are likely to fall into the same cluster, thereby maintaining stable color set assignments.

## Method

### Overall Architecture

Before generation: apply k-means clustering to codebook tokens. During generation: compute the hash based on the cluster of the previous token (rather than the token itself), partition clusters (not individual tokens) into green/red sets, and bias logits to favor green tokens. During verification: encode the image into tokens, compute the proportion of green tokens, and perform a one-sided binomial test.

### Key Designs

1. **Cluster-Based Green/Red Set Partitioning**:

    - Function: Improve watermark robustness against image perturbations.
    - Mechanism: Apply k-means to partition the codebook $\mathbb{V}$ into $k$ clusters $(C_1, ..., C_k)$. The hash is computed based on cluster indices rather than token indices: $o_i = \text{hash}(\kappa, c(q_{i-1}))$. The green set is the union of selected clusters: $G_i = \bigcup_{C_i \in G_i^{\text{cluster}}} C_i$.
    - Design Motivation: After perturbation, tokens may change but are likely to remain within the same cluster (since clustering is based on Euclidean distances in codebook vector space), ensuring stable color assignments.

2. **Token/Cluster Classifier Fine-Tuning**:

    - Function: Further improve token reconstruction accuracy under perturbation.
    - Mechanism: A copy of the VQ-VAE encoder is augmented with a classification output head and trained on unwatermarked images with adversarial perturbation augmentation. The token classifier predicts original token indices; the cluster classifier directly predicts cluster indices.
    - Design Motivation: The standard VQ-VAE encoder is insufficiently accurate at reconstructing tokens from perturbed images; the fine-tuned version learns to "undo" the effects of perturbations.

3. **Prefix Tuning**:

    - Function: Select the optimal hash prefix to avoid false positives.
    - Mechanism: Certain prefixes $\kappa$ produce anomalously high green token ratios on images with large uniform regions, leading to false positives. This is mitigated by selecting the best prefix across multiple $\kappa$ values.
    - Design Motivation: This issue is more severe with a small number of clusters, as specific cluster transition patterns more easily introduce bias in uniform regions.

### Loss & Training

Token classifier training: cross-entropy loss with perturbation-augmented unwatermarked images. Cluster classifier: cross-entropy loss for predicting cluster indices. Training runs for 30 epochs with linearly increasing perturbation strength.

## Key Experimental Results

### Main Results

| Model/Method | Clean AUC/TPR | JPEG20 | Gaussian Blur | Salt-and-Pepper | Regeneration |
|---|---|---|---|---|---|
| No clustering (baseline) | 1.0/0.999 | 0.692 | 0.068 | 0.069 | 0.710 |
| Clustering k=64 | 1.0/1.0 | 0.956 | 0.663 | 0.402 | 0.972 |
| +Cluster classifier | 1.0/1.0 | 0.893 | 0.925 | 0.999 | 0.935 |

### Ablation Study

| Configuration | JPEG | Gaussian Noise | Notes |
|---|---|---|---|
| k=8 | Highest robustness | Highest robustness | Notable FID degradation and high variance |
| k=64 | High robustness | Good | Best quality-robustness trade-off |
| k=128 | Moderate | Moderate | Approaches no-clustering baseline |
| δ=5 vs δ=2 | Significantly better | Significantly better | Stronger bias = stronger watermark |
| γ=0.25 vs 0.5 | Significantly better | Better | Smaller green set = stronger signal |

### Key Findings

- Clustering substantially improves robustness even without fine-tuning, particularly against JPEG compression and regeneration attacks.
- The cluster classifier most effectively addresses salt-and-pepper noise (TPR: 40.2% → 99.9%).
- Verification is extremely fast (~12ms/image), orders of magnitude faster than diffusion model watermarking.

## Highlights & Insights

- **Training-free baseline is already effective**: Robustness is substantially improved through k-means clustering alone, achieving remarkable simplicity.
- **Cluster-level hashing**: Elevating the hash granularity from the token level to the cluster level is the key innovation, providing inherent fault tolerance against token-level perturbations.
- **Practical verification efficiency**: Verification time is comparable to lightweight post-processing watermarking schemes (~12ms), far faster than diffusion model watermarking.

## Limitations & Future Work

- The method remains vulnerable to geometric transformations such as rotation and cropping, requiring image synchronization layers for mitigation.
- Clustering reduces the effective codebook size, and excessively small values of $k$ lead to image quality degradation.
- Validation is currently limited to AR models that decode in left-to-right sequential order.

## Related Work & Insights

- **vs. IndexMark**: IndexMark pairs similar tokens but assigns them to different color sets; ClusterMark places similar tokens into the same color set.
- **vs. WMAR**: WMAR simultaneously fine-tunes the VAE decoder, adding complexity; ClusterMark does not modify the decoder, yielding a simpler design.

## Rating

- Novelty: ⭐⭐⭐⭐ Cluster-level watermarking is a concise and effective innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three models, multiple perturbation types, comprehensive ablation, and runtime comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with complete algorithmic pseudocode.
- Value: ⭐⭐⭐⭐ A practical solution for watermarking AR image models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces](recovermark_robust_watermarking_for_localization_and_recovery_of_manipulated_fac.md)
- [\[CVPR 2026\] AdvMark: Decoupling Defense Strategies for Robust Image Watermarking](decoupling_defense_strategies_for_robust_image_watermarking.md)
- [\[CVPR 2026\] Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection](tutor-student_reinforcement_learning_a_dynamic_curriculum_for_robust_deepfake_de.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)

</div>

<!-- RELATED:END -->
