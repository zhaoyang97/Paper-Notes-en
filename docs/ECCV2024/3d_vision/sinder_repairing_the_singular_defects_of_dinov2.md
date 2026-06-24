---
title: >-
  [Paper Note] SINDER: Repairing the Singular Defects of DINOv2
description: >-
  [ECCV 2024][3D Vision][DINOv2] Reveals that the root cause of high-norm defect tokens in DINOv2 feature maps is the principal left singular vector of network weights (singular defect), and proposes SINDER—which repairs the defects by fine-tuning singular values on a small dataset while preserving feature quality.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "DINOv2"
  - "Singular Defects"
  - "Self-Supervised Learning"
  - "Vision Transformer"
  - "Unsupervised Segmentation"
date: 2026-05-08
content_hash: 880d6efda2ea1b39
---

# SINDER: Repairing the Singular Defects of DINOv2

**Conference**: ECCV 2024  
**arXiv**: [2407.16826](https://arxiv.org/abs/2407.16826)  
**Code**: [GitHub](https://github.com/haoqiwang/sinder)  
**Area**: 3D Vision  
**Keywords**: DINOv2, Singular Defects, Self-Supervised Learning, Vision Transformer, Unsupervised Segmentation

## TL;DR

Reveals that the root cause of high-norm defect tokens in DINOv2 feature maps is the principal left singular vector of network weights (singular defect), and proposes SINDER—which repairs the defects by fine-tuning singular values on a small dataset while preserving feature quality.

## Background & Motivation

### Approach

**Goal**: **Background**: Large-scale self-supervised ViT models like DINOv2 generate abnormal, high-norm patch tokens in feature maps (mean norm of 434 vs. 57.6 for normal tokens), which severely compromises dense prediction tasks. The only prior solution (DINOv2-Register) requires retraining the entire model from scratch and adding extra register tokens, which is extremely costly. This study conducts an in-depth analysis and reveals that these defect tokens possess two properties: (1) their directions are nearly independent of the input (with an angle of only 5.5° across different images); (2) they can be predicted by the principal singular vectors of the network weights.

## Method

### Overall Architecture

Consists of two parts: theoretical analysis and practical repair. First, the theoretical prediction of defect directions is derived by linearizing the network blocks, and then a lightweight fine-tuning strategy is designed for repair.

### Key Designs

**Singular Defect Direction Theory**: Linearly approximates the Attention Block and MLP Block as $Ax+b$ and $Cx+d$. Their combination yields the $E_i$ matrix. Combining multiple layers yields $G_i = E_i E_{i-1} \cdots E_0$, where the principal left singular vector is the theoretical singular defect direction of the $i$-th layer. Experiments demonstrate that starting from the 20th layer, the theoretical prediction matches the actual defect direction with high accuracy.

**Defect Detection**: Computes the absolute value of the inner product $l_t$ between the normalized patch token and the singular defect direction as the logit. Tokens exceeding the threshold of mean $\mu + 4\sigma$ standard deviations are identified as defects.

**Smooth Regularization Repair (SINDER)**: For each defect token, its $3 \times 3$ spatial neighborhood weighted average is used as the learning target, with weights determined by the softmax of the logits and a Gaussian kernel. Only the singular values of the network's linear layers are learned (freezing $U$ and $V$), and in each iteration, only the parameters of the 10 layers preceding the current defective layer are unfrozen.

### Loss & Training

$$L = \frac{1}{|\mathcal{D}|} \sum_{t \in \mathcal{D}} \|x_t - \tilde{x}_t\|$$

where $\tilde{x}_t$ is the smoothed target based on neighboring tokens. Finetuning is performed for only one epoch on a 30K ImageNet subset.

## Key Experimental Results

### Unsupervised Segmentation (CAUSE Method)

| Backbone | Cityscapes mIoU↑ | Cityscapes Acc↑ | VOC2012 mIoU↑ | VOC2012 Acc↑ |
|----------|-------------------|-----------------|---------------|--------------|
| DINOv2 | 31.4 | 85.2 | 55.8 | 91.7 |
| DINOv2-Register | 33.3 | 87.6 | 48.9 | 90.9 |
| **DINOv2-SINDER** | **35.6** | **88.4** | **62.9** | **93.6** |

### Supervised Segmentation and Classification

| Backbone | ADE20k mIoU↑ (Linear) | ADE20k mIoU↑ (Multi-scale) | ImageNet KNN Top1↑ | NYUd Depth (Linear 1)↓ |
|----------|------------------------|----------------------------|--------------------|-----------------------|
| DINOv2 | 48.83 | 53.24 | 83.53 | 0.370 |
| DINOv2-Register | 49.03 | 53.62 | 83.69 | 0.367 |
| **DINOv2-SINDER** | **51.11** | **54.78** | **83.51** | **0.337** |

### Ablation Study

Impact of constraining learnable parameters:

| Setting | KNN Top1↑ | ADE20k mIoU↑ |
|------|-----------|--------------|
| Singular values + bias (all layers) | 6.64 | 13.77 |
| Singular values (excl. QK) | 80.12 | 45.53 |
| Singular values (excl. QK) 15 layers | 82.81 | 49.85 |
| **Singular values (excl. QK) 10 layers** | **83.51** | **51.11** |
| Singular values (excl. QK) 5 layers | 83.53 | 50.61 |

### Key Findings

- SINDER achieves a +14% mIoU improvement over DINOv2-Register on VOC2012 (62.9 vs. 48.9), whereas the latter requires complete retraining.
- Classification performance experiences almost no degradation (KNN Top1 drops by only 0.02%).
- Significant advantages in carbon emissions and cost with only 6 hours of V100 training compared to the full retraining of DINOv2-Register.

## Highlights & Insights

1. **Outstanding theoretical contribution**: For the first time, provides a clear explanation for the cause of ViT defect tokens from an SVD perspective, decoupling them from the network weights.
2. Extremely efficient repair solution: fine-tuning only the singular value parameters on 30K images for 1 epoch.
3. The finding that limiting the number of learnable parameters actually helps preserve feature quality is widely generalizable.

## Limitations & Future Work

- Only validated on the DINOv2 Giant model; other ViT variants require further verification.
- Theoretical analysis is based on a single-token simplification assumption, without modeling interactions under multi-token scenarios.
- Repair effectiveness depends on the accuracy of the pre-computed singular defect directions.

## Related Work & Insights

This work offers a new perspective on understanding the internal mechanisms of large-scale ViT models. The singular value fine-tuning strategy can be generalized to feature repair scenarios in other models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Usefulness: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary](../../CVPR2025/3d_vision/nopain_no-box_point_cloud_attack_via_optimal_transport_singular_boundary.md)
- [\[ECCV 2024\] Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)

</div>

<!-- RELATED:END -->
