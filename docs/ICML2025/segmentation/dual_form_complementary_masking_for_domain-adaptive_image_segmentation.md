---
title: >-
  [Paper Note] Dual form Complementary Masking for Domain-Adaptive Image Segmentation
description: >-
  [ICML 2025][Segmentation][Unsupervised Domain Adaptation] Proposes the MaskTwins framework, which theorizes masked reconstruction as a sparse signal reconstruction problem, proves that dual form complementary masks have theoretical advantages in extracting domain-invariant features, and achieves domain-adaptive segmentation through complementary mask consistency constraints in end-to-end training.
tags:
  - "ICML 2025"
  - "Segmentation"
  - "Unsupervised Domain Adaptation"
  - "Complementary Mask"
  - "Masked Image Modeling"
  - "Sparse Signal Reconstruction"
  - "Consistency Regularization"
  - "Semantic Segmentation"
date: 2026-05-08
content_hash: f40334967fd2b2b9
---

# Dual form Complementary Masking for Domain-Adaptive Image Segmentation

**Conference**: ICML 2025

**arXiv**: [2507.12008](https://arxiv.org/abs/2507.12008)

**Authors**: Jiawen Wang, Yinda Chen, Xiaoyu Liu, Che Liu, Dong Liu, Jianqing Gao, Zhiwei Xiong

**Area**: Segmentation (Domain-Adaptive Semantic Segmentation)

**Keywords**: Unsupervised Domain Adaptation, Complementary Mask, Masked Image Modeling, Sparse Signal Reconstruction, Consistency Regularization, Semantic Segmentation

## TL;DR

Proposes the MaskTwins framework, which theorizes masked reconstruction as a sparse signal reconstruction problem, proves that dual form complementary masks have theoretical advantages in extracting domain-invariant features, and achieves domain-adaptive segmentation through complementary mask consistency constraints in end-to-end training.

## Background & Motivation

Unsupervised Domain Adaptation (UDA) aims to bridge the domain gap by utilizing labeled source domain data and unlabeled target domain data. Existing methods mainly fall into three categories: statistical moment alignment, adversarial learning, and self-training. Recent works like MIC incorporate Masked Image Modeling (MIM) with consistency regularization for UDA, but suffer from two key limitations:

**Lack of Theoretical Foundation**: Existing methods only treat masking as a special form of perturbation to the input image, lacking a theoretical analysis of the effectiveness of masked reconstruction.

**Underutilization of Complementarity**: Prior work (e.g., MIC) uses random masking and does not deeply explore the potential of complementary masks in single-modal scenarios.

Starting from Compressed Sensing theory, this paper reformulates masked reconstruction as a sparse signal reconstruction problem, and theoretically proves for the first time that complementary masking outperforms random masking in three aspects: information preservation, generalization bounds, and feature consistency.

## Method

### Overall Architecture

MaskTwins adopts a teacher-student architecture with the following core pipeline:

1. Calculate supervised segmentation loss on source domain data.
2. Generate a complementary mask pair $(D, 1-D)$ for target domain images, producing two complementary views.
3. The teacher model (updated via EMA) generates pseudo-labels for the unmasked target image.
4. The student model predicts on the two complementary masked images separately, constrained by consistency loss and complementary mask loss.

### Theory Analysis

**Visual Data Model**: The input image is modeled as $X = S + E + N$, where $S$ is the sparse signal, $E$ is the environmental factor, and $N \sim \mathcal{N}(0, \sigma^2 I)$ is Gaussian noise.

**Complementary Mask Definition**: $D \in \{0,1\}^{H \times W}$, where each element is independently sampled from $\text{Bernoulli}(0.5)$, and the complementary mask pair is $(D, \mathbf{1}-D)$.

**Core Theorems**:

- **Information Preservation (Theorem 1)**: $\mathbb{E}[\text{IP}(X_D, X_{1-D})] \geq \mathbb{E}[\text{IP}(X_{R_1}, X_{R_2})]$
- **Generalization Bound (Theorem 2)**: The generalization bound of the complementary mask is tighter, containing no extra term of $\sqrt{HWC}$.
- **Feature Consistency (Theorem 3)**: The feature consistency error of complementary masks does not contain the environmental factor $\|E\|_F$ term.

### Key Designs: Complementary Mask Learning

For the target domain image $X^T$, generate a complementary masked image pair:

$$X^T_{cm} = \{D \odot X^T, (1-D) \odot X^T\}$$

The masks are sampled at the patch level: $D_{mb+1:(m+1)b, nb+1:(n+1)b} \sim \text{Bernoulli}(1-r)$

**Complementary Mask Loss**:

$$\mathcal{L}^T_{cm} = \mathbb{E}[\|p^T_{j,D}, p^T_{j,1-D}\|_2]$$

**Mask Consistency Learning Loss**:

$$\mathcal{L}^T_{cl} = \mathbb{E}[\lambda \cdot \mathcal{L}_{ce}(p^T_{j,D}, \hat{y}^T_j) + (1-\lambda) \cdot \mathcal{L}_{ce}(p^T_{j,1-D}, \hat{y}^T_j)]$$

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}^S_{sup} + \mathcal{L}^T_{cl} + \lambda_{cm} \mathcal{L}^T_{cm}$$

where $\mathcal{L}^S_{sup} = \mathbb{E}[-y^S_i \log(p^S_i)]$ is the source domain supervised cross-entropy loss. The teacher model is updated via EMA: $\phi_{t+1} \leftarrow \alpha \phi_t + (1-\alpha)\theta_t$.

## Key Experimental Results

### Main Results: SYNTHIA→Cityscapes Semantic Segmentation (mIoU, %)

| Method | Road | SW | Build | TL | TS | Veg. | Sky | PR | Rider | Car | Bus | Motor | Bike | mIoU |
|------|------|------|-------|------|------|------|------|------|-------|------|------|-------|------|------|
| DAFormer | 84.5 | 40.7 | 88.4 | 55.0 | 54.6 | 86.0 | 89.8 | 73.2 | 48.2 | 87.2 | 53.2 | 53.9 | 61.7 | 67.4 |
| HRDA | 85.2 | 47.7 | 88.8 | 65.7 | 60.9 | 85.3 | 92.9 | 79.4 | 52.8 | 89.0 | 64.7 | 63.9 | 64.9 | 72.4 |
| MIC | 86.6 | 50.5 | 89.3 | 66.7 | 63.4 | 87.1 | 94.6 | 81.0 | 58.9 | 90.1 | 61.9 | 67.1 | 64.3 | 74.0 |
| **MaskTwins** | **96.0** | **70.1** | **89.5** | **66.8** | 62.1 | **89.1** | 94.3 | **81.5** | **59.7** | **90.5** | **66.6** | **67.7** | 63.6 | **76.7** |

**Key Findings**: MaskTwins outperforms the SOTA (MIC) by **+2.7 mIoU**, showing surprising improvements on the sidewalk category (50.5 → 70.1, +19.6 IoU) and the road category (+4.8 IoU).

### Biological Image Segmentation: Mitochondrial Semantic Segmentation (IoU, %)

| Method | V2L1 | V2L2 | R2H | H2R |
|------|------|------|------|------|
| DA-ISC | 68.7 | 74.3 | 74.8 | 79.4 |
| CAFA | 71.8 | 75.4 | 76.3 | 80.6 |
| **MaskTwins** | **75.0** | **78.6** | **78.4** | **81.9** |

### Ablation Study

| CL | CMask | RMask | EMA | AdaIN | mIoU |
|----|-------|-------|-----|-------|------|
| - | - | - | - | - | 53.7 |
| ✓ | - | ✓ | - | - | 74.3 |
| ✓ | - | ✓ | ✓ | ✓ | 75.2 |
| ✓ | ✓ | - | - | - | 76.0 |
| ✓ | ✓ | - | ✓ | ✓ | **76.7** |

**Key Findings**: Complementary Mask (CMask) vs. Random Mask (RMask): 76.0 vs. 74.3, obtaining a +1.7 mIoU gain by simply replacing the masking strategy.

### Hyperparameter Ablation

| Mask Ratio $r$ | mIoU | | Patch Size $b$ | mIoU |
|-------------|------|-|---------------|------|
| 0.1 | 72.0 | | 32 | 76.2 |
| 0.2 | 74.6 | | 64 | **76.7** |
| 0.3 | 75.4 | | 128 | 75.9 |
| 0.4 | 76.5 | | 256 | 75.6 |
| 0.5 | **76.7** | | 512 | 75.0 |

Optimal configuration: $r=0.5$, $b=64$ (approx. 1/16 of the input size).

## Highlights & Insights

1. **Theory-Driven Method Design**: For the first time, masked reconstruction is connected with Compressed Sensing theory. The superiority of complementary masks is strictly proven across three dimensions: information preservation, generalization bounds, and feature consistency, demonstrating a perfect alignment between theory and experiments.
2. **Simple and Efficient**: MaskTwins introduces no extra learnable parameters, achieving significant performance gains solely by changing the masking strategy (random → complementary).
3. **Cross-Domain Generalization**: Achieves SOTA on natural images (Cityscapes), EM mitochondrial segmentation, and 3D synapse detection, demonstrating effectiveness across both 2D and 3D scenarios.
4. **Significant Boost on Sidewalk**: The hardest-to-adapt sidewalk category improves by +19.6 IoU, indicating that complementary masking is exceptionally effective at learning categories that heavily rely on contextual relationships.

## Limitations & Future Work

1. Limited improvement on small-object categories, as the complementary mask may fully obscure small targets.
2. The theoretical analysis is based on linear feature extraction assumptions, which deviates from the non-linear nature of deep networks.
3. Only validated on synthetic → real UDA scenarios, lacking real → real domain adaptation experiments.
4. The mask patch size and ratio need to be manually adjusted for different tasks.

## Related Work & Insights

- **MIC (CVPR 2023)**: First to utilize mask consistency in UDA, but relies solely on random masks and lacks theoretical analysis.
- **HRDA (ECCV 2022)**: A multi-resolution domain adaptation framework, which serves as the base architecture for MaskTwins.
- **Compressed Sensing Theory**: The signal reconstruction advantage of complementary masks can be generalized to other missions requiring domain-invariant features.

**Insight**: The theoretical framework of complementary masking can be extended to multi-view learning (the paper has proven a multi-view theorem for $K$ complementary masks), providing theoretical guidance for directions such as video domain-adaptive segmentation and multi-modal fusion.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Provides a theoretical foundation for complementary masking from a compressed sensing perspective, offering a novel viewpoint |
| Technical Depth | 5 | Complete theoretical proofs (5 theorems) + detailed experimental validation |
| Experimental Thoroughness | 5 | 6 datasets, natural/biological/3D scenarios, comprehensive ablation |
| Writing Quality | 4 | Clear structure, rigorous theoretical derivation |
| Value | 4 | Plug-and-play masking strategy with zero extra parameter overhead |
| Overall | 4.4 | A model exemplar combining theory and practice |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Balanced Learning for Domain Adaptive Semantic Segmentation](balanced_learning_for_domain_adaptive_semantic_segmentation.md)
- [\[CVPR 2026\] Masked Representation Modeling for Domain-Adaptive Segmentation](../../CVPR2026/segmentation/mrm_masked_representation_modeling_domain_adaptive.md)
- [\[NeurIPS 2025\] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](../../NeurIPS2025/segmentation/towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)
- [\[CVPR 2025\] Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](../../CVPR2025/segmentation/dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)
- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](../../CVPR2026/segmentation/heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)

</div>

<!-- RELATED:END -->
