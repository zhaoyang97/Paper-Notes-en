---
title: >-
  [Paper Note] Simple Unsupervised Knowledge Distillation With Space Similarity
description: >-
  [ECCV 2024][Model Compression][Unsupervised Knowledge Distillation] CoSS proposes that in unsupervised knowledge distillation, in addition to the conventional **feature-dimension cosine similarity**, an additional **space-dimension cosine similarity (Space Similarity)** loss is introduced. By transposing the feature matrix and aligning it along the dimension direction, this loss compensates for the loss of manifold structure information caused by $L_2$ normalization…
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Unsupervised Knowledge Distillation"
  - "Space Similarity"
  - "Manifold Learning"
  - "Cosine Similarity"
  - "Homeomorphism"
date: 2026-05-08
content_hash: 096eeee17fda4ed0
---

# Simple Unsupervised Knowledge Distillation With Space Similarity

**Conference**: ECCV 2024  
**arXiv**: [2409.13939](https://arxiv.org/abs/2409.13939)  
**Code**: None (not provided)  
**Area**: Model Compression / Knowledge Distillation  
**Keywords**: Unsupervised Knowledge Distillation, Space Similarity, Manifold Learning, Cosine Similarity, Homeomorphism

## TL;DR

CoSS proposes that in unsupervised knowledge distillation, in addition to the conventional **feature-dimension cosine similarity**, an additional **space-dimension cosine similarity (Space Similarity)** loss is introduced. By transposing the feature matrix and aligning it along the dimension direction, this loss compensates for the loss of manifold structure information caused by $L_2$ normalization, achieving SOTA on multiple UKD benchmarks in a minimalist manner.

## Background & Motivation

Self-supervised learning (SSL) has achieved excellent general representations on large models, but small models cannot fully benefit from SSL due to limited parameters. Unsupervised knowledge distillation (UKD) addresses this issue by transferring knowledge from a large self-supervised teacher to a small student. Existing UKD methods (such as SEED, BINGO, DisCo, SMD, etc.) usually construct handcrafted similarity relationships between samples for distillation, which may ignore other crucial structural information in the teacher's mapping.

**Key Challenge**: All existing UKD methods rely on $L_2$-normalized features for operations (to compute cosine similarity), but $L_2$ normalization is an **irreversible mapping**: it projects all points onto a hypersphere, destroying the rich structure of the original manifold. Specifically, all points on the same ray from the origin are mapped to the same hypersphere point, rendering operations after normalization unable to recover the topological structure of the original embedded manifold.

**Mathematical Proof**: $L_2$ normalization is not a homeomorphism because it is not a continuous bijection and lacks a continuous inverse mapping. Therefore, merely minimizing the objective function on normalized features cannot preserve the original ununormalized manifold structure.

**Key Insight**: Rather than handcrafting sample relationships to be preserved, it is better to directly let the student model the teacher's embedding manifold. If two manifolds are similar, all inter-sample relationships are naturally preserved indirectly.

**Core Idea**: Use transposing of the feature matrix to compute cosine similarity in the space dimension (Space Similarity), recovering the structural information lost by normalization, which is complementary to conventional feature similarity.

## Method

### Overall Architecture

CoSS is a two-stage UKD framework:
1. **Offline Preprocessing**: Computes the $k$-nearest neighbors index of the training set using the teacher model.
2. **Distillation Training**: Jointly optimizes Feature Similarity and Space Similarity on a mini-batch augmented with neighbor samples.

### Key Designs

1. **Offline k-Nearest Neighbors Preprocessing**:

    - **Function**: Uses the teacher encoder to compute $k$ nearest neighbors for each sample in the training set.
    - **Mechanism**: First uses the teacher to generate $L_2$-normalized features for all training samples, computes the similarity matrix $S_{ij} = \hat{f}_t(x_i) \cdot \hat{f}_t(x_j)$, and then takes the top-k as the neighbor set $\Omega_i^k = \arg\max(S_{i\cdot}, k)$.
    - **Design Motivation**: In a standard randomly sampled mini-batch, local neighborhood information is missing. By appending neighbor samples into the batch, the student can capture the local structure of the manifold. This is essential for manifold modeling—not only matching global structure but also preserving local details.

2. **Feature Similarity (Cosine Similarity)**:

    - **Function**: Maximizes the cosine similarity between the normalized feature vectors of the teacher and student for each sample.
    - **Mechanism**:

    $\mathcal{L}_{co} = -\frac{1}{bk} \sum_{i=0}^{bk} \text{cosine}(\hat{A}_s^i, \hat{A}_t^i)$

   where $\hat{A}^i$ is the $L_2$-normalized feature vector of sample $i$. This is a standard loss widely used in UKD.
    - **Design Motivation**: Ensures that the teacher and student have consistent representation directions for the same sample, performing alignment on the normalized manifold. However, relying on this alone cannot recover the structure prior to normalization.

3. **Space Similarity**:

    - **Function**: Transposes the feature matrix and then computes the cosine similarity in the **space dimension** (i.e., the sample response vector corresponding to each dimension of the feature).
    - **Mechanism**: Constructs the transposed matrix $Z = A^T$ (dimension $d \times bk$), and computes after normalization:

    $\mathcal{L}_{ss} = -\frac{1}{d} \sum_{i=0}^{d} \text{cosine}(\hat{Z}_s^i, \hat{Z}_t^i)$

   Here, each $Z^i$ is the response vector of the $i$-th dimension of the feature space across all samples in the batch. Minimizing this loss matches the response patterns of each feature dimension to the samples between the teacher and the student.
    - **Mathematical Guarantee**: Normalization along the space dimension scales all data points under the same dimension identically, thus preserving bijectivity and continuity. When $\mathcal{L}_{ss}$ is minimized, $f_s(x_i) = \frac{\alpha}{\beta} f_t(x_i)$, where $\alpha, \beta > 0$ are dimension-level scaling vectors, making the mapping continuous, bijective, and invertible, which satisfies the homeomorphism condition.
    - **Design Motivation**: Recovers the structural information lost in the feature dimension due to $L_2$ normalization. Feature Similarity ensures consistent sample-level alignment directions, while Space Similarity ensures consistent dimension-level response patterns, with the two complementarily covering the complete manifold structure.

### Loss & Training

The final loss function is extremely simple:

$$\mathcal{L}_{CoSS} = \mathcal{L}_{co} + \lambda \mathcal{L}_{ss}$$

where $\lambda = 1.0$ (ablation studies show consistent effects for $\lambda \in \{0.5, 1.0\}$). The overall loss is scaled by a factor of 70.0 (as slow convergence was observed without scaling).

Training setup: $k=15$ neighbors, $N=31$ neighbor candidates, batch size $B=64$, initial learning rate of 0.03 with cosine learning rate decay, distilled for 25 epochs (using 4 GPUs), utilizing mocov2_aug augmentation. The teacher is a Moco-v2 pre-trained ResNet-50, and the student is ResNet-18/34 or EfficientNet-B0, with a projection head added to the student to align the output dimension with the teacher's 2048 dimensions.

**Advantage of minimalist design**: Compared to existing UKD methods, CoSS **does not require** feature queues, contrastive learning objectives, or heavy data augmentations.

## Key Experimental Results

### Main Results

| Student | Metric | CoSS | DisCo | BINGO | SEED | Moco-v2(baseline) |
|---------|------|------|-------|-------|------|-------------------|
| ResNet-18 | Top-1 | **62.35** | 60.60 | 61.40 | 57.60 | 52.20 |
| ResNet-18 | KNN-10 | **53.78** | 52.03 | 54.16 | 50.12 | 36.70 |
| ResNet-34 | Top-1 | **64.01** | 62.50 | 63.50 | 58.50 | 56.80 |
| EfficientNet-B0 | Top-1 | **67.36** | 66.50 | 63.74 | 61.30 | 42.20 |
| EfficientNet-B0 | KNN-10 | **58.33** | 54.78 | 54.75 | 53.11 | 30.00 |

EfficientNet-B0 has only 4M parameters (16.3% of the teacher) yet achieves a top-1 accuracy of 67.36%, which is only 0.04% lower than the teacher (67.40%), almost catching up.

### Ablation Study

| Configuration | Top-1 (R18) | KNN-10 (R18) | Explanation |
|------|------------|-------------|------|
| $\mathcal{L}_{co}$ only ($\lambda=0$) | ~60.0 | ~51.5 | Equivalent to SimReg |
| $\mathcal{L}_{ss}$ only | Slightly lower | Slightly lower | Space similarity alone is insufficient |
| CoSS ($\lambda=1.0$) | **62.35** | **53.78** | Complementary combination performs best |
| $k=0$ (No neighbor sampling) | Decreased | Decreased | Local information is important for manifold modeling |
| $k=15$ (Neighbor sampling) | **62.35** | **53.78** | Optimal configuration |

### Key Findings

- **Space Similarity and Feature Similarity are complementary**: Using either alone is inferior to their joint use, verifying that the manifold structure needs to be constrained from both the feature and space directions simultaneously.
- **Neighbor sampling is crucial**: There is a significant gap between $k=0$ (no neighbors) vs $k=15$, indicating that capturing local manifold structure relies on neighbor information.
- **CoSS outperforms SLD (Soft-label Distillation)**: ResNet-18 top-1 accuracy of 62.35% vs 59.88%, showing that directly modeling the manifold is more effective than mimicking the output distribution.
- **Generalization across teachers**: Distillation from ResNet-101 to ResNet-18 is also effective (63.74%), surpassing DisCo (62.30%).

## Highlights & Insights

- **Minimalist yet effective**: The core innovation is simply "transposing the feature matrix before calculating cosine similarity." Implementation requires only a single line of code (`transpose`), yet brings consistent improvements.
- **Solid theoretical motivation**: Rigorously argues the information loss issue of $L_2$ normalization from the perspective of homeomorphism, presenting a complete chain of reasoning.
- **No feature queue required**: Methods like SEED/BINGO need to maintain a feature queue of length ~100K, whereas CoSS does not require it at all, making training simpler and more efficient.
- **Transferable design principles**: The concept of Space Similarity can be easily embedded into any distillation framework based on cosine similarity, showing strong universality.

## Limitations & Future Work

- Only CNN architectures (ResNet, EfficientNet) are verified, without verification on Transformer architectures like ViTs (the authors note that AttnDistill focuses on ViTs, but did not compare against it).
- The teacher only uses Moco-v2 (an earlier SSL method), and stronger teachers (e.g., DINO, MAE) are not evaluated.
- Homeomorphic constraints only guarantee structure alignment "up to a scale"; stronger constraints (like isometric mapping) might bring further improvements.
- Neighbor precomputation requires extra storage of the entire similarity matrix, which might pose a bottleneck for ultra-large-scale datasets.
- Distillation was performed for only 25 epochs, and whether further scaling of training epochs yields additional gains remains unknown.

## Related Work & Insights

- **vs SEED**: SEED minimizes the divergence of the teacher-student similarity distribution over a shared embedding queue, requiring a large feature queue. CoSS needs no queue and is simpler.
- **vs BINGO**: BINGO is two-stage (clustering to construct a bag first, followed by contrastive distillation), which is complex and introduces noisy label issues. CoSS is single-stage.
- **vs DisCo**: DisCo employs consistency regularization across augmented views + contrastive distillation, while CoSS outperforms it without relying on contrastive objectives.
- **vs SimReg**: SimReg only uses $\mathcal{L}_{co}$. CoSS consistently outperforms it after adding $\mathcal{L}_{ss}$, proving that the gain of space similarity stems from essential complementarity rather than hyperparameter tuning.

## Rating

- Novelty: ⭐⭐⭐⭐ The core idea is simple yet deeply insightful, identifying the loss of normalization information from the perspective of manifold homeomorphism and providing an elegant fix.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on ImageNet classification, transfer learning, detection, and retrieval, along with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The motivation is clearly argued, originating from topologic concepts with a tight logical chain, offering excellent readability.
- Value: ⭐⭐⭐⭐ High practical value; the minimalist design is plug-and-play and highly inspiring for the UKD field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Improving Knowledge Distillation via Regularizing Feature Direction and Norm](improving_knowledge_distillation_via_regularizing_feature_direction_and_norm.md)
- [\[ECCV 2024\] A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging](a_simple_lowbit_quantization_framework_for_video_snapshot_co.md)
- [\[ICCV 2025\] Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning](../../ICCV2025/model_compression/soft_separation_and_distillation_toward_global_uniformity_in_federated_unsupervi.md)
- [\[ICCV 2025\] Cross-Architecture Distillation Made Simple with Redundancy Suppression](../../ICCV2025/model_compression/cross-architecture_distillation_made_simple_with_redundancy_suppression.md)
- [\[ICCV 2025\] Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification](../../ICCV2025/model_compression/competitive_distillation_a_simple_learning_strategy_for_improving_visual_classif.md)

</div>

<!-- RELATED:END -->
