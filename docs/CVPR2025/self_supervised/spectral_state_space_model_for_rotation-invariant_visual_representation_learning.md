---
title: >-
  [Paper Note] Spectral State Space Model for Rotation-Invariant Visual Representation Learning
description: >-
  [CVPR 2025][Self-Supervised Learning][State Space Models] Proposed Spectral VMamba, which orders the patch traversal sequence using eigenvectors of the spectral graph Laplacian (instead of predefined scan lines) and combines it with a Rotation Feature Normalizer (RFN, aggregating features of 4 canonical rotations) to achieve 87.86% accuracy on miniImageNet with complete invariance to canonical rotations.
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "State Space Models"
  - "Rotation Invariance"
  - "Spectral Graph Traversal"
  - "VMamba"
  - "Graph Laplacian"
date: 2026-05-08
content_hash: 4d038c7b511a1b85
---

# Spectral State Space Model for Rotation-Invariant Visual Representation Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.06369](https://arxiv.org/abs/2503.06369)  
**Code**: Yes  
**Area**: Self-Supervised Learning / Vision Architectures  
**Keywords**: State Space Models, Rotation Invariance, Spectral Graph Traversal, VMamba, Graph Laplacian

## TL;DR

Proposed Spectral VMamba, which orders the patch traversal sequence using eigenvectors of the spectral graph Laplacian (instead of predefined scan lines) and combines it with a Rotation Feature Normalizer (RFN, aggregating features of 4 canonical rotations) to achieve 87.86% accuracy on miniImageNet with complete invariance to canonical rotations.

## Background & Motivation

**Background**: Vision Mamba (VMamba) flattens images into sequences to process them using State Space Models. However, the flattening order (e.g., raster/zigzag scanning) depends on the spatial orientation of the image. When an image is rotated by 90°, the scanning sequence changes entirely, leading to inconsistent features.

**Limitations of Prior Work**: Self-attention in ViTs is permutation-invariant (encoding spatial relationships via position embeddings), but SSMs/Mamba depend on sequence order, which is an inherent limitation of sequence models. Even though VMamba uses multi-directional scanning, its features for rotated images remain unstable (accuracy can drop by 30+% for 90° rotations).

**Key Challenge**: SSMs require a fixed sequence order, but rotation changes the spatial arrangement $\rightarrow$ sequence order $\rightarrow$ extracted features.

**Key Insight**: Define a rotation-invariant traversal order using spectral decomposition of graphs. By modeling the similarity between patches as a graph adjacency matrix, the sorting of eigenvectors of the graph Laplacian is independent of rotation (as rotation does not change the relative relationships between patches).

**Core Idea**: Sort patches using eigenvectors of the spectral graph Laplacian $\rightarrow$ rotation-invariant traversal $\rightarrow$ SSM robust to rotation.

## Method

### Key Designs

1. **Spectral Traversal Scan (STS)**:

    - Function: Generates a rotation-invariant patch traversal order.
    - Mechanism: Constructs a k-NN adjacency graph $\mathbf{W}$ for image patches, computes the symmetric normalized Laplacian $\mathbf{L}_{sym} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{W}\mathbf{D}^{-1/2}$, and sorts the patches according to the primary $m$ eigenvectors ordered by their eigenvalues. Since the eigenvalues of the Laplacian depend only on the graph topology (rather than spatial orientation), rotated images produce the same sorting order.
    - Design Motivation: Spectral clustering theory guarantees rotation invariance of the eigenvector ordering (exactly invariant under canonical rotations).

2. **Rotation Feature Normalizer (RFN)**:

    - Function: Handles non-canonical rotation angles that STS cannot cover.
    - Mechanism: Rotates the image by 4 canonical angles {0°, 90°, 180°, 270°}, patchifies and extracts features for each, and applies a patch-wise max pooling: $\mathbf{F}_{i,j} = \max_{r \in \{1,...,4\}} [\mathcal{R}_{-\theta_r}(\text{Patchify}(\mathcal{R}_{\theta_r}(\mathbf{I})))]_{i,j}$.
    - Design Motivation: Since STS is exactly invariant under canonical rotations, RFN further eliminates patchify boundary effects.

### Loss & Training

Standard supervised classification training. The computational overhead of spectral decomposition is extremely small (~2MB FLOPs, as the number of patches is only 196). Optimal hyperparameters: $m=4$ eigenvectors, $k=5$ nearest neighbors.

## Key Experimental Results

### Main Results

| Model | 0° Accuracy | 90° Accuracy | 180° Accuracy |
|------|----------|-----------|------------|
| VMamba-T | 86.25% | ~55% | ~60% |
| **Spectral VMamba-T** | **87.86%** | **~87%** | **~87%** |

### Ablation Study

| Configuration | 0° | 90° |
|------|-----|------|
| VMamba + RFN (w/o STS) | 86.5% | 52% |
| STS (w/o RFN) | 87.5% | 85% |
| **STS + RFN** | **87.86%** | **~87%** |

### Key Findings
- **STS is the core of rotation invariance**: Improving performance from 55% to 85% under 90° rotation.
- **RFN complements patchify boundary effects**: Providing an additional 2% improvement.
- **Improvement on 0° as well**: 87.86% vs 86.25%, indicating spectral sorting itself is superior to raster scanning.

## Highlights & Insights
- **Theoretical Elegance**—Translates the rotation invariance problem into spectral invariance in graph theory, with solid mathematical foundations.
- **Fundamental Improvement for SSMs**—Addresses the rotation sensitivity issue of all sequence models (not just Mamba) in spatial tasks.

## Limitations & Future Work
- Exact invariance is only achieved under the 4 canonical rotations; degradation still occurs at non-canonical angles (~78%).
- Spectral decomposition does not handle graphs with disconnected components well.
- Only evaluated on miniImageNet; exploration on large-scale datasets is left for future work.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Cross-disciplinary innovation of spectral graph theory × SSM.
- Experimental Thoroughness: ⭐⭐⭐ Only evaluated on miniImageNet, limited scale.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical exposition.
- Value: ⭐⭐⭐⭐ Provides an elegant solution to a fundamental limitation of SSMs in spatial tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Exemplar-Free Continual Learning for State Space Models](../../CVPR2026/self_supervised/exemplar-free_continual_learning_for_state_space_models.md)
- [\[ICCV 2025\] Scaling Language-Free Visual Representation Learning](../../ICCV2025/self_supervised/scaling_languagefree_visual_representation_learning.md)
- [\[ICLR 2026\] Unsupervised Representation Learning - An Invariant Risk Minimization Perspective](../../ICLR2026/self_supervised/unsupervised_representation_learning_-_an_invariant_risk_minimization_perspectiv.md)
- [\[ICLR 2026\] CARL: Camera-Agnostic Representation Learning for Spectral Image Analysis](../../ICLR2026/self_supervised/carl_camera-agnostic_representation_learning_for_spectral_image_analysis.md)
- [\[CVPR 2026\] Weight Space Representation Learning via Neural Field Adaptation](../../CVPR2026/self_supervised/weight_space_representation_learning_via_neural_field_adaptation.md)

</div>

<!-- RELATED:END -->
