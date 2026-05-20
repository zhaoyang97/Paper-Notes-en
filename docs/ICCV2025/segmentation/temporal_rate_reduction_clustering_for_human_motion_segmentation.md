---
title: >-
  [Paper Note] Temporal Rate Reduction Clustering for Human Motion Segmentation
description: >-
  [ICCV 2025][Segmentation][human motion segmentation] This paper proposes Temporal Rate Reduction Clustering (TR²C), which integrates the Maximal Coding Rate Reduction (MCR²) principle with temporal continuity regularizat…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "human motion segmentation"
  - "subspace clustering"
  - "maximal coding rate reduction"
  - "temporal consistency"
  - "unsupervised temporal clustering"
date: 2026-05-08
content_hash: 6b88ed957e18f8f9
---

# Temporal Rate Reduction Clustering for Human Motion Segmentation

**Conference**: ICCV 2025
**arXiv**: [2506.21249](https://arxiv.org/abs/2506.21249)  
**Code**: [GitHub](https://github.com/mengxianghan123/TR2C)  
**Area**: Image Segmentation
**Keywords**: human motion segmentation, subspace clustering, maximal coding rate reduction, temporal consistency, unsupervised temporal clustering

## TL;DR

This paper proposes Temporal Rate Reduction Clustering (TR²C), which integrates the Maximal Coding Rate Reduction (MCR²) principle with temporal continuity regularization to jointly learn temporally consistent representations and affinity matrices conforming to the Union of Subspaces (UoS) distribution, achieving substantial state-of-the-art improvements on five HMS benchmarks.

## Background & Motivation

Human Motion Segmentation (HMS) aims to partition a sequence of video frames into distinct, non-overlapping motion segments. Due to the high cost of annotation, HMS is typically formulated as an unsupervised temporal clustering task. Existing methods are largely grounded in the subspace clustering assumption, where video frame features are approximated as lying on a Union of Subspaces (UoS).

The paper identifies the following core bottlenecks in prior work:

**Data-assumption mismatch**: Features extracted from videos with complex human motions and cluttered backgrounds rarely conform well to the UoS distribution. Existing representation learning approaches (e.g., auto-encoders, graph consistency methods) attempt to learn better features, but **there is no evidence that the learned representations genuinely align with UoS structure**.

**Insufficient exploitation of temporal priors**: Although methods such as OSC and TSC leverage the prior that adjacent frames likely belong to the same motion, their effectiveness is limited when the feature space is misaligned.

**Limitations of transfer learning approaches**: Cross-domain alignment strategies have been introduced, yet performance bottlenecks persist, fundamentally because the distribution alignment problem is not addressed at the representation level.

The key insight of this paper is that **jointly learning** UoS-structured representations and segmentation affinity matrices—while incorporating temporal consistency constraints—allows the learned features to naturally align to the desired geometric structure during optimization.

## Method

### Overall Architecture

TR²C comprises three network components: an encoder $f(\cdot)$, a feature head $g(\cdot)$, and a clustering head $h(\cdot)$. Input features are mapped to a shared representation via the encoder, which is then passed to the feature head and clustering head to produce a structured representation $\boldsymbol{Z}$ and an affinity matrix $\boldsymbol{\Gamma}$, respectively. Spectral clustering is subsequently applied to $\boldsymbol{\Gamma}$ to obtain the final segmentation.

### Key Designs

1. **MCR² principle for joint representation and segmentation learning**: Based on the Maximal Coding Rate Reduction principle, the core optimization objective consists of three terms. $\rho(\boldsymbol{Z}, \epsilon)$ denotes the total coding rate, measuring the overall volume of the representation (via the $\log\det$ function); $\rho^c(\boldsymbol{Z}, \epsilon | \boldsymbol{\Pi})$ denotes the sum of within-class coding rates. Maximizing the total volume while minimizing the within-class volume naturally drives the representations toward a UoS distribution with mutually orthogonal subspaces. This is the first application of the MCR² principle to temporal sequence clustering. The geometric intuition is that $\log\det(\cdot)$, as a concave relaxation of $\text{rank}(\cdot)$, effectively measures the volume of the representation space.

2. **Temporal Laplacian regularization**: A temporal graph Laplacian regularization term $r(\boldsymbol{Z}) = \text{tr}(\boldsymbol{Z}\boldsymbol{L}\boldsymbol{Z}^\top)$ is introduced, where $\boldsymbol{L}$ is the graph Laplacian constructed via a sliding window of size $s$. This regularization encourages adjacent frames to maintain similar representations, thereby enforcing temporal consistency. The design motivation is that optimizing MCR² alone disregards the temporal continuity of video frames, potentially causing adjacent frames belonging to the same motion to be assigned to different subspaces.

3. **Total coding rate maximization to prevent collapse**: Directly minimizing $\rho^c + \lambda r(\boldsymbol{Z})$ admits a trivial solution (collapse of all embeddings), analogous to over-smoothing in graph neural networks. The term $-\rho(\boldsymbol{Z}, \epsilon)$ is therefore incorporated as regularization to prevent excessive compression of representations by maximizing the total coding rate. The final optimization objective is:

$$\min_{\boldsymbol{Z}, \boldsymbol{\Pi}} -\rho(\boldsymbol{Z}, \epsilon) + \lambda_1 \rho^c(\boldsymbol{Z}, \epsilon | \boldsymbol{\Pi}) + \lambda_2 r(\boldsymbol{Z})$$

subject to $\|\boldsymbol{z}_i\|_2^2 = 1$.

4. **Differentiable optimization framework**: The discrete assignment matrix $\boldsymbol{\Pi}$ is relaxed to a doubly stochastic affinity matrix $\boldsymbol{\Gamma}$ via Sinkhorn projection to satisfy the constraints. Both $\boldsymbol{Z}$ and $\boldsymbol{\Gamma}$ are parameterized by the network and updated via backpropagation, enabling end-to-end differentiable training.

### Loss & Training

The final loss function is the sum of three terms:

$$\mathcal{L} = -\mathcal{L}_\rho + \lambda_1 \mathcal{L}_{\bar{\rho}^c} + \lambda_2 \mathcal{L}_r$$

- $\mathcal{L}_\rho$: maximizes total coding rate to prevent representation collapse
- $\mathcal{L}_{\bar{\rho}^c}$: minimizes within-class coding rate to promote subspace separation
- $\mathcal{L}_r$: temporal Laplacian regularization to maintain temporal consistency

The network architecture is lightweight (two-layer MLP encoder with FC heads). $\lambda_1$ and $\lambda_2$ are tuned independently per dataset; the sliding window is fixed at $s=2$; training runs for 500 iterations.

## Key Experimental Results

### Main Results

Comparison on five HMS benchmarks using HoG features (ACC / NMI):

| Method | Weiz ACC | Keck ACC | UT ACC | MAD ACC | YouTube ACC |
|--------|---------|---------|--------|---------|------------|
| TSC | 61.11 | 47.81 | 53.40 | 55.56 | 90.40 |
| CDMS (transfer learning) | 65.05 | 62.07 | 66.43 | 65.36 | 67.98 |
| GCTSC (Prev. SOTA) | 85.01 | 78.64 | 87.00 | 82.97 | 95.79 |
| **TR²C (Ours)** | **94.12** | **83.50** | **93.54** | **83.08** | **97.96** |

Without employing transfer learning, TR²C outperforms transfer learning-based methods by approximately 20% in clustering accuracy and surpasses the previous state-of-the-art GCTSC by 5–9 percentage points.

### Ablation Study

| Loss Configuration | $\mathcal{L}_\rho$ | $\mathcal{L}_{\bar{\rho}^c}$ | $\mathcal{L}_r$ | Weiz ACC | Keck ACC | UT ACC |
|-------------------|:---:|:---:|:---:|---------|---------|--------|
| MCR² only (no temporal) | ✓ | ✓ | × | 37.30 | 47.29 | 45.79 |
| w/o total coding rate | × | ✓ | ✓ | 53.14 | 47.91 | 63.13 |
| w/o within-class coding rate | ✓ | × | ✓ | 64.68 | 58.60 | 65.67 |
| **Full TR²C** | ✓ | ✓ | ✓ | **94.07** | **86.78** | **94.05** |

All three loss terms are indispensable. Removing $\mathcal{L}_\rho$ leads to excessive representation compression; removing $\mathcal{L}_{\bar{\rho}^c}$ results in over-segmentation; removing $\mathcal{L}_r$ eliminates temporal consistency.

### Key Findings

- **Representation quality**: PCA visualization shows that raw HoG features lie on a one-dimensional manifold and cannot be clearly segmented, whereas TR²C-learned representations exhibit a distinct orthogonal UoS structure.
- **Robustness**: Under Gaussian noise perturbation, the clustering accuracy of TR²C degrades by at most 15%, compared to 45% for GCTSC, demonstrating that UoS alignment confers significant noise robustness.
- **CLIP features**: Replacing HoG with CLIP pre-trained features, TR²C+CLIP achieves 96.32 on Weiz and 90.86 on Keck.
- **Computational efficiency**: With GPU acceleration, TR²C is more than 100× faster than GCTSC (YouTube dataset: 41s vs. 8475s).

## Highlights & Insights

- **Theoretical contribution**: This is the first work to extend the MCR² principle to temporal clustering, adapting it for HMS via temporal regularization and collapse prevention mechanisms.
- **Clear geometric interpretation**: The $\log\det$ function measures subspace volume, giving the optimization objective a well-defined geometric meaning—maximize total volume, minimize within-class volume.
- **Simplicity and effectiveness**: The network architecture consists of only a two-layer MLP with FC heads, trains rapidly, and substantially outperforms prior methods.

## Limitations & Future Work

- Validation is limited to HoG and CLIP features; end-to-end learning directly from raw video frames has not been explored.
- Datasets are relatively small (hundreds to thousands of frames); scalability to long videos or large-scale data remains to be verified.
- Hyperparameters $\lambda_1$ and $\lambda_2$ require per-dataset tuning; adaptive strategies are worth exploring.
- The method is validated only on HMS; generalization to broader temporal segmentation tasks (e.g., action recognition, activity detection) warrants investigation.

## Related Work & Insights

- MCR² (Ma et al.) proposed the coding rate reduction principle for supervised learning; MLC extended it to unsupervised clustering; this paper further generalizes it to temporal settings.
- TSC introduced temporal graph Laplacian regularization; TR²C builds upon this by additionally incorporating representation learning.
- Future work could explore combining TR²C with video foundation models (e.g., VideoMAE) for stronger feature extraction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to combine the MCR² principle with temporal clustering, with both theoretical and methodological innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across five benchmarks, including ablation studies, visualizations, robustness analysis, and multi-feature assessment.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous and motivation is clearly articulated, though the notation is dense.
- **Value**: ⭐⭐⭐⭐ Provides a novel theoretical framework for temporal clustering and substantially advances the performance frontier of HMS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation](skeleton_motion_words_for_unsupervised_skeleton-based_temporal_action_segmentati.md)
- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](move_motion-guided_few-shot_video_object_segmentation.md)
- [\[ICCV 2025\] What If: Understanding Motion Through Sparse Interactions](what_if_understanding_motion_through_sparse_interactions.md)
- [\[ICCV 2025\] A Plug-and-Play Physical Motion Restoration Approach for In-the-Wild High-Difficulty Motions](a_plugandplay_physical_motion_restoration_approach_for_inthe.md)
- [\[ICCV 2025\] 2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos](2handedafforder_learning_precise_actionable_bimanual_affordances_from_human_vide.md)

</div>

<!-- RELATED:END -->
