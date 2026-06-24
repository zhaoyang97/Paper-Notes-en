---
title: >-
  [Paper Note] Regor: Progressive Correspondence Regenerator for Robust 3D Registration
description: >-
  [CVPR 2025][Point Cloud Registration] Regor proposes a progressive correspondence regeneration strategy. Unlike traditional "top-down" outlier removal methods, it iteratively generates high-quality correspondences in local spheres using a "bottom-up" approach. The number of generated correct matches is up to 10 times that of existing methods, achieving robust registration even under weak feature conditions.
tags:
  - "CVPR 2025"
  - "Point Cloud Registration"
  - "Correspondence Regeneration"
  - "Outlier Removal"
  - "Tri-Point Consistency"
  - "Progressive Iteration"
date: 2026-05-08
content_hash: bc30d719180b3baa
---

# Regor: Progressive Correspondence Regenerator for Robust 3D Registration

**Conference**: CVPR 2025  
**arXiv**: [2502.02163](https://arxiv.org/abs/2502.02163)  
**Code**: Coming soon  
**Area**: Others (3D Registration)  
**Keywords**: Point Cloud Registration, Correspondence Regeneration, Outlier Removal, Tri-Point Consistency, Progressive Iteration

## TL;DR

Regor proposes a progressive correspondence regeneration strategy. Unlike traditional "top-down" outlier removal methods, it iteratively generates high-quality correspondences in local spheres using a "bottom-up" approach. The number of generated correct matches is up to 10 times that of existing methods, achieving robust registration even under weak feature conditions.

## Background & Motivation

Point cloud registration is a fundamental task in 3D computer vision, aiming to estimate the rigid transformation between two point clouds. In scenarios with low overlap, high noise, and strong self-similarity, feature matching often generates a massive number of outliers.

Fundamental limitations of prior work:
- **Geometric methods** (RANSAC and its variants, SC2-PCR): Filter correspondences through geometric consistency, but struggle to identify inliers under extreme outlier ratios.
- **Learning-based methods** (PointDSC, VBReg): Treat outlier removal as a classification problem, but essentially still rely on the quality of initial correspondences.
- **Common deficiency**: All methods perform "subtraction"—removing outliers from the initial correspondence set. When initial inliers are extremely sparse, even perfectly identifying all of them is insufficient to support robust pose estimation.

Key Insight: Instead of "removing wrong matches", it is better to "generate more correct matches"—shifting the paradigm from subtraction to addition.

## Method

### Overall Architecture

Regor adopts a progressive iterative framework where each iteration contains three modules: (1) Prior-guided local grouping + generalized mutual matching, which generates new correspondences in local regions; (2) Center-aware tri-point consistency for local correspondence correction; (3) Global correspondence refinement to optimize from a holistic perspective. Through multiple iterations, the quantity and quality of correspondences are progressively increased.

### Key Designs

**Design 1: Prior-Guided Local Grouping + Generalized Mutual Matching (GMM)**

* **Function**: Efficiently generate new correspondences in small local spaces, avoiding the massive search space of global matching.
* **Mechanism**: Guided by the position prior of the previous iteration's correspondences $\mathcal{G}^{t-1}$, a nearest neighbor search is performed within radius $r^t$ to construct local region pairs $(\mathbf{P}_i^t, \mathbf{Q}_i^t)$. Within local regions, generalized mutual matching is proposed: compute bidirectional NN and MNN matching matrices, obtain the mutual matching matrix through the Hadamard product, and relax the strict mutual constraint using a logical OR operation: $\mathbf{M}^* = (\mathbf{M}_1^{P \to Q} \odot \mathbf{M}_2^{Q \to P}) \otimes (\mathbf{M}_1^{Q \to P} \odot \mathbf{M}_2^{P \to Q})$.
* **Design Motivation**: Due to high feature similarity in local areas, standard NN matching produces massive mismatches, whereas strict mutual NN matching might yield no correspondences. GMM balances precision and recall by relaxing strict mutual constraints. As the iteration progresses, the local sphere radius $r^t$ gradually decreases to ensure precise convergence.

**Design 2: Center-Aware Tri-Point Consistency (CTC) — Local Correspondence Correction**

* **Function**: Utilize seed point prior information to identify inliers and correct mismatches within local regions.
* **Mechanism**: Utilizing the high-accuracy prior of the seed correspondence $g_i^{t-1}$ (center point), CTC is computed as: $s_\Delta(g_j, g_k) = (s_\sigma(g_j, g_i^{t-1}) \cdot s_\sigma(g_i^{t-1}, g_k)) \| s_{\sigma/2}(g_j, g_k)$, where $s_\sigma(g_i, g_j) = \mathbb{1}(|\|\mathbf{p}_i - \mathbf{p}_j\|_2 - \|\mathbf{q}_i - \mathbf{q}_j\|_2| \leq \sigma)$. High-scoring pairs are used to estimate the local transformation $\mathbf{T}_i^t$, followed by a nearest neighbor search to correct mismatches.
* **Design Motivation**: Traditional two-point consistency cannot distinguish between mismatches with similar structures. Introducing a center point (accurate prior) to construct a tri-point constraint significantly improves discriminative power. An additional strict point-pair constraint $s_{\sigma/2}$ prevents error propagation in case of inaccurate seed points.

**Design 3: Progressive Iteration + Global Refinement — Simultaneous Improvement of Density and Accuracy**

* **Function**: Progressively increase inlier count and improve accuracy through multiple iterations.
* **Mechanism**: In each iteration, local regeneration and correction are performed first, followed by merging local correspondences into global ones using a hash table: $\mathcal{G}^t = \bigcup_{i=1}^n \mathcal{G}_i^t$. The global stage utilizes second-order consistency to identify high-quality correspondences and correct abnormal matches. The local sphere radius decreases as iterations progress.
* **Design Motivation**: Single-shot matching is constrained by search space and feature quality. The progressive strategy allows each round to utilize the improved results of the previous round, forming a positive feedback loop. Local operations drastically reduce problem scale and computational overhead.

### Loss & Training

Regor is a non-learning-based method and requires no training. The final pose estimation is directly solved via least squares from the refined correspondences using SVD.

## Key Experimental Results

### Main Results: Registration on 3DMatch Dataset

| Method | Registration Recall (%) | Number of Correct Correspondences |
|------|-------------|------------|
| RANSAC | Baseline | Baseline |
| SC2-PCR | High | N |
| PointDSC | High | N |
| MAC | High | N |
| **Regor** | **SOTA** | **~10×N** |

(Note: The number of correct correspondences generated by Regor is 10 times that of outlier removal methods)

### Ablation Study: Contribution of Each Module

| Configuration | Effect |
|------|------|
| Without GMM (using standard mutual matching) | Significant reduction in local matches |
| Without CTC (using standard second-order consistency) | Decrease in local accuracy |
| Without global refinement | Insufficient cross-region consistency |
| Fewer iterations | Decreased quantity and accuracy of correspondences |

### Key Findings

- Regor achieves SOTA performance on both 3DMatch and KITTI datasets, with significant advantages especially in low-overlap scenarios.
- The number of generated correct correspondences is **10 times** that of outlier removal methods, which is critical for robust pose estimation.
- Even when using weak feature descriptors (non-SOTA feature extractors), Regor can still achieve robust registration.
- Quantity and accuracy of correspondences improve simultaneously during progressive iterations.

## Highlights & Insights

1. **Paradigm Shift**: Shifting from "subtraction" (outlier removal) to "addition" (correspondence regeneration), fundamentally resolving the issue of sparse initial inliers.
2. **Prior-Guided Progressive Strategy**: The output of each iteration serves as the prior for the next, establishing a virtuous cycle of self-improvement.
3. **High Practical Value**: Independent of specific feature extractors, it can be plugged into any registration pipeline as a post-processing module.

## Limitations & Future Work

- The iterative process introduces additional computational overhead, although local operations help control complexity.
- Key hyperparameters (e.g., sphere radius decay strategy, threshold $\sigma$) require tuning.
- Future work can explore learning-guided iterative strategies to further improve efficiency and robustness.

## Related Work & Insights

- **SC2-PCR**: Outlier removal based on second-order spatial consistency; this work proposes center-aware tri-point consistency on top of it.
- **GeoTransformer**: A coarse-to-fine Transformer matching framework.
- **TEASER**: Truncated Least Squares (TLS)-based robust registration using translation-invariant measurements.
- Insight: When "subtraction" solutions to a problem fail, reflecting on whether to switch to "addition" provides a highly inspiring paradigm shift.

## Rating

⭐⭐⭐⭐ — The paradigm shift from outlier removal to correspondence regeneration is elegant and profound, and the 10$\times$ correct matches result is impressive. The methodology is clearly designed with well-justified motivations for each module. It offers a fresh perspective to the 3D registration field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Towards In-the-Wild 3D Plane Reconstruction from a Single Image](towards_in-the-wild_3d_plane_reconstruction_from_a_single_image.md)
- [\[CVPR 2025\] MagicArticulate: Make Your 3D Models Articulation-Ready](magicarticulate_make_your_3d_models_articulation-ready.md)
- [\[CVPR 2026\] Progressive Neural Architecture Generation](../../CVPR2026/others/progressive_neural_architecture_generation.md)
- [\[ICCV 2025\] EDFFDNet: Towards Accurate and Efficient Unsupervised Multi-Grid Image Registration](../../ICCV2025/others/edffdnet_towards_accurate_and_efficient_unsupervised_multi-grid_image_registrati.md)
- [\[CVPR 2026\] PAUL: Uncertainty-Guided Partition and Augmentation for Robust Cross-View Geo-Localization under Noisy Correspondence](../../CVPR2026/others/paul_uncertainty-guided_partition_and_augmentation_for_robust_cross-view_geo-loc.md)

</div>

<!-- RELATED:END -->
