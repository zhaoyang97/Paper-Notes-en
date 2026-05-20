---
title: >-
  [Paper Note] DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration
description: >-
  [CVPR 2026][Model Compression][Rigid registration] DualReg proposes a dual-space registration paradigm that progressively filters feature-space correspondences via lightweight 1-point RANSAC followed by 3-point RANSAC…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "Rigid registration"
  - "dual-space optimization"
  - "RANSAC"
  - "point cloud correspondence"
  - "geometric proxy"
date: 2026-05-08
content_hash: c61d0c56f8c70596
---

# DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration

**Conference**: CVPR 2026
**arXiv**: [2508.17034](https://arxiv.org/abs/2508.17034)  
**Code**: [https://ustc3dv.github.io/DualReg/](https://ustc3dv.github.io/DualReg/) (project page available)  
**Area**: Model Compression
**Keywords**: Rigid registration, dual-space optimization, RANSAC, point cloud correspondence, geometric proxy

## TL;DR

DualReg proposes a dual-space registration paradigm that progressively filters feature-space correspondences via lightweight 1-point RANSAC followed by 3-point RANSAC, then constructs geometric proxy point sets from the filtered anchor correspondences for joint dual-space optimization. The method achieves state-of-the-art accuracy on 3DMatch while running 32× faster than MAC.

## Background & Motivation

Rigid registration aims to estimate the rigid transformation (rotation + translation) from a source point cloud to a target point cloud, with broad applications in SLAM, robotics, 3D reconstruction, and AR/VR. Due to partial overlap, noise, and outliers in point clouds, robust and accurate registration remains a challenging problem.

**Two correspondence establishment paradigms each have shortcomings**:
- **Feature-space correspondences** (FPFH, FCGF, etc.): Global matching based on descriptor similarity handles large transformation differences, but correspondence accuracy is limited, precluding fine-grained alignment.
- **Local geometric-space correspondences** (ICP and variants): Nearest-neighbor search in Euclidean space enables high-accuracy local alignment, but is severely sensitive to initial transformation and prone to local optima under large transformation differences.

**The intuitive combination**—coarse feature matching followed by ICP refinement—also underperforms in low-overlap scenarios, since ICP's spatial proximity search introduces numerous erroneous nearest-point pairs.

**The computational cost of existing outlier rejection methods**: MAC achieves substantial accuracy gains through maximum clique search but at prohibitive computational cost. TCF introduces 1-point RANSAC for acceleration but at the expense of accuracy. A solution that is both fast and accurate is needed.

**Core Idea**: Rather than a serial coarse-to-fine pipeline, the paper constructs a **joint dual-space optimization framework** that simultaneously incorporates filtered feature-space correspondences and dynamically established local geometric-space correspondences **within a single objective function**, enabling the two sources of information to complement each other.

## Method

### Overall Architecture

Input source point cloud $\mathcal{V}$ and target point cloud $\mathcal{U}$ → Feature extraction (FPFH/FCGF) yields initial correspondences $\mathcal{C}_0$ → **Stage 1: 1-point RANSAC fast filtering** → **Stage 2: 3-point RANSAC refinement** → High-confidence correspondences $\mathcal{C}_{II}$ → **Geometric proxy set construction** → **Joint dual-space optimization** → Output optimal rigid transformation $(R^*, t^*)$

### Key Designs

1. **Lightweight 1-Point RANSAC Fast Filtering**:

    - Function: Rapidly eliminates a large proportion of outliers from the initial correspondence set $\mathcal{C}_0$.
    - Mechanism: Uses a **single correspondence** as the sampling unit (rather than the conventional 3 points), reducing sampling complexity from $\mathcal{O}(n^3)$ to $\mathcal{O}(n)$. For each sampled correspondence $\mathbf{c}_j$, a consistency set is defined as:
    $\mathcal{I}(\mathbf{c}_j) = \{\mathbf{c}_i \in \mathcal{C}_0 \mid D_L(\mathbf{c}_i, \mathbf{c}_j) < \tau \;\text{and}\; D_N(\mathbf{c}_i, \mathbf{c}_j) < \nu \}$
      where $D_L$ denotes length consistency (difference in inter-correspondence distances) and $D_N$ denotes normal consistency (tangential distance difference). An iterative search maintains a confidence score for each correspondence, and the consistency set with the highest cumulative score is selected.
    - Symmetry removal: When source and target points are symmetric about a plane, consistency constraints may be satisfied even when no valid rigid transformation exists in 3D space; reflection transformations are detected via SVD and filtered out.
    - Design Motivation: TCF also proposed 1-point RANSAC, but this work improves upon it in consistency set definition (incorporating normal consistency), optimal subset selection strategy (cumulative confidence scoring), and symmetry detection, achieving both speed and accuracy.

2. **Probability-Weighted 3-Point RANSAC Refinement**:

    - Function: Further purifies the correspondence set obtained after 1-point RANSAC filtering.
    - Mechanism: Assigns an inlier probability to each correspondence, updates probabilities via a dynamic Bayesian network, and employs probability-weighted sampling in place of random sampling. This substantially reduces unnecessary iterations compared to classical RANSAC.
    - Design Motivation: The length and normal constraints of 1-point RANSAC may still retain certain outliers; stricter 3-point transformation consistency verification is required for further refinement.

3. **Geometric Proxy Set Construction**:

    - Function: Extracts local neighborhoods from the original point cloud around filtered correspondences (anchors) to serve as proxy point sets.
    - Mechanism: For each anchor correspondence $\mathbf{c}_j = (\mathbf{v}_j, \mathbf{u}_j) \in \mathcal{C}_{II}$, the neighborhood within radius $\beta$ is extracted:
    $\mathcal{P}^s_{\mathbf{c}_j} = \{\mathbf{v}_i \in \mathcal{V} \mid \|\mathbf{v}_i - \mathbf{v}_j\|_2 < \beta\}$
    - Design Motivation: Compared to the full original point cloud, geometric proxy sets around anchor points exhibit **significantly higher overlap ratios**, addressing the fundamental weakness of ICP in low-overlap scenarios.

4. **Joint Dual-Space Optimization**:

    - Function: Simultaneously leverages feature-space and geometric-space correspondences to estimate the optimal transformation.
    - Core formulation:
    $E(\mathbf{R}, \mathbf{t}) = \frac{\lambda}{|\mathcal{C}_{II}|} \sum_{\mathbf{c}_j \in \mathcal{C}_{II}} w_j \|\mathbf{R}\mathbf{v}_j + \mathbf{t} - \mathbf{u}_j\|^2 + \frac{1}{|\mathcal{P}^s|} \sum_{\tilde{\mathbf{v}}_i \in \mathcal{P}^s} \tilde{w}_i \|\mathbf{R}\tilde{\mathbf{v}}_i + \mathbf{t} - \tilde{\mathbf{u}}_{\rho_i}\|^2$
      The first term is the alignment error over feature-space correspondences; the second term is the nearest-point alignment error over the geometric proxy set. Weights $w_j, \tilde{w}_i$ are robustified via a Gaussian function $\exp(-\|e\|^2 / 2\sigma^2)$.
    - Design Motivation: Feature correspondences provide global constraints to prevent local optima, while geometric correspondences provide fine-grained alignment accuracy—the two are complementary within a unified objective.

### Loss & Training

This is a non-learning method solved via alternating optimization:
1. Fix the transformation; update geometric correspondences $\{\rho_i\}$ via nearest-neighbor search.
2. Fix correspondences and transformation; compute robust weights using the Gaussian weighting function.
3. Fix correspondences and weights; solve for the optimal rigid transformation in closed form via SVD.
- Convergence criterion: $\|\mathbf{T}^{(k)} - \mathbf{T}^{(k-1)}\|_F < 0.001$ or a maximum of 200 iterations.

## Key Experimental Results

### Main Results

| Dataset | Metric | DualReg (FPFH) | MAC (FPFH) | MAC++ (FPFH) | Gain |
|--------|------|----------------|------------|--------------|------|
| 3DMatch | RR ↑ | 84.41% | 83.92% | 83.73% | +0.5% |
| 3DMatch | RMSE ↓ | 4.55cm | 4.94cm | 4.78cm | Best accuracy |
| 3DMatch | RE ↓ | 1.75° | 2.11° | 2.11° | −17% |
| 3DMatch | Time ↓ | 0.14s | 2.10s | 4.28s | **15–30× speedup** |
| 3DLoMatch | RMSE ↓ | 7.98cm | 9.18cm | 9.54cm | −13% |
| KITTI | RR ↑ | 98.20% | 97.48% | 98.02% | SOTA |
| KITTI | RMSE ↓ | 12.26cm | 15.57cm | 25.25cm | −21% |

### Ablation Study

| Configuration | 3DLoMatch RR/RMSE/Time | KITTI RR/RMSE/Time | Note |
|------|------------------------|---------------------|------|
| Full method | 41.7/7.98/0.11 | 98.2/12.26/0.12 | Best overall balance |
| w/o fast filtering | 29.1/7.31/0.68 | 84.0/12.81/0.78 | RR drops significantly; 6× slower |
| w/o refinement | 41.3/8.76/0.11 | 97.7/13.51/0.12 | Accuracy degrades; speed maintained |
| w/o dual-space optimization | 32.1/12.04/0.07 | 98.2/28.22/0.12 | RMSE increases drastically |
| w/o anchors | 41.6/7.99/0.13 | 98.0/19.16/0.13 | Accuracy drops without feature constraints |
| w/o geometric proxies | 37.0/9.51/0.11 | 98.2/16.36/0.13 | No fine geometric refinement |

Correspondence filtering module ablation (inlier ratio after filtering):

| Variant | 3DLoMatch FPFH | KITTI FPFH | Note |
|------|-----------------|------------|------|
| $D_L$ only | 33.00% | 45.65% | Basic length consistency |
| + $D_N$ | 33.55% | 46.17% | Normal consistency contributes modestly but stably |
| + Symmetry detection | 36.00% | 46.13% | Eliminates symmetric ambiguity |
| + Weighted sampling (ours) | 37.22% | 85.52% | Substantial gain on KITTI |

### Key Findings

- **Dual-space synergy is critical for accuracy**: Removing dual-space optimization degrades RMSE from 12.26cm to 28.22cm on KITTI, demonstrating that filtered feature correspondences alone are insufficient for high accuracy.
- **Optimal speed–accuracy trade-off**: DualReg completes 3DMatch registration in 0.14s on CPU, 15× faster than MAC and 30× faster than MAC++, while achieving superior accuracy.
- **Robust performance in low-overlap scenarios**: On 3DLoMatch (10%–30% overlap), RMSE of 7.98cm outperforms all compared methods.

## Highlights & Insights

1. **Precise problem formulation**: The paper identifies the complementarity between feature-space and geometric-space correspondences and formalizes them into a unified optimization objective, rather than a simple coarse-to-fine serial pipeline.
2. **Elegant improvements to 1-point RANSAC**: The introduction of normal consistency, cumulative confidence scoring, and symmetry detection enables the simplified sampling strategy to maintain high-quality filtering at very high speed.
3. **Design of geometric proxy point sets**: Extracting anchor neighborhoods serves as an effective bridge connecting feature matching and geometric alignment, cleanly addressing the core failure mode of ICP under low overlap.
4. **CPU-only implementation competitive with GPU methods**: The C++ implementation of DualReg achieves speed and accuracy on CPU comparable to learning-based methods requiring GPU.

## Limitations & Future Work

- Some dependence on feature descriptor quality—if initial descriptors are very poor, filtering may not retain sufficient inliers.
- The neighborhood radius $\beta$ of geometric proxies requires tuning based on point cloud density.
- Currently limited to rigid registration; extension to non-rigid scenarios requires additional consideration.
- No comparative evaluation combining with recent Transformer-based end-to-end registration methods (e.g., GeoTransformer).

## Related Work & Insights

- **MAC** [54]: Improves registration accuracy through maximum clique search on a compatibility graph, but at high computational cost; DualReg achieves comparable accuracy with significantly reduced computation.
- **TCF** [38]: First introduces 1-point RANSAC into registration but with degraded registration quality; DualReg compensates for the accuracy loss through an improved consistency set definition and dual-space optimization.
- **FRICP** [51]: Robust ICP using the Welsch function, but still constrained by initial transformation quality; DualReg's geometric proxy mechanism provides better initialization conditions.
- Insight: The joint dual-space optimization paradigm combining feature space and geometric space is generalizable to other tasks requiring global–local synergy, such as scene flow estimation and non-rigid registration.

## Rating

- Novelty: ⭐⭐⭐⭐ The dual-space joint optimization framework and improved 1-point RANSAC are valuable contributions, though individual components are relatively independent.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers indoor/outdoor datasets, multiple feature descriptors, comprehensive ablation studies, and runtime comparisons.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; experimental setup is fair and thorough.
- Value: ⭐⭐⭐⭐ Balances speed and accuracy, with direct practical value for real-world SLAM and 3D reconstruction systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CBRS: Cognitive Blood Request System with Bilingual Dataset and Dual-Layer Filtering](../../ACL2026/model_compression/cbrs_cognitive_blood_request_system_with_bilingual_dataset_and_dual-layer_filter.md)
- [\[ICLR 2026\] Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](../../ICLR2026/model_compression/null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)
- [\[CVPR 2026\] RL-ScanIQA: Reinforcement-Learned Scanpaths for Blind 360° Image Quality Assessment](rl-scaniqa_reinforcement-learned_scanpaths_for_blind_360image_quality_assessment.md)
- [\[CVPR 2026\] DAGE: Dual-Stream Architecture for Efficient and Fine-Grained Geometry Estimation](dage_dual-stream_architecture_for_efficient_and_fine-grained_geometry_estimation.md)
- [\[CVPR 2026\] Memory-Efficient Transfer Learning with Fading Side Networks via Masked Dual Path Distillation](memory_efficient_transfer_learning_with_fading_side_networks.md)

</div>

<!-- RELATED:END -->
