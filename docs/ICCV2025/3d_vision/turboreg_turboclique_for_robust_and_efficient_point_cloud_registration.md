---
title: >-
  [Paper Note] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration
description: >-
  [ICCV 2025][3D Vision][Point cloud registration] TurboReg is proposed, a framework that replaces traditional maximum clique enumeration with lightweight 3-cliques (TurboCliques) and introduces a highly parallelizable Pivot-Guided Search (PGS) algorithm, achieving state-of-the-art registration accuracy while delivering over 208× speedup.
tags:
  - ICCV 2025
  - 3D Vision
  - Point cloud registration
  - graph matching
  - maximum clique search
  - spatial consistency
  - robust estimation
date: 2026-05-08
content_hash: d40aaa0134423064
---

# TurboReg: TurboClique for Robust and Efficient Point Cloud Registration

**Conference**: ICCV 2025
**arXiv**: [2507.01439](https://arxiv.org/abs/2507.01439)
**Code**: [GitHub](https://github.com/ShaochengYan/TurboReg)
**Area**: 3D Vision
**Keywords**: Point cloud registration, graph matching, maximum clique search, spatial consistency, robust estimation

## TL;DR

TurboReg is proposed, a framework that replaces traditional maximum clique enumeration with lightweight 3-cliques (TurboCliques) and introduces a highly parallelizable Pivot-Guided Search (PGS) algorithm, achieving state-of-the-art registration accuracy while delivering over 208× speedup.

## Background & Motivation

Point cloud registration (PCR) is a fundamental task that aligns 3D scans from different viewpoints into a common coordinate system, with broad applications in SLAM, virtual reality, and beyond. Correspondence-based PCR first establishes 3D keypoint correspondences via feature matching, then identifies inliers and computes the optimal transformation through robust estimation.

Existing methods face a core tension: **accuracy vs. efficiency**.

**RANSAC-based methods**: Estimate transformations by randomly sampling hypothesis subsets; convergence is extremely slow under high outlier ratios, requiring millions of iterations.

**Deep learning methods** (e.g., PointDSC, VBReg): Improve efficiency by learning sampling probabilities, but suffer from limited generalization and high training costs.

**Graph-based methods (GPCR)**: Identify inliers by searching for maximum cliques on compatibility graphs. For instance, 3DMAC achieves breakthrough performance under low inlier ratios via maximum clique enumeration (MCE). However, MCE has two critical drawbacks:
- **Exponential time complexity**: $\mathcal{O}(d(N-d)3^{d/3})$, growing exponentially with the number of correspondences.
- **Parallelization difficulty**: Cliques vary in size and may grow unboundedly, making efficient parallelization challenging.

The authors' core insight is that maximum cliques are effective due to two mechanisms: (1) data-scale stability (more matches reduce variance) and (2) pairwise compatibility stability (spatial consistency constraints suppress noise). The key observation is that by tightening the compatibility threshold to strengthen the second form of stability, fixed-size lightweight 3-cliques can replace variable-size maximum cliques, enabling efficient parallel search.

## Method

### Overall Architecture

The TurboReg pipeline proceeds as follows: input correspondences → construct an Ordered SC2 Graph (O2Graph) → apply the PGS algorithm to search for TurboCliques → estimate transformations for each TurboClique using the Kabsch solver → select the highest-scoring transformation as the final output.

### Key Designs

1. **TurboClique (Lightweight Clique Structure)**:

    - Function: Defines a novel clique structure fixed at size 3 (exactly three mutually compatible correspondence pairs).
    - Mechanism: Grounded in variance analysis of least-squares estimation, $\text{Var}(\hat{\beta}|X) = \sigma^2(X'X)^{-1}$, the stability of transformation estimation depends on the noise level $\sigma$ and the number of observations $N$. Maximum cliques reduce variance by maximizing $N$, but at the cost of exponential search complexity. TurboClique takes the opposite approach: fixing $N=3$ (the minimum required for rigid transformation estimation) and then tightening the compatibility threshold $\tau$ to reduce $\sigma$, compensating for the loss of data-scale stability. Experiments show that $\tau = 0.25 \times \text{pr}$ (one quarter of the point cloud resolution) achieves a good balance between stability and recall.
    - Design Motivation: Fixed-size 3-cliques are naturally suited for parallel computation, and the tightened compatibility constraint makes three correspondences sufficient for estimating stable transformations.

2. **Pivot-Guided Search (PGS)**:

    - Function: Efficiently searches for TurboCliques with high inlier ratios.
    - Mechanism: The $K_1$ edges with the highest SC2 scores are selected as "pivots." Each pivot $\pi = (i,j)$ guides the search over its compatible neighbors $\mathcal{N}(i,j) = \{z | \hat{\mathbf{G}}_{iz} > 0 \wedge \hat{\mathbf{G}}_{jz} > 0\}$, forming TurboCliques $(i,j,z)$. For each pivot, the $K_2$ TurboCliques with the highest aggregated weights are retained, yielding $K_1 \cdot K_2$ candidates in total. The verification condition reduces to element-wise matrix multiplication: $\bar{\mathbf{G}}_{ij} \cdot \bar{\mathbf{G}}_{iz} \cdot \bar{\mathbf{G}}_{jz} > 0$, which is naturally amenable to GPU parallelization.
    - Design Motivation: Edges with high SC2 scores are more likely to connect inlier pairs, and high scores indicate greater local TurboClique density, facilitating the discovery of more high-quality candidates. The time complexity is only $\mathcal{O}(K_1 N)$—linear complexity with $K_1$ treated as a constant.

3. **Ordered SC2 Graph (O2Graph)**:

    - Function: Eliminates redundant detection of TurboCliques.
    - Mechanism: The SC2 graph is converted into a directed graph where edges only point from lower-index nodes to higher-index nodes, forming an upper-triangular matrix $\tilde{\mathbf{G}}_{ij} = 0$ for $i \geq j$. This guarantees that each TurboClique is detected by at most one pivot (Unique Assignment Property), avoiding redundant computation.
    - Design Motivation: On a naive SC2 graph, the same TurboClique may be discovered repeatedly by multiple pivots (e.g., $(1,2,4)$ could be found via both pivot $(1,2)$ and pivot $(1,4)$). O2Graph eliminates this redundancy entirely by enforcing directionality.

### Loss & Training

TurboReg is a training-free method. During the estimation stage, the Kabsch solver estimates a rigid transformation from the three correspondence pairs of each TurboClique. The optimal transformation is then selected by an inlier number (IN) metric: $\mathbf{T}^\star = \arg\max_{\mathbf{T}_z \in \mathcal{T}} g(\mathbf{T}_z)$.

Key hyperparameters: the number of pivots $K_1$ is set to 1K/2K (indoor) or 0.25K/0.5K (outdoor); the number of TurboCliques per pivot is $K_2 = 2$.

## Key Experimental Results

### Main Results

| Dataset | Metric (RR%) | TurboReg(1K) | TurboReg(2K) | 3DMAC | FastMAC@50 | SC2-PCR |
|--------|-----------|-------------|-------------|-------|------------|---------|
| 3DMatch+FPFH | RR | 83.92 | **84.10** | 83.92 | 82.87 | 83.73 |
| 3DMatch+FCGF | RR | **93.59** | **93.59** | 92.79 | 92.67 | 93.16 |
| 3DLoMatch+FPFH | RR | 40.76 | **40.99** | 40.88 | 38.46 | 38.57 |
| 3DLoMatch+FCGF | RR | 59.40 | **59.74** | 59.85 | 58.23 | 58.73 |
| 3DMatch+Predator | RR | **94.89** | 94.60 | 94.60 | 93.72 | 93.59 |
| 3DLoMatch+Predator | RR | **73.07** | 72.95 | 70.90 | 69.12 | 69.57 |

**Speed comparison (GPU FPS, 3DMatch)**: TurboReg(1K) = 61.25, FastMAC@20 = 26.32, SC2-PCR = 15.84; 3DMAC has no GPU implementation. TurboReg is **208×** faster than 3DMAC.

### Ablation Study

| Configuration | 3DMatch RR(%) | 3DLoMatch RR(%) | GPU FPS | Notes |
|------|--------------|----------------|---------|------|
| TurboReg w/ SC2 Graph | 93.59 | 59.40 | ~55 | Naive SC2 graph with redundant detection |
| TurboReg w/ O2Graph | 93.59 | 59.40 | 61.25 | Ordered graph eliminates redundancy, faster |
| $K_1=0.5$K | 92.85 | 57.50 | Faster | Too few pivots reduce recall |
| $K_1=1$K | **93.59** | **59.40** | 61.25 | Best balance |
| $K_1=2$K | 93.59 | 59.74 | 54.04 | More pivots slightly improve low-overlap performance |

### Key Findings

1. Under strict compatibility constraints, 3-cliques are sufficient to support high-quality registration without requiring maximum cliques.
2. The linear time complexity and high parallelism of PGS make TurboReg faster than all competing methods, achieving real-time performance on GPU (>60 FPS).
3. The advantage is more pronounced in low-overlap scenarios (3DLoMatch): TurboReg(2K) outperforms FastMAC@50 by 2.53% RR while being 10× faster.
4. On outdoor data (KITTI), TurboReg(0.25K) generates only 500 pivots yet achieves 98.56% RR, far fewer than the millions of hypotheses required by RANSAC.

## Highlights & Insights

1. **Theoretical elegance**: The complementary relationship between "reducing clique size" and "tightening compatibility threshold" is derived from variance analysis, providing a clear theoretical framework for future work.
2. **Strong practical value**: No training is required; real-time performance is achievable on CPU (2.73 FPS), making the method friendly for embedded deployment.
3. **Elegant parallel design**: TurboClique verification reduces to element-wise matrix multiplication, and PGS offers two levels of parallelism (pivot-level and search-level), making it highly amenable to GPU acceleration.

## Limitations & Future Work

1. The compatibility threshold $\tau$ must be adjusted according to point cloud resolution; adaptive strategies may be needed in scenes with large resolution variation.
2. RE and TE are slightly higher than some methods—because more challenging samples are recalled—which may require post-processing in precision-sensitive applications.
3. The fixed setting $K_2=2$ may lack flexibility in certain scenarios; dynamic adjustment strategies are worth exploring.
4. Performance under extremely high outlier ratios (>99%) is not discussed.

## Related Work & Insights

- **3DMAC**: Pioneered maximum clique enumeration for low-inlier registration, but exponential complexity is the bottleneck → TurboClique addresses this with fixed-size cliques.
- **SC2-PCR**: The concept of second-order compatibility is directly inherited by TurboReg and applied to pivot selection.
- **FastMAC**: Reduces MCE cost by downsampling correspondences at the expense of accuracy → PGS resolves this fundamentally via linear search.
- Starting from the geometric foundation that triangulation requires only 3 points, the 3-clique design is theoretically the minimal sufficient solution.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The justification for lightweight cliques derived from variance analysis is theoretically rigorous and yields significant practical gains.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple indoor/outdoor datasets and descriptors, with comprehensive speed analysis and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Logically clear, with intuitive figures and formally rigorous definitions.
- **Value**: ⭐⭐⭐⭐⭐ A 208× speedup is a substantial breakthrough; the training-free design is highly attractive for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](buffer-x_towards_zero-shot_point_cloud_registration_in_diverse_scenes.md)
- [\[ICCV 2025\] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis](upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)
- [\[CVPR 2026\] APC: Transferable and Efficient Adversarial Point Counterattack for Robust 3D Point Cloud Recognition](../../CVPR2026/3d_vision/apc_adversarial_point_counterattack.md)

</div>

<!-- RELATED:END -->
