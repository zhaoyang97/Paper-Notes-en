---
title: >-
  [Paper Note] Fully Dynamic Algorithms for Chamfer Distance
description: >-
  [NeurIPS 2025][3D Vision][Chamfer distance] This paper proposes the first fully dynamic algorithm for maintaining Chamfer distance, reducing the problem to approximate nearest neighbor (ANN) queries to achieve a $(1+\epsilon)$ approximation with update time $\tilde{O}(\epsilon^{-d})$, significantly surpassing the linear-time lower bound of static recomputation. On real-world datasets, the algorithm achieves <10% relative error while running orders of magnitude faster than nai…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Chamfer distance"
  - "dynamic algorithms"
  - "approximate nearest neighbor"
  - "importance sampling"
  - "point cloud similarity"
date: 2026-05-08
content_hash: 8b3b56c5921dcdcf
---

# Fully Dynamic Algorithms for Chamfer Distance

**Conference**: NeurIPS 2025
**arXiv**: [2512.16639](https://arxiv.org/abs/2512.16639)  
**Code**: None  
**Area**: 3D Vision / Algorithm Theory / Point Cloud
**Keywords**: Chamfer distance, dynamic algorithms, approximate nearest neighbor, importance sampling, point cloud similarity

## TL;DR
This paper proposes the first fully dynamic algorithm for maintaining Chamfer distance, reducing the problem to approximate nearest neighbor (ANN) queries to achieve a $(1+\epsilon)$ approximation with update time $\tilde{O}(\epsilon^{-d})$, significantly surpassing the linear-time lower bound of static recomputation. On real-world datasets, the algorithm achieves <10% relative error while running orders of magnitude faster than naive approaches.

## Background & Motivation

**Background**: Chamfer distance is the most widely used dissimilarity measure between point clouds, defined as $\text{dist}_{\text{CH}}(A,B) = \sum_{a \in A} \min_{b \in B} \text{dist}(a,b)$. It is broadly applied as a loss function in 3D point cloud completion and upsampling, object reconstruction from video sequences, and anatomical structure tracking in medical imaging.

**Limitations of Prior Work**: Many applications require repeatedly computing Chamfer distance over dynamically changing point sets (e.g., evolving model predictions during training iterations), yet no dynamic maintenance algorithm previously existed. Naive approaches require recomputation from scratch after each update: exact computation costs $O(n^2 d)$, and even a $(1+\epsilon)$ approximation requires $O(nd\log n \cdot \epsilon^{-2})$.

**Key Challenge**: Static algorithms cannot break the linear-time update lower bound, while practical applications demand point-wise updates.

**Goal**: Given dynamic point sets $A, B \subset \mathbb{R}^d$ with insertions and deletions, how can one efficiently maintain an approximation of Chamfer distance?

**Key Insight**: Reduce the dynamic maintenance of Chamfer distance to dynamic ANN queries, and estimate the total distance using an importance sampling framework.

**Core Idea**: Implicit distance estimation via randomly shifted quadtrees combined with importance sampling, enabling fully dynamic Chamfer distance maintenance with sublinear update time.

## Method

### Overall Architecture
**Input**: Two dynamic point sets $A, B \subset \mathbb{R}^d$ (each with at most $n$ points), supporting insertions and deletions. **Output**: A $(1+\alpha+\epsilon)$ approximation of Chamfer distance. **Pipeline**: (1) Maintain a hierarchical spatial decomposition of all points using a randomly shifted quadtree; (2) implicitly maintain an $O(\log^2 n)$-approximate distance estimate $\hat{d}_a$ for each $a \in A$; (3) at query time, draw a small number of sample points via importance sampling and refine each sample's distance estimate using an ANN oracle.

### Key Designs

1. **Dynamic Maintenance via Randomly Shifted Quadtrees**:

    - **Function**: Maintains a hierarchical grid decomposition of the input space, with each node recording the count of points from $A$ and $B$.
    - **Mechanism**: A random shift vector $z \in [0,U]^d$ is applied to all points, constructing an $O(\log n)$-level grid where cell side lengths halve at each level. Each node $v$ stores $\gamma_A(v)$ (number of points from $A$), $\gamma_B(v)$ (number of points from $B$), and $\gamma(v)$ (number of points in $A$ "matched" at $v$—i.e., $v$ is the smallest cell simultaneously containing some $a$ and some $b$).
    - **Design Motivation**: The cell side length $L(v_a)$ is an $O(\log^2 n)$ approximation of $\min_{b \in B} \text{dist}(a,b)$ (core lemma), and updates only require traversing the path from the inserted/deleted point to the root, taking $O(d \log n)$ time.

2. **Implicit Matching and Importance Sampler**:

    - **Function**: Explicitly maintaining the nearest-neighbor assignment for each point is infeasible (a single update may alter $\Omega(n)$ assignments); instead, a sampler implicitly maintains this information.
    - **Mechanism**: A global Tree-Sampler samples nodes with weight $w_T(v) = L(v) \cdot \gamma(v)$, then a Node-Sampler recursively samples child nodes level by level until reaching a leaf, returning point $a \in A$. The resulting sampling probability is $P(a) = L(v_a) / \sum_{v \in T} L(v) \cdot \gamma(v)$, which is approximately proportional to $a$'s contribution to the Chamfer distance.
    - **Design Motivation**: The variance of importance sampling is bounded by $O(\log^2 n \cdot \alpha^2)$, requiring only $m = O(\log^2 n \cdot \max\{1,\alpha^2\} \cdot \epsilon^{-2})$ samples to achieve a $(1+\epsilon)$ approximation.

3. **Chamfer Distance Estimator**:

    - **Function**: Queries an ANN oracle for each sampled point to obtain a precise distance, then computes a weighted average as the total estimate.
    - **Mechanism**: For $m$ sampled points, the estimated weight for each point $a$ is $x_a = \text{NN}(a,B,\alpha/4) \cdot \frac{\sum_v \gamma(v) \cdot L(v)}{L(v_a)}$, and the algorithm returns $\frac{\tilde{\mu}}{m \cdot (1+\epsilon/2)}$. Taking the median of $O(\log n)$ independent queries boosts the success probability to $1-1/\text{poly}(n)$.
    - **Design Motivation**: Decouples estimation accuracy from the ANN oracle's approximation ratio—the sample count controls $\epsilon$-precision, while the ANN determines $\alpha$-precision.

### Theoretical Guarantees

**Core Theorem** (simplified): Given a $(1+\Theta(\alpha))$-ANN oracle with update/query time $\tau$, the algorithm supports:
- **Update time**: $O(d \cdot \log n + \log^2 n + \tau(\Theta(\alpha)))$ (worst case)
- **Query time**: $\tilde{O}((\tau + d) \cdot \epsilon^{-2} \cdot \max\{1, \alpha^2\})$
- Substituting low-dimensional ANN bounds: $(1+\epsilon)$ approximation, update time $\tilde{O}(\epsilon^{-d})$
- Substituting high-dimensional ANN bounds: $O(1/\epsilon)$ approximation, update time $\tilde{O}(dn^{\epsilon^2} \epsilon^{-4})$

**Negative Result**: Any dynamic algorithm maintaining an $\alpha$-approximate assignment requires $\Omega(n)$ recourse (a single update may require changing $\Omega(n)$ assignments), making explicit assignment maintenance infeasible.

## Key Experimental Results

### Main Results: Relative Error on Real-World Datasets

| Dataset | Dim. | \|A\| | \|B\| | Samples | Algorithm Error | Uniform Error |
|--------|------|-------|-------|--------|---------|-------------|
| Text Embedding | 300 | ~1.9k | ~1.2k | 150 | <10% | <10% |
| ShapeNet | 3 | ~2k | ~2k | 150 | <10% | <10% |
| Fashion-MNIST | 784 | 60k | 10k | 200 | <10% | <10% |
| SIFT | 128 | 1000k | 10k | 300 | <10% | <10% |

### Ablation Study: Robustness Under Injected Outliers

| Dataset | Algorithm Error (w/ outliers) | Uniform Error (w/ outliers) | Note |
|--------|----------------------|--------------------------|------|
| ShapeNet | <10% (stable) | Significant degradation | Importance sampling is robust to outliers |
| Fashion-MNIST | <10% (stable) | Significant degradation | Uniform sampling heavily affected by outliers |
| Text Embedding | <10% (stable) | Notable degradation | Outlier contributions suppressed by importance sampling |
| SIFT | <10% | <10% | Uniform distance distribution; uniform sampling near-optimal |

### Key Findings
- The algorithm achieves <10% relative error using only a few hundred samples across all datasets.
- Runtime is orders of magnitude faster than naive recomputation, especially on large datasets.
- **Importance sampling vs. uniform sampling**: Performance is comparable in the absence of outliers; however, after injecting outliers, uniform sampling degrades significantly while importance sampling remains stable.
- SIFT is a special case—its distance distribution is highly uniform, making uniform sampling already near-optimal.

## Highlights & Insights
- **Elegance of the reduction**: Reducing dynamic Chamfer distance maintenance to ANN queries plus importance sampling yields a clean technical core; improvements to ANN directly translate to improvements in Chamfer distance maintenance.
- **Clever handling of implicit matching**: Since explicit assignment maintenance is infeasible ($\Omega(n)$ recourse), sampling probabilities are implicitly maintained via matching counts in the quadtree, requiring only $O(\log^2 n)$ time per update.
- **Theoretical value of the negative result**: The impossibility of maintaining approximate assignments ($\Omega(n)$ recourse) is formally established, demonstrating that implicit estimation is the only viable path.
- The algorithm can be directly applied to accelerate Chamfer loss computation in point cloud completion and upsampling training.

## Limitations & Future Work
- In low dimensions, the update time $\tilde{O}(\epsilon^{-d})$ for a $(1+\epsilon)$ approximation grows exponentially with dimension, potentially degrading in high-dimensional ($d > 10$) settings.
- Experiments use exact nearest neighbor search rather than approximate NN, leaving the practical effectiveness of the theoretical ANN component unverified.
- A sliding window is used to simulate dynamic updates, without validating performance on arbitrary update sequences.
- Only $\ell_1$ and $\ell_2$ norms are supported; extension to other distance metrics is not addressed.
- No speed comparison is made against modern GPU-accelerated batch Chamfer distance computation.

## Related Work & Insights
- **vs. Static Chamfer algorithms [Bakshi et al.]**: The static optimum is $O(nd\log n \cdot \epsilon^{-2})$; the dynamic algorithm in this paper incurs only $\tilde{O}(d)$ plus ANN overhead per update, far superior to recomputation.
- **vs. Dynamic EMD [Goranci et al.]**: The dynamic algorithm for EMD achieves only an $O(1/\epsilon)$ approximation in $d=2$ with update time $O(n^{1/\epsilon})$. This paper's dynamic Chamfer distance offers stronger guarantees in arbitrary dimensions.
- **vs. Dynamic clustering algorithms**: This work falls within the broader vision of dynamic algorithms for machine learning, contributing point cloud metric maintenance as a new primitive.
- The algorithmic framework may extend to other metrics based on summed nearest-neighbor distances.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First fully dynamic Chamfer distance algorithm with solid theoretical contributions, including an impossibility result.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four real-world datasets with outlier robustness tests, but lacking GPU baseline comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretically rigorous and complete; experiments are clearly presented, though lengthy proofs somewhat hinder readability.
- **Value**: ⭐⭐⭐⭐ — Theoretically important as a first result; practical value in ML training loops awaits further empirical validation.

## Related Papers

- [\[NeurIPS 2025\] DGH: Dynamic Gaussian Hair](dgh_dynamic_gaussian_hair.md)
- [\[CVPR 2026\] Long-SCOPE: Fully Sparse Long-Range Cooperative 3D Perception](../../CVPR2026/3d_vision/long_scope_fully_sparse_long_range_cooperative_3d_perception.md)
- [\[CVPR 2025\] GaussianUDF: Inferring Unsigned Distance Functions through 3D Gaussian Splatting](../../CVPR2025/3d_vision/gaussianudf_inferring_unsigned_distance_functions_through_3d_gaussian_splatting.md)
- [\[ECCV 2024\] Ray-Distance Volume Rendering for Neural Scene Reconstruction](../../ECCV2024/3d_vision/ray-distance_volume_rendering_for_neural_scene_reconstruction.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](../../ECCV2024/3d_vision/wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](../../AAAI2026/3d_vision/parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)
- [\[CVPR 2025\] FSHNet: Fully Sparse Hybrid Network for 3D Object Detection](../../CVPR2025/3d_vision/fshnet_fully_sparse_hybrid_network_for_3d_object_detection.md)
- [\[NeurIPS 2025\] DGH: Dynamic Gaussian Hair](dgh_dynamic_gaussian_hair.md)
- [\[CVPR 2025\] GaussianUDF: Inferring Unsigned Distance Functions through 3D Gaussian Splatting](../../CVPR2025/3d_vision/gaussianudf_inferring_unsigned_distance_functions_through_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)

</div>

<!-- RELATED:END -->
