---
title: >-
  [Paper Note] RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] RegGS is proposed as a framework that incrementally aligns locally generated 3D Gaussians from a feed-forward network into a globally consistent 3D representation via a diffe…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "pose-free reconstruction"
  - "sparse views"
  - "optimal transport"
  - "Gaussian registration"
date: 2026-05-08
content_hash: c1bd842ec3b9b5ed
---

# RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration

**Conference**: ICCV 2025
**arXiv**: [2507.08136](https://arxiv.org/abs/2507.08136)
**Code**: [Project Page](https://3dagentworld.github.io/reggs/)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, pose-free reconstruction, sparse views, optimal transport, Gaussian registration

## TL;DR

RegGS is proposed as a framework that incrementally aligns locally generated 3D Gaussians from a feed-forward network into a globally consistent 3D representation via a differentiable 3DGS registration module based on the optimal-transport MW2 distance, enabling high-quality 3D reconstruction from unposed sparse views.

## Background & Motivation

Reconstructing 3D scenes from sparse, pose-free images is a highly challenging problem. Existing methods fall into three categories, each with its own limitations:

**Optimization-based 3DGS methods** (e.g., CF-3DGS): integrate pose estimation into the 3DGS optimization loop, but struggle under sparse views due to the lack of geometric priors—manifesting as topological discontinuities and severe scale ambiguity.

**Feed-forward Gaussian methods** (e.g., NoPoSplat, pixelSplat): leverage large-scale training data to learn 3D priors and predict 3D Gaussians directly, with strong cross-dataset generalization. However, they can only handle a limited number of input images (typically 2) and do not scale to more views.

**Traditional methods** (COLMAP + 3DGS): SfM pipelines frequently fail under sparse views.

**Key Challenge**: Feed-forward methods possess strong 3D priors but are limited in the number of input views; optimization-based methods handle arbitrary numbers of views but lack priors. The key question is whether the local Gaussian representations from feed-forward models can be merged into a globally consistent representation via registration.

The proposed solution is **3DGS Registration**: the problem is reformulated as generating local 3D Gaussians for each image (or image pair) using a feed-forward model, then incrementally aligning them into a unified coordinate system through registration.

A key technical challenge is that the centers of 3D Gaussians do not accurately reflect the scene's geometric structure—the full distribution of each Gaussian (mean + covariance) must be considered. This motivates the use of a Gaussian Mixture Model (GMM) statistical framework to measure structural similarity between sets of Gaussians.

## Method

### Overall Architecture

1. A pretrained feed-forward Gaussian model (NoPoSplat) generates main Gaussians from two initial images.
2. Sub-Gaussians are generated for each additional input image.
3. A Sim(3) transformation is estimated via joint optimization of MW2 distance, photometric consistency, and depth geometry.
4. Sub-Gaussians are transformed and merged into the main Gaussians.
5. After all frames are registered, global refinement is performed.

### Key Designs

1. **Optimal Transport MW2 Distance**

   - **Function**: Measures the structural discrepancy between two sets of 3D Gaussian distributions.
   - **Mechanism**: Each set of 3D Gaussians is modeled as a GMM, and the 2-Wasserstein distance is used to measure the discrepancy between individual Gaussian pairs:
     $W_2^2 = \|\mu_i^A - \mu_k^{B'}\|^2 + \text{Tr}(\Sigma_i^A + \Sigma_k^{B'} - 2(\Sigma_i^A \Sigma_k^{B'})^{1/2})$
     Since directly computing the $W_2$ distance between GMMs requires solving an infinite-dimensional optimization problem, the transport plan is restricted to the Gaussian mixture subspace, yielding a tractable upper bound—the Mixture W2 (MW2) distance:
     $\text{MW}_2^2(P,Q) = \inf_{\pi \in \Pi(w^A, w^B)} \sum_{i,k} \pi_{ik} C_{ik}$
     This is efficiently solved via entropy-regularized Sinkhorn iterations: $W_{2,\epsilon}^2 = \min_\pi [\sum_{i,k} \pi_{ik} C_{ik} + \epsilon \sum_{i,k} \pi_{ik} \log \pi_{ik}]$, converging through alternating scaling.
   - **Design Motivation**: The MW2 distance accounts not only for positional offsets of Gaussian centers but also for covariance matrices (shape and orientation), providing a more complete alignment metric than ICP on centers alone. Entropy regularization avoids local optima, accelerates convergence, and renders the entire computation differentiable. Computational complexity is $O(MN)$.

2. **Differentiable Joint 3DGS Registration Module**

   - **Function**: Jointly optimizes Sim(3) transformation parameters to align sub-Gaussians to main Gaussians.
   - **Mechanism**: The Sim(3) transformation is parameterized with quaternions, translation, and log-scale as $\boldsymbol{\theta} = [\mathbf{q}; \mathbf{t}; \log s] \in \mathbb{R}^8$. Three losses are jointly optimized:
     $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{MW}_2} + \lambda_2 \mathcal{L}_{\text{Photo}} + \lambda_3 \mathcal{L}_{\text{Depth}}$
     - The MW2 loss drives global distribution alignment.
     - The photometric loss $\mathcal{L}_{\text{Photo}}$ enforces pixel-level RGB consistency via the 3DGS rendering pipeline.
     - The depth loss $\mathcal{L}_{\text{Depth}}$ constrains depth consistency, suppressing scale drift and topological deformation.
   - **Design Motivation**: The MW2 loss alone is prone to local optima (since Sinkhorn yields an approximate solution); the photometric loss provides fine-grained local alignment; the depth loss stabilizes geometry and mitigates scale issues. The three terms are complementary, achieving coarse-to-fine registration.

3. **Incremental Registration and Global Refinement**

   - **Function**: Registers frames sequentially, followed by global optimization.
   - **Mechanism**: Sub-Gaussians produced by the feed-forward model exhibit significant scale variation; scale normalization (based on mean depth) and initial scale estimation are applied before coarse-to-fine incremental registration. After registration, the global Gaussians undergo adaptive pruning and refinement to improve final rendering quality.
   - **Design Motivation**: The incremental approach allows handling an arbitrary number of input images, overcoming the input limitation of feed-forward models. Global refinement corrects locally accumulated inconsistencies from the registration process.

### Loss & Training

The three registration losses are jointly optimized, with gradients backpropagated to the quaternion rotation parameters via automatic differentiation. For computational efficiency, Sinkhorn iterations, Cholesky decomposition, and Wasserstein distance computations are all mapped to GPU tensor operations. Covariance matrices are regularized with $10^{-6}I$ to ensure positive definiteness, and log-space Sinkhorn iterations prevent exponential overflow.

## Key Experimental Results

### Main Results

**NVS Results on RE10K**:

| Method | 2-view PSNR↑ | 8-view PSNR↑ | 16-view PSNR↑ | 32-view PSNR↑ |
|--------|-------------|-------------|--------------|--------------|
| NoPoSplat (2-view only) | 23.247 | - | - | - |
| CF-3DGS | 19.326 | 20.329 | 23.034 | 25.596 |
| MASt3R* | 16.036 | 24.249 | 27.024 | 28.309 |
| VideoLifter | 14.526 | 16.651 | 14.765 | 15.268 |
| **RegGS** | **24.272** | **26.691** | **28.663** | **28.332** |

**Pose Estimation (ATE↓)**:

| Method | RE10K 8x | RE10K 16x | ACID 8x | ACID 16x |
|--------|----------|-----------|---------|----------|
| CF-3DGS | 0.237 | 0.254 | 0.278 | 0.195 |
| VideoLifter | 0.335 | 0.291 | 0.272 | 0.206 |
| **RegGS** | **0.023** | **0.041** | **0.020** | **0.038** |

### Ablation Study

| Configuration | ATE↓ | PSNR↑ | SSIM↑ | LPIPS↓ | MW2↓ |
|--------------|------|-------|-------|--------|------|
| w/o Photo | 1.184 | 16.06 | 0.52 | 0.44 | 58.8 |
| w/o Depth | 0.160 | 20.97 | 0.72 | 0.29 | 57.8 |
| w/o MW2 | 1.151 | 19.41 | 0.67 | 0.31 | 67.7 |
| w/o Joint Registration Module | 1.164 | 11.41 | 0.34 | 0.60 | 100.0 |
| **Full RegGS** | **0.098** | **23.09** | **0.79** | **0.23** | **56.5** |

### Key Findings

1. Removing any single loss term leads to significant degradation—removing MW2 or the photometric loss causes ATE to increase by more than 10×.
2. The joint registration module is the core of the system—without it, PSNR drops from 23.09 to 11.41, rendering the scene nearly unrecoverable.
3. Pose estimation accuracy far exceeds competing methods—RE10K 8-view ATE = 0.023 vs. 0.237 for CF-3DGS (a 10× improvement).
4. Under the 2-view setting, RegGS (24.272) even surpasses NoPoSplat (23.247), demonstrating that global refinement further improves feed-forward predictions.
5. RegGS also leads on the ACID dataset (UAV aerial footage), confirming that the method generalizes beyond indoor scenes.
6. The MW2 distance effectively quantifies Gaussian distribution alignment, serving as a reliable indicator of registration quality.

## Highlights & Insights

1. **An elegant unification of feed-forward and optimization-based approaches**: the feed-forward model provides strong 3D priors (addressing the sparse-view problem), while the registration mechanism enables multi-view fusion (overcoming input limitations)—the two are complementary.
2. **Optimal transport framework**: formalizing 3DGS registration as an optimal transport problem between GMMs is more theoretically grounded than naive ICP or center-point matching.
3. **Sim(3)-space registration**: incorporating scale as a degree of freedom accommodates the varying scales of outputs from feed-forward models, which is critical in practice.
4. MW2 as a differentiable registration quality metric has potential for generalization to other tasks requiring alignment of sets of distributions.

## Limitations & Future Work

1. Performance is bounded by the quality of the feed-forward Gaussian model—poor local Gaussian generation may cause registration failure.
2. Computation time increases significantly with the number of input views (MW2 distance computation is $O(MN)$), requiring further optimization.
3. Registration may fail to converge under large inter-frame motion, necessitating better initialization strategies.
4. The current framework uses a fixed feed-forward backbone (NoPoSplat); joint fine-tuning could yield further improvements.
5. Performance in dynamic scenes is not discussed—all experiments involve static scenes.

## Related Work & Insights

- **NoPoSplat**: provides the foundation for pose-free feed-forward Gaussian prediction → RegGS extends its 2-view output to arbitrary numbers of views.
- **CF-3DGS**: optimization-based pose-free 3DGS → RegGS replaces end-to-end optimization with registration, achieving greater stability under sparse views.
- **MASt3R**: point cloud matching + 3DGS reconstruction → RegGS performs alignment at the distribution level rather than the point level.
- **Sinkhorn algorithm**: a classical optimal transport solver → applied here for the first time to 3DGS registration.
- **Insight**: the Gaussian attributes of 3DGS (mean + covariance + opacity) are naturally amenable to registration within a statistical framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Introducing optimal transport into 3DGS registration is a novel idea; the feed-forward + registration framework is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two datasets, multiple view-count settings, and complete ablations, though evaluation on large-scale outdoor scenes is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — The method is clearly presented, though the notation is dense.
- **Value**: ⭐⭐⭐⭐ — Provides a practical solution for pose-free multi-view 3DGS reconstruction with notably strong pose estimation accuracy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [\[ICCV 2025\] Sparfels: Fast Reconstruction from Sparse Unposed Imagery](sparfels_fast_reconstruction_from_sparse_unposed_imagery.md)
- [\[ICCV 2025\] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](3dgslm_faster_gaussiansplatting_optimization_with_levenbergm.md)
- [\[ICCV 2025\] SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)

</div>

<!-- RELATED:END -->
