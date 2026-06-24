---
title: >-
  [Paper Note] Toward Robust Neural Reconstruction from Sparse Point Sets
description: >-
  [CVPR 2025][3D Vision][Sparse Point Cloud Reconstruction] Proposes a neural SDF learning method based on the distributionally robust optimization (DRO) framework. By defining uncertainty sets through Wasserstein and Sinkhorn distances, it samples from model uncertainty regions to regularize training, achieving robust 3D reconstruction on sparse and noisy point clouds.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Sparse Point Cloud Reconstruction"
  - "Signed Distance Function"
  - "Distributionally Robust Optimization"
  - "Wasserstein Distance"
  - "Adversarial Examples"
date: 2026-05-08
content_hash: 9943c162ddda1632
---

# Toward Robust Neural Reconstruction from Sparse Point Sets

**Conference**: CVPR 2025  
**arXiv**: [2412.16361](https://arxiv.org/abs/2412.16361)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Sparse Point Cloud Reconstruction, Signed Distance Function, Distributionally Robust Optimization, Wasserstein Distance, Adversarial Examples

## TL;DR

Proposes a neural SDF learning method based on the distributionally robust optimization (DRO) framework. By defining uncertainty sets through Wasserstein and Sinkhorn distances, it samples from model uncertainty regions to regularize training, achieving robust 3D reconstruction on sparse and noisy point clouds.

## Background & Motivation

Learning signed distance functions (SDF) from sparse and noisy 3D point clouds is a core challenge in 3D reconstruction. Traditional methods (such as Poisson Surface Reconstruction) require dense, clean point clouds and accurate normals. Deep learning methods like Neural Pull perform well when learning SDFs from dense point clouds, but suffer from shape loss and hallucinations due to overfitting under sparse, noisy inputs.

Core Problem: The approximation error of the SDF tends to concentrate in low-density and noisy regions of the point cloud. The existing method, NAP, regularizes training by generating adversarial samples through local perturbations at each query point. However, it only performs pointwise independent perturbations (hard-ball projection), lacking a globally optimal "worst-case distribution."

Core Idea: Instead of independently perturbing query points, this work seeks the **worst-case distribution** within the Wasserstein ball neighborhood of the query point distribution—where the expected loss under this distribution is maximized—and optimizes the SDF on this distribution. Feasible solution is achieved through the dual formulation of DRO, and entropy regularization of the Sinkhorn distance is further employed to accelerate convergence and generate a smoother adversarial distribution.

## Method

### Overall Architecture

Learns the SDF $f_\theta$ based on the query-pull strategy of Neural Pull. A DRO regularization term is added to the standard empirical risk minimization. Two schemes are proposed: SDF WDRO (Wasserstein DRO) and SDF SDRO (Sinkhorn DRO). The final training objective combines the standard loss and the DRO loss, adaptively balanced using learnable weights $\lambda_1, \lambda_2$.

### Key Design 1: Wasserstein Distributionally Robust Optimization (WDRO)

**Function**: Finds the worst-case distribution within the Wasserstein ball neighborhood of the query distribution.

**Mechanism**: The optimization problem is formulated as $\inf_\theta \sup_{Q': \mathcal{W}_c(Q', Q) < \epsilon} \mathbb{E}_{q' \sim Q'} \mathcal{L}(\theta, q')$. It is reformulated into a feasible form via dual representation:

$$\inf_{\theta, \lambda \geq 0} \left\{\lambda \epsilon + \mathbb{E}_{q \sim Q}\left[\sup_{q'}\{\mathcal{L}(\theta, q') - \lambda c(q', q)\}\right]\right\}$$

Given current $\theta$ and $\lambda$, the worst-case spatial inquiries $q'$ are found via several gradient ascent steps on the perturbed query points $q$, and then $\lambda$ is updated.

**Design Motivation**: Compared to the pointwise independent perturbations in NAP (local information), WDRO captures global information by updating the dual variable $\lambda$. The soft-ball projection (rather than a hard ball) adaptively adjusts via $\lambda$ during training, yielding stronger adversarial samples.

### Key Design 2: Sinkhorn DRO Entropy Regularization (SDRO)

**Function**: Accelerates the convergence of WDRO and generates smoother worst-case distributions.

**Mechanism**: Replaces the Wasserstein distance with the Sinkhorn distance (adding a relative entropy penalty). The dual formulation is:

$$\mathcal{L}_{\text{SDRO}}(\theta, Q) = \lambda \rho \mathbb{E}_{q \sim Q}\left[\log \mathbb{E}_{q' \sim \mathbb{Q}_{q,\rho}}\left[e^{\mathcal{L}(\theta, q') / (\lambda \rho)}\right]\right]$$

where the density of $\mathbb{Q}_{q,\rho}$ is proportional to $e^{-c(q,z)/\rho}$, which is equivalent to a Gaussian distribution $\mathcal{N}(q, \rho \mathbf{I}_3)$ when the cost is $c = \frac{1}{2}\|\cdot\|^2$. For each query $q$, $N_s = 5$ adversarial samples are sampled.

**Design Motivation**: WDRO converges slowly, and the worst-case distribution is discrete (due to the finite support of the nominal distribution), which can be overly conservative. Entropy regularization yields a **continuous and diffused** adversarial distribution, allowing the SDF approximation error to be more evenly distributed across the entire shape rather than concentrating on a few discrete points.

### Key Design 3: Multi-task Weighted Training Objective

**Function**: Adaptively balances the standard loss and the DRO regularization loss.

**Mechanism**:

$$\mathfrak{L}(\theta, q) = \frac{1}{2\lambda_1}\mathcal{L}(\theta, q) + \frac{1}{2\lambda_2}\mathcal{L}_{\text{DRO}}(\theta, q) + \ln(1+\lambda_1) + \ln(1+\lambda_2)$$

$\lambda_1, \lambda_2$ are learnable weights optimized alongside the network parameters $\theta$.

**Design Motivation**: The standard Neural Pull loss ensures accurate SDFs on the point clouds, while the DRO loss enhances robustness in uncertain regions. Adaptive weighting of the two avoids manual hyperparameter tuning.

### Loss & Training

Neural Pull base loss $\mathcal{L}(\theta, q) = \|q - f_\theta(q) \cdot \frac{\nabla f_\theta(q)}{\|\nabla f_\theta(q)\|_2} - p\|_2^2$ + DRO regularization loss $\mathcal{L}_{\text{SDRO}}$ + Eikonal constraint.

## Key Experimental Results

### Main Results: ShapeNet Sparse Noisy Point Cloud Reconstruction (1024 points + Gaussian noise)

| Method | CD1↓ | CD2↓ | NC↑ | FS↑ |
|------|------|------|-----|-----|
| Neural Pull | 1.16 | 0.074 | 0.84 | 0.75 |
| NAP | 0.76 | 0.020 | 0.87 | 0.83 |
| SparseOcc | 0.76 | 0.020 | 0.88 | 0.83 |
| NTPS | 1.11 | 0.067 | 0.88 | 0.74 |
| **Ours (WDRO)** | **0.77** | **0.015** | 0.87 | 0.83 |
| **Ours (SDRO)** | **0.63** | **0.012** | **0.90** | **0.86** |

### Comparison with Supervised Methods

| Method | Type | CD1↓ |
|------|------|------|
| POCO (Supervised) | Feed-forward Generalization | Higher (drops on OOD data) |
| CONet (Supervised) | Feed-forward Generalization | Higher |
| **Ours SDRO (Unsupervised)** | **Per-scene Optimization** | **Lower** |

### Key Findings

- SDRO improves **CD1 by -17%** compared to NAP and SparseOcc (0.63 vs 0.76), proving that distribution-level adversarial optimization is superior to pointwise adversarial optimization.
- WDRO already outperforms NAP on CD2 (0.015 vs 0.020), but SDRO further reduces it to 0.012.
- Outperforms SOTA on both Faust real human scans and 3D Scene large-scale scenes.
- **Unsupervised method outperforms supervised generalization models**: On out-of-distribution sparse data, the per-scene optimized DRO method outperforms supervised methods that rely on feed-forward generalization.
- SDRO converges significantly faster than WDRO (about 2-3 times faster), validating the training efficiency improvement brought by entropy regularization.

## Highlights & Insights

1. **Theoretical Depth**: Systematically introduces theoretical tools of optimal transport and distributionally robust optimization to 3D point cloud reconstruction for the first time.
2. **From Points to Distributions**: Upgrades from NAP's pointwise adversarial perturbation to distribution-level worst-case optimization, providing stronger regularization.
3. **Elegant Role of Entropy Regularization**: The use of the Sinkhorn distance not only accelerates convergence but also theoretically yields a more suitable continuous worst-case distribution.

## Limitations & Future Work

- The training time of the WDRO version increases significantly (though mitigated by SDRO).
- Hyperparameters ($\rho, \lambda, \epsilon$) search still needs to be conducted on benchmarks.
- Only processes unoriented point clouds, without utilizing potential normal information.
- Future work can explore applying the DRO framework to new representations like Gaussian Splatting.

## Related Work & Insights

- **Neural Pull**: The base framework, pulling query points to the nearest input points via SDF gradients.
- **NAP**: A pioneer introducing pointwise adversarial perturbation regularization; SDRO is its theoretical generalization.
- **Wasserstein DRO Literature**: Mathematical foundations of distributionally robust optimization in machine learning/operations research.

## Rating

⭐⭐⭐⭐ — Rigorous theoretical framework that elegantly integrates optimal transport theory with 3D reconstruction. The upgrade from NAP's "point-level adversarial" to SDRO's "distribution-level adversarial" offers both theoretical depth and practical performance improvements. The result of outperforming supervised generalization methods on sparse point clouds is particularly prominent. However, the method is highly complex and hyperparameters need meticulous tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Neural Compression for 3D Geometry Sets](../../ICCV2025/3d_vision/neural_compression_for_3d_geometry_sets.md)
- [\[CVPR 2025\] 4Deform: Neural Surface Deformation for Robust Shape Interpolation](4deform_neural_surface_deformation_for_robust_shape_interpolation.md)
- [\[CVPR 2025\] Spectral Informed Mamba for Robust Point Cloud Processing](spectral_informed_mamba_for_robust_point_cloud_processing.md)
- [\[CVPR 2025\] Sparse Point Cloud Patches Rendering via Splitting 2D Gaussians](sparse_point_cloud_patches_rendering_via_splitting_2d_gaussians.md)
- [\[CVPR 2025\] ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion](shapeshifter_3d_variations_using_multiscale_and_sparse_point-voxel_diffusion.md)

</div>

<!-- RELATED:END -->
