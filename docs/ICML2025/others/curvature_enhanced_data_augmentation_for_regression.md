---
title: >-
  [Paper Note] Curvature Enhanced Data Augmentation for Regression
description: >-
  [ICML 2025][Data Augmentation] Proposes CEMS (Curvature-Enhanced Manifold Sampling), which utilizes the second-order approximation (curvature information) of the data manifold to generate synthetic samples for data augmentation in regression tasks, achieving state-of-the-art (SOTA) or near-SOTA performance in both in-distribution and out-of-distribution scenarios.
tags:
  - "ICML 2025"
  - "Data Augmentation"
  - "Regression Tasks"
  - "Manifold Learning"
  - "Curvature"
  - "Second-order Approximation"
date: 2026-05-08
content_hash: f2dc06bf8d47445a
---

# Curvature Enhanced Data Augmentation for Regression

**Conference**: ICML 2025  
**arXiv**: [2506.06853](https://arxiv.org/abs/2506.06853)  
**Code**: [azencot-group/CEMS](https://github.com/azencot-group/CEMS)  
**Area**: Others  
**Keywords**: Data Augmentation, Regression Tasks, Manifold Learning, Curvature, Second-order Approximation

## TL;DR

Proposes CEMS (Curvature-Enhanced Manifold Sampling), which utilizes the second-order approximation (curvature information) of the data manifold to generate synthetic samples for data augmentation in regression tasks, achieving state-of-the-art (SOTA) or near-SOTA performance in both in-distribution and out-of-distribution scenarios.

## Background & Motivation

Data augmentation has been widely studied and has achieved great success in classification tasks, but its application in **regression tasks** is relatively lacking. The labels of classification tasks are discrete, making it relatively easy to define label-preserving transformations; conversely, the outputs of regression tasks are continuous values, and how to maintain the validity of input-output pairs after transformation is a key challenge.

Existing regression data augmentation methods are mainly based on the Mixup family (such as C-mixup, RegMix), which generate new data through convex combinations of samples. However, Mixup performs unstably in regression tasks. The recent FOMA method approaches the problem from a manifold learning perspective, using the **first-order approximation (tangent space)** of the data manifold to sample new points. However, first-order methods deviate from the true manifold in high-curvature regions, generating lower-quality samples.

The core motivation of this paper is: **first-order approximations are insufficient to capture complex, curved real-world data structures, whereas second-order approximations (incorporating curvature/Hessian information) can achieve a better balance between effectiveness and computational cost.**

## Method

### Overall Architecture

CEMS formulates data augmentation as a manifold approximation and sampling problem. Given a regression training set $\mathcal{D} = \{(x^i, y^i)\}_{i=1}^{N}$, the inputs and labels are concatenated as $z^i = [x^i, y^i] \in \mathbb{R}^D$, assuming these points lie on a low-dimensional manifold $\mathcal{M}$ with an intrinsic dimension $d \ll D$.

The four core steps of CEMS:

1. **Neighborhood Extraction**: For each point $z$, find its $k$-nearest neighbors $N_z$ in the joint input-output space.
2. **Basis Construction and Projection**: Construct orthonormal bases $B_u = [B_{\mathcal{T}_u}, B_{\mathcal{N}_u}]$ for the tangent space $\mathcal{T}_u\mathcal{M}$ and normal space $\mathcal{N}_u\mathcal{M}$ via SVD, and project the neighborhood into local coordinates.
3. **Solving Linear Systems**: Estimate the gradient $\nabla g(u)$ and Hessian $H(u)$ by constructing and solving a linear system using second-order Taylor expansion.
4. **Sampling and Back-projection**: Sample a new point $\eta$ in the tangent space, calculate its normal space component $g(\eta)$ via the second-order approximation, and then back-project it to the original space.

### Key Designs

**Second-Order Manifold Representation**: The core innovation of CEMS lies in using the second-order Taylor expansion of the embedding map $g: \mathcal{T}_u\mathcal{M} \rightarrow \mathcal{N}_u\mathcal{M}$:

$$g(\eta) = \eta^T \nabla g(u) + \frac{1}{2} \eta^T H(u) \eta$$

where $\nabla g(u)$ is the gradient (first-order term) and $H(u)$ is the Hessian (second-order term). Unlike FOMA, which only uses tangent space projection and scales the normal space components, CEMS accurately estimates both terms by solving a least-squares problem.

**Essential Difference from FOMA**: FOMA performs simple scaling on the normal space components $\tilde{g}_j = \lambda g_j$ ($\lambda \in (0,1)$) without using the embedding map $g$, whereas CEMS explicitly estimates the gradient and Hessian, using Taylor expansion to calculate the normal space component at the newly sampled point. This allows CEMS to better fit the manifold structure in high-curvature regions.

**Batch Adaptation**: To improve computational efficiency, CEMS shares the neighborhood and basis construction for all points within the same batch. Specifically, for an anchor point $z$ and its neighborhood $N_z$ in a batch, all $z_j \in N_z$ share the same orthonormal basis $B_u$, and only the linear solver is computed individually for each point.

**Intrinsic Dimension Estimation**: Although $d$ can be treated as a hyperparameter, in practice, a robust estimator (Facco et al., 2017) is used to determine it automatically.

**Fully Differentiable**: All steps (SVD, least-squares solving, sampling, and back-projection) are differentiable, allowing for end-to-end training.

### Loss & Training

CEMS is an **online data augmentation method**: in each training mini-batch, an augmented sample $\tilde{z}$ is generated for each sample $z$. Then, normal training is conducted on both original and augmented samples using standard regression losses (e.g., MSE).

The sampling process utilizes a simple Gaussian sampler: $\eta \sim \mathcal{N}(0, \sigma I_d)$, where $\sigma$ is the sole sampling hyperparameter that controls the distance of new samples from the original points.

The back-projection formula is:

$$z_\eta = f(\eta) = B_u \cdot [\eta, g(\eta)] + z$$

**Computational Complexity**: The overall time complexity is $\mathcal{O}(b^2 D)$, where $b$ is the batch size and $D$ is the data dimension. Since $d \ll D$, the extra computational overhead of the second-order method is only a minimal increment compared to first-order methods.

## Key Experimental Results

### Main Results

**In-Distribution Generalization** (4 datasets, RMSE / MAPE, ↓ is better):

| Method | Airfoil RMSE | Airfoil MAPE | NO2 RMSE | NO2 MAPE | Exchange RMSE | Electricity RMSE |
|------|-------------|-------------|---------|---------|--------------|-----------------|
| ERM | 2.901 | 1.753 | 0.537 | 13.615 | 0.024 | 0.058 |
| Mixup | 3.730 | 2.327 | 0.528 | 13.534 | 0.024 | 0.058 |
| C-Mixup | 2.717 | 1.610 | 0.509 | 12.998 | 0.020 | 0.057 |
| ADA | 2.360 | 1.373 | 0.515 | 13.128 | 0.021 | 0.059 |
| FOMA | 1.471 | 0.816 | 0.512 | 12.894 | **0.013** | 0.058 |
| **CEMS** | **1.455** | **0.809** | **0.507** | **12.807** | 0.014 | **0.058** |

**Out-of-Distribution Generalization** (5 datasets):

| Method | RCF Avg↓ | Crimes Avg↓ | SkillCraft Avg↓ | DTI Avg R↑ | Poverty Avg R↑ |
|------|---------|------------|----------------|-----------|---------------|
| ERM | 0.164 | 0.136 | 6.147 | 0.483 | 0.80 |
| C-Mixup | 0.146 | 0.123 | 5.201 | 0.498 | 0.81 |
| FOMA | 0.159 | 0.128 | - | 0.503 | 0.78 |
| **CEMS** | **0.146** | 0.128 | **5.142** | **0.511** | **0.81** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| CEMS (First-order) | Sampling deviates from the manifold in high-curvature regions | Degenerates to FOMA-like behavior |
| CEMS (Second-order) | Accurate sampling even in high-curvature regions | Full method, Hessian term captures curvature |
| Pre-computation vs. Online computation | Pre-computation is faster but uses more memory | Online computation sacrifices speed for flexibility |
| Point-level (CEMSp) vs. Batch-level (CEMS) | Batch-level sharing basis construction is more efficient | Slight loss in accuracy but significantly more practical |

### Key Findings

- **First-order vs. Second-order**: A toy-sine-wave experiment visually demonstrates that first-order methods (FOMA, CEMS first-order mode) deviate from the manifold during sampling at high-curvature regions, whereas CEMS's second-order approximation accurately tracks the curvature.
- **In-Distribution**: CEMS achieves optimal results on Airfoil and NO2, and near-optimal performance on Exchange-Rate and Electricity.
- **Out-of-Distribution**: CEMS achieves the best results in 6 out of 9 tests, notably improving Avg by 1% and Worst by 8% on SkillCraft.
- **Minimal Computational Overhead**: Since the intrinsic dimension $d \ll D$, the additional cost of second-order computations is negligible.

## Highlights & Insights

1. **Unifying Regression DA under the Manifold Learning Framework**: Rather than being a simple mixup variant, the paper provides a theoretical foundation based on manifold assumptions, bringing first-order (FOMA) and second-order (CEMS) methods under a unified perspective.
2. **Clever Utilization of $d \ll D$**: Second-order methods are usually prohibitive due to high computational costs, but this paper leverages the low intrinsic dimension of the manifold to reduce the computation scale of the Hessian from $D^2$ to $d^2$.
3. **Domain Agnosticism**: CEMS does not rely on specific data domains (applicable to images, time series, and tabular data), showing wider applicability than Mixup-based methods.
4. **Fully Differentiable Design**: Both SVD and least-squares are differentiable, allowing seamless integration into end-to-end training pipelines.

## Limitations & Future Work

1. **Scalability to Large Intrinsic Dimension $d$**: The linear system can become underdetermined, requiring $\mathcal{O}(d^2)$ neighbors, which is impractical for datasets with large $d$ (though this can be alleviated via ridge regression).
2. **SVD Memory Requirements**: When $d$ is large, full SVD might be required, leading to $\mathcal{O}(bD^2)$ memory consumption.
3. **Neighborhood Sharing Assumption**: Batch-level CEMS assumes all points in a neighborhood share the same basis, which may introduce errors in regions of non-uniform curvature.
4. **Overly Simple Sampler**: Currently, an isotropic Gaussian sampler is used, which does not consider the local geometric structure of the manifold (e.g., the sampling magnitude should differ along different principal curvature directions).
5. **Lack of Adaptive Order Selection**: Different local regions may be suited for approximations of different orders.

## Related Work & Insights

- **FOMA** (Kaufman & Azencot, 2024): The direct predecessor of this work, featuring first-order manifold sampling. CEMS can be viewed as its natural generalization.
- **Mixup Family**: Mixup (Zhang et al. 2018) and its variants (C-Mixup, ADA) are effective in classification but unstable in regression.
- **Hessian Eigenmaps** (Donoho & Grimes, 2003): Utilizes second-order information in dimensionality reduction; this paper applies a similar idea to data augmentation.
- **VRM Theory**: The theoretical foundation of data augmentation is Vicinal Risk Minimization. CEMS provides a geometrically aware vicinal distribution.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Second-order manifold sampling for DA is novel, but the core mathematical tools already exist |
| Theoretical Depth | 4 | Features a complete manifold learning theoretical framework and sampling error analysis |
| Experimental Thoroughness | 4 | 9 datasets covering ID/OOD, though missing large-scale deep learning scenarios |
| Practicality | 3.5 | Domain-agnostic with small extra overhead, but exhibits limitations on data with large intrinsic dimensions |
| Writing Quality | 4 | Clear structure with equal emphasis on theory and experiments |
| **Overall Score** | **4/5** | A unified framework for regression DA from a manifold learning perspective, mathematically sound with thorough experiments |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Is Linguistically-Motivated Data Augmentation Worth It?](../../ACL2025/others/is_linguistically-motivated_data_augmentation_worth_it.md)
- [\[ICML 2025\] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks](improved_exploration_in_gflownets_via_enhanced_epistemic_neural_networks.md)
- [\[ECCV 2024\] FreeAugment: Data Augmentation Search Across All Degrees of Freedom](../../ECCV2024/others/freeaugment_data_augmentation_search_across_all_degrees_of_freedom.md)
- [\[ICCV 2025\] Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents](../../ICCV2025/others/adversarial_data_augmentation_for_single_domain_generalization_via_lyapunov_expo.md)
- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)

</div>

<!-- RELATED:END -->
