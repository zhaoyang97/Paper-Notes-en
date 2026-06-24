---
title: >-
  [Paper Note] PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI
description: >-
  [AAAI 2026][Medical Imaging][4D Flow MRI] This paper proposes PINGS-X, a framework that transfers the explicit representation paradigm of 3D Gaussian Splatting (3DGS) into the domain of physics-informed super-resolution. Through three key innovations—Normalized Gaussian Splatting (NGS), axes-aligned Gaussians, and a Gaussian merging strategy—PINGS-X achieves training speeds an order of magnitude faster than PINNs while maintaining superior super-resolution accuracy on both sy…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "4D Flow MRI"
  - "Gaussian Splatting"
  - "Physics-Informed Learning"
  - "Super-Resolution"
  - "Navier-Stokes Equations"
  - "Normalized Kernel Regression"
date: 2026-05-08
content_hash: 43208ca324428007
---

# PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI

**Conference**: AAAI 2026
**arXiv**: [2511.11048](https://arxiv.org/abs/2511.11048)  
**Code**: [GitHub](https://github.com/SpatialAILab/PINGS-X)  
**Area**: Medical Imaging / 4D Flow MRI Super-Resolution
**Keywords**: 4D Flow MRI, Gaussian Splatting, Physics-Informed Learning, Super-Resolution, Navier-Stokes Equations, Normalized Kernel Regression

## TL;DR

This paper proposes PINGS-X, a framework that transfers the explicit representation paradigm of 3D Gaussian Splatting (3DGS) into the domain of physics-informed super-resolution. Through three key innovations—Normalized Gaussian Splatting (NGS), axes-aligned Gaussians, and a Gaussian merging strategy—PINGS-X achieves training speeds an order of magnitude faster than PINNs while maintaining superior super-resolution accuracy on both synthetic CFD and real 4D Flow MRI datasets.

## Background & Motivation

4D Flow MRI is a non-invasive technique for measuring blood flow velocity. By encoding velocity information along three orthogonal directions via phase-contrast MRI, it yields time-resolved three-dimensional velocity fields. High-resolution 4D Flow MRI is critical for detecting cardiovascular pathologies such as stenoses and aneurysms; however, acquiring high spatiotemporal resolution requires prohibitively long scan times, giving rise to a fundamental **trade-off between acquisition speed and predictive accuracy**.

Existing super-resolution methods fall into two main categories:

**Data-driven deep learning methods**: These approaches learn low-to-high-resolution mappings via CNN-based architectures, but require large paired training datasets and generalize poorly to unseen cardiovascular domains.

**Physics-informed neural network (PINN) methods**: These embed physical constraints such as the Navier-Stokes equations into the optimization process and do not require large datasets, but rely on implicit MLP representations that must be retrained per patient, resulting in severe computational bottlenecks.

The authors observe that in novel view synthesis, **3DGS has already achieved state-of-the-art quality at training speeds orders of magnitude faster than NeRF**. This motivates the central research question: can Gaussian representations be incorporated into physics-informed super-resolution to overcome the computational bottleneck of PINNs while retaining theoretical convergence guarantees?

## Method

### Core Idea

PINGS-X transfers the explicit Gaussian representation of 3DGS from the rendering domain to physics field modeling. Directly adopting 3DGS is infeasible, since physics-informed super-resolution does not involve camera projection or alpha compositing, but instead approximates physical fields directly in the spatiotemporal domain. Accordingly, three key innovations are proposed.

### Innovation 1: Normalized Gaussian Splatting (NGS)

A naïve approach would use an unnormalized Gaussian-weighted sum to predict the physical field: $\hat{\mathbf{v}}(\mathbf{x}) = \sum_i z_i(\mathbf{x}) \mathbf{v}_i$. This suffers from a fundamental flaw—predictions decay to zero in regions far from all Gaussian centers, forcing the model to either learn excessively large Gaussians (over-smoothing) or continuously add new Gaussians (oscillation).

PINGS-X addresses this by adopting **normalized weights**:

$$\hat{\mathbf{v}}(\mathbf{x}) = \sum_{i=1}^{N} w_i(\mathbf{x}) \mathbf{v}_i, \quad w_i(\mathbf{x}) = \frac{z_i(\mathbf{x})}{\sum_{j=1}^{N} z_j(\mathbf{x})}$$

This ensures that predictions always constitute a **convex combination** of Gaussian attributes, providing intrinsic stability. The authors prove **uniform convergence** of this scheme (Theorem 1) with an explicit convergence rate, consistent with the theoretical foundation of the classical Nadaraya-Watson estimator, while achieving greater flexibility through learnable centers and anisotropic bandwidths.

### Innovation 2: Axes-Aligned Gaussians (NGS-X)

Optimizing a full anisotropic covariance matrix in 4D spatiotemporal space is challenging due to the difficulty of parameterizing high-dimensional rotations. The authors constrain the Gaussians to be **axes-aligned**, i.e., the covariance matrix is diagonal: $\Sigma_i = \mathrm{diag}(h_{i1}^2, \cdots, h_{iq}^2)$. This avoids the parameterization of 4D rotations and reduces the number of covariance parameters from 10 to 4. Corollary 1 establishes that axes-aligned Gaussians retain convergence guarantees and achieve minimax-optimal convergence rates under optimal bandwidth selection.

### Innovation 3: Gaussian Merging Strategy

Unlike 3DGS, which implicitly disambiguates overlapping Gaussians via multi-view geometric constraints, PINGS-X directly approximates physical fields and is susceptible to degenerate solutions in which multiple Gaussians accumulate in the same region. Pruning is ineffective since such redundant Gaussians may contribute equally; instead, the authors design a **cosine-similarity-based merging scheme**. The normalized influence vector of each Gaussian over all training points is computed, an undirected graph is constructed by connecting pairs whose cosine similarity exceeds a threshold (0.9), and connected components are merged. The merging operation is performed every 100 epochs.

### Loss & Training

$$L = L_{data} + \lambda L_{PDE}$$

- $L_{data}$: MSE between predictions and low-resolution observations (unobservable pressure components are masked via a binary vector $\omega$)
- $L_{PDE}$: physical constraints based on the dimensionless Navier-Stokes equations and the continuity equation

A key advantage is that, unlike PINNs which compute PDE derivatives via backpropagation, the explicit Gaussian representation enables **analytical computation of derivatives**, eliminating the cascaded overhead of automatic differentiation.

## Key Experimental Results

### Experiment 1: Synthetic 2D CFD Datasets

Three steady incompressible 2D flow fields (lid-driven cavity, L-shaped channel, Y-shaped channel) are used, with low-resolution data generated by spatial averaging.

| Method | Lid-driven (Time / Error%) | Y-shaped (Time / Error%) | L-shaped (Time / Error%) |
|--------|---------------------------|--------------------------|--------------------------|
| PINN | 51.4 min / 12.20% | 45.0 min / 11.54% | 43.6 min / 11.84% |
| Siren | 59.8 min / 3.20% | 54.0 min / 4.34% | 50.3 min / 2.49% |
| XPINN (sin) | 105.8 min / 2.71% | 195.5 min / 3.89% | 158.0 min / 2.15% |
| PIG | 820.1 min / 2.33% | 292.9 min / 7.06% | 297.8 min / 1.48% |
| **PINGS-X** | **21.9 min / 1.13%** | **4.8 min / 2.62%** | **6.3 min / 1.15%** |

PINGS-X achieves the lowest relative error across all datasets with substantially reduced training time. On the Y-shaped channel, it is approximately 40× faster than the second-most-accurate method, XPINN (sin).

### Experiment 2: Real 4D Flow MRI Dataset

Experiments are conducted on high-resolution 4D Flow MRI data of a carotid artery phantom (0.35 mm spatial resolution, 25 ms temporal resolution), evaluating ×8 and ×64 super-resolution tasks.

| Method | ×8 (Time / Error% / RMSE) | ×64 (Time / Error% / RMSE) |
|--------|--------------------------|---------------------------|
| PINN | 30.1 hr / 25.93% / 2.75 cm/s | 259.0 min / 39.70% / 4.21 cm/s |
| Siren | 30.8 hr / 10.63% / 1.13 cm/s | 299.2 min / 18.49% / 1.96 cm/s |
| **PINGS-X** | **2.6 hr / 8.98% / 0.95 cm/s** | **3.9 min / 17.59% / 1.87 cm/s** |

For the ×8 task, PINGS-X requires only 1/12 the training time of Siren while achieving lower error; for the ×64 task, it is approximately 77× faster.

### Ablation Study

| Variant | Lid-driven (Time / Error) | Y-shaped (Time / Error) |
|---------|--------------------------|-------------------------|
| PINGS-X (full) | 21.9 min / 1.13% | 4.8 min / 2.62% |
| w/o normalization | 23.3 min / diverges | 3.4 min / diverges |
| w/o axes alignment | 70.3 min / 1.01% | 23.3 min / 2.27% |
| w/o merging | 12.1 min* / 1.38%* | 35.1 min* / 1.66%* |

(*denotes early termination due to OOM.) Normalization is a necessary condition for convergence; axes alignment substantially reduces training time with negligible accuracy loss; merging prevents memory bloat.

## Highlights & Insights

1. **Cross-domain transfer**: 3DGS is successfully transferred from the rendering domain to physics-informed super-resolution, opening a new direction for explicit representations in scientific computing.
2. **Theoretical guarantees**: Formal convergence proofs are provided for normalized Gaussian splatting (grounded in Nadaraya-Watson kernel regression theory), with axes-aligned simplification shown to preserve optimal convergence rates.
3. **Substantial efficiency gains**: PINGS-X is 10–77× faster than PINN/Siren on real 4D Flow MRI data, greatly improving clinical feasibility.
4. **Analytical derivative computation**: The explicit Gaussian representation avoids the cascaded overhead of automatic differentiation, enabling more efficient computation of PDE loss terms.

## Limitations & Future Work

1. **Validation limited to steady and quasi-steady flows**: All CFD experiments involve steady-state conditions, and the 4D Flow MRI evaluation uses only 5 temporal frames; applicability to highly unsteady turbulent flows remains unclear.
2. **Spatial averaging assumption**: Low-resolution data in the experiments are generated by spatial averaging, whereas real MRI acquisition involves more complex k-space sampling and signal processing that may introduce additional error sources.
3. **Absence of pressure field evaluation**: Since 4D Flow MRI does not directly measure pressure, pressure prediction is evaluated only on CFD data; the quality of pressure estimation in clinical settings is unknown.
4. **Single-phantom validation**: The 4D Flow MRI experiments are conducted on a single carotid artery phantom, leaving generalization across multiple patients and anatomical structures unverified.

## Related Work & Insights

- **4D Flow MRI super-resolution**: CNN-based method of Ferdian et al. (2023), PINN-based method of Saitta et al. (2024), temporal super-resolution deep learning method of Callmer et al. (2025).
- **Physics-informed Gaussian representations**: PIGS (Max Rensen and Eisemann 2024) uses unnormalized Gaussian weights and lacks convergence guarantees; PIG (Kang et al. 2024) retains an MLP prediction pathway and remains fundamentally an implicit method.
- **3D Gaussian Splatting**: The seminal work of Kerbl et al. (2023) in novel view synthesis, serving as the primary methodological inspiration for this paper.
- **PINNs**: The classical framework of Raissi et al. (2019) and its domain decomposition extension XPINN (Jagtap and Karniadakis 2021).

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

The cross-domain transfer is elegant, the theoretical analysis is rigorous (convergence proofs and optimal rates), and the experiments are convincing on both speed and accuracy. The primary limitations are that real MRI validation is restricted to a single phantom and direct comparisons with data-driven methods are absent. Overall, this is a high-quality interdisciplinary contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Adaptive Anisotropic Gaussian Splatting for Multi-contrast MRI Arbitrary-Scale Super-Resolution with Anatomy Guidance](../../CVPR2026/medical_imaging/adaptive_anisotropic_gaussian_splatting_for_multi-contrast_mri_arbitrary-scale_s.md)
- [\[AAAI 2026\] CD-DPE: Dual-Prompt Expert Network Based on Convolutional Dictionary Feature Decoupling for Multi-Contrast MRI Super-Resolution](cd-dpe_dual-prompt_expert_network_based_on_convolutional_dictionary_feature_deco.md)
- [\[AAAI 2026\] MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss](maisi-v2_accelerated_3d_high-resolution_medical_image_synthesis_with_rectified_f.md)
- [\[ECCV 2024\] Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis](../../ECCV2024/medical_imaging/radiative_gaussian_splatting_for_efficient_x-ray_novel_view_synthesis.md)
- [\[AAAI 2026\] Multivariate Gaussian Representation Learning for Medical Action Evaluation](multivariate_gaussian_representation_learning_for_medical_action_evaluation.md)

</div>

<!-- RELATED:END -->
