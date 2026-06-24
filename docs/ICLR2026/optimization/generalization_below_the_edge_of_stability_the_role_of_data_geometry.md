---
title: >-
  [Paper Note] Generalization Below the Edge of Stability: The Role of Data Geometry
description: >-
  [ICLR 2026][Optimization][Generalization Theory] This paper proposes the "data shatterability" principle to provide a unified explanation of how data geometry controls the strength of implicit regularization for gradient descent near the Edge of Stability (EoS). It derives a spectrum of $\alpha$-dependent generalization upper and lower bounds for the Beta(α) radial distribution family and proves that generalization rates adapt to the intrinsic dimension $m$ rather than the am…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Generalization Theory"
  - "Edge of Stability"
  - "Data Geometry"
  - "ReLU Networks"
  - "Implicit Regularization"
date: 2026-05-08
content_hash: 0848e797b7d9d45d
---

# Generalization Below the Edge of Stability: The Role of Data Geometry

**Conference**: ICLR 2026  
**arXiv**: [2510.18120](https://arxiv.org/abs/2510.18120)  
**Code**: None  
**Area**: Learning Theory / Optimization  
**Keywords**: Generalization Theory, Edge of Stability, Data Geometry, ReLU Networks, Implicit Regularization

## TL;DR

This paper proposes the "data shatterability" principle to provide a unified explanation of how data geometry controls the strength of implicit regularization for gradient descent near the Edge of Stability (EoS). It derives a spectrum of $\alpha$-dependent generalization upper and lower bounds for the Beta(α) radial distribution family and proves that generalization rates adapt to the intrinsic dimension $m$ rather than the ambient dimension $d$ for mixtures of low-dimensional subspace distributions.

## Background & Motivation

**Background**: Over-parameterized neural networks generalize well even without explicit regularization (e.g., weight decay), a phenomenon that classical statistical learning theory fails to explain. Recently, the discovery of the Edge of Stability (EoS) phenomenon—where the maximum eigenvalue of the Hessian $\lambda_{\max}(\nabla^2\mathcal{L}) \approx 2/\eta$ during gradient descent (GD) training with large step sizes—has provided a new perspective for understanding implicit regularization.

**Limitations of Prior Work**:

1. Prior studies proved that the EoS condition is equivalent to a data-dependent weighted path-norm constraint. However, generalization bounds derived for uniform spherical distributions suffer from the curse of dimensionality, which contradicts the practical success of deep learning.
2. There is no unified theoretical framework to distinguish which data geometries lead to generalization and which lead to memorization.
3. Existing generalization bounds are distribution-independent and fail to capture the impact of different data geometries.

**Key Challenge**: The strength of EoS-induced data-dependent regularization varies drastically across different data distributions—data on the surface of a sphere can be memorized by the network without cost, whereas data inside the sphere is subject to strong regularization constraints. A unified principle is needed to explain this discrepancy.

**Goal**: This paper introduces the concept of "data shatterability"—the difficulty for ReLU half-spaces to fragment the data distribution—as the core geometric quantity controlling generalization behavior.

## Method

### Overall Architecture

This work conducts a purely theoretical analysis of two-layer ReLU networks $f_{\boldsymbol{\theta}}(\boldsymbol{x}) = \sum_{k=1}^K v_k \phi(\boldsymbol{w}_k^\top \boldsymbol{x} - b_k) + \beta$ where $\phi(z)=\max\{z,0\}$. It adopts the fact that gradient descent stays Below the Edge of Stability (BEoS)—meaning the maximum Hessian eigenvalue satisfies $\lambda_{\max}(\nabla^2_{\boldsymbol{\theta}}\mathcal{L}) \leq 2/\eta$, equivalent to a data-dependent weighted path-norm upper bound—as the sole assumption for generalization. The technical pipeline involves: first, using half-space depth to partition the input space into "shatterable" and "unshatterable" regions to provide their respective generalization contributions; then, instantiating these general bounds for Beta(α) radial distributions and low-dimensional subspace mixtures to quantitatively translate "data shatterability" into generalization rates.

### Key Designs

**1. Half-space Depth Stratification: Translating Data Geometry into Shatterability**

The difficulty lies in the fact that BEoS provides a distribution-independent path-norm constraint; direct use of global metric entropy leads to a dimensionality explosion. This paper introduces the Tukey half-space depth $\text{depth}(\boldsymbol{x}, \mathcal{P}_X) = \inf_{\boldsymbol{u} \in \mathbb{S}^{d-1}} \mathbb{P}(\boldsymbol{u}^\top(\boldsymbol{X} - \boldsymbol{x}) \geq 0)$ to stratify the input space: within a $T$-deep region $\Omega_T$, any ReLU activation boundary passing through it must retain at least a $T$ proportion of data on either side. Thus, the weight function $g(\boldsymbol{u}, t)$ in the weighted path-norm has a positive lower bound, effectively compressing the unweighted path-norm of neurons to $O(1/g_{\min}(T))$. This is the core of "shatterability"—deeper data makes half-spaces harder to "shatter" it, leading to stronger regularization. This yields a key decomposition of the generalization gap: $\sup_{\boldsymbol{\theta} \in \Theta_{\text{BEoS}}} \text{Gap}(f_{\boldsymbol{\theta}}, \mathcal{D}) \leq \tilde{O}(\mathbb{P}(\boldsymbol{X} \notin \Omega_T)) + \tilde{O}(g_{\min}(T)^{-d/(2d+3)} n^{-(d+3)/(4d+6)})$.

**2. Generalization Spectrum of Beta(α) Radial Distributions: A Parameterized Transition**

To map the general bound to analytical geometry, the authors construct isotropic radial distributions $\boldsymbol{X} = h(R)\boldsymbol{U}$ where $h(r) = 1 - (1-r)^{1/\alpha}$, $R \sim \text{Uniform}[0,1]$, and $\boldsymbol{U} \sim \text{Uniform}(\mathbb{S}^{d-1})$. A single parameter $\alpha$ controls mass concentration: high $\alpha$ concentrates mass at the center (low shatterability, good generalization); low $\alpha$ pushes mass toward the surface, allowing the network to pack many disjoint spherical caps and memorize with almost no cost. In the limit $\alpha \to 0$ (on the sphere), networks with width $\leq n$ can perfectly interpolate under BEoS with $\lambda_{\max} \leq 1 + (D^2+2)/n$. The paper provides matching upper bounds (Theorem 3.4, rate $n^{-\alpha(d+3)/(2(d^2+4\alpha d+3\alpha))}$) and lower bounds (Theorem 3.5, rate $n^{-2\alpha/(d-1+2\alpha)}$).

**3. Adaptive Low-dimensional Structures: Breaking the Curse of Dimensionality**

Real-world data often resides on low-dimensional manifolds. This is modeled using a mixture distribution $\mathcal{P}_X = \sum_{j=1}^J \pi_j \mathcal{P}_{X,j}$ where each component is a uniform spherical distribution on an $m$-dimensional affine subspace ($m < d$). The mechanism is that when a network is restricted to subspace $V_j$, neuron activations depend only on the projection $\text{proj}_{V_j} \boldsymbol{w}_k$, and the $d-1$ dimensional separating hyperplanes degenerate into low-dimensional "knots." The proven generalization rate $\text{Gap} \lessapprox_d \left(\frac{1}{\eta} - \frac{1}{2} + 4M\right)^{\frac{m}{m^2+4m+3}} M^2 J^{4/m} n^{-1/(2m+4)}$ shows that the exponent depends only on the intrinsic dimension $m$, explaining why low-dimensional manifold data (like images) is harder to overfit than ambient Gaussian noise.

## Key Experimental Results

### Main Results: Generalization Rate Verification on Isotropic Distributions

In a $d=5$ dimensional space, two-layer ReLU networks with width 1000 were trained on Beta(α) radial distributions for $\alpha \in \{0.1, 0.3, 1.5, 5.0\}$.

| Distribution Parameter $\alpha$ | log-log Slope (Measured) | Theoretical Prediction Trend |
|:---:|:---:|:---:|
| 0.1 | ≈ -0.05 (Minimal generalization) | Mass on sphere → Memorization |
| 0.3 | ≈ -0.12 | Weak generalization |
| 1.5 | ≈ -0.25 | Moderate generalization |
| 5.0 | ≈ -0.38 (Steepest) | Mass at center → Strong generalization |

Higher $\alpha$ corresponds to steeper log-log slopes and faster generalization, consistent with the theory.

### Ablation Study: Intrinsic Dimension Adaptability

20 one-dimensional lines were embedded into $\mathbb{R}^d$ ($d \in \{10, 50, 100, 500\}$):

| Ambient Dimension $d$ | log-log Slope | Total Change |
|:---:|:---:|:---:|
| 10 | ≈ -0.22 | Baseline |
| 50 | ≈ -0.21 | +0.01 |
| 100 | ≈ -0.21 | +0.01 |
| 500 | ≈ -0.20 | +0.02 |

The slope remains nearly constant, indicating that generalization adapts to the intrinsic dimension ($m=1$) and is largely unaffected by the ambient dimension $d$. In contrast, generalization for uniform spherical distributions ($\alpha=1$) significantly worsens as $d$ increases.

### Key Findings: Verification on MNIST

| Data Type | Clean MSE after 20k steps | Behavior |
|:---:|:---:|:---:|
| $\mathcal{N}(0, I_{784})$ | ≈ 1.0 (Noise level) | Rapid Memorization |
| MNIST Images | ≈ 0.2 | Overfitting Resistance |

Gaussian distributions concentrate on a thin shell (high shatterability), leading to quick memorization. MNIST possesses an approximate low-dimensional structure, resisting overfitting. Points with shallower depth in MNIST exhibit higher prediction errors, as predicted by the theory.

## Highlights & Insights

### Novelty
The paper provides an excellent theoretical depth by unifying EoS implicit regularization, data geometry, and generalization under the "shatterability" framework, providing both upper bounds and matching lower bounds. It offers a breakthrough insight by explaining why real data (low-dimensional manifolds) is more difficult to overfit than random Gaussian data—a formal theoretical response to Zhang et al. (2017) "Rethinking Generalization."

### Mechanism
The use of half-space depth stratification avoids the global metric entropy explosion, overcoming the limitations of distribution-independent bounds. This provides theoretical justification for common practices like Mixup data augmentation and activation frequency pruning.

### Limitations & Future Work
1. The analysis is limited to two-layer ReLU networks; extending this to deep networks faces challenges regarding the propagation of EoS regularization.
2. The concentration index $\mathsf{S}_{\text{DQ}}$ for half-space depth is accurately characterized only for isotropic distributions; quantifying shatterability for non-isotropic data remains heuristic.
3. Experimental scale is limited to synthetic data and MNIST; predictive power for more complex data like CIFAR/ImageNet has not been fully verified.

### Rating

⭐⭐⭐⭐⭐

This is a theoretical work of significant depth and breadth, establishing the first quantitative link between EoS implicit regularization and data geometry. The concept of "data shatterability" elegantly unifies previously scattered empirical observations and provides a solid theoretical foundation for understanding why deep learning can break the curse of dimensionality in practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Conflicting Biases at the Edge of Stability: Norm versus Sharpness Regularization](../../ICML2026/optimization/conflicting_biases_at_the_edge_of_stability_norm_versus_sharpness_regularization.md)
- [\[ICLR 2026\] How Muon's Spectral Design Benefits Generalization: A Study on Imbalanced Data](how_muons_spectral_design_benefits_generalization_a_study_on_imbalanced_data.md)
- [\[NeurIPS 2025\] A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](../../NeurIPS2025/optimization/a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)
- [\[ICLR 2026\] FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability–Plasticity Tradeoff](fire_frobenius-isometry_reinitialization_for_balancing_the_stabilityplasticity_t.md)
- [\[ICLR 2026\] Matched Data, Better Models: Target Aligned Data Filtering with Sparse Autoencoders](matched_data_better_models_target_aligned_data_filtering_with_sparse_autoencoder.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] How Muon's Spectral Design Benefits Generalization: A Study on Imbalanced Data](how_muons_spectral_design_benefits_generalization_a_study_on_imbalanced_data.md)
- [\[ICML 2026\] Conflicting Biases at the Edge of Stability: Norm versus Sharpness Regularization](../../ICML2026/optimization/conflicting_biases_at_the_edge_of_stability_norm_versus_sharpness_regularization.md)
- [\[NeurIPS 2025\] A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](../../NeurIPS2025/optimization/a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)
- [\[ICLR 2026\] FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability–Plasticity Tradeoff](fire_frobenius-isometry_reinitialization_for_balancing_the_stabilityplasticity_t.md)
- [\[ICLR 2026\] Matched Data, Better Models: Target Aligned Data Filtering with Sparse Autoencoders](matched_data_better_models_target_aligned_data_filtering_with_sparse_autoencoder.md)

</div>

<!-- RELATED:END -->
