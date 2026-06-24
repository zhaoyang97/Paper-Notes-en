---
title: >-
  [Paper Note] Accurate Differential Operators for Hybrid Neural Fields
description: >-
  [CVPR 2025][Physics & Scientific Computing][Hybrid Neural Fields] This paper reveals that gradients and curvatures computed via automatic differentiation in hybrid neural fields (e.g., Instant NGP) suffer from severe high-frequency noise. It proposes a post-processing differential operator based on local polynomial fitting and a self-supervised fine-tuning method, reducing gradient and curvature errors by 4x, which significantly eliminates artifacts in rendering and physical…
tags:
  - "CVPR 2025"
  - "Physics & Scientific Computing"
  - "Hybrid Neural Fields"
  - "Differential Operators"
  - "Local Polynomial Fitting"
  - "High-frequency Noise"
  - "SDF"
date: 2026-05-08
content_hash: 23cf3dae189b4eac
---

# Accurate Differential Operators for Hybrid Neural Fields

**Conference**: CVPR 2025  
**arXiv**: [2312.05984](https://arxiv.org/abs/2312.05984)  
**Code**: [https://justachetan.github.io/hnf-derivatives/](https://justachetan.github.io/hnf-derivatives/)  
**Area**: Scientific Computing / 3D Vision  
**Keywords**: Hybrid Neural Fields, Differential Operators, Local Polynomial Fitting, High-frequency Noise, SDF

## TL;DR

This paper reveals that gradients and curvatures computed via automatic differentiation in hybrid neural fields (e.g., Instant NGP) suffer from severe high-frequency noise. It proposes a post-processing differential operator based on local polynomial fitting and a self-supervised fine-tuning method, reducing gradient and curvature errors by 4x, which significantly eliminates artifacts in rendering and physical simulations.

## Background & Motivation

**Background**: Hybrid neural fields represent spatial signals using small MLPs with explicit feature grids (such as hash grids) to achieve fast training and scale to large scenes. Instant NGP is a representative method.

**Limitations of Prior Work**: Although hybrid neural fields achieve high-fidelity fitting for zero-order signals (values), their first- and second-order derivatives (gradients, curvatures) obtained via automatic differentiation (autodiff) suffer from severe noise, causing graying normal maps in rendering and abnormal behaviors in physical simulations.

**Key Challenge**: The high resolution of feature grids enables hybrid neural fields to capture fine details, but also introduces high-frequency noise components. Although this noise has a negligible amplitude in zero-order signals, the differentiation operation amplifies it proportionally to the frequency ($\frac{d \sin(2\pi\nu x)}{dx} = 2\pi\nu\cos(2\pi\nu x)$), leading to severe noise amplification in derivatives.

**Goal**: To design differential operators robust to high-frequency noise that can be applied to any pre-trained hybrid neural field.

**Key Insight**: A classic method in signal processing to handle differentiation of noisy signals is smoothing prior to differentiation. For continuous neural fields, direct differentiation can be replaced by local low-order polynomial fitting.

**Core Idea**: Sample around the query point, fit a local low-order polynomial (linear/quadratic), and replace the autodiff derivative of the neural field with the analytical derivative of the polynomial.

## Method

### Overall Architecture

Two complementary schemes: (1) **Post-processing operator**—obtains accurate derivatives at any query point via local polynomial fitting on pre-trained neural fields, without modifying them; (2) **Self-supervised fine-tuning**—uses accurate derivatives from polynomial fitting as supervisory signals to fine-tune the neural field, making its own autodiff derivatives accurate.

### Key Designs

1. **Local Polynomial Fitting Differential Operator**:

    - **Function**: Obtains accurate spatial derivatives without modifying the pre-trained neural field.
    - **Mechanism**: Samples $k$ points $\mathbf{x}_i$ within the local neighborhood $N(\mathbf{q})$ of a query point $\mathbf{q}$, queries the neural field to get $y_i = F_\Theta(\mathbf{x}_i)$, and then uses least squares to fit a low-order polynomial (linear fitting for gradient $\hat{\mathbf{g}}$, quadratic fitting for Hessian and curvature). Polynomial fitting essentially performs local smoothing, automatically filtering out high-frequency noise. The sampling neighborhood size controls the smoothing scale.
    - **Design Motivation**: Low-order polynomials naturally do not contain high-frequency components, making the fitting process equivalent to a joint operation of smoothing and differentiation. The closed-form solution (least squares) is computationally efficient.

2. **Self-Supervised Fine-Tuning**:

    - **Function**: Makes the neural field's own autodiff derivatives accurate without changing downstream pipelines.
    - **Mechanism**: During the fine-tuning stage, two objectives are optimized simultaneously: (1) reconstruction loss to keep the original signal unchanged; (2) derivative consistency loss—enforcing the autodiff gradient $\nabla F_\Theta(\mathbf{q})$ to align with the polynomial-fitted gradient $\hat{\mathbf{g}}(\mathbf{q})$ of the query point. The fine-tuned neural field can then directly provide accurate derivatives via autodiff.
    - **Design Motivation**: The post-processing operator requires modifying downstream code, whereas the fine-tuning scheme keeps the interface unchanged, making it transparent to existing rendering/simulation pipelines.

### Loss & Training

Fine-tuning loss: $\mathcal{L} = \mathcal{L}_\text{recon} + \lambda \mathcal{L}_\text{grad}$, where $\mathcal{L}_\text{grad} = \|\nabla F_\Theta(\mathbf{q}) - \hat{\mathbf{g}}(\mathbf{q})\|^2$.

## Key Experimental Results

### Main Results

| Method | Gradient Angular Error ↓ | Curvature Error ↓ |
|------|-------------|----------|
| Autodiff | 4.2° | 0.83 |
| Finite Difference | 3.1° | 0.64 |
| Eikonal Regularization | 3.8° | 0.71 |
| **Polynomial Fitting (Ours)** | **1.0°** | **0.16** |
| **Fine-tuning (Ours)** | **1.5°** | **0.25** |

### Ablation Study

| Configuration | Gradient Error | Explanation |
|------|---------|------|
| Linear fitting | 1.0° | First-order derivative |
| Quadratic fitting | 0.8° | First- and second-order derivatives |
| Small neighborhood radius | 1.5° | Under-smoothing |
| Large neighborhood radius | 1.2° | Over-smoothing |

### Key Findings
- Polynomial fitting reduces gradient error by ~4x (4.2° → 1.0°) and curvature error by ~5x.
- Although finite difference yields partial improvements, it performs poorly on curvatures (second-order derivatives).
- Eikonal regularization is effective for pure MLP neural fields but does not work for hybrid neural fields.
- In rendering and collision simulation, utilizing accurate derivatives visually eliminates artifacts.

## Highlights & Insights
- **Precise Problem Identification**: The paper clearly explains why hybrid neural field derivatives are noisy from a spectral analysis perspective, rather than simply blaming it on "insufficient training".
- **Novel Application of Classic Methods**: Local polynomial fitting is a classic signal processing tool, but applying it to the continuous query scenario of hybrid neural fields is novel.
- **Complementary Schemes**: The post-processing scheme offers higher accuracy but requires code modifications, while the fine-tuning scheme has slightly lower accuracy but is transparent to downstream tasks.

## Limitations & Future Work
- The post-processing operator requires multiple queries of neighboring points, which increases inference time.
- Selecting the neighborhood size requires a trade-off between smoothing extent and detail preservation.
- Currently verified primarily on SDFs; applicability to other signal types like radiance fields requires further investigation.

## Related Work & Insights
- **vs Instant NGP**: Direct use of autodiff derivatives in Instant NGP is noisy, which this method successfully fixes.
- **vs Li et al. (Finite Difference Regularization)**: They focus on training dynamics, whereas this study focuses on high-frequency noise, addressing different aspects.
- The core idea can be extended to other explicit-implicit hybrid representations, such as 3D Gaussian Splatting.

## Rating
- Novelty: ⭐⭐⭐⭐ Deep problem identification; the solution is based on classic methods but appropriately adapted.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across three downstream applications: rendering, simulation, and PDEs.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear and thorough spectral analysis and visualization.
- Value: ⭐⭐⭐⭐ Significantly advances the practical application of hybrid neural fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] INC: An Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers](../../NeurIPS2025/physics/inc_an_indirect_neural_corrector_for_auto-regressive_hybrid_pde_solvers.md)
- [\[ICLR 2026\] Scaling Laws and Symmetry, Evidence from Neural Force Fields](../../ICLR2026/physics/scaling_laws_and_symmetry_evidence_from_neural_force_fields.md)
- [\[ICLR 2026\] Fast training of accurate physics-informed neural networks without gradient descent](../../ICLR2026/physics/fast_training_of_accurate_physics-informed_neural_networks_without_gradient_desc.md)
- [\[NeurIPS 2025\] Towards Universal Neural Operators through Multiphysics Pretraining](../../NeurIPS2025/physics/towards_universal_neural_operators_through_multiphysics_pretraining.md)
- [\[ICLR 2026\] Adaptive Mamba Neural Operators](../../ICLR2026/physics/adaptive_mamba_neural_operators.md)

</div>

<!-- RELATED:END -->
