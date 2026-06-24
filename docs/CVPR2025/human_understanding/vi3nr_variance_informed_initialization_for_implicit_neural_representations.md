---
title: >-
  [Paper Note] VI3NR: Variance Informed Initialization for Implicit Neural Representations
description: >-
  [CVPR 2025][Human Understanding][INR Initialization] VI3NR, a variance-informed initialization method for implicit neural representations (INRs) applicable to arbitrary activation functions, is derived, generalizing Xavier/Kaiming initialization to non-standard activations such as Gaussian and Sinc. By controlling the variance consistency of forward and backward propagation, stability in both directions is simultaneously satisfied using a single degree of freedom $\sigma_p^2$…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "INR Initialization"
  - "Variance Propagation"
  - "General Activation"
  - "Xavier Generalization"
  - "Monte Carlo"
date: 2026-05-08
content_hash: 0402c380d66aa7a4
---

# VI3NR: Variance Informed Initialization for Implicit Neural Representations

**Conference**: CVPR 2025  
**arXiv**: [2504.19270](https://arxiv.org/abs/2504.19270)  
**Code**: To be released  
**Area**: Human Understanding / Neural Representations  
**Keywords**: INR Initialization, Variance Propagation, General Activation, Xavier Generalization, Monte Carlo

## TL;DR

VI3NR, a variance-informed initialization method for implicit neural representations (INRs) applicable to arbitrary activation functions, is derived, generalizing Xavier/Kaiming initialization to non-standard activations such as Gaussian and Sinc. By controlling the variance consistency of forward and backward propagation, stability in both directions is simultaneously satisfied using a single degree of freedom $\sigma_p^2$, significantly improving the convergence speed and reconstruction quality of INRs.

## Background & Motivation

### Background

**Background**: INRs utilize various novel activation functions (such as Sine, Gaussian, Sinc, and Wavelet) instead of ReLU to enhance frequency fitting capabilities. However, each activation function requires a specially derived initialization (e.g., SIREN derives a specific initialization for Sine), lacking a general approach.

**Limitations of Prior Work**: (1) Xavier and Kaiming initializations are only applicable to standard activations such as ReLU and tanh; (2) SIREN's initialization is designed exclusively for Sine and cannot be directly applied to Gaussian, Sinc, etc.; (3) Incorrect initialization leads to variance explosion or vanishing in forward propagation and unstable gradients in backward propagation, causing INR training to collapse.

**Key Challenge**: Forward propagation variance stability ($\text{Var}[z_l] = \sigma_p^2$) and backward propagation gradient stability ($\text{Var}[\frac{\partial L}{\partial z_l}]$ is constant) are two constraints, which usually leave one degree of freedom—how to choose this degree of freedom?

**Key Insight**: Deriving general forward and backward variance propagation equations (for arbitrary activation functions), and then using grid search or Monte Carlo estimation to find the optimal $\sigma_p^2$.

**Core Idea**: General variance propagation equations + grid search over a single degree of freedom = INR initialization applicable to arbitrary activation functions.

## Method

### Overall Architecture


### Key Designs

1. **General Forward Variance Equation**: $\sigma^2(W_i) = \frac{\sigma_p^2}{M_i(\mu^2(x_i) + \sigma^2(x_i))}$—Given the target pre-activation variance $\sigma_p^2$, the weight variance of each layer is derived. This applies to arbitrary activation functions (by calculating their mean and variance).

2. **Backward Stability Constraint**: $M_{i+1} \sigma^2(W_i)(\mu^2(f'(z_i)) + \sigma^2(f'(z_i))) = 1$—Leveraging the statistics of the activation function's derivative.

3. **Monte Carlo Statistical Estimation**: For non-standard activations (Gaussian, Sinc, etc.), Monte Carlo estimation with $\ge 10\text{K}$ samples is used to estimate statistics such as $\mu(f(z)), \sigma^2(f(z)), \mu(f'(z))$, which is more accurate than Taylor expansions.

### Loss & Training

Standard MSE reconstruction loss. Hyperparameter search overhead is 10-20 minutes (vs. 5+ hours for activation parameter search).

## Key Experimental Results

| Activation Function | Forward Error $E_f$ (VI3NR/Baseline) | Description |
|---------|-------------------------|------|
| Gaussian | **0.9** / 6.7 | 7.4$\times$ improvement |
| Sinc | Significant improvement | — |
| Sine | Matches SIREN | Consistency verified |

Audio and 3D surface reconstruction also show significant convergence acceleration and quality improvement.

### Ablation Study
- Monte Carlo > Taylor approximation—10K samples are sufficiently accurate.
- Selecting $\sigma_p^2$ based on backward conditions—minimizing both forward and backward errors simultaneously.
- PyTorch's built-in gain values are highly consistent with analytical predictions (tanh).

### Key Findings
- **A single degree of freedom $\sigma_p^2$ is sufficient to simultaneously satisfy forward and backward stability**—no separate design is needed.
- **New activations like Gaussian/Sinc benefit the most**—Sine already has SIREN initialization, hence showing minor improvement.
- **The initialization time overhead is negligible (10-20 minutes) vs. the resulting convergence acceleration (saving hours of training time)**.

## Highlights & Insights
- **Natural generalization of Xavier/Kaiming**—extending classical initialization theory to arbitrary activations in a completely general way.
- **Highly practical**—any new INR activation function can directly use this method to derive the optimal initialization.

## Limitations & Future Work
- Theoretical assumption of large network width (CLT convergence).
- The importance of the backward condition may diminish in shallow networks (typically 8 layers for INRs).
- Convolutional/recurrent architectures are not addressed.

## Rating
- Novelty: ⭐⭐⭐⭐ Natural generalization of Xavier/Kaiming
- Experimental Thoroughness: ⭐⭐⭐⭐ Image/audio/3D multimodal
- Writing Quality: ⭐⭐⭐⭐⭐ Elegant derivation
- Value: ⭐⭐⭐⭐ A practical fundamental tool for the INR community

## Related Work & Insights
- **vs. Representative methods in the same area**: Ours makes unique contributions to method design, complementing existing methods.
- **vs. Traditional methods**: Compared with traditional schemes, our method achieves significant improvements on key metrics.
- **Insights**: The technical route of this work has important reference value for future related work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Quaffure: Real-Time Quasi-Static Neural Hair Simulation](quaffure_real-time_quasi-static_neural_hair_simulation.md)
- [\[CVPR 2025\] NBAvatar: Neural Billboards Avatars with Realistic Hand-Face Interaction](nbavatar_neural_billboards_avatars_with_realistic_hand-face_interaction.md)
- [\[ICCV 2025\] ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling](../../ICCV2025/human_understanding/imhead_a_large-scale_implicit_morphable_model_for_localized_head_modeling.md)
- [\[ICCV 2025\] NGD: Neural Gradient Based Deformation for Monocular Garment Reconstruction](../../ICCV2025/human_understanding/ngd_neural_gradient_based_deformation_for_monocular_garment_reconstruction.md)
- [\[ECCV 2024\] Alignist: CAD-Informed Orientation Distribution Estimation by Fusing Shape and Correspondences](../../ECCV2024/human_understanding/alignist_cad-informed_orientation_distribution_estimation_by_fusing_shape_and_co.md)

</div>

<!-- RELATED:END -->
