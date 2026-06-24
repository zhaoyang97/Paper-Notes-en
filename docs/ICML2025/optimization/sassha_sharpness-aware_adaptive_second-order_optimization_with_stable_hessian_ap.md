---
title: >-
  [Paper Note] Sassha: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation
description: >-
  [ICML2025][Optimization][Second-order optimization] This work proposes the Sassha optimizer, which introduces sharpness-aware minimization (SAM) into the second-order optimization framework. Through stable Hessian approximation and a lazy update strategy, it enables second-order methods to comprehensively outperform first-order methods such as SGD, AdamW, and SAM in generalization performance for the first time.
tags:
  - "ICML2025"
  - "Optimization"
  - "Second-order optimization"
  - "sharpness-aware minimization"
  - "Hessian approximation"
  - "generalization"
  - "loss landscape"
date: 2026-05-08
content_hash: 642a38d3caa66c55
---

# Sassha: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation

**Conference**: ICML2025  
**arXiv**: [2502.18153](https://arxiv.org/abs/2502.18153)  
**Code**: [GitHub](https://github.com/LOG-postech/Sassha)  
**Area**: Optimization  
**Keywords**: Second-order optimization, sharpness-aware minimization, Hessian approximation, generalization, loss landscape

## TL;DR

This work proposes the Sassha optimizer, which introduces sharpness-aware minimization (SAM) into the second-order optimization framework. Through stable Hessian approximation and a lazy update strategy, it enables second-order methods to comprehensively outperform first-order methods such as SGD, AdamW, and SAM in generalization performance for the first time.

## Background & Motivation

Approximate second-order optimization methods (such as AdaHessian, Sophia-H, Shampoo) utilize curvature information to accelerate convergence, but often underperform compared to simple SGD in terms of generalization. This paper explains this phenomenon from the perspective of **loss landscape flatness**:

- The authors measure multiple sharpness metrics (maximum Hessian eigenvalue $\lambda_{\max}(H)$, trace $\text{tr}(H)$, sharpness along the gradient direction $\delta L_{\text{grad}}$, and average sharpness $\delta L_{\text{avg}}$) on ResNet-32 / CIFAR-100. They find that the minima to which existing second-order methods converge are **several orders of magnitude sharper** than those found by SGD.
- For instance, the $\lambda_{\max}(H)$ of Shampoo reaches as high as 436,374, while that of SGD is only 265, and Sassha reduces it further to 107.
- This negative correlation between sharp minima and generalization ability is well-supported by extensive theoretical and empirical studies.

**Core Problem**: Can the sharpness of the solutions found by second-order methods be explicitly minimized to unleash their generalization potential?

## Method

### 1. Sharpness-Aware Second-Order Optimization Framework

Sassha solves the following min-max problem:

$$\min_{x} \max_{\|\epsilon\|_2 \le \rho} f(x + \epsilon)$$

Similar to SAM, the perturbation direction is obtained by solving the inner maximization via a first-order approximation:

$$\epsilon_t^\star = \rho \frac{\nabla f(x_t)}{\|\nabla f(x_t)\|_2}$$

A second-order Taylor expansion is then applied to the perturbed objective function $\tilde{f}_t(x) = f(x + \epsilon_t^\star)$, yielding the following update rule:

$$x_{t+1} = x_t - H(x_t + \epsilon_t^\star)^{-1} \nabla f(x_t + \epsilon_t^\star)$$

### 2. Stable Hessian Approximation

Sharpness minimization tends to drive Hessian entries close to zero, leading to numerical instability in diagonal Hessian estimation. Sassha employs three key designs:

- **Square-Root Scaling**: Using $|{\hat{H}}|^{1/2}$ instead of the original Hessian smoothly scales up near-zero entries (since $\sqrt{h} > h$ when $0 < h < 1$), preserving the relative scale of each dimension without requiring extra hyperparameters (in contrast to damping/clipping).
- **Absolute Value Operation**: Using $|\hat{H}| = \sum_i |\hat{H}_{ii}| \mathbf{e}_i \mathbf{e}_i^\top$ flips the sign of negative curvature directions to prevent convergence to saddle points.
- **Exponential Moving Average**: Applying an EMA to smooth the Hessian diagonal and reduce stochastic estimation noise.

### 3. Lazy Hessian Updates

The Hessian is recomputed only once every $k$ steps (default $k=10$), significantly reducing computational overhead:

$$D_t = \begin{cases} \beta_2 D_{t-1} + (1-\beta_2)|\hat{H}(x_t+\epsilon_t^\star)| & \text{if } t \bmod k = 1 \\ D_{t-1} & \text{otherwise}\end{cases}$$

Key discovery: Sassha exhibits significantly stronger robustness to lazy updates than other second-order methods, because sharpness minimization guides the optimization trajectory through regions with low curvature variations, allowing historical Hessians to remain valid for more steps.

### 4. Complete Update Rule

$$x_{t+1} = x_t - \eta_t \bar{D}_t^{-1} \bar{m}_t - \eta_t \lambda x_t$$

Where $\bar{m}_t$ is the bias-corrected first-order moment of the gradient at the perturbed point, and $\bar{D}_t = \sqrt{D_t/(1-\beta_2^t)}$ is the bias-corrected Hessian square root.

## Key Experimental Results

### Image Classification (Validation Accuracy %)

| Method | CIFAR-10 ResNet-20 | CIFAR-100 ResNet-32 | CIFAR-100 WRN-28-10 | ImageNet ResNet-50 | ImageNet ViT-s-32 |
|---|---|---|---|---|---|
| SGD | 92.03 | 69.32 | 80.06 | 75.58 | 62.90 |
| AdamW | 92.04 | 68.78 | 79.09 | 75.38 | 66.46 |
| SAM_SGD | 92.85 | 71.99 | 83.14 | 76.36 | 64.54 |
| SAM_AdamW | 92.77 | 71.15 | 82.88 | 76.35 | 68.31 |
| AdaHessian | 92.00 | 68.06 | 76.92 | 73.64 | 66.42 |
| Sophia-H | 91.81 | 67.76 | 79.35 | 72.06 | 62.44 |
| Shampoo | 88.55 | 64.08 | 74.06 | — | — |
| **Sassha** | **92.98** | **72.14** | **83.54** | **76.43** | **69.20** |

### Language Model Pre-training (GPT1-mini, Wikitext-2 Perplexity ↓)

| Method | Perplexity |
|---|---|
| AdamW | 175.06 |
| SAM_AdamW | 158.06 |
| AdaHessian | 407.69 |
| Sophia-H | 157.60 |
| **Sassha** | **122.40** |

### Sharpness Comparison (ResNet-32 CIFAR-100)

| Method | $\lambda_{\max}(H)$ | $\text{tr}(H) \times 10^3$ | Validation Accuracy |
|---|---|---|---|
| SGD | 265 | 7.29 | 69.32% |
| AdaHessian | 11992 | 46.94 | 68.06% |
| Sophia-H | 22797 | 68.15 | 67.76% |
| Shampoo | 436374 | 6823 | 64.08% |
| **Sassha** | **107** | **1.87** | **72.14%** |

## Highlights & Insights

1. **First to diagnose sharpness as the root cause of poor generalization in second-order methods**: This work completely quantifies this phenomenon using four metrics, explaining the poor generalization performance of second-order methods that has puzzled the community for years.
2. **An elegant fusion of SAM and second-order optimization**: Rather than simple stacking, the authors discover that sharpness minimization destabilizes Hessian estimation, and propose square-root scaling as an elegant, hyperparameter-free solution.
3. **Unexpected benefits from Lazy Hessian Updates**: Sharpness minimization naturally causes the Hessian to vary more slowly along the trajectory, rendering the lazy update strategy highly effective and creating a positive cycle of efficiency and performance.
4. **Comprehensively outperforming first-order baselines**: Sassha outperforms SGD, AdamW, and SAM across six vision tasks and language pre-training, representing a pioneering milestone for second-order methods.
5. **Rigorous theoretical support**: Convergence proofs and linear stability analyses explain why Sassha inherently prefers flat minima.

## Limitations & Future Work

1. **Computational overhead still higher than first-order methods**: Even with $k=10$ lazy updates, an extra Hessian-vector product backpropagation step is required every 10 steps, resulting in a wall-clock time approximately 1.1-1.2x that of SGD.
2. **Inherent limitations of diagonal Hessian approximation**: By ignoring non-diagonal structural curvature information, the approximation may be less precise in highly non-diagonal loss landscapes.
3. **Hyperparameter selection**: The interaction effects between $\rho$ (perturbation radius) and $k$ (Hessian update interval) are not fully explored.
4. **Insufficient large-scale validation**: Language modeling experiments are only validated on GPT1-mini and SqueezeBERT, lacking evaluations on billion-parameter models.
5. **Convergence analysis limited to convex cases**: Theoretical guarantees assume twice-differentiability and convexity, which diverges from the non-convex reality of deep learning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The integration of SAM with second-order optimization and the stability designs are highly novel and insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across vision and language, CNNs, ViTs, and Transformers, utilizing multiple sharpness metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, rigorous logic, with theory and experiments validating each other.
- Value: ⭐⭐⭐⭐ — Opens up new directions for the practical application of second-order optimization in deep learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Tilted Sharpness-Aware Minimization](tilted_sharpness-aware_minimization.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICML 2025\] Efficient Curvature-Aware Hypergradient Approximation for Bilevel Optimization](efficient_curvature-aware_hypergradient_approximation_for_bilevel_optimization.md)
- [\[ICLR 2026\] MASAM: Multimodal Adaptive Sharpness-Aware Minimization for Heterogeneous Data Fusion](../../ICLR2026/optimization/masam_multimodal_adaptive_sharpness-aware_minimization_for_heterogeneous_data_fu.md)
- [\[ICLR 2026\] The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton](../../ICLR2026/optimization/the_potential_of_second-order_optimization_for_llms_a_study_with_full_gauss-newt.md)

</div>

<!-- RELATED:END -->
