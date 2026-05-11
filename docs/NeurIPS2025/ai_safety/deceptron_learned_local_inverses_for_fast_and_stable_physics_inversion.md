---
title: >-
  [Paper Note] Deceptron: Learned Local Inverses for Fast and Stable Physics Inversion
description: >-
  [NEURIPS2025][AI Safety][inverse problems] This paper proposes the Deceptron bidirectional module, which learns a local inverse of a differentiable forward surrogate and introduces a Jacobian Composition Penalty (JCP). B…
tags:
  - "NEURIPS2025"
  - "AI Safety"
  - "inverse problems"
  - "physics inversion"
  - "learned preconditioning"
  - "Jacobian composition"
  - "Gauss-Newton"
date: 2026-05-08
content_hash: 3cc54ac6046cefe4
---

# Deceptron: Learned Local Inverses for Fast and Stable Physics Inversion

**Conference**: NEURIPS2025  
**arXiv**: [2511.21076](https://arxiv.org/abs/2511.21076)  
**Code**: [aadityakachhadiya/deceptron-ml4ps2025](https://github.com/aadityakachhadiya/deceptron-ml4ps2025)  
**Area**: AI Safety  
**Keywords**: inverse problems, physics inversion, learned preconditioning, Jacobian composition, Gauss-Newton

## TL;DR
This paper proposes the Deceptron bidirectional module, which learns a local inverse of a differentiable forward surrogate and introduces a Jacobian Composition Penalty (JCP). By mapping output-space residuals back to the input space, the method achieves Gauss-Newton-like preconditioned gradient updates for physics inversion, dramatically reducing iteration counts (approximately 20× speedup on Heat-1D).

## Background & Motivation
Inverse problems in physical sciences (e.g., PDE inversion, system identification, imaging) typically require minimizing data misfit in the input space while satisfying physical constraints via projection. However, these optimization objectives are often severely ill-conditioned with poor gradient scaling, necessitating a large number of iterations to converge. Classical Gauss-Newton/Levenberg-Marquardt methods provide second-order curvature information but require solving a linear system at each step, incurring high computational cost. A method that can obtain quasi-second-order search directions with low per-step overhead is therefore desirable.

## Core Problem
How to provide effective preconditioning for physics inverse problems—without explicitly solving Hessian or Jacobian linear systems—such that the optimization update direction approximates the Gauss-Newton direction, thereby substantially reducing the number of iterations required for convergence?

## Method

### Deceptron Bidirectional Module
The forward mapping is defined as $f_W(x) = \sigma(Wx + b)$ and the inverse mapping as $g_V(y) = \tilde{\sigma}(Vy + c)$, where $V$ and $W^\top$ are untied, allowing $g$ to serve as a local inverse even when $W$ is non-orthogonal.

### Training Loss
The joint loss comprises seven terms:

1. **Task loss**: $\lambda_{\text{task}}\|f_W(x) - y^*\|^2$ — forward fitting
2. **Reconstruction loss**: $\lambda_{\text{rec}}\|g_V(f_W(x)) - x\|^2$ — inverse reconstruction consistency
3. **Cycle loss**: $\lambda_{\text{cyc}}\|f_W(g_V(\tilde{y})) - \tilde{y}\|^2$ — forward-inverse cycle consistency
4. **Spectral regularization**: $\beta_{\text{spec}}\|W^\top W - I\|_F^2$ — encourages $W$ to be near-orthogonal
5. **Bias tying**: $\lambda_{\text{tie}}\|b + c\|_2^2$ — soft symmetric bias constraint
6. **Composition regularization**: $\lambda_{\text{comp}}\|VW - I\|_F^2$ — linear-layer inverse constraint
7. **JCP (core)**: $\lambda_{\text{JCP}}\mathbb{E}_\xi\|J_g(f_W(x))J_f(x)\xi - \xi\|^2$ — Jacobian Composition Penalty

The JCP uses the Hutchinson identity to obtain an unbiased estimate of $\|J_g(f(x))J_f(x) - I\|_F^2$ with only 1–4 JVP/VJP probes, ensuring that $g$ acts as a local left inverse of $f$ over the training domain.

### D-IPG Inference Algorithm
When solving $\Phi(x) = \frac{1}{2}\|f_W(x) - y^*\|^2$, each iteration proceeds as follows:

1. Compute output residual $r_t = f_W(x_t) - y^*$
2. Gradient step in output space: $y_{t+1}^{\text{prop}} = y_t - \alpha r_t$
3. Pull back via inverse mapping: $x_{t+1}^{\text{prop}} = g_V(y_{t+1}^{\text{prop}})$
4. Relaxed projection: $x_{t+1} = \Pi_{\mathcal{C}}((1-\rho)x_t + \rho x_{t+1}^{\text{prop}})$
5. Armijo backtracking to verify acceptance condition

**Key theoretical connection**: Under first-order approximation, $g(y_t - \alpha r_t) \approx x_t - \alpha J_g(f(x_t)) r_t$. When $J_g(f(x)) \approx J_f(x)^+ = (J^\top J)^{-1}J^\top$, the D-IPG update direction is equivalent to that of Gauss-Newton. A lower JCP penalty implies closer adherence to this ideal behavior.

### DeceptronNet v0 (2D Extension)
A lightweight unrolled corrector for image inverse problems: the input three-channel feature $F_t = [\uparrow y, \uparrow r_t, x_t]$ is passed through a small U-Net to predict a correction $\Delta x_t$; a learnable gain $\alpha_t = \sigma(\gamma_t) \in (0,1)$ controls the step size, with a fixed unroll of $N=6$ steps.

## Key Experimental Results

### Heat-1D Initial Condition Recovery

| Method | Median Iterations [IQR] | Per-step Time | Total Time |
|--------|-------------------------|---------------|------------|
| x-GD | 49.0 [38.2, 80.0] | 0.43 ms | 0.026 s |
| D-IPG | **3.0 [2.0, 3.0]** | 0.51 ms | **0.001 s** |
| GN/LM | 3.0 [2.0, 3.0] | 3.82 ms | 0.011 s |

D-IPG matches GN/LM in iteration count, yet incurs only 1/7 the per-step cost of GN/LM, achieving an 11× total speedup.

### Damped Oscillator Inverse Problem

| Method | Median Iterations [IQR] | Per-step Time | Total Time |
|--------|-------------------------|---------------|------------|
| x-GD | 65.0 [1.0, 104.5] | 0.45 ms | 0.004 s |
| D-IPG | 28.0 [1.0, 34.0] | 1.28 ms | 0.001 s |
| GN/LM | 16.5 [1.0, 33.2] | 4.22 ms | 0.007 s |

### DeceptronNet v0 2D PSF Recovery

| Method | RMSE | Iterations | Time |
|--------|------|------------|------|
| LM | 0.0883 | 69.25 | - |
| x-GD | 0.1271 | 80.00 | - |
| DNet v0 | **0.0640** | **6.00** | - |

DNet v0 achieves the lowest error with a fixed 6 steps under an identical fair-comparison protocol.

### Ablation Study
- Removing JCP: composition residual rises from near zero to 457.7; iterations increase from 2.6 to 3.8.
- Tying $V = W^\top$: iterations increase dramatically from 2.8 to 16.2; acceptance rate drops from 0.58 to 0.061.
- Removing reconstruction/cycle losses: performance is largely unchanged, indicating that the preconditioning effect is primarily driven by JCP.

## Highlights & Insights
- **Theoretical elegance**: JCP combines the Hutchinson probe with local inverse learning to establish a theoretical equivalence between D-IPG and Gauss-Newton.
- **Fair comparison protocol**: All methods share the same projector, Armijo backtracking, relaxation parameters, initialization, and stopping criteria, eliminating experimental bias.
- **RJCP runtime diagnostic**: Provides a monitoring metric at inference time to detect when the surrogate model operates outside its valid regime.
- **Cost efficiency**: JCP introduces overhead only during training; at inference, D-IPG requires only one forward pass, one inverse pass, and one gradient computation—no linear solves needed.

## Limitations & Future Work
- **Locality**: The inverse mapping $g$ is valid only near the training distribution; out-of-distribution inputs may yield overconfident reconstructions.
- **Surrogate fidelity**: The current framework assumes a sufficiently accurate differentiable forward surrogate; deviations from true physics models are not thoroughly investigated.
- **Limited scale**: Core experiments are restricted to 1D Heat and low-dimensional oscillator problems; DeceptronNet v0 remains a single-scale prototype.
- **Limited nonlinearity**: The Deceptron itself is a shallow linear-plus-activation structure, which may lack expressiveness for highly nonlinear inverse problems.
- Future directions include multi-scale DeceptronNet, validation on real-world data, and integration with deep unrolling networks (e.g., LISTA, PnP).

## Related Work & Insights
- **vs. Gauss-Newton/LM**: Matches in iteration count but is significantly lighter per step (no CG linear solve required).
- **vs. PINNs**: Does not train directly on physical equations; instead learns a local inverse of an existing surrogate.
- **vs. learned unrolling (LISTA, PnP, RED)**: D-IPG retains the standard projection loop structure and only replaces the update direction; DNet v0 follows the unrolling paradigm but emphasizes fair comparison.
- **vs. L-BFGS**: On Kodak24 experiments, L-BFGS requires 80–100 steps, whereas DNet requires only 6.

## Connections & Inspirations
- The Hutchinson probe idea underlying JCP can be generalized to other settings requiring Jacobian-approximation regularization (e.g., invertibility constraints in generative models).
- The use of RJCP as a runtime diagnostic metric is broadly applicable—it can detect degradation in any system involving learned inverse mappings.
- The paradigm of *learning* optimization preconditioners rather than designing them by hand may inspire large-scale scientific computing applications such as climate model inversion and medical image reconstruction.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of JCP-based local inverse learning with preconditioned gradient updates is original.
- Experimental Thoroughness: ⭐⭐⭐ — The fair-comparison protocol is rigorous, but the problem scale is small and large-scale validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear, and the fair-comparison protocol is described in thorough detail.
- Value: ⭐⭐⭐ — The direction is promising, but the range of practically validated scenarios is currently limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Taught Well, Learned Ill: Towards Distillation-Conditional Backdoor Attack](taught_well_learned_ill_towards_distillation-conditional_backdoor_attack.md)
- [\[NeurIPS 2025\] Model Inversion with Layer-Specific Modeling and Alignment for Data-Free Continual Learning](model_inversion_with_layer-specific_modeling_and_alignment_for_data-free_continu.md)
- [\[NeurIPS 2025\] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)
- [\[NeurIPS 2025\] Boosting Adversarial Transferability with Spatial Adversarial Alignment](boosting_adversarial_transferability_with_spatial_adversarial_alignment.md)
- [\[NeurIPS 2025\] FLUX: Efficient Descriptor-Driven Clustered Federated Learning under Arbitrary Distribution Shifts](flux_efficient_descriptor-driven_clustered_federated_learning_under_arbitrary_di.md)

</div>

<!-- RELATED:END -->
