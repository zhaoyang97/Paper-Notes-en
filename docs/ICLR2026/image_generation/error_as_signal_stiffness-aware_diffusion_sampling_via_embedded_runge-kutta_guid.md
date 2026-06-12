---
title: >-
  [Paper Note] Error as Signal: Stiffness-Aware Diffusion Sampling via Embedded Runge-Kutta Guidance
description: >-
  [ICLR2026][Image Generation][diffusion sampling] This paper proposes ERK-Guid, which leverages the order-difference error of embedded Runge-Kutta solvers as a guidance signal to adaptively correct local truncation error…
tags:
  - "ICLR2026"
  - "Image Generation"
  - "diffusion sampling"
  - "stiffness"
  - "local truncation error"
  - "embedded Runge-Kutta"
  - "guidance"
date: 2026-05-08
content_hash: 5efa02ab1d8b8b0a
---

# Error as Signal: Stiffness-Aware Diffusion Sampling via Embedded Runge-Kutta Guidance

**Conference**: ICLR2026
**arXiv**: [2603.03692](https://arxiv.org/abs/2603.03692)  
**Code**: [mlvlab/ERK-Guid](https://github.com/mlvlab/ERK-Guid)  
**Area**: Image Generation
**Keywords**: diffusion sampling, stiffness, local truncation error, embedded Runge-Kutta, guidance

## TL;DR

This paper proposes ERK-Guid, which leverages the order-difference error of embedded Runge-Kutta solvers as a guidance signal to adaptively correct local truncation error (LTE) in stiff regions, improving diffusion model sampling quality without additional network evaluations.

## Background & Motivation

- Diffusion model sampling is essentially solving an ODE/SDE; sampling quality depends on both model accuracy and numerical solver accuracy.
- Methods such as Classifier-Free Guidance (CFG) and Autoguidance (AG) address **model error** (conditional/unconditional prediction discrepancy) but entirely neglect **solver error** (LTE).
- In **stiff regions** of the ODE, the drift direction changes sharply, causing the LTE of numerical solvers to significantly degrade sampling quality.
- Key observation: in stiff regions, the LTE aligns closely with the dominant eigenvector of the drift Jacobian, suggesting that this directional information can be exploited to correct errors.

## Core Problem

Existing guidance methods (CFG, AG, etc.) exploit only model-level signals to guide sampling, leaving the LTE induced by solvers in stiff regions unaddressed. The central question is: how can the solver's own error information be used as a guidance signal to reduce LTE without increasing the number of network evaluations?

## Method

### 1. Theoretical Foundation: Alignment of LTE with the Dominant Eigenvector

For the Heun method (a second-order Runge-Kutta scheme), both an Euler (first-order) and a Heun (second-order) solution are produced simultaneously, forming an embedded Runge-Kutta (ERK) pair. Define:

- **ERK solution difference**: $\Delta^{\mathbf{x}} = \mathbf{x}^{\text{Heun}} - \mathbf{x}^{\text{Euler}}$
- **ERK drift difference**: $\Delta^{\mathbf{f}} = f(\mathbf{x}^{\text{Heun}}; \sigma) - f(\mathbf{x}^{\text{Euler}}; \sigma)$

Under a local linearization assumption, both the LTE and the ERK solution difference can be decomposed in the Jacobian eigenbasis. When $|z_k| = |h\lambda_k|$ is large (i.e., in stiff regions), the component along the dominant eigenvector dominates these errors, achieving alignment.

### 2. Zero-Cost Estimators

**Stiffness estimator**: The ratio of the norm of the ERK drift difference to the norm of the ERK solution difference approximates the largest eigenvalue of the Jacobian:

$$\hat{\rho}_{\text{stiff}} = \frac{\|f(\mathbf{x}^{\text{Heun}}; \sigma) - f(\mathbf{x}^{\text{Euler}}; \sigma)\|_2}{\|\mathbf{x}^{\text{Heun}} - \mathbf{x}^{\text{Euler}}\|_2}$$

**Dominant eigenvector estimator**: The normalized ERK drift difference serves as an estimate of the dominant eigenvector, since the drift difference approximates the action of the Jacobian on the solution difference (equivalent to one step of Jacobian-vector product power iteration), which naturally amplifies the dominant eigendirection:

$$\hat{\mathbf{v}}_{\text{stiff}} = \frac{\Delta^{\mathbf{f}}}{\|\Delta^{\mathbf{f}}\|_2}$$

All quantities required by both estimators are already computed during the Heun update, requiring **no additional network calls**.

### 3. ERK-Guid Update Formula

$$\hat{\mathbf{x}}^{\text{Heun}}_{\sigma_{i+1}} = \mathbf{x}^{\text{Heun}}_{\sigma_{i+1}} - h \cdot \beta \cdot z^2 \cdot \langle f^{\text{Heun}}_{\sigma_i}, \hat{\mathbf{v}}_{\sigma_i} \rangle \cdot \hat{\mathbf{v}}_{\sigma_i}$$

where:
- $\beta = \mathbf{1}_{\{\hat{\rho} > w_{\text{con}}\}}$ is a confidence gate that activates guidance only when stiffness exceeds a threshold;
- $z = w_{\text{stiff}} \cdot h \cdot \hat{\rho}$ is an adaptive scaling factor;
- $w_{\text{stiff}}$ controls overall guidance strength and $w_{\text{con}}$ controls the activation threshold;
- $z^2$ replaces the theoretically motivated $\alpha(z)$ to avoid over-amplification under imprecise estimation.

This can equivalently be rewritten in the conventional guidance form: extrapolation along the direction of the difference between two drift evaluations, structurally analogous to CFG/AG but with an entirely different signal source.

## Key Experimental Results

### ImageNet 512×512 (EDM2 + Heun sampler)

| Steps | Method | FD-DINOv2↓ | FID↓ |
|-------|--------|-----------|------|
| 32 | No guidance | 90.1 | 2.58 |
| 32 | ERK-Guid ($w_{\text{stiff}}$=2.0) | **82.8** | 2.74 |
| 16 | No guidance | 97.4 | 2.79 |
| 16 | ERK-Guid ($w_{\text{stiff}}$=0.75) | **88.9** | **2.68** |
| 8 | No guidance | 161.2 | 7.06 |
| 8 | ERK-Guid ($w_{\text{stiff}}$=0.5) | **136.9** | **4.91** |

### Combined with CFG/Autoguidance (32 steps)

| Baseline | FD-DINOv2↓ | +ERK-Guid FD-DINOv2↓ |
|----------|-----------|---------------------|
| CFG | 88.5 | **83.9** |
| Autoguidance | 50.4 | **47.6** |

### Cross-Solver Adaptation (ImageNet 64×64, 6 NFEs)

| Solver | FID↓ | +ERK-Guid FID↓ |
|--------|------|---------------|
| Heun | 89.63 | **85.19** |
| DPM-Solver | 44.83 | **31.59** |
| DEIS | 12.57 | **9.56** |

Improvements are most pronounced in low-step regimes (8-step FID reduced from 7.06 to 4.91), consistent with the expectation that LTE dominates errors when fewer steps are used.

## Highlights & Insights

1. **Novel perspective**: This is the first work to use the truncation error of an ODE solver as a guidance signal, forming an orthogonal complement to model-error-based guidance methods such as CFG and AG.
2. **Zero computational overhead**: All required quantities are byproducts of the Heun update and necessitate no additional network forward passes.
3. **Plug-and-play**: Compatible with arbitrary Runge-Kutta solvers including Heun, DPM-Solver, and DEIS, and can be stacked with CFG and Autoguidance.
4. **Solid theoretical grounding**: The alignment between LTE and the dominant eigenvector is derived from ODE numerical analysis and validated through both 2D toy experiments and ImageNet experiments.
5. **Low-step advantage**: As the number of steps decreases, LTE accounts for a larger share of total error, making ERK-Guid increasingly beneficial.

## Limitations & Future Work

1. The method requires solvers that produce embedded pairs (e.g., Heun) and is not directly applicable to purely first-order solvers such as Euler or DDIM.
2. The hyperparameters $w_{\text{stiff}}$ and $w_{\text{con}}$ require tuning per model and step count; while experiments demonstrate robustness, this still adds a tuning burden.
3. The theoretical analysis relies on a local linearization assumption, which may be insufficiently accurate in highly nonlinear regions.
4. The current work addresses only deterministic ODE sampling and does not consider SDE sampler settings.
5. The main experiments are conducted on the EDM2 framework; although PixArt-α (DiT) is also evaluated, validation on other mainstream architectures (e.g., SD3, FLUX) remains limited.

## Related Work & Insights

| Method | Signal Source | Extra Cost | Complementarity |
|--------|--------------|------------|-----------------|
| CFG | Conditional/unconditional model discrepancy | 2× NFE | Complementary to ERK-Guid |
| Autoguidance | Strong/weak model discrepancy | Auxiliary network required | Complementary to ERK-Guid |
| PCG | Predictor-corrector interpretation of CFG | Same as CFG | Theoretically related |
| DPM-Solver | Higher-order numerical solver | None | Stackable with ERK-Guid |
| ERK-Guid (Ours) | Solver order-difference error | None | Orthogonal to model guidance |

### Further Connections

- Examining diffusion sampling from a numerical analysis perspective is a promising direction; stiffness-aware adaptive step-size scheduling warrants further exploration.
- Extending the "error as signal" paradigm to flow matching samplers or SDE solvers is worth investigating.
- In high-dimensional settings such as video generation, the impact of LTE may be more pronounced, suggesting potential applicability of ERK-Guid.
- Combining ERK-Guid with distillation methods (e.g., consistency models) is promising, as per-step error is more critical in few-step distilled models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First use of solver error as a guidance signal; highly original perspective)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-dataset and multi-solver validation on ImageNet/FFHQ/PixArt, though broader architectural coverage is lacking)
- Writing Quality: ⭐⭐⭐⭐⭐ (Theoretical derivations are clear, with a well-structured progression from 2D toy experiments to real data)
- Value: ⭐⭐⭐⭐ (A practical, zero-cost plug-and-play method with particular value in low-step regimes)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[NeurIPS 2025\] System-Embedded Diffusion Bridge Models](../../NeurIPS2025/image_generation/system-embedded_diffusion_bridge_models.md)
- [\[ICML 2026\] Quantifying Error Propagation and Model Collapse in Diffusion Models](../../ICML2026/image_generation/quantifying_error_propagation_and_model_collapse_in_diffusion_models.md)
- [\[ICLR 2026\] Contractive Diffusion Policies: Robust Action Diffusion via Contractive Score-Based Sampling with Differential Equations](contractive_diffusion_policies_robust_action_diffusion_via_contractive_score-bas.md)
- [\[ICLR 2026\] Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models](stochastic_self-guidance_for_training-free_enhancement_of_diffusion_models.md)

</div>

<!-- RELATED:END -->
