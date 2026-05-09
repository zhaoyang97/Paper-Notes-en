---
title: >-
  [Paper Note] Adaptive Discretization for Consistency Models
description: >-
  [NeurIPS 2025][Image Restoration][Consistency Model] This paper proposes ADCM, which formalizes the discretization step size of consistency models as a constrained optimization problem balancing local consistency (trainability) and global consistency (stability), derives a closed-form solution via the Gauss-Newton method, and achieves adaptive discretization that surpasses all prior CMs on CIFAR-10 using less than 25% of the training budget.
tags:
  - NeurIPS 2025
  - Image Restoration
  - Consistency Model
  - Adaptive Discretization
  - Training Efficiency
  - Single-Step Generation
  - Lagrange Multiplier Method
date: 2026-05-08
content_hash: e0f1679799461ada
---

# Adaptive Discretization for Consistency Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.17266](https://arxiv.org/abs/2510.17266)
**Code**: [GitHub](https://github.com/rainstonee/ADCM)
**Area**: Image Restoration
**Keywords**: Consistency Model, Adaptive Discretization, Training Efficiency, Single-Step Generation, Lagrange Multiplier Method

## TL;DR

This paper proposes ADCM, which formalizes the discretization step size of consistency models as a constrained optimization problem balancing local consistency (trainability) and global consistency (stability), derives a closed-form solution via the Gauss-Newton method, and achieves adaptive discretization that surpasses all prior CMs on CIFAR-10 using less than 25% of the training budget.

## Background & Motivation

- **Core Problem**: Consistency models (CMs) achieve single-step generation by mapping points on PF-ODE trajectories to endpoints, but their training is highly sensitive to the discretization strategy used for adjacent trajectory points.
- **Limitations of Prior Work**: (1) Discrete CMs (iCT, ECM) rely on manually designed discretization schedules that require repeated tuning across different noise schedules and datasets; (2) Continuous CMs (sCM) circumvent discretization by taking $\Delta t \to 0$, but suffer from severe training instability; (3) CCM iteratively solves via PSNR thresholds, incurring substantial computational cost.
- **Key Challenge**: Small $\Delta t$ yields good local consistency but large global denoising error, causing instability; large $\Delta t$ improves stability but degrades local consistency, making training difficult.
- **Key Insight**: Formalize the discretization step size selection as a constrained optimization problem that adaptively balances trainability and stability.

## Method

### Local Consistency and Global Consistency

- **Local Consistency** (optimization objective): $\mathcal{L}_\text{local} = \mathbb{E}[\|f_{\theta^-}(\mathbf{x}_t) - f_{\theta^-}(\mathbf{x}_{t-\Delta t})\|_2^2]$, minimization of which favors small $\Delta t$.
- **Global Consistency** (constraint): $\mathcal{L}_\text{global} = \mathbb{E}[\|f_{\theta^-}(\mathbf{x}_{t-\Delta t}) - \mathbf{x}_0\|_2^2] \leq \delta$, controlling denoising error, which favors large $\Delta t$.

The two objectives impose opposing constraints on $\Delta t$.

### Constrained Optimization and Lagrangian Relaxation

The two objectives are unified as:

$$\Delta t^* = \arg\min_{\Delta t} \mathbb{E}[\mathcal{L}_\text{local}(t, \Delta t) + \lambda \mathcal{L}_\text{global}(t, \Delta t)]$$

The Lagrange multiplier $\lambda$ balances trainability and stability, with $\lambda \ll 1$ in practice (prioritizing trainability).

### Unified Framework: Prior Methods as Special Cases

| Method | Corresponding $\lambda$ |
|--------|------------------------|
| DM (e.g., EDM) | $\lambda \to \infty$ (maximum step $\Delta t = t - \varepsilon$) |
| Continuous CM (sCM) | $\lambda = 0$ (minimum step $\Delta t \to 0$) |
| Discrete CM (iCT, ECM) | Empirically estimated |
| CCM | $\mathcal{L}_\text{local}$ set as constant |

### Closed-Form Solution via Gauss-Newton Method

A first-order Taylor expansion is used to approximate $f_{\theta^-}(\mathbf{x}_{t-\Delta t})$, and the Jacobian direction vector $\mathbf{v}$ is efficiently computed via JVP, yielding the closed-form solution:

$$\Delta t^* = \frac{\lambda}{1+\lambda} \frac{\mathbb{E}[\mathbf{v}^\top(f_{\theta^-}(\mathbf{x}_t) - \mathbf{x}_0)]}{\mathbb{E}[\mathbf{v}^\top \mathbf{v}]}$$

Three intuitive interpretations: (1) larger Jacobian → smaller step size (caution when outputs vary rapidly); (2) larger denoising error → larger step size (ensuring stability); (3) greater alignment between local and global optimization directions → larger step size.

### Adaptive Weighting Function and Loss

The weighting function $w(t) = 1/\mathcal{L}_\text{global}$ down-weights regions with large global error (avoiding instability) and up-weights regions with small error. The final loss replaces L2 with the Pseudo-Huber metric to reduce variance:

$$\min_\theta \mathbb{E}\left[\frac{\sqrt{\|f_\theta(\mathbf{x}_t) - f_{\theta^-}(\mathbf{x}_{t-\Delta t^*})\|_2^2 + c^2} - c}{\sqrt{\|f_{\theta^-}(\mathbf{x}_{t-\Delta t^*}) - \mathbf{x}_0\|_2^2 + c^2} - c}\right]$$

### Training Procedure

The time partition $\mathbb{T}$ and network parameters $\theta$ are optimized alternately: $\mathbb{T}$ is updated every 25,000 steps by iterating from $t = T$ to $t = \varepsilon$ via Eq. 10, with expectations estimated using a single mini-batch.

## Key Experimental Results

### CIFAR-10 Unconditional Generation (1-step FID↓)

| Method | Training Budget (Mimgs) | FID↓ |
|--------|------------------------|------|
| ECM | 12.8 | 4.54 |
| ECM | 51.2 | 3.60 |
| iCT | 409.6 | 2.83 |
| sCT (TrigFlow) | 204.8 | 2.85 |
| **ADCM** | **12.8** | **3.16** |
| **ADCM** | **76.8** | **2.80** |

ADCM with 12.8M images outperforms ECM with 51.2M images; with 76.8M images it surpasses iCT trained on 409.6M images (approximately 19% of the training budget).

### ImageNet 64×64 Class-Conditional Generation

| Method | Model Size | Training Budget | FID↓ |
|--------|-----------|----------------|------|
| iCT-deep | 2× | 1638.4M | 3.25 |
| ECM | 2× | 12.8M | 3.67 |
| **ADCM** | **2×** | **12.8M** | **3.49** |
| **ADCM** | **2×** | **51.2M** | **3.04** |

ADCM (2×, 12.8M) already surpasses ECM (2×, 12.8M) and approaches iCT-deep (2×, 1638.4M).

### Training Efficiency

- Additional computational overhead is approximately 4% (JVP computation + periodic updates of $\mathbb{T}$).
- Convergence speed is substantially faster than iCT, ECM, and sCT.
- Adapts to Flow Matching without manual tuning: FID 5.14 vs. ECM 5.82 (12.8M budget).

### Effect of $\lambda$

- Too small $\lambda$ → overemphasis on global consistency → fast convergence but poor final quality.
- Too large $\lambda$ → overemphasis on local consistency → instability and difficult convergence.
- Optimal $\lambda$ achieves a balance between training stability and final performance.

## Highlights & Insights

1. **Theoretical Elegance of the Unified Framework**: All prior CM discretization methods (iCT/ECM/sCM/CCM/DM) are unified as special cases under different values of $\lambda$, providing a clear theoretical perspective.
2. **Intuitive Interpretation of Adaptive Step Sizes**: The closed-form solution reveals how three factors—Jacobian magnitude, denoising error, and alignment of optimization directions—jointly determine the optimal step size.
3. **Exceptional Training Efficiency**: On CIFAR-10, ADCM surpasses all prior CMs using less than 25% of the training budget; on ImageNet, it approaches iCT-deep with approximately 3% of the budget, with only 4% additional overhead.
4. **No Manual Tuning Required**: The same framework automatically adapts to both VE SDE and Flow Matching without redesigning the discretization schedule for different noise strategies.

## Limitations & Future Work

1. **Dependence on Pretrained DM Initialization**: All CM experiments initialize from a pretrained EDM (following the ECM paradigm); training from scratch has not been validated.
2. **Insufficient Validation at High Resolutions**: ImageNet 512×512 experiments are limited and yield a relatively high FID (10.53/2×/6.4M), with a notable gap compared to sCT.
3. **Applicability of First-Order Taylor Approximation**: The first-order approximation may be inaccurate when network outputs vary rapidly.
4. **$\lambda$ Still Requires Manual Selection**: Despite the paper's claim of adaptivity, $\lambda$ itself remains a hyperparameter that requires tuning.
5. **Validation Limited to Image Generation**: The approach has not been extended to other diffusion model applications such as video, audio, or 3D generation.

## Related Work & Insights

- **CM Discretization Lineage**: iCT (exponentially decreasing step sizes) → ECM (decoupled step magnitude and distribution) → CCM (iterative PSNR-threshold solution) → sCM (continuous limit $\Delta t \to 0$) → ADCM (adaptive closed-form solution).
- **Connection to DM Optimization**: DM training essentially optimizes only global consistency ($\lambda \to \infty$); ADCM reveals that CMs must simultaneously account for local consistency.
- **Inspiration**: The adaptive discretization idea can be generalized to ODE solver step size selection in flow matching and teacher-student step matching in diffusion distillation.

## Rating

⭐⭐⭐⭐ — The theoretical framework is elegant (unifying prior methods with a closed-form solution), training efficiency improvements are substantial (4–20× budget savings), and the approach is broadly applicable. Weaknesses include insufficient high-resolution validation and the need for manual tuning of $\lambda$.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Audio Super-Resolution with Latent Bridge Models](audio_super-resolution_with_latent_bridge_models.md)
- [\[NeurIPS 2025\] Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates](rethinking_circuit_completeness_in_language_models_and_or_and_adder_gates.md)
- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](../../ICCV2025/image_restoration/enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [\[ICCV 2025\] Metric Convolutions: A Unifying Theory to Adaptive Image Convolutions](../../ICCV2025/image_restoration/metric_convolutions_a_unifying_theory_to_adaptive_image_convolutions.md)
- [\[NeurIPS 2025\] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)

<!-- RELATED:END -->
