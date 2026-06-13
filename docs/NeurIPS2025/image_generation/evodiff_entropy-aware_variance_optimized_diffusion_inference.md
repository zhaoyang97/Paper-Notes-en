---
title: >-
  [Paper Note] EVODiff: Entropy-aware Variance Optimized Diffusion Inference
description: >-
  [NeurIPS 2025][Image Generation][Diffusion models] This paper analyzes the inference process of diffusion models from an information-theoretic perspective and proposes EVODiff…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Diffusion models"
  - "inference acceleration"
  - "information theory"
  - "conditional entropy"
  - "variance optimization"
date: 2026-05-08
content_hash: 84e077379fedab81
---

# EVODiff: Entropy-aware Variance Optimized Diffusion Inference

**Conference**: NeurIPS 2025
**arXiv**: [2509.26096](https://arxiv.org/abs/2509.26096)  
**Code**: [ShiguiLi/EVODiff](https://github.com/ShiguiLi/EVODiff)  
**Area**: Image Generation
**Keywords**: Diffusion models, inference acceleration, information theory, conditional entropy, variance optimization

## TL;DR

This paper analyzes the inference process of diffusion models from an information-theoretic perspective and proposes EVODiff, a method that reduces conditional entropy by optimizing conditional variance, achieving significant sampling acceleration and quality improvement without modifying the underlying model.

## Background & Motivation

Diffusion models excel at image generation but suffer from slow inference and training–inference discrepancies. Existing acceleration methods (e.g., DPM-Solver, UniPC) treat denoising as ODE solving but lack an information-theoretic foundation—specifically, they overlook information transmission efficiency. The authors argue that successful denoising is fundamentally equivalent to reducing conditional entropy in the reverse transition, a principle that existing methods do not leverage in their algorithmic design.

## Core Problem

1. Existing ODE solvers lack information-theoretic guidance and therefore cannot optimally recover information during the denoising process.
2. There is no theoretical explanation for why data-prediction parameterization outperforms noise-prediction parameterization.
3. How conditional variance optimization can simultaneously reduce transition error and reconstruction error without access to reference data.

## Method

### Information-Theoretic Framework

The paper frames diffusion inference as a process of conditional entropy reduction. The mutual information between adjacent states in the reverse transition is:

$$I_p(\mathbf{x}_{t_i}; \mathbf{x}_{t_{i+1}}) = H_p(\mathbf{x}_{t_i}) - H_p(\mathbf{x}_{t_i} | \mathbf{x}_{t_{i+1}})$$

Under Gaussian assumptions, conditional entropy is directly related to conditional variance:

$$H_p(\mathbf{x}_{t_i} | \mathbf{x}_{t_{i+1}}) \propto \log\det(\mathrm{Var}(\mathbf{x}_{t_i} | \mathbf{x}_{t_{i+1}}))$$

Therefore, **minimizing conditional variance is equivalent to maximizing information transmission efficiency**.

### Reconstruction Error Decomposition

The reconstruction error is decomposed into a variance term and a bias term:

$$\mathbb{E}_q[\|\mathbf{x}_{t_i} - \mathbf{x}_0\|^2] = \underbrace{\mathbb{E}_q[\|\mathbf{x}_{t_i} - \boldsymbol{\mu}_{t_i|t_{i+1}}\|^2]}_{\text{variance term}} + \underbrace{\mathbb{E}_q[\|\boldsymbol{\mu}_{t_i|t_{i+1}} - \mathbf{x}_0\|^2]}_{\text{bias term}}$$

Since the true $\mathbf{x}_0$ is unavailable at inference time, optimizing the conditional variance becomes the only actionable mechanism.

### Data Prediction vs. Noise Prediction

**Theorem 3.4**: Data-prediction parameterization more effectively reduces reconstruction error and conditional entropy than noise-prediction parameterization. Data parameterization directly targets the data distribution, avoiding the error accumulation chain $\boldsymbol{\epsilon}_t \mapsto \mathbf{x}_t \mapsto \mathbf{x}_0$.

### EVODiff Algorithm

EVODiff performs multi-step iteration based on data prediction, achieving entropy-aware inference by optimizing two key parameters $\zeta_i$ and $\eta_i$:

**Step 1 — Unified Iteration**: Explicit and implicit corrections are unified as:

$$\frac{\mathbf{x}_{t_{i-1}}}{\sigma_{t_{i-1}}} - \frac{\mathbf{x}_{t_i}}{\sigma_{t_i}} = h_{t_i} \mathbf{x}_\theta(\mathbf{x}_{t_i}, t_i) + \frac{1}{2} h_{t_i}^2 \zeta_i \bar{B}_\theta(t_i; u_i)$$

**Step 2 — Solving for $\zeta_i$**: A closed-form solution is derived by minimizing the discrepancy between forward and backward estimates:

$$\zeta_i^* = -\frac{\text{vec}^T(D_i) \text{vec}(\tilde{P}_i)}{\sigma_{t_i} h_{t_i} \text{vec}^T(D_i) \text{vec}(D_i)}$$

**Step 3 — Solving for $\eta_i$**: Balancing implicit and explicit gradient errors:

$$\eta_i^* = -\frac{\text{vec}^T(\tilde{B}_i) \text{vec}(B_\theta(t_i, l_i))}{\text{vec}^T(\tilde{B}_i) \text{vec}(\tilde{B}_i)}$$

**Step 4 — Mapping to Usable Parameters**:

$$\eta_i = \text{Sigmoid}(|\eta_i^*|), \quad \zeta_i = \text{Sigmoid}(-(|\zeta_i^*| - \mu))$$

The algorithm achieves second-order global convergence with a local error of $\mathcal{O}(h_{t_i}^3)$.

## Key Experimental Results

### CIFAR-10 (EDM, 50k samples)

| Method | NFE=5 FID↓ | NFE=8 FID↓ | NFE=10 FID↓ | NFE=12 FID↓ |
|--------|-----------|-----------|------------|------------|
| DPM-Solver++ | 27.96 | 8.40 | 5.10 | 3.70 |
| UniPC | 27.03 | 7.67 | 3.98 | 2.76 |
| **EVODiff** | **17.84** | **3.98** | **2.78** | **2.30** |

### FFHQ-64 (EDM, 50k samples)

| Method | NFE=5 FID↓ | NFE=10 FID↓ | NFE=15 FID↓ | NFE=20 FID↓ |
|--------|-----------|------------|------------|------------|
| DPM-Solver++ | 25.08 | 6.81 | 3.80 | 3.00 |
| UniPC | 28.87 | 6.65 | 3.40 | 2.69 |
| **EVODiff** | **19.65** | **5.31** | **3.04** | **2.66** |

### ImageNet-256 (ADM, 10k samples)

| Method | NFE=5 FID↓ | NFE=10 FID↓ | NFE=15 FID↓ | NFE=20 FID↓ |
|--------|-----------|------------|------------|------------|
| DPM-Solver++ | 16.62 | 8.68 | 7.80 | 7.51 |
| DPM-Solver-v3 | 14.92 | 8.14 | 7.70 | 7.42 |
| **EVODiff** | **13.98** | **8.14** | **7.48** | **7.25** |

- On CIFAR-10 at NFE=10, FID decreases from 5.10 to 2.78, a **45.5% reduction**.
- On ImageNet-256, high-quality samples can be obtained at NFE=15 instead of NFE=20, **saving 25% of compute**.
- Artifacts are also reduced in text-to-image generation.

## Highlights & Insights

- ⭐ First systematic information-theoretic analysis of diffusion inference, establishing a conditional entropy reduction framework.
- ⭐ Theoretical proof that data-prediction parameterization outperforms noise-prediction parameterization.
- ⭐ The variance-optimization parameters $\zeta_i$ and $\eta_i$ admit closed-form solutions, incurring negligible computational overhead.
- Requires no additional training or reference data (unlike DPM-Solver-v3).
- Provides a unified explanation for the acceleration mechanisms of DPM-Solver and EDM Heun iteration.

## Limitations & Future Work

- The analysis assumes independence of estimated noise between denoising steps; in practice, shared parameters may introduce dependencies.
- Theoretical analysis is primarily grounded in Gaussian assumptions; applicability to non-Gaussian distributions remains to be verified.
- The method focuses exclusively on deterministic sampling (ODE) and has not been extended to stochastic sampling (SDE) trajectories.
- The shift parameter $\mu$ requires manual tuning.

## Related Work & Insights

| Property | DDIM | DPM-Solver | UniPC | DPM-Solver-v3 | EVODiff |
|----------|------|------------|-------|---------------|---------|
| Gradient-based | ✗ | ✓ | ✓ | ✓ | ✓ |
| Requires reference $\tilde{\mathbf{x}}_0$ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Variance term optimization | ✓ | ✓ | ✓ | ✓ | ✓ |
| Entropy-aware | ✗ | ✗ | ✗ | ✗ | ✓ |

The information-theoretic perspective is generalizable to multi-step diffusion settings such as video and 3D generation. The conditional entropy framework may also guide automatic design of sampling schedules. The variance optimization idea is potentially compatible with consistency models and rectified flow.

## Rating

- Novelty: ⭐⭐⭐⭐ (novel information-theoretic entry point)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (comprehensive comparison across multiple datasets, models, and NFE settings)
- Writing Quality: ⭐⭐⭐⭐ (clear theoretical derivations with a consistent notation system)
- Value: ⭐⭐⭐⭐ (plug-and-play inference acceleration with strong practical utility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Neural Entropy](neural_entropy.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)
- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[NeurIPS 2025\] Gradient Variance Reveals Failure Modes in Flow-Based Generative Models](gradient_variance_reveals_failure_modes_in_flow-based_generative_models.md)
- [\[NeurIPS 2025\] Tortoise and Hare Guidance: Accelerating Diffusion Model Inference with Multirate Integration](tortoise_and_hare_guidance_accelerating_diffusion_model_inference_with_multirate.md)

</div>

<!-- RELATED:END -->
