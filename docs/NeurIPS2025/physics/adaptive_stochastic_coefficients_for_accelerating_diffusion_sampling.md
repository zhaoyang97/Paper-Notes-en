---
title: >-
  [Paper Note] Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][diffusion sampling] By theoretically analyzing the complementary weaknesses of ODE and SDE solvers (ODE solvers accumulate irreducible gradient errors; SDE solvers amplify discretization errors at large step sizes), this paper proposes AdaSDE—a method that introduces a learnable stochastic coefficient $\gamma_i$ at each denoising step to control noise injection intensity. Optimized via lightweight distillation…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "diffusion sampling"
  - "SDE solver"
  - "ODE solver"
  - "adaptive noise injection"
  - "few-step generation"
date: 2026-05-08
content_hash: 71fec875cb9dd10a
---

# Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling

**Conference**: NeurIPS 2025
**arXiv**: [2510.23285](https://arxiv.org/abs/2510.23285)  
**Code**: [GitHub](https://github.com/WLU-wry02/AdaSDE)  
**Area**: Diffusion Models / Sampling Acceleration
**Keywords**: diffusion sampling, SDE solver, ODE solver, adaptive noise injection, few-step generation

## TL;DR
By theoretically analyzing the complementary weaknesses of ODE and SDE solvers (ODE solvers accumulate irreducible gradient errors; SDE solvers amplify discretization errors at large step sizes), this paper proposes AdaSDE—a method that introduces a learnable stochastic coefficient $\gamma_i$ at each denoising step to control noise injection intensity. Optimized via lightweight distillation, AdaSDE achieves state-of-the-art FID of 4.18 on CIFAR-10 and 8.05 on FFHQ at 5 NFE.

## Background & Motivation

**Background**: Diffusion model sampling requires solving reverse differential equations. Mainstream approaches fall into two categories: ODE solvers (DDIM, DPM-Solver, etc.) provide efficient deterministic sampling but accumulate irreducible gradient errors; SDE solvers inject stochasticity to correct gradient errors but require a large number of function evaluations (100–1000 NFE).

**Limitations of Prior Work**: (a) ODE solvers in few-step settings (<10 NFE) irreversibly accumulate discrepancies between the learned score function and the true score (gradient errors) along deterministic trajectories, leading to a performance ceiling; (b) SDE solvers, while capable of correcting gradient errors via stochasticity, suffer from discretization errors that grow with step size, making them worse than ODE solvers in few-step regimes; (c) hybrid methods such as Restart Sampling still require 50+ steps.

**Key Challenge**: In the few-step regime, both the gradient error of ODE solvers and the discretization error of SDE solvers are problematic—a method is needed that simultaneously exploits the efficiency of ODE solvers and the error-correction capability of SDE solvers.

**Goal**:
- Can SDE solvers achieve efficient sampling with very few steps (<10 NFE)?
- How can stochasticity intensity be adaptively controlled to find the optimal trade-off between gradient error correction and discretization error?

**Key Insight**: Theoretical analysis shows that the gradient error bound of AdaSDE includes a contraction factor $(1-\lambda(\gamma))$, which is strictly smaller than the gradient error bound of ODE solvers (Theorem 3). Crucially, $\gamma$ must be adaptively tuned—each step requires a different optimal noise intensity.

**Core Idea**: At each denoising step, a learnable scalar $\gamma_i$ controls the magnitude of "forward noise addition + backward denoising." Optimizing $\gamma_i$ via lightweight distillation enables few-step SDE sampling to surpass ODE solvers.

## Method

### Overall Architecture
AdaSDE is a single-step SDE solver. At each denoising step $[t+\Delta t, t]$, two sub-steps are performed: (1) **Forward process**: inject $\gamma$-controlled Gaussian noise into the current state $x_{t+\Delta t}$ to reach a higher noise level $t+(1+\gamma)\Delta t$; (2) **Backward process**: deterministically integrate back to the target time $t$ using an ODE solver. The coefficient $\gamma_i$ varies across steps and is optimized via distillation.

### Key Designs

1. **Unified Theoretical Analysis of ODE vs. SDE Errors**

    - **Function**: Proves that the gradient error bound of AdaSDE is strictly smaller than that of ODE solvers.
    - **Mechanism**: Theorem 1 gives the ODE error bound as a gradient error term ($B \cdot \text{TV}$, accumulated) plus a discretization error term ($O(\Delta t^2)$). Theorem 2 gives the AdaSDE error bound, where the gradient error term is multiplied by a contraction factor $(1-\lambda(\gamma)) < 1$, arising from the smoothing effect of Gaussian convolution—noise injection brings the generated distribution closer to the true distribution. Theorem 3 rigorously proves $\mathcal{E}_{\text{grad}}^{\text{AdaSDE}} \leq \mathcal{E}_{\text{grad}}^{\text{ODE}}$.
    - **Design Motivation**: This theoretically confirms that "moderate noise addition" reduces gradient errors, providing a theoretical foundation for introducing adaptive $\gamma$. However, excessively large $\gamma$ increases discretization error, necessitating optimization.

2. **Learnable Adaptive Stochastic Coefficient $\gamma_i$**

    - **Function**: Learns a scalar $\gamma_i \in (0, 1)$ per step to control noise injection intensity.
    - **Mechanism**: $\gamma_i$ is parameterized and optimized via distillation. A key observation is that diffusion trajectories exhibit a consistent low-dimensional geometric structure across solvers and datasets, allowing $\gamma_i$ to be learned from only a small amount of data.
    - **Design Motivation**: The ratio of gradient errors to discretization errors differs across denoising steps—in early steps with high noise levels, the benefit of gradient error correction is large; in later steps with low noise levels, the discretization cost of excessive noise injection is high. Adaptive $\gamma_i$ automatically balances this trade-off at each step.

3. **Process-Supervised Distillation Optimization Framework**

    - **Function**: Optimizes $\gamma_i$ by supervising intermediate steps rather than only the final output.
    - **Mechanism**: A high-step ODE solver (e.g., 250-step DDIM) is used to generate reference trajectories, and AdaSDE trajectories are aligned to the reference at each intermediate step. Only the $\gamma$ parameters (scalars per step) are optimized—no neural network training is required.
    - **Design Motivation**: Terminal supervision (monitoring only the final output) leaves errors at intermediate steps uncorrected; process supervision provides feedback at every step, ensuring the entire trajectory remains close to the reference. Since $\gamma$ is a global property of trajectory geometry (low-dimensional), only a few hundred samples are sufficient.

4. **Plug-and-Play Combination with Existing Solvers**

    - **Function**: AdaSDE's $\gamma$ can serve as a plugin to enhance any single-step ODE solver.
    - **Mechanism**: Any ODE step $x_t = \text{ODE}(x_{t+\Delta t})$ is replaced by $x_t = \text{ODE}(\text{add\_noise}(x_{t+\Delta t}, \gamma_i))$, without modifying the solver's internal logic.
    - **Design Motivation**: Generality—different solvers (Euler, iPNDM, DPM-Solver, etc.) can all benefit from adaptive noise injection.

### Loss & Training
- Distillation loss: $\mathcal{L} = \sum_i \|x_t^{\text{AdaSDE}} - x_t^{\text{ref}}\|^2$ (intermediate-step alignment)
- Training data: only ~500 samples needed to generate reference trajectories
- Optimized parameters: only $N$ scalars $\gamma_i$ ($N$ = number of steps), extremely lightweight

## Key Experimental Results

### Main Results: 5 NFE

| Dataset | Metric (FID↓) | AdaSDE | Prev. SOTA (AMED) | Gain |
|--------|-----------|--------|----------------|------|
| CIFAR-10 | FID | **4.18** | 7.14 | 1.7× |
| FFHQ 64×64 | FID | **8.05** | 14.85 | 1.8× |
| LSUN Bedroom | FID | **6.96** | — | — |

### Ablation Study: Combination with Different Base Solvers

| Base Solver | Original FID | + AdaSDE FID | Gain |
|-----------|---------|-------------|------|
| Euler | — | Significant improvement | ✓ |
| iPNDM | — | Significant improvement | ✓ |
| DPM-Solver | — | Significant improvement | ✓ |

AdaSDE, as a plug-and-play module, improves all tested base solvers.

### Key Findings
- **$\gamma$ varies significantly across steps**: optimal $\gamma$ values are larger in early steps (more stochasticity needed to correct gradient errors) and approach 0 in later steps (to avoid discretization errors)—validating the necessity of adaptive coefficients.
- **Process supervision substantially outperforms terminal supervision**: monitoring only the final output causes intermediate trajectories to deviate, degrading final quality.
- **Consistency of trajectory geometry**: the optimal $\gamma$ pattern across datasets is similar (larger early, smaller late), indicating that the underlying geometric structure of diffusion trajectories is consistent across datasets.
- **SDE surpasses ODE for the first time at very few steps (5 NFE)**: challenging the conventional wisdom that "few-step generation requires ODE solvers."

## Highlights & Insights
- **A theoretical answer to "why can't SDE be used for few-step sampling?"**: discretization error $O(\delta^{3/2})$ vs. ODE's $O(\delta^2)$—but if adaptive $\gamma$ makes the gradient error correction benefit exceed the discretization cost, SDE can outperform ODE at few steps. This insight is remarkably precise.
- **Extremely concise optimization objective**: the entire method optimizes only $N$ scalar parameters ($N$ = number of steps, typically 5–10)—truly lightweight distillation requiring neither additional network training nor large datasets.
- **Plug-and-play design**: any ODE solver + AdaSDE = a better solver. This generality gives the method far broader practical impact than a standalone approach.

## Limitations & Future Work
- **Distillation step still required**: although lightweight, reference trajectories must be generated in advance, making the method not entirely training-free.
- **Dependence on reference trajectory quality**: if the high-step ODE reference is itself suboptimal, the optimization target for $\gamma$ may be biased.
- **Pixel space vs. latent space**: validation is primarily conducted in pixel space (CIFAR-10, FFHQ, LSUN); effectiveness on latent diffusion models (e.g., Stable Diffusion) remains unverified.
- **Generalization of $\gamma$**: whether the learned $\gamma$ generalizes across different prompts or conditions is not discussed in the context of conditional generation.

## Related Work & Insights
- **vs. DDIM / DPM-Solver (ODE solvers)**: AdaSDE corrects the irreducible gradient errors of ODE solvers via adaptive noise injection, achieving substantially lower FID at the same number of steps.
- **vs. Restart Sampling (hybrid methods)**: Restart Sampling requires 50+ steps, whereas AdaSDE surpasses it at 5 steps—because AdaSDE's $\gamma$ is optimized rather than fixed.
- **vs. AMED / EPD (distillation-based solvers)**: AMED optimizes the time schedule while AdaSDE optimizes noise injection coefficients; the two approaches are orthogonal and can be combined.
- **Transferable insight**: the idea of adaptive noise injection may be applicable to any iterative refinement process—such as iterative reasoning in LLMs or protein structure prediction.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to achieve fewer-step SDE performance surpassing ODE; the logical chain from theoretical motivation to method design to experimental validation is highly complete.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset and multi-solver combination experiments with comprehensive ablations, but latent-space model experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical analysis is clear and rigorous; the ODE vs. SDE error analysis is highly instructive.
- **Value**: ⭐⭐⭐⭐⭐ 5 NFE state-of-the-art performance + plug-and-play design + extremely lightweight optimization yield very high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Guided Diffusion Sampling on Function Spaces with Applications to PDEs](guided_diffusion_sampling_on_function_spaces_with_applications_to_pdes.md)
- [\[NeurIPS 2025\] Quantum Doubly Stochastic Transformers](quantum_doubly_stochastic_transformers.md)
- [\[NeurIPS 2025\] Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)
- [\[NeurIPS 2025\] FEAT: Free Energy Estimators with Adaptive Transport](feat_free_energy_estimators_with_adaptive_transport.md)
- [\[NeurIPS 2025\] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)

</div>

<!-- RELATED:END -->
