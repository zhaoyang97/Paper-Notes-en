---
title: >-
  [Paper Note] CHORDS: Diffusion Sampling Accelerator with Multi-Core Hierarchical ODE Solvers
description: >-
  [ICCV 2025][Image Generation][Diffusion model acceleration] This paper proposes CHORDS, a diffusion sampling acceleration framework based on multi-core hierarchical ODE solvers. Through a slow-to-fast inter-core rectification mechanism, CHORDS achieves 2.1×–2.9× speedup on 4–8 GPUs without sacrificing generation quality.
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Diffusion model acceleration"
  - "multi-core parallelism"
  - "ODE solvers"
  - "training-free"
  - "video generation"
date: 2026-05-08
content_hash: e5f06ac4ef3f83fc
---

# CHORDS: Diffusion Sampling Accelerator with Multi-Core Hierarchical ODE Solvers

**Conference**: ICCV 2025
**arXiv**: [2507.15260](https://arxiv.org/abs/2507.15260)  
**Code**: [Project Page](https://hanjq17.github.io/CHORDS)  
**Area**: Image Generation
**Keywords**: Diffusion model acceleration, multi-core parallelism, ODE solvers, training-free, video generation

## TL;DR

This paper proposes CHORDS, a diffusion sampling acceleration framework based on multi-core hierarchical ODE solvers. Through a slow-to-fast inter-core rectification mechanism, CHORDS achieves 2.1×–2.9× speedup on 4–8 GPUs without sacrificing generation quality.

## Background & Motivation

Diffusion models have become the dominant paradigm for high-quality image and video generation, yet their inference process is inherently iterative and computationally expensive. Existing acceleration approaches fall into two broad categories:

**Distillation-based methods** (progressive/consistency distillation): require additional training and generalize poorly across models.

**Fast ODE solvers** (DDIM, DPM-Solver, etc.): single-core methods suffer severe quality degradation when the number of steps is reduced.

**Why is parallelization a natural direction?** In classical numerical computation, multigrid methods have long accelerated ODE solving through multi-level parallelism. However, transferring this idea to diffusion models presents challenges:

- Existing multi-core methods (e.g., ParaDIGMS) rely on sliding-window Picard iteration, which converges slowly and imposes architectural constraints.
- SRDS fixes the number of cores at $\sqrt{N}$, limiting flexibility and requiring hand-crafted pipeline designs.
- No unified theoretical framework exists to guide optimal core allocation strategies.

**Core Motivation**: Can one design a general acceleration framework that is (1) training-free, (2) model-agnostic, and (3) flexibly adaptable to varying numbers of cores?

## Method

### Overall Architecture

The core idea of CHORDS is to treat multi-core diffusion sampling as a pipeline of ODE solvers, where slower but more accurate solvers progressively rectify faster but less accurate ones through a theoretically grounded inter-core communication mechanism.

The framework consists of four components:

1. **Parameterization**: An initialization sequence $\mathbf{I} = [t^{(1)}, \ldots, t^{(K)}]$ determines the starting time for each core.
2. **Initialization**: Core $k$ is initialized at time $t^{(k)}$ by jumping from $t=0$.
3. **Termination**: Upon reaching $t=1$, core $k$ outputs its result; users may select the output of any core according to quality requirements.
4. **Communication**: Every $\delta^{(k)} = t^{(k)} - t^{(k-1)}$ time units, the slower core rectifies the faster one.

### Key Designs

#### 1. Multi-Core Rectification

This is the fundamental operation in CHORDS. Consider two cores: slow core 1 starting from $t$ and fast core 2 starting from $t'$ (where $t < t'$). After $\delta_t = t' - t$ time units:

- Slow core 1 obtains a more accurate estimate $x^1_{t'}$.
- Fast core 2 has already advanced to $x^2_{t'+\delta_t}$.

The rectification rule transfers accurate information from the slow core to the fast core:

$$x^2_{t'+\delta_t} \leftarrow x^2_{t'+\delta_t} + r_\theta(x^1_{t'}, x^2_{t'}, t', \delta_t)$$

where the rectification term $r_\theta$ is computed by performing two single-step jumps at time $t'$ to approximate the continuous solution and taking their difference to correct accumulated error.

**Theoretical guarantee (Proposition 2.1)**: Under sufficient smoothness of $f_\theta$, the post-rectification error is a higher-order infinitesimal of the pre-rectification error:

$$\|\tilde{x}_{t'} + r_\theta - x_{t'}\|_2 = o(\|\tilde{x}_{t'} - x_{t'}\|_2)$$

**Why does this rectification work?** Intuitively, the slow core travels more finely and accurately, so its latent at any given time is closer to the true ODE trajectory. The rectification term exploits the local linearity of $f_\theta$ by comparing one-step estimates departing from different starting points, thereby canceling accumulated error.

#### 2. Optimal Initialization Sequence Selection

Framework performance depends critically on the choice of initialization sequence $\mathbf{I}$. The authors define:

- **Speedup ratio** $S(\mathbf{I}) = 1/(1 - t^{(K)})$: the acceleration factor provided by the fastest core.
- **Reward function** $R(\mathbf{I})$: a proxy metric measuring output quality.

**Why use a reward function rather than direct optimization?** Since $f_\theta$ is a high-dimensional neural network, directly precomputing errors is infeasible. The reward function captures the essential characteristics of parallel ODE solving through three properties: optimality, monotonicity, and trade-off.

**Theorem 2.5** (Optimal initialization for three cores): For $K=3$ and speedup ratio $s$, the optimal sequence is:
- $s \leq 3$: $t^{(2)} = t^{(3)}/2$
- $s > 3$: $t^{(2)} = 2t^{(3)} - 1$

For general $K$ cores, a greedy recursive strategy proceeding from fast to slow cores is employed.

#### 3. Discrete Instantiation

The continuous framework is translated into a practical algorithm via:

- **Initialization**: Core $k$ reaches $t(i_k)$ by jumping $k-1$ times from $t(i_1)$.
- **Scheduler** $\text{Scheduler}(N, \text{step}, k)$: determines the time interval each core processes at each step.
- **Communication condition** $\text{Communicate}(k, \text{prev}, \text{cur})$: rectification is triggered when $(\text{cur} - \text{prev})$ is divisible by $i_k - i_{k-1}$.
- **Streaming output**: cores output results in order of arrival at $t=1$, with quality increasing progressively.

### Loss & Training

CHORDS is a **training-free** method and involves no training procedure. It directly uses existing diffusion samplers (DDIM, Euler, etc.) as subroutines and achieves acceleration through inter-core communication. The only core hyperparameter is the initialization sequence $\hat{\mathbf{I}}$.

For $N=50$ steps and $K=4/6/8$ cores, the initialization sequences used in practice are:
- $K=4$: $[0, 8, 16, 32]$
- $K=6$: $[0, 3, 6, 12, 24, 36]$
- $K=8$: $[0, 2, 4, 8, 16, 24, 32, 40]$

## Key Experimental Results

### Main Results

Comprehensive evaluation is conducted on three video diffusion models and two image diffusion models.

| Model | Method | 4-core Speedup | 8-core Speedup | VBench/CLIP | Latent RMSE |
|-------|--------|----------------|----------------|-------------|-------------|
| HunyuanVideo | Sequential | 1.0× | 1.0× | 84.4% | — |
| HunyuanVideo | ParaDIGMS | 1.3× | 1.4× | 84.2% | 0.202 |
| HunyuanVideo | SRDS | 1.4× | 2.6× | 84.2% | 0.068 |
| **HunyuanVideo** | **CHORDS** | **2.1×** | **2.9×** | **84.1%** | **0.068** |
| Wan2.1 | CHORDS | 1.8× | 2.7× | 85.1% | 0.043 |
| CogVideoX1.5 | CHORDS | 2.0× | 2.4× | 81.8% | 0.055 |
| SD-3.5-Large | CHORDS | 2.0× | 2.4× | 32.5 | 0.211 |
| Flux | CHORDS | 2.0× | 2.6× | 31.0 | 0.179 |

Key finding: at 4 cores, CHORDS is on average 67% faster than SRDS; at 8 cores, 50% faster.

### Ablation Study

| Experiment | Setting | HunyuanVideo Speedup | Flux Speedup |
|------------|---------|----------------------|--------------|
| Init sequence | CHORDS (theoretically optimal) | 2.9× (8 cores) | 2.4× (8 cores) |
| Init sequence | Uniform distribution | 2.6× (8 cores) | 2.2× (8 cores) |
| Init sequence | CHORDS (theoretically optimal) | 2.1× (4 cores) | 2.0× (4 cores) |
| Init sequence | Uniform distribution | 1.8× (4 cores) | 1.8× (4 cores) |

Effect of varying step count $N$ (HunyuanVideo, 8 cores):

| $N$ | Speedup | VBench | Latent RMSE |
|-----|---------|--------|-------------|
| 50 | 2.9× | 84.1% | 0.068 |
| 75 | 3.4× | 84.4% | 0.073 |
| 100 | 3.6× | 84.6% | 0.076 |

### Key Findings

1. **Theory-guided initialization outperforms uniform distribution**: the theoretically optimal sequence consistently yields 10–15% higher speedup across all settings.
2. **Speedup scales with both core count and step count**: speedup further increases with larger $N$ (reaching 3.6× at $N=100$).
3. **Negligible quality degradation**: VBench/CLIP Score fluctuations are minimal across all settings, and Latent RMSE is substantially lower than that of ParaDIGMS.
4. **Unified framework**: both ParaDIGMS and SRDS can be viewed as special instantiations of the CHORDS framework.

## Highlights & Insights

1. **Seamless integration of theory and practice**: the top-down methodology—deriving optimal initialization from continuous ODE multigrid theory and then discretizing to a practical algorithm—is a valuable methodological reference.
2. **Diffusion streaming capability**: the hierarchical multi-core structure naturally supports progressive "fast-then-fine" output, well-suited for interactive applications.
3. **Orthogonality to other acceleration methods**: CHORDS can be combined with model parallelism (e.g., attention partitioning in DiTs) and distillation-based methods.
4. **Unifying framework**: subsuming multiple prior methods under a single framework clearly delineates directions for further improvement.

## Limitations & Future Work

1. Inter-core communication relies on synchronization barriers, which may introduce waiting overhead in heterogeneous computing environments.
2. Theoretical derivations are based on a simplified reward function ($f_\theta(x, t) = x$), which may diverge from the behavior of actual neural networks.
3. The rectification term requires additional network forward passes (though parallelized with normal steps), incurring non-trivial computational overhead.
4. The framework focuses primarily on ODE-based sampling; extension to SDE-based (stochastic) sampling warrants further investigation.

## Related Work & Insights

- **ParaDIGMS** [Shih et al.]: sliding-window Picard iteration; a special case of CHORDS.
- **SRDS** [Liu et al.]: multigrid parallel diffusion solver with a fixed core count of $\sqrt{N}$.
- **Classical multigrid methods** [Brandt 1977]: the theoretical foundation of CHORDS.
- **Insight**: the framework may generalize to accelerating other iterative generative models (e.g., speculative decoding for autoregressive models shares analogous principles).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (unified theoretical framework + practical algorithm design)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (5 models, 3 core-count configurations)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (clear theoretical derivations with well-structured progression)
- **Value**: ⭐⭐⭐⭐⭐ (plug-and-play, applicable to diverse models)
- **Overall**: ⭐⭐⭐⭐⭐ (a significant contribution to multi-core acceleration of diffusion models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Accelerating Diffusion Sampling via Exploiting Local Transition Coherence](accelerating_diffusion_sampling_via_exploiting_local_transition_coherence.md)
- [\[AAAI 2026\] Hierarchical Schedule Optimization for Fast and Robust Diffusion Model Sampling](../../AAAI2026/image_generation/hierarchical_schedule_optimization_for_fast_and_robust_diffusion_model_sampling.md)
- [\[ICCV 2025\] HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation](hypdae_hyperbolic_diffusion_autoencoders_for_hierarchical_few-shot_image_generat.md)
- [\[CVPR 2026\] Visual Diffusion Models are Geometric Solvers](../../CVPR2026/image_generation/visual_diffusion_models_are_geometric_solvers.md)
- [\[CVPR 2026\] Coupled Diffusion Sampling for Training-Free Multi-View Image Editing](../../CVPR2026/image_generation/coupled_diffusion_sampling_for_training-free_multi-view_image_editing.md)

</div>

<!-- RELATED:END -->
