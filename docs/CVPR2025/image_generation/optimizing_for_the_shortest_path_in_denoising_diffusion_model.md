---
title: >-
  [Paper Note] Optimizing for the Shortest Path in Denoising Diffusion Model
description: >-
  [CVPR 2025][Image Generation][Diffusion model acceleration] Models the denoising process of diffusion models as a shortest path problem in graph theory. By optimizing the initial residual to compress the reverse diffusion path, it achieves a generation quality with 2-step sampling that matches or even exceeds that of 10-step DDIM.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion model acceleration"
  - "shortest path optimization"
  - "residual propagation"
  - "graph theory"
  - "DDIM"
date: 2026-05-08
content_hash: db3753e79506f381
---

# Optimizing for the Shortest Path in Denoising Diffusion Model

**Conference**: CVPR 2025  
**arXiv**: [2503.03265](https://arxiv.org/abs/2503.03265)  
**Code**: [GitHub](https://github.com/UnicomAI/ShortDF)  
**Area**: Image Generation/Diffusion Model Acceleration  
**Keywords**: Diffusion model acceleration, shortest path optimization, residual propagation, graph theory, DDIM

## TL;DR

Models the denoising process of diffusion models as a shortest path problem in graph theory. By optimizing the initial residual to compress the reverse diffusion path, it achieves a generation quality with 2-step sampling that matches or even exceeds that of 10-step DDIM.

## Background & Motivation

Diffusion models have achieved outstanding performance in image generation, but their iterative sampling process leads to high computational costs, limiting real-time applications.

Limitations of prior work:
- **Distillation methods** (PD, InstaFlow, etc.): Compress the multi-step process into a single-step student network, but accurate approximation of the high-dimensional complex sampling process is difficult.
- **Fast samplers** (DDIM, DPM-Solver, etc.): Reduce steps through better numerical solvers, but quality degrades significantly when steps are further reduced.
- **Key Challenge**: The trade-off between fast sampling and high-quality generation.

The authors observe a residual propagation phenomenon in the DDIM sampling process: the estimation error at each step accumulates and propagates to subsequent steps. If the initial residual (estimation error of the first step) can be optimized, the cumulative error of the entire path can be reduced. This is highly analogous to the shortest path problem in graph theory.

## Method

### Overall Architecture

ShortDF is based on the DDIM framework, treating each step of reverse diffusion as a graph node and the transition error between steps as the edge weight. By constructing a reverse-step graph and applying shortest path relaxation optimization, it finds the optimal (lowest error) path from $x_T$ to $x_0$.

### Key Design 1: Residual Propagation Analysis (Residual Propagation)

- **Function**: Formalizes the multi-step error of the diffusion process as an optimizable path problem.
- **Mechanism**: Defines the initial residual $R(k_1, 0) = \frac{\sqrt{1-\bar{\alpha}_{k_1}}}{\sqrt{\bar{\alpha}_{k_1}}}(\epsilon_\theta(x_{k_1}, k_1) - \epsilon)$, and derives the path residual $R(k_n, 0) = R(k_1, 0) - \sum_{i=1}^{n-1} R(k_i, k_{i+1})$. Optimizing the full path residual is equivalent to optimizing the initial residual.
- **Design Motivation**: Directly optimizing the cumulative multi-step residual is infeasible, but a smaller initial residual benefits all subsequent steps since each step is constructed based on the estimate from the previous step. This provides theoretical support for path optimization.

### Key Design 2: Graph Construction & Relaxation

- **Function**: Dynamically selects the optimal transition path via relaxation conditions.
- **Mechanism**: Defines edge weight $\text{edge}(k,t) = |x_0 - \hat{x}'_{0|k}| - |x_0 - \hat{x}_{0|k}|$, utilizing ground-truth $x_0$ to eliminate path residuals so that the optimal solution at step $k$ is not interfered with by the residual at step $t$. When the relaxation condition $dist(x_t, t) > dist(x_k, k) + edge(k,t)$ holds, the path selection is updated.
- **Design Motivation**: Directly defining edge weights with residuals introduces inter-step interference, making optimal propagation untrackable. Introducing the ground-truth $x_0$ as a reference eliminates path residuals, which is then progressively optimized using graph-theoretic relaxation methods.

### Key Design 3: Multi-State Optimization Strategy (Multi-State Optimization)

- **Function**: Stabilizes training and achieves the practical implementation of graph-theoretic optimization.
- **Mechanism**: Maintains three model instances—Base Model $\epsilon_\theta$ for noise prediction and residual updates; EMA Model $\epsilon_{\theta,ema}$ to provide a stable optimal estimate for step $k$; Graph Model $\epsilon_{\theta,graph}$ to compute edge weights with delayed updates to capture global information.
- **Design Motivation**: Training instability caused by random noise makes direct optimization difficult. Three models each serve specific roles—the base model learns, the EMA model provides stable targets, and the graph model offers consistent edge weights.

### Loss & Training

The total loss is $L = \lambda \cdot L_\epsilon + cond \cdot L_r$, where $L_\epsilon$ is the standard noise prediction loss, and $L_r = \|dist(x_k, k) + edge(k, t) - dist(x_t, t)\|_2$ is the relaxation loss. $cond$ is an indicator function for the relaxation condition, which activates path optimization only when the condition is met.

## Key Experimental Results

### Main Results: CIFAR-10 FID Comparison

| Method | 1 Step | 2 Steps | 3 Steps | 4 Steps | 5 Steps | 10 Steps |
|------|-----|-----|-----|-----|-----|------|
| DDIM | >100 | >100 | 123.54 | 66.13 | 26.91 | 11.14 |
| DPM-solver | - | - | 90.38 | 33.29 | 23.31 | 5.09 |
| DPM-solver++ | - | - | 107.02 | 30.46 | 18.87 | 7.83 |
| SDDPM | - | - | - | 19.20 | - | - |
| **ShortDF** | - | **9.08** | - | - | - | - |

### Efficiency Comparison

| Metric | DDIM (10 Steps) | ShortDF (2 Steps) |
|------|-----------|-------------|
| FID | 11.14 | **9.08** |
| Speedup | 1.0x | **5.0x** |
| Time per Step | 1.2ms | 1.2ms |

### Key Findings

- 2-step ShortDF (FID=9.08) outperforms 10-step DDIM (FID=11.14), improving quality by 18.5% and achieving a 5x speedup.
- Performs equally well on CelebA and LSUN Churches.
- Path optimization is most advantageous at 2 steps, as the initial residual has the greatest impact on the final result at this stage.
- Removing graph modeling degrades the model back to DDIM, verifying the effectiveness of the shortest path optimization.

## Highlights & Insights

1. **Novel graph-theoretic perspective**: Formulates the reverse process of diffusion models as a shortest path problem on a weighted graph for the first time, providing an elegant theoretical framework.
2. **Clear intuition for path compression**: Through iterative relaxation, multi-hop paths like $x_0 \to x_k \to x_t$ are compressed into direct paths like $x_0 \to x_t$.
3. **Maintains sampler universality**: Does not modify the network architecture; only introduces additional loss during training, while inference uses standard DDIM sampling.

## Limitations & Future Work

- Requires retraining the diffusion model and cannot be directly applied to already-trained pre-trained models.
- The multi-state optimization strategy increases training overhead (requires maintaining three model copies).
- Limited validation on complex conditional generation tasks such as text-to-image.
- Lacks quantitative analysis of the gap between the theoretical optimal path and its practical approximation.

## Related Work & Insights

- **DDIM**: The foundational framework for ShortDF, which accelerates sampling through a flexible sampling path.
- **DPM-Solver**: A specialized solver that utilizes the semi-linear structure of ODEs.
- **RDDM**: Decomposes denoising into residual diffusion and noise diffusion, resonating with the residual analysis ideas in ShortDF.
- The concept of shortest path optimization could potentially be generalized to other iterative generative models.

## Rating

⭐⭐⭐⭐ — Unique graph-theoretic perspective with rigorous theoretical derivations; the result of 2-step sampling surpassing 10-step DDIM is highly impressive. However, the requirement of retraining and the lack of friendliness to existing pre-trained models present significant limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](../../CVPR2026/image_generation/dpcache_denoising_path_planning_diffusion_accel.md)
- [\[CVPR 2025\] Fractals made Practical: Denoising Diffusion as Partitioned Iterated Function Systems](fractals_made_practical_denoising_diffusion_as_partitioned_iterated_function_sys.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](../../NeurIPS2025/image_generation/denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [\[ICCV 2025\] Fewer Denoising Steps or Cheaper Per-Step Inference: Towards Compute-Optimal Diffusion Model Deployment](../../ICCV2025/image_generation/fewer_denoising_steps_or_cheaper_per-step_inference_towards_compute-optimal_diff.md)
- [\[CVPR 2025\] Random Conditioning for Diffusion Model Compression with Distillation](random_conditioning_for_diffusion_model_compression_with_distillation.md)

</div>

<!-- RELATED:END -->
