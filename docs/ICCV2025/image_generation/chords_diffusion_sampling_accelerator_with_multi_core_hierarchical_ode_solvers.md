---
title: >-
  [Paper Note] CHORDS: Diffusion Sampling Accelerator with Multi-Core Hierarchical ODE Solvers
description: >-
  [ICCV 2025][Image Generation][Diffusion Models] This paper proposes CHORDS, a training-free and model-agnostic diffusion sampling acceleration framework based on multi-core hierarchical ODE solvers. By employing a slow-to-fast solver hierarchy and an inter-core rectification mechanism, CHORDS achieves up to 2.9× speedup across 4–8 GPU cores without sacrificing generation quality.
tags:
  - ICCV 2025
  - Image Generation
  - Diffusion Models
  - Multi-Core Parallel Sampling
  - ODE Solvers
  - Training-Free Acceleration
  - Video Generation
date: 2026-05-08
content_hash: d602c52a42ad02c5
---

# CHORDS: Diffusion Sampling Accelerator with Multi-Core Hierarchical ODE Solvers

**Conference**: ICCV 2025
**arXiv**: [2507.15260](https://arxiv.org/abs/2507.15260)
**Code**: [https://hanjq17.github.io/CHORDS](https://hanjq17.github.io/CHORDS)
**Area**: Image/Video Generation Acceleration
**Keywords**: Diffusion Models, Multi-Core Parallel Sampling, ODE Solvers, Training-Free Acceleration, Video Generation

## TL;DR

This paper proposes CHORDS, a training-free and model-agnostic diffusion sampling acceleration framework based on multi-core hierarchical ODE solvers. By employing a slow-to-fast solver hierarchy and an inter-core rectification mechanism, CHORDS achieves up to 2.9× speedup across 4–8 GPU cores without sacrificing generation quality.

## Background & Motivation

Diffusion models have become the dominant approach for high-fidelity image and video generation, yet their iterative sampling process incurs substantial computational overhead, severely limiting deployment in latency-sensitive scenarios such as real-time editing and streaming applications. Existing acceleration methods fall into two main categories: (1) distillation-based methods require additional training and may compromise quality; (2) efficient ODE solver-based methods offer limited further speedup under single-core settings.

**Root Cause**: How can multi-GPU parallel resources be leveraged for significant acceleration without retraining or restricting model architecture? Existing multi-core methods either depend on specific architectures, require additional training, or lack flexible resource allocation.

**Starting Point**: Inspired by classical multigrid ODE acceleration algorithms, CHORDS treats multi-core diffusion sampling as an ODE solving pipeline, where fast but imprecise solvers can be hierarchically corrected using information from slow, accurate solvers—theoretically guaranteeing acceleration without degrading precision.

## Method

### Overall Architecture

CHORDS organizes $K$ compute cores into a slow-to-fast hierarchical solver sequence. The slowest core solves the full ODE from $t=0$ (pure noise), while faster cores start from later time points, initialized with coarse single-step jumps and subsequently refined via pipelined inter-core rectification to achieve accuracy comparable to the slowest core.

### Key Designs

1. **Multi-Core Rectification**: The central operation. When a slow core $k{-}1$ reaches a time point corresponding to fast core $k$, the discrepancy between both cores at the same timestep is used to correct the fast core's trajectory. The rectification term is defined as:

    $\mathbf{r}_\theta(\mathbf{x}_t, \tilde{\mathbf{x}}_t, t, \delta_t) = \delta_t \cdot (\mathbf{f}_\theta(\mathbf{x}_t, t) - \mathbf{f}_\theta(\tilde{\mathbf{x}}_t, t)) + \mathbf{x}_t - \tilde{\mathbf{x}}_t$

   **Theoretical Guarantee** (Proposition 2.1): Under sufficient smoothness of $f$, the post-rectification error is a higher-order infinitesimal relative to the pre-rectification error, i.e., $\|\text{error}_\text{after}\| = o(\|\text{error}_\text{before}\|)$. This ensures that information propagates from the most accurate core to the fastest core layer by layer, effectively suppressing error accumulation.

2. **Initialization Sequence Selection**: A proxy reward function $\mathcal{R}(\mathbf{I})$ is defined (satisfying three axioms: optimality, monotonicity, and efficiency–accuracy trade-off), decomposing the optimal initialization time allocation problem for $K$ cores into a series of three-core optimization subproblems. Theorem 2.5 provides a closed-form solution: when the speedup ratio $s \leq 3$, the intermediate core initializes at $t^{(2)} = t^{(3)}/2$; when $s > 3$, $t^{(2)} = 2t^{(3)} - 1$. The general $K$-core case is determined via a fast-to-slow recursive procedure.

3. **Discretization Implementation**: The continuous framework is converted into a practical algorithm (Algorithm 1). Each core determines its current and next discrete timesteps according to a scheduler and performs forward solving simultaneously. Communication conditions are governed by the divisibility of inter-core timestep differences, ensuring a bubble-free pipeline.

### Loss & Training

CHORDS is entirely training-free—no parameters of the pre-trained diffusion model are modified. Acceleration is achieved solely by scheduling multi-core forward inference. The method is compatible with arbitrary ODE solvers such as DDIM and Euler.

## Key Experimental Results

### Main Results

Comparisons against ParaDIGMS and SRDS across three video diffusion models and two image diffusion models:

| Model | Cores | Method | Speedup | Quality (VBench/CLIP) | Latent RMSE |
|-------|-------|--------|---------|----------------------|-------------|
| HunyuanVideo | 4 | Sequential | 1.0× | 84.4% | - |
| HunyuanVideo | 4 | SRDS | 1.4× | 84.2% | 0.068 |
| HunyuanVideo | 4 | **CHORDS** | **2.1×** | 84.1% | 0.066 |
| HunyuanVideo | 8 | SRDS | 2.6× | 84.2% | 0.068 |
| HunyuanVideo | 8 | **CHORDS** | **2.9×** | 84.1% | 0.068 |
| Flux | 8 | SRDS | 2.3× | 31.0 | 0.183 |
| Flux | 8 | **CHORDS** | **2.6×** | 31.0 | 0.179 |

### Ablation Study

| Cores | Initialization | HunyuanVideo Speedup | VBench | Flux Speedup | CLIP |
|-------|---------------|----------------------|--------|--------------|------|
| 8 | Ours | 2.9× | 84.1% | 2.4× | 31.0 |
| 8 | Uniform | 2.6× | 84.0% | 2.2× | 30.9 |
| 4 | Ours | 2.1× | 84.1% | 2.0× | 31.1 |
| 4 | Uniform | 1.8× | 84.2% | 1.8× | 31.0 |

The theory-driven non-uniform initialization sequence consistently outperforms uniform distribution across all settings, validating the practical utility of Theorem 2.5.

### Key Findings

- At 4 cores, CHORDS is approximately 50% faster than SRDS (2.1× vs. 1.4×), with the advantage further increasing at 8 cores.
- As the total number of diffusion steps $N$ increases (50→100), the speedup ratio improves from 2.9× to 3.6×, with a slight concurrent improvement in quality.
- The method naturally supports a "diffusion streaming" paradigm—fast cores first output coarse results, while slow cores subsequently provide higher-quality outputs.

## Highlights & Insights

- **Strong alignment between theory and practice**: The optimal initialization is derived from continuous ODE analysis and then discretized into a practical algorithm, yielding a rigorous end-to-end design pipeline.
- **Fully training-free and model-agnostic**: Any diffusion model can be accelerated with multi-GPU inference alone, presenting an extremely low deployment barrier.
- **Streaming output**: The hierarchical structure naturally supports progressive quality refinement, making it well-suited for interactive applications.

## Limitations & Future Work

- Multiple GPU cores must be simultaneously available; single-GPU scenarios are not supported.
- The theoretically optimal initialization is derived from a simplified linear ODE proxy, leaving a gap with the actual nonlinear neural network ODE.
- The combination with orthogonal acceleration methods such as model distillation and attention parallelism has not been explored.
- Communication overhead and load balancing across different models have not been thoroughly analyzed.

## Related Work & Insights

- **ParaDIGMS** (Picard iteration parallelism) and **SRDS** (self-rectified multigrid) can be viewed as special cases of the proposed framework.
- The method is orthogonal to model parallelism approaches (e.g., attention distribution in xDiT) and can be combined with them.
- The streaming output mechanism may offer particular value for applications such as interactive video editing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — A novel perspective grounded in multigrid ODE theory
- Theoretical Depth: ⭐⭐⭐⭐⭐ — Complete theoretical analysis with optimality proofs
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five models, multiple core configurations, thorough ablations
- Practicality: ⭐⭐⭐⭐ — Requires multiple GPUs but is straightforward to deploy

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Accelerating Diffusion Sampling via Exploiting Local Transition Coherence](accelerating_diffusion_sampling_via_exploiting_local_transition_coherence.md)
- [\[ICCV 2025\] End-to-End Multi-Modal Diffusion Mamba](end-to-end_multi-modal_diffusion_mamba.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)
- [\[ICCV 2025\] Golden Noise for Diffusion Models: A Learning Framework](golden_noise_for_diffusion_models_a_learning_framework.md)

<!-- RELATED:END -->
