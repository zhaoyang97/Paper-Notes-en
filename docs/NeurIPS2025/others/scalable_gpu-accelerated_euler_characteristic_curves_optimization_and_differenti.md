---
title: >-
  [Paper Note] Scalable GPU-Accelerated Euler Characteristic Curves: Optimization and Differentiable Learning for PyTorch
description: >-
  [NeurIPS 2025 (NeurReps Workshop)][Euler characteristic curves] This paper proposes an ECC CUDA kernel optimized for modern Ampere GPUs, achieving 16–2000× speedup over prior GPU implementations…
tags:
  - "NeurIPS 2025 (NeurReps Workshop)"
  - "Euler characteristic curves"
  - "GPU acceleration"
  - "differentiable programming"
  - "CUDA kernels"
  - "topological data analysis"
date: 2026-05-08
content_hash: af8bed90c2d41ee6
---

# Scalable GPU-Accelerated Euler Characteristic Curves: Optimization and Differentiable Learning for PyTorch

**Conference**: NeurIPS 2025 (NeurReps Workshop)
**arXiv**: [2510.20271](https://arxiv.org/abs/2510.20271)  
**Code**: Available  
**Area**: Topological Deep Learning / GPU Computing
**Keywords**: Euler characteristic curves, GPU acceleration, differentiable programming, CUDA kernels, topological data analysis

## TL;DR

This paper proposes an ECC CUDA kernel optimized for modern Ampere GPUs, achieving 16–2000× speedup over prior GPU implementations, and introduces a differentiable PyTorch layer supporting end-to-end topological feature learning on dense grid images via DECT-style sigmoid relaxation.

## Background & Motivation

**Background**: Topological features such as Euler characteristic curves (ECC) and the Euler characteristic transform (ECT) capture global geometric structure in image data and are theoretically sufficient statistics for shape description. Recent work such as DECT has enabled differentiable topological feature learning on graphs and point clouds.

**Limitations of Prior Work**: (1) Existing GPU implementations (Wang et al., 2023) were designed for legacy hardware and employ a chunking strategy to cope with limited shared memory, incurring repeated kernel launches, inter-block synchronization, redundant boundary computations, and frequent global atomic operations. (2) DECT supports only graphs and point clouds, not dense 2D/3D image grids. (3) No existing implementation is simultaneously efficient and differentiable.

**Key Challenge**: While topological features are theoretically attractive, computational cost and non-differentiability are the two main obstacles to integration into deep learning pipelines. The absence of algorithm–hardware co-design (from a hardware lottery perspective) prevents theoretically promising topological invariants from becoming practical components.

**Goal**: (1) Eliminate the overhead of the chunking strategy by leveraging the large memory and high bandwidth of modern GPUs. (2) Provide a differentiable ECC implementation for dense cubical complexes (2D/3D image grids).

**Key Insight**: Redesign the ECC computation kernel from the perspective of GPU architectural characteristics—replacing chunked iteration with a single full-grid scan, using 128-byte aligned memory access and hierarchical shared-memory accumulation.

**Core Idea**: Eliminate chunking overhead through a full-scan CUDA kernel design tailored to the Ampere GPU architecture, and achieve a differentiable ECC layer via sigmoid relaxation.

## Method

### Overall Architecture

The work comprises two components: (1) an efficient ECC computation kernel that processes the entire dense grid in a single kernel launch, assigning one thread per voxel and achieving high throughput via hierarchical accumulation using shared-memory histograms and global atomic operations; (2) a differentiable PyTorch layer that replaces the hard-threshold indicator function with a sigmoid, exposes learnable threshold and direction parameters, and supports gradient-descent optimization.

### Key Designs

1. **Full-Scan CUDA Kernel**

    - **Function**: Efficient computation of ECC on dense 2D/3D grids.
    - **Mechanism**: Each voxel is assigned exactly one thread, ensuring globally 128-byte aligned (coalesced) memory access. Each thread block maintains a private shared-memory histogram with $H$ bins; threads accumulate local ECC contributions in shared memory via `__syncthreads()` barriers. A single global atomic-add operation is performed only after all voxels in a block have been processed, substantially reducing atomic operation count. Boundary conditions are handled at the warp level to maintain convergence without redundant passes.
    - **Design Motivation**: The chunking strategy of prior implementations is an unnecessary constraint on modern GPUs with 48 GB VRAM and high bandwidth. The full-scan design eliminates multiple kernel launches, inter-block synchronization, and redundant boundary computations, yielding latency that scales more gracefully with data size.

2. **Differentiable ECC Layer (Sigmoid Relaxation)**

    - **Function**: Enable ECC for end-to-end gradient-based optimization.
    - **Mechanism**: The hard-threshold indicator $\mathbf{1}[X(p) \leq \tau]$ is replaced by the sigmoid $\sigma_\lambda(\tau - X(p))$, yielding a soft ECC. As $\lambda \to \infty$, the original ECC is recovered; for finite $\lambda$, gradients exist. For the single-direction variant, learnable direction $u$ and scale $\alpha$ are incorporated, enabling gradient backpropagation with respect to both direction and threshold. A custom CUDA backward kernel computes analytic gradients, avoiding the overhead of automatic differentiation.
    - **Design Motivation**: The sigmoid relaxation strategy from DECT has been validated but targets graphs and point clouds. Specializing it to dense cubical grids with efficient CUDA forward/backward kernels makes topological feature learning on dense images feasible.

3. **Local Computation of Euler Characteristic Coefficients**

    - **Function**: Compute each pixel/voxel's contribution to topology.
    - **Mechanism**: The ECC decomposes into a sum of local per-pixel contributions, where the coefficient $c(p) \in \{-1, 0, 1\}$ is computed from $p$'s local $3 \times 3$ (2D) or $3 \times 3 \times 3$ (3D) binary neighborhood. These coefficients encode how each pixel contributes to global topology—creating a connected component (+1), forming a hole (−1), or leaving topology unchanged (0).
    - **Design Motivation**: The pointwise decomposition makes ECC naturally suited for GPU parallelism: each thread independently computes one pixel's contribution without cross-thread communication.

### Loss & Training

The PyTorch layer exposes learnable threshold and optional direction parameters. It computes ECC via a custom CUDA forward kernel and analytic gradients via a custom CUDA backward kernel, and can be directly embedded into standard PyTorch training loops for arbitrary downstream loss optimization.

## Key Experimental Results

### 2D Synthetic Grid Speedup Comparison

| Grid Size | Baseline (ms) | Ours (ms) | Speedup |
|-----------|--------------|-----------|---------|
| $128^2$ | 7.7 | 0.0165 | **466×** |
| $256^2$ | 24 | 0.0174 | **1379×** |
| $512^2$ | 39.68 | 0.021 | **1900×** |
| $1024^2$ | 65 | 0.032 | **2031×** |
| $2048^2$ | 92 | 0.11 | 836× |
| $4096^2$ | 154 | 0.408 | 377× |
| $8192^2$ | 267 | 1.6 | 166× |

### 3D Synthetic Grid Speedup Comparison

| Grid Size | Baseline (ms) | Ours (ms) | Speedup |
|-----------|--------------|-----------|---------|
| $128^3$ | 27 | 0.1 | **270×** |
| $256^3$ | 91 | 0.71 | **128×** |
| $512^3$ | 140 | 5.61 | 25× |
| $1024^3$ | 800 | 48.47 | 16.5× |

### Key Findings

- **Peak 2D speedup at $1024^2$**: 2031× acceleration with 0.032 ms makes real-time topological feature extraction fully practical.
- **Speedup increases then decreases with grid size**: Mid-range grids ($512^2$–$1024^2$) yield the greatest gains due to optimal thread utilization under the full-scan strategy; very large grids ($8192^2$) are memory-bandwidth bound yet still achieve 166×.
- **Lower 3D speedups**: Only 16.5× at $1024^3$, as 3D voxel memory grows cubically, constraining the full-scan strategy on very large 3D grids.
- **Single GPU suffices**: All experiments run on an NVIDIA RTX A6000 (48 GB, Ampere), demonstrating that expensive data-center hardware is unnecessary.
- **Negligible end-to-end latency**: In practical deep learning pipelines, ECC computation requires sub-millisecond time, negligible compared to other forward-pass components.

## Highlights & Insights

- **A successful case of algorithm–hardware co-design**: This work vividly illustrates the hardware lottery perspective—the same algorithm (ECC) can differ by up to 2000× depending on hardware assumptions. The old chunking design was a pragmatic compromise for legacy hardware; the new full-scan design fully exploits modern GPU memory and bandwidth.
- **Ultra-low latency enables real-time applications**: A 0.032 ms ECC computation on $1024^2$ images enables topological feature extraction as a module in real-time vision pipelines—previously inconceivable.
- **Differentiable implementation lowers the barrier to adoption**: The native PyTorch interface allows topological features to be embedded in deep learning models like any convolutional layer, without requiring expertise in topological data analysis.
- **Transferable to scientific imaging**: In scientific and medical imaging contexts such as cosmic microwave background (CMB) analysis and digital breast tomosynthesis (VICTRE), fast differentiable ECC can serve as a geometric prior complementing learned features.

## Limitations & Future Work

- **Validation only on synthetic grids**: Downstream task experiments on real images or medical data (e.g., segmentation, classification) are absent, leaving the practical utility of topological features unconfirmed.
- **Single-GPU limitation**: Multi-GPU data parallelism and batched execution are not supported, limiting applicability to large-scale training scenarios.
- **Ampere-only evaluation**: Performance on newer architectures (e.g., Hopper/H100) has not been assessed, and compatibility with older architectures (e.g., Volta/Turing) is not verified.
- **Single-direction differentiable ECC**: A complete ECT requires integration over multiple directions to constitute a sufficient statistic for shape; only the single-direction case is implemented, leaving theoretical descriptive capacity incomplete.
- **No comparison with persistent homology**: The information richness and downstream task performance of ECC are not compared against persistence diagrams, barcodes, or other TDA methods.
- **Future directions**: Multi-GPU scaling, batched execution support, broader GPU architecture compatibility, and end-to-end integration with the VICTRE medical imaging benchmark.

## Related Work & Insights

- **vs. Wang et al. (2023) GPU ECC**: That work employs a chunking strategy adapted to legacy GPU memory constraints; this paper achieves 16–2000× speedup on modern GPUs via a full-scan strategy, representing a direct technical advancement.
- **vs. DECT (Roell & Rieck, 2024)**: DECT (ICLR 2024) implements differentiable Euler characteristic transforms on graphs and point clouds; this paper extends differentiability to dense cubical grids (2D/3D), covering image and volumetric data scenarios.
- **vs. Persistent Homology**: Persistent homology provides richer topological information (multi-scale birth–death pairings) but is computationally more expensive and harder to differentiate. ECC is a lightweight alternative suited to latency-sensitive settings.

## Rating

- Novelty: ⭐⭐⭐ — The CUDA kernel optimization is engineering-driven innovation; the sigmoid relaxation is adapted from DECT.
- Experimental Thoroughness: ⭐⭐⭐ — Speedup benchmarks are thorough, but downstream task and real-data experiments are missing.
- Writing Quality: ⭐⭐⭐⭐ — Technical details are clear; the chunking-vs-full-scan comparative analysis is well presented.
- Value: ⭐⭐⭐⭐ — High engineering value, substantially lowering the barrier to incorporating topological features into deep learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Exact Learning of Arithmetic with Differentiable Agents](exact_learning_of_arithmetic_with_differentiable_agents.md)
- [\[NeurIPS 2025\] Learning to Condition: A Neural Heuristic for Scalable MPE Inference](learning_to_condition_a_neural_heuristic_for_scalable_mpe_inference.md)
- [\[NeurIPS 2025\] A Differentiable Model of Supply-Chain Shocks](a_differentiable_model_of_supply-chain_shocks.md)
- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[NeurIPS 2025\] Overfitting in Adaptive Robust Optimization](overfitting_in_adaptive_robust_optimization.md)

</div>

<!-- RELATED:END -->
