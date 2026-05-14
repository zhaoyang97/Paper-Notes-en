---
title: >-
  [Paper Note] DiffBMP: Differentiable Rendering with Bitmap Primitives
description: >-
  [CVPR2026][differentiable rendering] This paper proposes DiffBMP — the first general-purpose differentiable rendering engine for **bitmap primitives** — which enables efficient gradient-based optimization of position…
tags:
  - "CVPR2026"
  - "differentiable rendering"
  - "bitmap primitives"
  - "CUDA kernel"
  - "soft rasterization"
  - "alpha compositing"
  - "creative workflow"
date: 2026-05-08
content_hash: e659c3d09c651c6a
---

# DiffBMP: Differentiable Rendering with Bitmap Primitives

**Conference**: CVPR2026  
**arXiv**: [2602.22625](https://arxiv.org/abs/2602.22625)  
**Code**: [diffbmp.com](https://diffbmp.com)  
**Area**: others (Differentiable Rendering / Computer Graphics)  
**Keywords**: differentiable rendering, bitmap primitives, CUDA kernel, soft rasterization, alpha compositing, creative workflow

## TL;DR

This paper proposes DiffBMP — the first general-purpose differentiable rendering engine for **bitmap primitives** — which enables efficient gradient-based optimization of position, rotation, scale, color, and opacity across thousands of bitmap primitives via a custom CUDA parallel pipeline, filling the gap left by 2D differentiable rendering methods that are restricted to vector graphics.

## Background & Motivation

**Core requirement of differentiable rendering**: Large-scale optimization problems rely on first-order gradient methods, requiring the rendering process to be differentiable with respect to scene parameters. While mature solutions exist in 3D (NeRF, 3DGS), 2D differentiable rendering remains confined to vector graphics.

**Existing methods support only vector primitives**: DiffVG and its follow-up works perform well on vector paths, but the vast majority of real-world 2D assets are bitmaps, which cannot directly participate in gradient-based optimization.

**Challenges of differentiable bitmap rendering**: Bitmaps are discrete, high-dimensional pixel arrays that impose substantial memory and computational overhead. Although Spatial Transformer Networks (STN) introduced differentiable image sampling, this idea has not been generalized to general bitmap composition optimization.

**Limitations of Prior Work on bitmaps**: Reddy et al. represent the only prior attempt at differentiable bitmap rendering, but their approach lacks transparency support, parallel acceleration, and is restricted to narrow tasks such as repeating opaque patterns.

**DiffVG cannot handle complex primitives**: Experiments show that DiffVG suffers a sharp drop in PSNR and a dramatic increase in runtime when confronted with complex SVG curves, rendering even a vectorize-then-optimize pipeline infeasible.

**Missing creative workflow**: No existing tool supports exporting optimization results as layered PSD files for seamless integration into designer workflows.

## Method

### Overall Architecture

The DiffBMP pipeline proceeds as follows: given a set of bitmap primitives and a target image as **input**, differentiable forward rendering (coordinate transformation + bilinear interpolation sampling + Porter-Duff alpha compositing) produces a rendered result; a loss is computed; gradients are efficiently computed via custom CUDA backward kernels; each primitive's parameters $(x_i, y_i, s_i, \theta_i, \nu_i, \mathbf{c}_i)$ are updated; and upon convergence, a layered PSD file is exported.

### Forward Pass

- **Coordinate transformation and sampling**: For each canvas pixel $(x,y)$, a rotation matrix combined with translation and scale transforms maps the pixel to the primitive's normalized coordinates $(u,v) \in [-1,1]^2$, which are then converted to discrete coordinates $(U,V)$ and sampled via **bilinear interpolation**, making spatial transformations fully differentiable.
- **Alpha compositing**: The alpha value of each primitive is defined as $\alpha_{\max} \cdot \sigma(\nu_i) \cdot M_i(x,y)$, and standard Porter-Duff over compositing is applied to accumulate transmittance and final color.
- **Tile-based CUDA parallelism**: The canvas is divided into $T \times T$ tiles (default $T=32$), with each tile handled by one CUDA thread block, achieving full pixel-level parallelism.

### Key Designs

1. **Soft Rasterization**: A Gaussian blur is applied to each primitive to extend the spatial support of gradients, addressing the sparsity of bilinear interpolation gradients (which are non-zero only near object boundaries). Ablation studies confirm consistent PSNR improvements (Tab. 3).
2. **Structure-aware Initialization**: The local variance of the target image (computed over a $7 \times 7$ window) guides primitive placement — high-variance regions receive densely placed small primitives, while low-variance regions receive sparsely placed large ones; colors are initialized to target pixel values plus noise.
3. **Noisy Canvas**: The background is set to uniform random noise $\mathbf{b}(x,y) \sim \mathcal{U}[0,1]^3$, forcing primitives to cover regions whose color matches the background and thereby preventing coverage holes.
4. **Color Constraint**: An optional parameter $\mu_{\text{blend}}$ controls whether the original primitive colors are preserved; setting $\mu_{\text{blend}}=1$ fully retains original colors (e.g., for brand logo mosaic applications).

### Loss & Training

- **Base loss**: $\| I - I^{\text{target}} \|_2^2$ (pixel-wise MSE).
- **Spatial constraint loss** (Eq. 9): $\mathcal{L} = \|(I_\alpha^{\text{target}} > 0) \odot (I - I^{\text{target}})\|_2^2 + \lambda_\alpha \|I_\alpha - I_\alpha^{\text{target}}\|_2^2$, used for foreground rendering to enforce primitive disappearance in background regions.
- **CLIP loss**: Can be combined with CLIP to enable text-driven bitmap composition.

### Backward Pass and Efficiency

- Gradients are propagated via the chain rule from rendered outputs to all primitive parameters (position, scale, rotation, color, opacity) and can be computed exactly without approximation.
- **FP16 half-precision** arithmetic combined with `__half2` packing and `atomicAdd` substantially reduces bandwidth and memory consumption.
- Dedicated export CUDA kernels support high-resolution PSD export (optimizing at low resolution and exporting at 2×/4× higher resolution).

## Experiments

### Main Results

| Implementation | Resolution 512² | Resolution 1024² (tile=32) | Resolution 2048² |
|---|---|---|---|
| PyTorch (RTX 3090) | 1360/2337 ms, 6.4 GB | 1393/2477 ms, 5.0 GB | 5405/9483 ms, 9.0 GB |
| CUDA-FP32 (RTX 3090) | 3.9/11.6 ms, 1.0 GB | 7.6/9.3 ms, 2.0 GB | 16.1/10.0 ms, 6.1 GB |
| CUDA-FP16 (RTX 3090) | **2.3/6.2 ms, 1.1 GB** | **4.3/5.5 ms, 1.6 GB** | **9.0/6.4 ms, 3.8 GB** |

> CUDA-FP16 is approximately **350–600× faster** than the PyTorch baseline, with memory reduced by approximately **2.5–6×**.

### Ablation Study

| Soft Rasterization | Structure-aware Init | Scenario 1 (PSNR) | Scenario 2 | Scenario 3 |
|---|---|---|---|---|
| ✗ | ✗ | 24.4 | 20.6 | 25.9 |
| ✓ | ✗ | 24.7 | 21.5 | 26.5 |
| ✗ | ✓ | 25.5 | 21.0 | 27.1 |
| ✓ | ✓ | **25.7** | **21.7** | **27.4** |

The combination of both techniques achieves the best PSNR across all scenarios.

### Key Findings

- **DiffVG fails on complex SVGs**: When confronted with bitmap-level complexity in vector primitives, DiffVG exhibits significant PSNR degradation and sharp runtime increases, demonstrating the necessity of DiffBMP.
- **Dynamic video**: Combining sequential initialization, removal of stuck primitives, and freezing of unchanged regions, DiffBMP achieves the best temporal consistency (tOF=1.84) across 17 video sequences while maintaining competitive per-frame fidelity (PSNR=24.38).
- **Noisy canvas** effectively eliminates coverage holes in regions whose color matches the background.
- **Spatial constraints** combined with opacity loss and re-initialization of low-opacity primitives yield the cleanest foreground rendering.

## Highlights & Insights

- **Filling a gap**: DiffBMP is the first general-purpose, high-efficiency differentiable rendering engine for arbitrary bitmap primitives, serving as the bitmap counterpart to DiffVG.
- **Engineering excellence**: A custom tile-based CUDA kernel with FP16 mixed precision enables optimization of thousands of primitives within one minute on a consumer-grade GPU.
- **Strong practicality**: Layered PSD export, a Python interface, and CLIP-driven text-guided creation allow direct integration into designer workflows.
- **Complete optimization toolkit**: Soft rasterization, structure-aware initialization, and noisy canvas are each validated by ablation studies, with notable combined gains.
- **Diverse applications**: Brand logo mosaics, video modeling, foreground-constrained rendering, and text-driven creation are demonstrated.

## Limitations & Future Work

- **GPU dependency**: Unlike DiffVG, which can run on CPU, DiffBMP is CUDA-based and requires an NVIDIA GPU.
- **Hyperparameter sensitivity**: The generality of the framework makes hyperparameter selection and initialization strategy highly influential on results, with a risk of local optima and no automatic tuning mechanism.
- **Autoregressive/RL not explored**: The paper notes that differentiable bitmap rendering could serve as a foundation for autoregressive painting and reinforcement learning, but neither is implemented.
- **Trade-offs remain in video**: Dynamic DiffBMP still exhibits a trade-off between flicker suppression and per-frame fidelity that has not been fully resolved.

## Related Work & Insights

- **Vector differentiable rendering**: DiffVG [Li et al., 2020] and its extensions (image vectorization, text-driven SVG generation); Bézier Splatting [Liu et al., 2025] is likewise restricted to vector primitives.
- **Bitmap differentiable rendering**: STN [Jaderberg et al., 2015] introduced differentiable spatial transformations; Reddy et al. applied this to pattern composition but without parallelism or transparency support.
- **3D differentiable rendering**: NeRF, 3DGS, and their accelerated variants (Plenoxels, 3D Convex Splatting) inform DiffBMP's tile-based parallelism design.
- **Neural painting**: RL- or feedforward-based painting methods such as Paint Transformer [Liu et al., 2021] and CLIPDraw [Frans et al., 2022]; DiffBMP offers a gradient-optimization alternative to these approaches.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Fills the gap in differentiable bitmap rendering; the problem is clearly defined and previously unsolved
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers performance comparisons, ablation studies, and multi-scenario applications, though quantitative comparisons with more baselines are limited
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, complete mathematical derivations, and rich, intuitive figures
- Value: ⭐⭐⭐⭐ — Opens a new paradigm for bitmap gradient optimization with strong practical utility; actual impact depends on community adoption

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Differentiable Model of Supply-Chain Shocks](../../NeurIPS2025/others/a_differentiable_model_of_supply-chain_shocks.md)
- [\[NeurIPS 2025\] Exact Learning of Arithmetic with Differentiable Agents](../../NeurIPS2025/others/exact_learning_of_arithmetic_with_differentiable_agents.md)
- [\[AAAI 2026\] Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning](../../AAAI2026/others/expandable_and_differentiable_dual_memories_with_orthogonal_regularization_for_e.md)
- [\[NeurIPS 2025\] Scalable GPU-Accelerated Euler Characteristic Curves: Optimization and Differentiable Learning for PyTorch](../../NeurIPS2025/others/scalable_gpu-accelerated_euler_characteristic_curves_optimization_and_differenti.md)
- [\[CVPR 2026\] Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)

</div>

<!-- RELATED:END -->
