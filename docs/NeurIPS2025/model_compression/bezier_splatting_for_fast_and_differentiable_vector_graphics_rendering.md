---
title: >-
  [Paper Note] Bézier Splatting for Fast and Differentiable Vector Graphics Rendering
description: >-
  [NeurIPS 2025][Model Compression][Differentiable Rendering] Bézier Splatting integrates the Gaussian Splatting framework with Bézier curves by uniformly sampling 2D Gaussian points along each curve and rendering via α-blending to achieve differentiable vector graphics. The method achieves 30× forward and 150× backward speedups over DiffVG while maintaining or surpassing the image quality of methods such as LIVE.
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Differentiable Rendering"
  - "Bézier Curves"
  - "Gaussian Splatting"
  - "Vectorization"
  - "Image Optimization"
date: 2026-05-08
content_hash: a5d98b6ede800d7e
---

# Bézier Splatting for Fast and Differentiable Vector Graphics Rendering

**Conference**: NeurIPS 2025
**arXiv**: [2503.16424](https://arxiv.org/abs/2503.16424)  
**Code**: [https://xiliu8006.github.io/Bezier_splatting_project](https://xiliu8006.github.io/Bezier_splatting_project)  
**Area**: Vector Graphics Rendering
**Keywords**: Differentiable Rendering, Bézier Curves, Gaussian Splatting, Vectorization, Image Optimization

## TL;DR
Bézier Splatting integrates the Gaussian Splatting framework with Bézier curves by uniformly sampling 2D Gaussian points along each curve and rendering via α-blending to achieve differentiable vector graphics. The method achieves 30× forward and 150× backward speedups over DiffVG while maintaining or surpassing the image quality of methods such as LIVE.

## Background & Motivation

**Background**: Differentiable vector graphics renderers (e.g., DiffVG, LIVE) optimize Bézier curve parameters for image vectorization. DiffVG performs per-pixel color accumulation, while LIVE adds curves layer by layer; both require hours of optimization for high-resolution images.

**Limitations of Prior Work**: DiffVG's backward pass requires sampling at every boundary pixel, making computation proportional to the number of curves and image resolution—extremely slow. LIVE's layer-wise optimization takes 2.6–5.1 hours for 512 curves. No existing method achieves high-quality vectorization within minutes.

**Key Challenge**: Traditional differentiable rendering must precisely handle anti-aliased gradients at curve boundaries, which creates a computational bottleneck. A rendering approach that bypasses per-pixel boundary sampling is needed.

**Goal**: Substantially accelerate differentiable vector graphics rendering while preserving output quality, reducing optimization time from hours to minutes.

**Key Insight**: Gaussian Splatting has demonstrated efficient differentiable rendering in 3D scenes. By parameterizing Bézier curves as a set of 2D Gaussian points distributed along each curve, the efficient Gaussian Splatting rendering pipeline can be directly reused.

**Core Idea**: Uniformly sample 2D Gaussian points along each Bézier curve, derive rotation from the local tangent direction and scale from inter-point distances, and achieve fast differentiable vectorization via α-blending rendering combined with adaptive pruning and densification.

## Method

### Overall Architecture
Input image → Initialize a set of Bézier curves → Uniformly sample points along each curve → Compute 2D Gaussian parameters per sample (position = sampled point, rotation = tangent angle, scale = distance to neighboring points) → α-blending rendering → Optimize curve control points and colors with L2 + Xing loss → Adaptively prune low-opacity curves and densify new curves in high-error regions.

### Key Designs

1. **Parameterization of Bézier Curves as 2D Gaussians**:

    - **Function**: Convert each Bézier curve into a sequence of 2D Gaussian points.
    - **Mechanism**: Position = uniformly sampled points on the curve; rotation $\theta$ = local tangent angle; $\sigma_x$ (along the curve) = distance to neighboring sample points; $\sigma_y$ (perpendicular to the curve) = learnable width parameter. For closed curves, a "paired Bézier curve" structure linearly interpolates control points between two boundary curves to fill the interior region.
    - **Design Motivation**: 2D Gaussians are the native representation in Gaussian Splatting, enabling direct reuse of its efficient rendering pipeline. Automatically deriving parameters from tangent direction and inter-point distances avoids additional learning overhead.

2. **α-blending Rendering (Non-accumulative)**:

    - **Function**: Correctly handle occlusion relationships between curves.
    - **Mechanism**: Curves are sorted by depth and rendered front-to-back as $C = \sum_i T_i \alpha_i c_i$, where $T_i = \prod_{j<i}(1-\alpha_j)$. Unlike DiffVG's per-pixel color accumulation, α-blending naturally supports occlusion.
    - **Design Motivation**: DiffVG's accumulative rendering leads to color aliasing, whereas α-blending better reflects the physical occlusion model.

3. **Adaptive Pruning and Densification**:

    - **Function**: Dynamically adjust the number and distribution of curves.
    - **Mechanism**: Pruning—remove curves whose opacity falls below a threshold or whose area is too small; Densification—randomly initialize new curves in regions with high residual error; executed periodically.
    - **Design Motivation**: Initially random curves may be unevenly distributed; the adaptive strategy concentrates curves in regions requiring fine detail.

### Loss & Training
- $\mathcal{L} = \lambda_1 \|\hat{I} - I\|_2^2 + \lambda_2 \mathcal{L}_{Xing}$
- The Xing loss constrains Bézier curves to be self-intersection-free (preserving convexity).
- Optimization completes in 3.2–8.6 minutes (vs. 2.6–5.1 hours for LIVE).

## Key Experimental Results

### Main Results

| Method | SSIM↑ | PSNR↑ | LPIPS↓ | Forward Speedup | Backward Speedup | Optimization Time |
|--------|-------|-------|--------|-----------------|------------------|-------------------|
| DiffVG | — | — | — | 1× | 1× | Hours |
| LIVE | 0.601 | 21.70 | 0.521 | — | — | 2.6–5.1 h |
| **Bézier Splatting** | **0.607** | **22.11** | **0.528** | **30×** | **150×** | **3.2–8.6 min** |

### Ablation Study

| Configuration | SSIM | PSNR |
|---------------|------|------|
| w/o pruning/densification | 0.590 | 21.10 |
| w/ pruning/densification | **0.607** | **22.11** |
| Layer-wise strategy | 0.613 | — (2× training time) |

### Key Findings
- The backward pass speedup is most pronounced (150×), as it entirely bypasses DiffVG's per-boundary sampling.
- The paired-curve structure for closed curves enables gradient optimization over filled regions.
- The method generalizes across diverse image types including photographs, artistic paintings, and cartoons.
- Adaptive densification contributes substantially to quality (SSIM +0.017, PSNR +1.01).

## Highlights & Insights
- **Creative reuse of the Gaussian Splatting framework**: Mapping the vector graphics problem onto the Gaussian Splatting pipeline cleverly leverages existing CUDA-accelerated implementations.
- **Natural parameterization design**: Encoding tangent direction as rotation and inter-point distance as scale transfers curve geometry to Gaussian parameters with minimal information loss.
- **150× backward speedup opens the door to interactive vectorization**: Reducing optimization from hours to minutes enables real-time application scenarios.

## Limitations & Future Work
- LPIPS scores are comparable to or slightly worse than LIVE, leaving room for perceptual quality improvement.
- The paired-curve structure for closed curves limits shape expressiveness to strip-like regions.
- No comparison is made against recent neural vectorization methods such as VectorFusion.
- The Xing loss enforcing curve convexity may restrict the representation of complex shapes.

## Related Work & Insights
- **vs. DiffVG**: DiffVG samples boundary gradients per pixel, which is prohibitively slow; the proposed method substitutes Gaussians to bypass boundary computation entirely.
- **vs. LIVE**: LIVE's layer-wise curve addition yields high quality but is slow; this work's global optimization with adaptive strategies achieves comparable quality at over 100× higher speed.
- **vs. 3DGS**: 3DGS targets 3D scene reconstruction; this work combines 2D Gaussians with Bézier curves for vector graphics rendering.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of Gaussian Splatting and Bézier curves is a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset evaluation with ablations and speed comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear and well-structured.
- **Value**: ⭐⭐⭐⭐ Substantially advances the efficiency of differentiable vector graphics rendering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] All You Need is One: Capsule Prompt Tuning with a Single Vector](all_you_need_is_one_capsule_prompt_tuning_with_a_single_vector.md)
- [\[NeurIPS 2025\] Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression](learning_grouped_lattice_vector_quantizers_for_low-bit_llm_compression.md)
- [\[ICLR 2026\] Light Differentiable Logic Gate Networks](../../ICLR2026/model_compression/light_differentiable_logic_gate_networks.md)
- [\[NeurIPS 2025\] Uni-LoRA: One Vector is All You Need](uni-lora_one_vector_is_all_you_need.md)
- [\[NeurIPS 2025\] Deterministic Continuous Replacement: Fast and Stable Module Replacement in Pretrained Transformers](deterministic_continuous_replacement_fast_and_stable_module_replacement_in_pretr.md)

</div>

<!-- RELATED:END -->
