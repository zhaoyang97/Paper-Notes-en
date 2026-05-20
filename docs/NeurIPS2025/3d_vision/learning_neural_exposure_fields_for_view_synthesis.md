---
title: >-
  [Paper Note] Learning Neural Exposure Fields for View Synthesis
description: >-
  [NeurIPS 2025][3D Vision][Neural Radiance Fields] This paper proposes Neural Exposure Fields (NExF), which achieves 3D-consistent high-quality view synthesis by learning optimal exposure values per 3D point rather than p…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Neural Radiance Fields"
  - "View Synthesis"
  - "Exposure Compensation"
  - "3D Consistency"
  - "High Dynamic Range"
date: 2026-05-08
content_hash: 4bd2a2b339ebfafd
---

# Learning Neural Exposure Fields for View Synthesis

**Conference**: NeurIPS 2025
**arXiv**: [2510.08279](https://arxiv.org/abs/2510.08279)  
**Code**: [https://m-niemeyer.github.io/nexf](https://m-niemeyer.github.io/nexf)  
**Area**: 3D Vision / View Synthesis
**Keywords**: Neural Radiance Fields, View Synthesis, Exposure Compensation, 3D Consistency, High Dynamic Range

## TL;DR

This paper proposes Neural Exposure Fields (NExF), which achieves 3D-consistent high-quality view synthesis by learning optimal exposure values per 3D point rather than per image. On HDR scenes, NExF surpasses the state-of-the-art by 3.5+ dB in PSNR while being 50× faster.

## Background & Motivation

**Background**: Standard NeRF benchmarks exclude exposure variation, yet real-world scenes (e.g., mixed indoor/outdoor environments, windowed rooms) frequently exhibit strong exposure changes that severely degrade reconstruction quality.

**Limitations of Prior Work**: HDRNeRF requires professional HDR post-processing software and performs tone mapping only at the 2D image level, resulting in color inconsistencies for the same 3D point across different views. GLO embeddings are robust to small exposure changes but fail under large variations.

**Key Challenge**: Conventional cameras select a single exposure per image (a 2D operation), whereas the ideal approach is to learn an optimal exposure per 3D point (a 3D operation).

**Core Idea**: Elevating exposure modeling from the 2D image level to the 3D point level, thereby ensuring that the same 3D point yields consistent color across all views.

## Method

### Overall Architecture

Two components are added on top of a standard NeRF architecture: (1) latent exposure conditioning — log exposure is injected at the NeRF bottleneck layer rather than the input layer; and (2) a neural exposure field — an additional MLP that learns optimal exposure values in 3D space.

### Key Designs

1. **Latent Exposure Conditioning (Section 3.1)**

    - **Function**: Conditions the NeRF on log exposure at the bottleneck layer rather than at the input layer.
    - **Mechanism**: $f_\theta(\mathbf{x}, \mathbf{d}, \Delta t) = f_\theta^{view}(f_\theta^{pos}(\mathbf{x}) + \ln \Delta t(\mathbf{r}), \mathbf{d})$. Since the positional encoder $f_\theta^{pos}$ already predicts log radiance, injecting exposure at the intermediate layer is more stable.
    - **Design Motivation**: Mid-layer conditioning improves performance by 5%+ over direct input conditioning, as the positional encoding already encodes radiance information.

2. **Neural Exposure Field (Section 3.2)**

    - **Function**: Learns a neural field for 3D exposure values $e_\phi: \mathbb{R}^3 \to \mathbb{R}$.
    - **Mechanism**: A fully-connected MLP (4 layers, hidden dimension 128) that updates exposure only when the rendered color is "well-exposed and saturated" — well-exposure weight $w_{exp}(\mathbf{c}) = \prod_i \exp(-(c_i - 1/2)^2/\sigma_{exp})$; saturation weight $w_{sat}(\mathbf{c}) = \sqrt{\frac{1}{3}\sum_i (c_i - \bar{\mu}_c)^2}$.
    - **Design Motivation**: 3D consistency is guaranteed by construction — the exposure value of a given 3D point is view-independent. A 3D smoothness regularizer $\|\Delta t_{diff}\|_2^2$ encourages neighboring points to have similar exposure values.

3. **Joint Optimization (Section 3.3)**

    - **Function**: End-to-end joint training of NeRF parameters $\theta$ and exposure field parameters $\phi$.
    - **Mechanism**: Selective per-pixel backpropagation — exposure is updated only when the color is well-exposed and saturated, ignoring under- and over-exposed pixels.
    - **Total Loss**: $\mathcal{L}(\theta, \phi) = \mathcal{L}_f(\theta) + \mathcal{L}_e(\phi)$

### Loss & Training

- **NeRF Reconstruction Loss**: Standard MSE.
- **Exposure Field Loss**: Weighted L2 + 3D smoothness regularization.
- **Weighting**: Well-exposure × saturation, to prevent learning erroneous exposure values from over- or under-exposed regions.

## Key Experimental Results

### Main Results (HDRNeRF Dataset)

| Method | Inference Time | ID-PSNR↑ | OOD-PSNR↑ | ID-LPIPS↓ |
|--------|---------------|-----------|------------|-----------|
| NeRF | 405min | 13.97 | 14.51 | 0.376 |
| ZipNeRF | 11min | 19.00 | 19.73 | 0.142 |
| NeRF-W | 437min | 29.83 | 29.22 | 0.047 |
| HDRNeRF | 542min | 39.07 | 37.53 | 0.026 |
| HDR-GS* | 34min | 41.10 | 36.33 | 0.011 |
| **NExF** | **11min** | **42.54** | **38.36** | **0.014** |

### Ablation Study

| Configuration | PSNR(ID) | SSIM(ID) | LPIPS(ID) | Notes |
|---------------|----------|----------|-----------|-------|
| w/o View MLP | 33.85 | 0.928 | 0.104 | Base NeRF |
| w/o Latent Conditioning | 39.88 | 0.979 | 0.038 | Direct conditioning |
| **Full NExF** | **42.54** | **0.988** | **0.014** | **Best** |

### Key Findings

- Compared to HDRNeRF: PSNR +3.5 (ID) / +2.2 (OOD), with 50× speedup (11 vs. 542 minutes).
- Latent-layer conditioning contributes +2.66 PSNR; view-aware MLP contributes +8.69 PSNR.
- The 3D consistency problem is fully resolved in qualitative experiments — the same scene region exhibits consistent color across viewpoints.
- On the Eyeful Tower real-world dataset, NExF generalizes significantly better than baselines under extreme exposures (too dark / too bright).

## Highlights & Insights

- **Theoretical Elegance**: Lifting exposure modeling from 2D images to 3D points is a natural and elegant formulation. This idea is transferable to other image processing tasks such as white balance and color correction.
- **3D Consistency**: Resolved by design rather than post-hoc alignment, fundamentally eliminating tone-mapping inconsistencies.
- **Speed and Quality**: A 50× speedup with simultaneous quality improvement, enabled by the lightweight exposure field MLP.
- **Implementation Simplicity**: Only one additional 4-layer MLP is required, making integration into any NeRF variant straightforward.

## Limitations & Future Work

- The method assumes that most 3D points are well-exposed in at least one view; it may fail in scenes where virtually all views are severely under-exposed.
- Only static 3D scenes are handled; dynamic illumination changes are not addressed.
- Integration with 3D Gaussian Splatting has not been explored, and a Gaussian-based variant may raise different considerations.
- The method assumes accurate exposure metadata (ISO/shutter values) from the camera, which may be erroneous in practice.

## Related Work & Insights

- **vs. HDRNeRF**: HDRNeRF performs tone mapping at the 2D level, whereas NExF learns exposure at the 3D level, fundamentally avoiding cross-view inconsistency.
- **vs. NeRF-W**: NeRF-W uses GLO embeddings to handle appearance variation but is ineffective under large exposure changes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The introduction of 3D exposure fields is an elegant and innovative contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dataset evaluation with comprehensive ablations and both quantitative and qualitative results.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear, fluent, and formally rigorous.
- Value: ⭐⭐⭐⭐⭐ Highly practical for real-world scenes with mixed illumination; strong deployment potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis](hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)
- [\[ICCV 2025\] SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing](../../ICCV2025/3d_vision/sehdr_single-exposure_hdr_novel_view_synthesis_via_3d_gaussian_bracketing.md)
- [\[NeurIPS 2025\] Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework](copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)
- [\[NeurIPS 2025\] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)
- [\[ICLR 2026\] Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields](../../ICLR2026/3d_vision/learning_physics-grounded_4d_dynamics_with_neural_gaussian_force_fields.md)

</div>

<!-- RELATED:END -->
