---
title: >-
  [Paper Note] Pyramidal Patchification Flow for Visual Generation
description: >-
  [ICLR 2026][Image Generation][Pyramidal patchification] This paper proposes Pyramidal Patchification Flow (PPFlow), which employs larger patches at high-noise timesteps and smaller patches at low-noise timesteps…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Pyramidal patchification"
  - "flow matching"
  - "DiT"
  - "inference acceleration"
  - "variable token count"
date: 2026-05-08
content_hash: ea524f72eb091bbe
---

# Pyramidal Patchification Flow for Visual Generation

**Conference**: ICLR 2026
**arXiv**: [2506.23543](https://arxiv.org/abs/2506.23543)
**Code**: [GitHub](https://github.com/fudan-generative-vision/PPFlow)
**Area**: Diffusion Model Acceleration / Image Generation
**Keywords**: Pyramidal patchification, flow matching, DiT, inference acceleration, variable token count

## TL;DR

This paper proposes Pyramidal Patchification Flow (PPFlow), which employs larger patches at high-noise timesteps and smaller patches at low-noise timesteps, achieving 1.6–2.0× denoising speedup while preserving generation quality, without requiring any re-noising tricks.

## Background & Motivation

- DiT applies a fixed patch size (typically $2\times2$) across all timesteps, resulting in wasteful computation at high-noise timesteps.
- Limitations of existing acceleration methods:
    - **Step reduction** (DDIM, distillation): sacrifices generation quality.
    - **Per-step cost reduction** (quantization, pruning): limited potential gains.
    - **Pyramidal/cascaded generation** (Pyramidal Flow): introduces "jump points" requiring complex re-noising tricks.
- **Core observation**: Spatial details are unimportant at high noise levels and can be represented with fewer tokens.

## Method

### Pyramidal Patchification Scheme

The timestep range is divided into multiple stages, each using a different patch size.

**Three-stage example**:
- $[0, t_{s_1})$: patch size $4\times4$ (high noise, $L = (I/4)^2$ tokens)
- $[t_{s_1}, t_{s_2})$: patch size $4\times2$
- $[t_{s_2}, 1]$: patch size $2\times2$ (low noise, standard DiT)

### Parameter Sharing Strategy

- **Patchify/Unpatchify**: each stage has an independent linear projection matrix $\mathbf{W}_{s_i} \in \mathbb{R}^{d \times d_{s_i}}$.
- **DiT blocks**: parameters are shared across all stages.
- **Key point**: The cost of Patchify is independent of patch size ($L_s \times d_s \times d = I^2 C d$), while the cost of DiT blocks scales with token count.

### Computational Complexity

Per DiT block complexity: $\mathcal{O}(L_s^2 d + L_s d)$

In DiT-XL/2, approximately 99.8% of FLOPs reside in DiT blocks — reducing token count directly reduces computation.
- Two-stage PPFlow: 37.8% FLOPs reduction
- Three-stage PPFlow: 50.6% FLOPs reduction

### Initialization from Pretrained DiT

**Patchify initialization** (averaging): extends the $2\times2$ patch projection to $4\times4$:

$$\mathbf{W}_2 = \frac{1}{4}[\mathbf{W}, \mathbf{W}, \mathbf{W}, \mathbf{W}]$$

**Unpatchify initialization** (replication):

$$\mathbf{W}_2^u = [(\mathbf{W}^u)^\top, (\mathbf{W}^u)^\top, (\mathbf{W}^u)^\top, (\mathbf{W}^u)^\top]^\top$$

### Key Differences from Pyramidal Flow

| Property | PPFlow | Pyramidal Flow |
|----------|--------|----------------|
| Operating resolution | Full-resolution latent space | Pyramidal (multi-resolution) |
| Continuity equation | Satisfied | Not satisfied |
| Jump points | None | Present (requires re-noising tricks) |
| Inference pipeline | Identical to standard DiT | Requires special handling |

## Key Experimental Results

### Training from Scratch (ImageNet 256×256, SiT-B)

| Method | Training Steps | FID-50K(↓) | IS(↑) | Test FLOPs(%) | Speedup |
|--------|---------------|-----------|-------|--------------|---------|
| SiT-B/2 | 7M | 4.46 | - | 100% | 1.00× |
| PPF-B-2 | 7M | **3.83** | - | ~62% | **1.61×** |
| PPF-B-3 | 7M | 4.43 | - | ~49% | **2.04×** |

### Fine-tuning from Pretrained SiT-XL/2

| Method | Additional Training FLOPs | FID-50K(↓) | Test Speedup |
|--------|--------------------------|-----------|-------------|
| SiT-XL/2 | Baseline | ~2.06 | 1.00× |
| PPF-XL-2 | +8.9% | Comparable | **1.60×** |
| PPF-XL-3 | +7.1% | Comparable | **2.02×** |

### Text-to-Image (Based on FLUX.1-dev)

| Resolution | Speedup | GenEval | DPG-bench |
|-----------|---------|---------|-----------|
| 512×512 | 1.61× | Comparable | Comparable |
| 1024×1024 | 1.76× | Comparable | Comparable |
| 2048×2048 | 1.86× | Comparable | Comparable |

### Key Findings

1. Two-stage and three-stage PPFlow achieve 1.6× and 2.0× inference speedup, respectively, with quality preserved.
2. Fine-tuning from a pretrained model requires only ~8% additional training cost.
3. PPF-B-2 trained from scratch even surpasses the SiT-B/2 baseline (FID: 3.83 vs. 4.46).
4. Speedup is more pronounced at higher resolutions (1.86× at 2048), as the large-patch stage reduces tokens more aggressively.
5. Stage-aware CFG scheduling (e.g., [1.5, 3.5, 4.0]) further improves generation quality.

## Highlights & Insights

1. **Minimalist design**: only the linear projections in Patchify/Unpatchify are modified; DiT blocks are fully shared across stages.
2. **No re-noising tricks**: the method always operates in the full-resolution latent space, eliminating the complexity inherent in Pyramidal Flow.
3. **Training–inference consistency**: each patch size is trained only at its corresponding timesteps (unlike FlexiDiT/Lumina-Video, which train all patch sizes across all timesteps).
4. **Patch n' Pack**: variable-length token packing is exploited to reduce training FLOPs.

## Limitations & Future Work

- More aggressive pyramidal schemes with more than three stages remain underexplored.
- The selection of patch sizes and timestep split points is largely heuristic.
- At small resolutions (e.g., 256×256), large patches may discard excessive spatial information.
- Stage-aware CFG scheduling expands the hyperparameter search space.

## Related Work & Insights

- **Inference acceleration**: DDIM, Progressive Distillation, Consistency Models
- **Pyramidal/cascaded generation**: Pyramidal Flow, PixelFlow, Cascaded Diffusion
- **Variable patch size**: FlexiViT, FlexiDiT, Lumina-Video
- **DiT architectures**: DiT, SiT, FLUX

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea is straightforward, yet the full-resolution operation that distinguishes PPFlow from Pyramidal Flow constitutes a key innovation.
- **Technical Depth**: ⭐⭐⭐ — The method is intuitive and the theoretical analysis is relatively simple.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Validated via both from-scratch training and pretrained fine-tuning, covering class-conditional and text-to-image settings.
- **Practical Value**: ⭐⭐⭐⭐⭐ — Plug-and-play; significant speedup achievable at low fine-tuning cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Next Visual Granularity Generation](next_visual_granularity_generation.md)
- [\[ICLR 2026\] SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation](ssg_scaled_spatial_guidance_for_multi-scale_visual_autoregressive_generation.md)
- [\[ICLR 2026\] Purrception: Variational Flow Matching for Vector-Quantized Image Generation](purrception_variational_flow_matching_for_vector-quantized_image_generation.md)
- [\[ICLR 2026\] Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation](flow2gan_hybrid_flow_matching_and_gan_with_multi-resolution_network_for_few-step.md)
- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)

</div>

<!-- RELATED:END -->
