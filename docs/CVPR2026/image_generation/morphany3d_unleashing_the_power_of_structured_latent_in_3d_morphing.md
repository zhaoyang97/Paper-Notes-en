---
title: >-
  [Paper Note] MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing
description: >-
  [CVPR 2026][Image Generation][3D Morphing] MorphAny3D is proposed as the first training-free 3D morphing framework based on Structured Latent (SLAT) representations. It achieves state-of-the-art quality in cross-category 3D morphing through Morphing Cross-Attention (MCA) for structurally coherent source/target fusion, Temporal-Fused Self-Attention (TFSA) for temporal consistency, and a direction correction strategy to eliminate abrupt orientation jumps.
tags:
  - CVPR 2026
  - Image Generation
  - 3D Morphing
  - SLAT
  - Attention Mechanism
  - Training-Free
  - Trellis
date: 2026-05-08
content_hash: 114a629cdf08c8ca
---

# MorphAny3D: Unleashing the Power of Structured Latent in 3D Morphing

**Conference**: CVPR 2026
**arXiv**: [2601.00204](https://arxiv.org/abs/2601.00204)
**Code**: [https://xiaokunsun.github.io/MorphAny3D.github.io/](https://xiaokunsun.github.io/MorphAny3D.github.io/)
**Area**: Image Generation / 3D Vision
**Keywords**: 3D Morphing, SLAT, Attention Mechanism, Training-Free, Trellis

## TL;DR
MorphAny3D is proposed as the first training-free 3D morphing framework based on Structured Latent (SLAT) representations. It achieves state-of-the-art quality in cross-category 3D morphing through Morphing Cross-Attention (MCA) for structurally coherent source/target fusion, Temporal-Fused Self-Attention (TFSA) for temporal consistency, and a direction correction strategy to eliminate abrupt orientation jumps.

## Background & Motivation

**State of the Field**: 3D morphing is a foundational technique in animation, film, and gaming. Traditional methods rely on dense correspondence matching followed by interpolation to generate intermediate shapes. 2D morphing has advanced significantly with diffusion models, but 3D morphing remains a substantially harder problem.

**Limitations of Prior Work**: (a) Matching-based methods handle only geometric deformation while ignoring texture, and cross-category correspondences are unreliable; (b) 2D morphing followed by per-frame 3D reconstruction breaks temporal consistency; (c) Direct interpolation in noise or conditioning spaces lacks structural plausibility constraints.

**Root Cause**: Achieving smooth, high-fidelity, and temporally consistent cross-category morphing within a 3D generative framework remains an open challenge.

**Paper Goals**: How can the structural advantages of SLAT representations be exploited to enable high-quality 3D morphing?

**Starting Point**: A key observation is that directly fusing source and target SLAT features within Trellis's attention mechanism produces more plausible morphing than interpolating at the noise or conditioning level. However, naively applying KV fusion in both cross-attention and self-attention leads to mutual interference.

**Core Idea**: Compute source and target attention outputs separately in cross-attention and then fuse them with a weighted sum (MCA); fuse features from the previous frame in self-attention (TFSA); apply statistics-based direction correction.

## Method

### Overall Architecture
Given a source object $x^{src}$ and a target object $x^{tgt}$, the framework generates a morphing sequence of $N=50$ frames $\{x^n\}_{n=0}^{N}$, with $\alpha^n = n/N$ controlling the interpolation linearly. The Trellis Image-to-3D pipeline is used without any retraining.

### Key Designs

1. **Analysis of SLAT Fusion Modes**:

    - Function: Understand the effects of different attention fusion strategies.
    - Key Findings: (a) KV-Fused CA significantly improves structural plausibility (lowest FID) but introduces local distortions; (b) KV-Fused SA improves smoothness (lowest PPL); (c) combining both simultaneously degrades plausibility.
    - Design Motivation: Naive combination leads to conflicts, necessitating separate fusion strategies for CA and SA.

2. **Morphing Cross-Attention (MCA)**:

    - Function: Fuse source/target conditioning information to ensure structural consistency.
    - Mechanism: Rather than mixing K/V before computing attention, attention outputs are computed **separately** and then fused with a weighted sum: $\text{MCA}(Q^n, K^{src/tgt}, V^{src/tgt}) = (1-\alpha^n)\text{Attn}(Q^n, K^{src}, V^{src}) + \alpha^n\text{Attn}(Q^n, K^{tgt}, V^{tgt})$
    - Design Motivation: In KV-Fused CA, mixing DINOv2 features patch-by-patch conflates semantics with different spatial correspondences — e.g., head SLAT features incorrectly attend to background regions, causing distortions. MCA preserves the semantic correctness of each attention map.
    - t-SNE Validation: Feature trajectories under MCA are stable and smooth, whereas KV-Fused CA trajectories are disordered and discontinuous.

3. **Temporal-Fused Self-Attention (TFSA)**:

    - Function: Enhance inter-frame temporal consistency.
    - Mechanism: When generating frame $n$, the K/V from the previous frame are incorporated into self-attention: $\text{TFSA} = (1-\beta)\text{Attn}(Q^n, K^n, V^n) + \beta\text{Attn}(Q^n, K^{n-1}, V^{n-1})$, with $\beta=0.2$.
    - Design Motivation: Unlike KV-Fused SA, which mixes endpoint features of source and target (potentially harming plausibility), TFSA fuses features from already-plausible neighboring frames, yielding higher fidelity.
    - Distinction from KV-Fused SA: The latter mixes source/target endpoints; TFSA leverages validated intermediate results from adjacent frames.

4. **Direction Correction Strategy**:

    - Function: Eliminate abrupt orientation jumps during morphing.
    - Mechanism: Analysis of 200 sequences reveals that (a) jumps concentrate around $\alpha\approx 0.5$; (b) jumps are almost exclusively yaw rotations of 90°/180°/270°; (c) orientations of objects generated by Trellis cluster at the same discrete angles. Four yaw-rotated candidates $\{P^n, P_{90°}^n, P_{180°}^n, P_{270°}^n\}$ are created from the sparse structure (SS) stage output $P^n$, and the candidate with minimum Chamfer Distance to $P^{n-1}$ is selected.
    - Design Motivation: Orientation jumps originate from the discrete pose prior learned by Trellis, not from randomness. The correction strategy is non-invasive — when no jump occurs, the unrotated version is naturally selected.

## Key Experimental Results

### Main Results

| Method | FID↓ | PPL↓ | PDV↓ | AS(%)↑ | UP(%)↑ |
|------|------|------|------|--------|--------|
| 3DInterp | 409.1 | 2.55 | 0.0006 | 1.0 | 0.6 |
| DiffMorpher→3D | 208.1 | 6.65 | 0.0021 | 5.0 | 0.8 |
| DirectInterp | 150.9 | 3.72 | 0.0039 | 2.0 | 5.5 |
| MorphFlow | 285.0 | 2.41 | 0.0009 | 0.0 | 1.6 |
| **MorphAny3D** | **112.0** | 2.47 | **0.0006** | **81.0** | **86.7** |

### Ablation Study

| Method | FID↓ | PPL↓ | PDV↓ |
|------|------|------|------|
| KV-Fused CA | 125.5 | 3.82 | 0.0013 |
| MCA | 112.2 | 3.66 | 0.0010 |
| MCA + TFSA | 113.2 | 2.87 | 0.0007 |
| MCA + TFSA + OC | **112.0** | **2.47** | **0.0006** |

### Key Findings
- MorphAny3D achieves 86.7% user preference, substantially outperforming all baselines.
- MCA is the critical component for plausibility (FID reduced from 125.5 to 112.2).
- TFSA is the critical component for smoothness (PPL reduced from 3.66 to 2.87).
- Direction correction further reduces PPL to 2.47, approaching the lower bound of matching-based methods (2.41).
- The framework transfers directly to Hi3DGen and Text-to-3D Trellis, demonstrating generalizability.

## Highlights & Insights
- **Post-attention output fusion vs. KV fusion** is the central insight: the former preserves the semantic correctness of individual attention maps. This design pattern is transferable to any attention-based generative model requiring multi-source conditional fusion.
- **Statistics-driven direction correction**: the correction strategy is derived from empirical data statistics, is simple and effective, and introduces no side effects.
- **Decoupled morphing**: by selectively applying MCA to the SS and SLAT stages, global structure and local detail morphing can be decoupled, enabling dual-target morphing and style transfer.

## Limitations & Future Work
- Inherits Trellis's limitations in generating fine-grained structures.
- Rotation correction may fail for yaw-symmetric objects.
- Runtime is high: approximately 30 seconds per frame with 24 GB VRAM.

## Related Work & Insights
- **vs. 3DMorpher**: Built on 3DGS, it cannot handle complex geometry and is incompatible with commercial 3D software; MorphAny3D is more general by virtue of its SLAT foundation.
- **vs. DiffMorpher/FreeMorph**: These methods perform 2D morphing followed by per-frame 3D lifting, resulting in temporal inconsistency; MorphAny3D operates directly within the 3D generative framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First training-free SLAT-based 3D morphing framework with thorough attention fusion analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage including quantitative evaluation, user study, ablation, applications, and transfer experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ The reasoning chain from observation to validation to design is complete and well-structured.
- Value: ⭐⭐⭐⭐ Direct applicability in 3D content creation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)
- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](../../ICCV2025/image_generation/freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [\[CVPR 2026\] SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](slice_semantic_latent_injection_via_compartmentali.md)
- [\[ICLR 2026\] DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing](../../ICLR2026/image_generation/dragflow_unleashing_dit_priors_with_region_based_supervision_for_drag_editing.md)
- [\[CVPR 2026\] PhysGen: Physically Grounded 3D Shape Generation for Industrial Design](physgen_physically_grounded_3d_shape_generation_for_industrial_design.md)

<!-- RELATED:END -->
