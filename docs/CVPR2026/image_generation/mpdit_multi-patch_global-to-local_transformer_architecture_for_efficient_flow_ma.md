---
title: >-
  [Paper Note] MPDiT: Multi-Patch Global-to-Local Transformer Architecture for Efficient Flow Matching
description: >-
  [CVPR 2026][Image Generation][Diffusion Transformer] This paper proposes MPDiT, a multi-scale patch global-to-local diffusion Transformer architecture. Early layers process global context using large patches (4×4) with o…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Transformer"
  - "flow matching"
  - "multi-scale patch"
  - "efficient architecture"
date: 2026-05-08
content_hash: 2b34b43d11f414c3
---

# MPDiT: Multi-Patch Global-to-Local Transformer Architecture for Efficient Flow Matching

**Conference**: CVPR 2026
**arXiv**: [2603.26357](https://arxiv.org/abs/2603.26357)  
**Code**: [https://github.com/quandao10/MPDiT](https://github.com/quandao10/MPDiT)  
**Area**: Diffusion Models
**Keywords**: Diffusion Transformer, flow matching, multi-scale patch, efficient architecture, image generation

## TL;DR

This paper proposes MPDiT, a multi-scale patch global-to-local diffusion Transformer architecture. Early layers process global context using large patches (4×4) with only 64 tokens, and later layers upsample to small patches (2×2) with 256 tokens for local detail refinement. This reduces GFLOPs by up to 50%, while the XL model achieves FID 2.05 (with CFG) at only 240 training epochs.

## Background & Motivation

1. **Background**: Diffusion models and flow matching models have become the dominant paradigm for visual generation. Transformer architectures (DiT/SiT) are increasingly replacing UNet as the backbone due to their superior scalability. However, DiT's isotropic design processes the same number of patch tokens at every layer, resulting in high computational cost.

2. **Limitations of Prior Work**: Training efficiency remains a critical bottleneck. Linear attention methods (e.g., SANA, LIT) reduce computation but with significant performance degradation. Mamba/SSM approaches show limited advantage at the token counts typical of diffusion models (<1K). MaskDiT suffers severe performance collapse at high masking ratios (75% mask rate → FID ≈ 100).

3. **Key Challenge**: The failure of MaskDiT and the relative success of DiT-XL/4 provide a key observation: a small number of large-patch tokens, while lacking local detail, can effectively capture global structural information. In contrast, MaskDiT's random masking causes each training sample to learn relationships among only a subset of tokens, resulting in poor modeling of both global and local information.

4. **Goal**: How to substantially reduce the computational cost and training overhead of diffusion Transformers while preserving generation quality?

5. **Key Insight**: Inspired by global-local attention, but rather than implementing it at the attention layer level (which yields negligible gains), the paper implements this principle at the full architecture level — earlier layers perceive coarse-grained global structure, while later layers perceive fine-grained local detail.

6. **Core Idea**: Transform the isotropic DiT into a coarse-to-fine hierarchical structure — the majority of Transformer blocks efficiently capture global semantics on large patches (64 tokens), while a small number of tail blocks refine local details on small patches (256 tokens).

## Method

### Overall Architecture

The input is a VAE-encoded latent $z \in \mathbb{R}^{32 \times 32 \times 4}$ (ImageNet 256×256). Standard DiT uses patch size=2 to obtain 256 tokens. MPDiT instead applies the following scheme: the first $N-k$ Transformer blocks process only 64 tokens using patch size=4 (25% of the standard count), followed by an upsampling module that expands to 256 tokens, and finally $k$ blocks perform local refinement. The output is converted back via reverse patchify and VAE decoding to produce the generated image.

### Key Designs

1. **Multi-Patch Design**:

    - **Function**: Efficiently model global information with large patches and refine local details with small patches.
    - **Mechanism**: Of the $N$ total Transformer blocks, the first $N-k$ blocks receive patch size=4 embeddings (64 tokens). Since self-attention complexity scales quadratically with token count, this is only $\frac{1}{16}$ the cost of standard DiT. The remaining $k$ blocks ($k=4\sim6$ is sufficient) receive the upsampled 256 tokens for refinement. Because most blocks process only 64 tokens, MPDiT-XL reduces GFLOPs from 118.66 to 59.30 (50% reduction). For higher resolutions (512²), a three-level patch hierarchy $\{8, 4, 2\}$ can be applied.
    - **Design Motivation**: MaskDiT achieves FID ≈ 100 at 75% masking, whereas DiT-XL/4 (processing a comparable number of tokens) achieves FID ≈ 40. This demonstrates that global modeling via large patches is far superior to partial modeling via random masking. The lack of local detail in large patches can be compensated by a small number of refinement blocks.

2. **Upsample Block**:

    - **Function**: Expand 64 coarse-grained tokens into 256 fine-grained tokens.
    - **Mechanism**: Image tokens and class tokens are first separated. The image tokens undergo linear projection followed by pixel-unshuffle for 4× spatial expansion (64→256 tokens), activated by GELU, and then recombined with the class tokens. A LayerNorm and linear layer then repair the class-image relationship. Crucially, a skip connection from the original patch size=2 embeddings is added directly to the upsampled output to preserve fine-grained spatial detail.
    - **Design Motivation**: The preceding blocks model class-image interactions over 64 tokens; after upsampling, the change in token count disrupts this relationship, necessitating an additional linear layer to reconstruct it. The skip connection ensures that fine-grained information is not lost.

3. **FNO Time Embedding + Multi-Token Class Embedding**:

    - **Function**: Provide richer conditioning signals for timestep and class.
    - **Mechanism**: **FNO Time Embedding** — the scalar timestep $t$ is added to a 32-point uniform 1D grid to form a 1D signal, which is lifted to 32 channels via a linear layer and processed by 3 MixedFNO blocks (mixing SpectralConv1D and Conv1D) to learn smooth temporal structure, followed by global average pooling and linear projection. Inspired by Neural Operators, this better captures the continuous dynamics of the flow field. **Multi-Token Class Embedding** — each class is represented by $m=16$ learnable tokens instead of 1, prepended to the image tokens as a prefix, replacing AdaIN modulation.
    - **Design Motivation**: Traditional sinusoidal + MLP time embeddings have limited expressive power; the FNO design yields approximately 4 FID points of improvement. A single class token is overly compressed; 16 tokens provide a more distributed semantic representation, accelerating convergence by approximately 7 FID points.

### Loss & Training

- Standard flow matching objective: $L_{FM} = \|f_\theta(z_t, t, c) - (n - z)\|_2^2$
- AdaIN parameters are shared across all Transformer blocks (reducing parameters from 130M to ~90M with only a 0.4 FID increase)
- Training setup: 8×A100-40GB, fixed learning rate $2 \times 10^{-4}$, batch size 1024, EMA 0.9999
- Sampling uses a 250-step Euler solver

## Key Experimental Results

### Main Results

| Model | Epochs | GFLOPs | FID↓ (non-cfg) | FID↓ (cfg) | IS↑ (cfg) |
|------|--------|--------|----------------|------------|-----------|
| DiT-XL/2 | 1400 | 118.66 | 9.62 | 2.27 | 278.24 |
| SiT-XL/2 | 1400 | 118.66 | 9.35 | 2.15 | 258.09 |
| DiG-XL/2 | 240 | 89.40 | 8.60 | 2.07 | 278.95 |
| DiCo-XL | 80 | 87.30 | 11.67 | - | - |
| **MPDiT-XL** | **240** | **59.30** | **7.36** | **2.05** | **278.73** |

### Ablation Study

| Component | Params(M) | GFLOPs | FID↓ |
|------|-----------|--------|------|
| DiT-B/2 baseline | 130.0 | 23.0 | 34.84 |
| + Shared AdaIN | 90.3 | 22.9 | 35.31 |
| + Multi-token Class (m=16) | 101.9 | 24.3 | 28.56 |
| + FNO Time Embedding | 101.2 | 24.3 | 24.52 |
| + MPDiT (k=6) | 104.8 | 16.6 | **24.74** |

| k value (XL) | GFLOPs | FID↓ |
|-----------|--------|------|
| k=4 | 53.2 | 11.11 |
| k=6 (default) | 59.3 | 9.92 |
| k=8 | 65.4 | 9.73 |

| Class Token Count m | FID↓ |
|-------------------|------|
| m=1 | 32.31 |
| m=4 | 30.91 |
| m=8 | 28.12 |
| m=16 (default) | 24.74 |
| m=32 | 24.47 |

### Key Findings

- **k=6 is the optimal trade-off**: Only 6 refinement blocks suffice for the best efficiency-quality balance. k=4 results in a notable FID increase (XL: 11.11 vs. 9.92), while k=8 yields marginal improvement at 10% additional GFLOPs.
- **Multi-token class embedding provides substantial gains**: Increasing from m=1 to m=16 reduces FID from 32.31 to 24.74 (a drop of 7.5 points), with m=32 showing almost no further improvement, indicating that 16 tokens sufficiently encode class semantics.
- **FNO time embedding consistently improves FID by ~4 points**: 3 MixedFNO blocks is optimal (2 blocks perform slightly worse; 4 blocks introduce instability).
- **Upsample module design is critical**: Linear+Linear (default) achieves FID=24.74 vs. ConvTranspose at 29.45, demonstrating the significant impact of the upsampling strategy.
- **Training throughput doubles**: MPDiT-XL achieves more than 2× the sampling speed of DiT-XL/2.

## Highlights & Insights

- The **coarse-to-fine architectural design** is both elegant and effective. The contrast with MaskDiT's failure is particularly compelling — structured downsampling (large patches) is far superior to random downsampling (masking). This insight is transferable to any Transformer architecture seeking to reduce token count.
- The **FNO time embedding** is a noteworthy contribution: applying Neural Operator principles to model the continuous temporal dynamics of the diffusion process is both novel and conceptually well-motivated, given that flow matching is inherently an ODE/SDE problem.
- The **shared AdaIN finding** has practical value: sharing time/class modulation layers across all blocks reduces parameters by 30% with only a 0.4 FID increase, which is highly practical in resource-constrained settings.

## Limitations & Future Work

- Experiments are limited to ImageNet 256×256; validation on text-to-image generation or higher resolutions is absent.
- The three-level patch hierarchy for 512² is proposed conceptually but not experimentally validated.
- The upsample module design is relatively simple (linear projection); more sophisticated designs may yield further improvements.
- The cause of instability at dimension 128 in the FNO time embedding is not analyzed in depth.
- Integration with representation alignment methods such as REPA remains unexplored and may offer additional speedups.

## Related Work & Insights

- **vs. DiT/SiT**: The standard isotropic design processes 256 tokens at every layer. MPDiT compresses the majority of computation to 64 tokens via hierarchical patching, halving GFLOPs while achieving superior FID.
- **vs. MaskDiT**: Both approaches aim to reduce the number of processed tokens, but MaskDiT's random masking fails severely at high ratios (75% mask → FID ≈ 100), whereas MPDiT's structured downsampling is substantially more effective.
- **vs. DiCo/DiC**: These convolution-based diffusion models have comparable GFLOPs, but MPDiT achieves better FID at the same training epochs, indicating that Transformers retain an advantage in global modeling.
- **vs. SANA/LIT**: Linear attention approaches require initialization from pretrained full-attention models, whereas MPDiT can be trained from scratch.

## Rating

- Novelty: ⭐⭐⭐⭐ The multi-scale patch concept is not entirely new (inspired by global-local attention), but its application to diffusion Transformers and the accompanying empirical validation are valuable contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations on ImageNet are highly detailed, but validation across other domains and resolutions is lacking.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clearly articulated, and the comparative analysis with MaskDiT is persuasive.
- Value: ⭐⭐⭐⭐ A 50% GFLOPs reduction without quality degradation represents a meaningful practical contribution to diffusion model training efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](ditic_aligned_diffusion_transformer_for_efficient.md)
- [\[CVPR 2026\] Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)
- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](freqflow_frequency_aware_flow_matching.md)
- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](renderflow_single-step_neural_rendering_via_flow_matching.md)

</div>

<!-- RELATED:END -->
