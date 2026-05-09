---
title: >-
  [Paper Note] InnoAds-Composer: Efficient Condition Composition for E-Commerce Poster Generation
description: >-
  [CVPR 2026][Image Generation][e-commerce poster generation] This paper proposes InnoAds-Composer, a single-stage e-commerce poster generation framework built on MM-DiT. It maps three types of conditions — product subject, glyph text, and background style — into a unified token space via unified tokenization, and combines a Text Feature Enhancement Module (TFEM) with an importance-aware condition injection strategy to maintain high-quality generation while significantly reducing inference cost.
tags:
  - CVPR 2026
  - Image Generation
  - e-commerce poster generation
  - multi-condition composition
  - MM-DiT
  - text rendering
  - condition importance analysis
date: 2026-05-08
content_hash: aaf92a665bce2cb3
---

# InnoAds-Composer: Efficient Condition Composition for E-Commerce Poster Generation

**Conference**: CVPR 2026
**arXiv**: [2603.05898](https://arxiv.org/abs/2603.05898)
**Code**: N/A
**Area**: Image Generation / Controllable Generation
**Keywords**: e-commerce poster generation, multi-condition composition, MM-DiT, text rendering, condition importance analysis

## TL;DR

This paper proposes InnoAds-Composer, a single-stage e-commerce poster generation framework built on MM-DiT. It maps three types of conditions — product subject, glyph text, and background style — into a unified token space via unified tokenization, and combines a Text Feature Enhancement Module (TFEM) with an importance-aware condition injection strategy to maintain high-quality generation while significantly reducing inference cost.

## Background & Motivation

E-commerce poster generation must simultaneously satisfy **product fidelity, text accuracy, and style consistency**, yet existing methods exhibit notable shortcomings:

**Unreliable multi-stage pipelines**: Approaches that first synthesize the scene and then render text suffer from style inconsistency and degraded subject fidelity.

**Difficulty rendering Chinese text**: Single-stage methods struggle to accurately render complex scripts and small glyphs.

**Prompt-dependent style control**: Models easily drift from global style or semantic constraints.

**Scarce training data**: No dataset with joint annotations covering subject + text + style exists.

**Core gap**: No existing method can jointly and end-to-end control background style, product subject, and text within a single model, and concatenating multi-condition tokens causes quadratic complexity growth in attention.

## Method

### Overall Architecture

InnoAds-Composer is built on the MM-DiT backbone and comprises three core modules:
- **Multi-Condition Tokenization**: Maps style/subject/glyph conditions into a unified token space.
- **Importance-Aware Condition Injection**: Routes conditions to the most responsive layers and timesteps.
- **Decoupled Attention**: Removes the redundant cross-attention path from condition queries to noise latent keys.

### Key Designs

1. **Multi-Condition Tokenization**

    - **Background style control**: A style image is VAE-encoded and patchified to obtain visual tokens $h^i$, or alternatively represented as text tokens $h^p$; $h^b = \mathcal{C}(h^i, h^{p_0})$, where $h^{p_0}$ is a fixed anchor prompt.
    - **Product subject control**: Regions outside the subject are masked to black, then VAE-encoded to obtain $h^s$, suppressing background leakage.
    - **Glyph control + TFEM**: A dual-branch design — Branch 1: full-image glyph VAE encoding yields $h^{c1}$; Branch 2: individual character crops are processed by an OCR backbone with triple positional encoding (absolute position, font size, local position) to yield $h^{c2}$; a lightweight character encoder fuses the two: $h^c = \mathbf{GlyphEnc}(h^{c1}, h^{c2})$.

2. **Importance-Aware Condition Injection**

   Attention weights from the fully-conditioned pretrained model are analyzed to extract per-layer $b$ and per-timestep $t$ condition importance:

    $S_{ci}(b,t) = \mathbf{Mean}(A^{b,t,c} \odot mask_{ci})$

   The analysis reveals a **non-uniform complementary pattern** across the three condition types: background style dominates early layers and early timesteps; the subject forms a high-intensity band in mid-to-deep layers; glyphs gradually increase in mid layers and later timesteps. Accordingly, condition tokens are injected only at the most responsive positions (retaining by default ~40% for style, ~50% for subject, ~20% for glyphs), substantially shortening effective sequence length.

3. **Decoupled Attention**

   The attention path from condition queries to noise keys is removed (as condition tokens evolve slowly, this path is redundant), retaining only the noise query → condition key path:

    $O_n = \mathbf{Attn}(Q_n, [K_n; K_{ci}], [V_n; V_{ci}])$
    $O_{ci} = \mathbf{Attn}(Q_c, K_{ci}, V_{ci})$

   The condition branch is timestep-independent, enabling its activations to be cached and reused.

### Loss & Training

**Two-stage training**: Stage I retains all condition tokens to train the complete poster generator; Stage II prunes tokens by importance and fine-tunes, with timestep sampling weighted by a quality distribution derived from the global importance map to mitigate performance degradation from pruning.

## Key Experimental Results

### Main Results

Evaluation on InnoComposer-Bench (300 samples):

| Method | Sen. Acc↑ | NED↑ | DINO↑ | IoU↑ | CSD↑ | FID↓ |
|--------|-----------|------|-------|------|------|------|
| Flux-Kontext | - | - | 0.831 | 0.793 | 0.573 | 76.76 |
| PosterMaker | 0.765 | 0.848 | 0.916 | 0.954 | - | 60.55 |
| Qwen-Image-Edit | 0.831 | 0.960 | 0.922 | 0.903 | 0.722 | 69.86 |
| Seedream 4.0 | 0.865 | 0.972 | 0.864 | 0.837 | 0.700 | 64.21 |
| **Ours (Stage I)** | **0.857** | **0.976** | **0.923** | **0.972** | **0.729** | **54.39** |
| Ours (Stage II) | 0.847 | 0.969 | 0.914 | 0.960 | 0.727 | 55.24 |

Efficiency comparison:

| Method | Latency (s) | FLOPs (T) | Memory (G) |
|--------|-------------|-----------|------------|
| Flux-Kontext | 76.02 | 218.45 | 55.29 |
| Ours (Stage I) | 55.87 | 165.56 | 39.71 |
| **Ours (Stage II)** | **47.32** | **135.25** | **39.41** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| w/o TFEM | Sen. Acc drops ~5% | Noticeable degradation in text rendering quality |
| Random pruning vs. uniform pruning vs. importance pruning | Importance pruning greatly outperforms the other two | Glyphs tolerate ~80% pruning; subject ~50%; style ~60% |
| Stage I vs. Stage II | Slight quality drop with large efficiency gain | Latency reduced by 37.8%, FLOPs by 38.1% |

### Key Findings

- Stage I achieves the best performance on nearly all metrics, with FID 54.39 substantially outperforming all open-source and commercial competitors.
- Stage II sacrifices minimal quality for ~40% inference acceleration, demonstrating the effectiveness of selective injection.
- The dual-branch glyph encoding in TFEM is particularly critical for Chinese text rendering.

## Highlights & Insights

- **Condition importance visualization**: The first systematic analysis of condition importance distributions across layers and timesteps in MM-DiT, revealing a non-uniform complementary pattern.
- **Decoupled attention + condition caching**: The condition branch is timestep-independent, enabling precomputation and caching so that inference overhead amounts only to the main attention stream.
- **Companion dataset InnoComposer-80K**: The first e-commerce poster dataset with joint annotations covering subject + text + style.

## Limitations & Future Work

- Training data is constructed via a synthetic pipeline; the diversity of background styles may be constrained by the quality of the generative model.
- Importance analysis is based on fixed attention patterns from the fully-conditioned pretrained model; whether learnable dynamic routing is feasible warrants further exploration.
- The framework has not been extended to video posters or dynamic content.

## Related Work & Insights

- **Flux series**: Provides the base text-to-image capability upon which this work builds multi-condition control.
- **PosterMaker**: A prior poster generation method capable of producing subject + text but with poor style consistency.
- **Seedream 4.0**: A closed-source commercial model with strong text capability but copy-paste-style style transfer.

## Rating

- **Novelty**: ★★★★☆ — The combination of importance-aware injection and decoupled attention is innovative.
- **Technical Depth**: ★★★★☆ — The condition analysis framework and TFEM design are thorough.
- **Experimental Thoroughness**: ★★★★☆ — Multi-dimensional metrics, efficiency analysis, and ablations are provided, though the test set contains only 300 samples.
- **Practicality**: ★★★★★ — Directly applicable to e-commerce scenarios with significant efficiency gains.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation](posteriq_a_design_perspective_benchmark_for_poster_understanding_and_generation.md)
- [\[CVPR 2026\] ConsistCompose: Unified Multimodal Layout Control for Image Composition](consistcompose_multimodal_layout_control.md)
- [\[ICLR 2026\] Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss](../../ICLR2026/image_generation/condition_errors_refinement_in_autoregressive_image_generation_with_diffusion_lo.md)
- [\[CVPR 2026\] EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation](edgedit_hardware-aware_diffusion_transformers_for_efficient_on-device_image_gene.md)
- [\[CVPR 2026\] EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation](evatok_adaptive_length_video_tokenization_for_eff.md)

<!-- RELATED:END -->
