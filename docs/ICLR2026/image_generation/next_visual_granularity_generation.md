---
title: >-
  [Paper Note] Next Visual Granularity Generation
description: >-
  [ICLR 2026][Image Generation][Visual Granularity] This paper proposes the Next Visual Granularity (NVG) generation framework, which decomposes images into structured sequences at different granularity levels and generates from global layout to fine-grained details progressively, achieving consistent FID improvements over the VAR family.
tags:
  - ICLR 2026
  - Image Generation
  - Visual Granularity
  - Autoregressive Generation
  - Structured Sequence
  - Coarse-to-Fine Generation
  - ImageNet
date: 2026-05-08
content_hash: 34b1ced4bcc3bb1b
---

# Next Visual Granularity Generation

**Conference**: ICLR 2026
**arXiv**: [2508.12811](https://arxiv.org/abs/2508.12811)
**Code**: [Project Page](https://yikai-wang.github.io/nvg)
**Area**: Image Generation / Visual Autoregression
**Keywords**: Visual Granularity, Autoregressive Generation, Structured Sequence, Coarse-to-Fine Generation, ImageNet

## TL;DR

This paper proposes the Next Visual Granularity (NVG) generation framework, which decomposes images into structured sequences at different granularity levels and generates from global layout to fine-grained details progressively, achieving consistent FID improvements over the VAR family.

## Background & Motivation

- **Limitations of Prior Work**:
    - Token serialization methods neglect rich 2D spatial structures and suffer from exposure bias
    - In VAR's visual pyramid, tokens at early stages represent large, semantically diverse regions, causing representational ambiguity
    - Diffusion models lack explicit structural control and require additional modules
- **Mechanism**: Images are represented using varying numbers of unique tokens at the same spatial resolution, forming a granularity hierarchy

## Method

### Overall Architecture

NVG represents an image as a structured sequence $\mathcal{T} = \{(\boldsymbol{c}_i, \boldsymbol{s}_i)\}_{i=0}^K$, where:
- $\boldsymbol{c}_i$: content tokens at stage $i$ ($|c_i| = n_i$ unique tokens drawn from a shared codebook $\mathcal{V}$)
- $\boldsymbol{s}_i$: structure map ($h \times w$ matrix indicating the token index assigned to each position)

### 1. Visual Granularity Sequence Construction

**Structure Construction** (bottom-up clustering):
- Starts from the finest granularity (one unique token per position)
- Greedy strategy: computes pairwise $\ell_2$ distances and merges the top-$k$ most similar tokens into a cluster
- With $k=2$, the token count halves at each stage, yielding a sequence $\{2^i\}_{i=0}^8$ (for a $16^2$ latent space)

**Content Construction** (residual manner): Similar to VAR's visual pyramid, but compression is guided by structure maps rather than spatial downscaling

**Structure Embedding**: A $K$-dimensional vector encodes the hierarchical relationships across all stages; one bit (0 or 2) is appended per stage, with 1 used as padding

### 2. Generation Pipeline

Each stage generates structure first, then content:
- **Structure Generator**: A lightweight rectified flow model using v-prediction with Gumbel-top-$k$ sampling
    - Input: $\boldsymbol{z}_s(t) = t \cdot \boldsymbol{\varepsilon} + (1-t) \cdot \boldsymbol{s}_e$; known portions are replaced with ground-truth values
- **Content Generator**: Predicts the final canvas $f_c(\boldsymbol{x}_i) \rightarrow \boldsymbol{x}$, obtaining current-stage tokens via residuals

Content generator training loss:

$$\ell(\boldsymbol{x}_i) = \|\boldsymbol{x} - f_c(\boldsymbol{x}_i)\|_2^2 + \text{CE}(\hat{\boldsymbol{c}}_i, \boldsymbol{c}_i)$$

### 3. Structure-Aware RoPE

The 64-dimensional attention features are partitioned as:
- [8] text/image identifier
- [2]×8 structure encoding
- [20]×2 spatial position

Tokens within the same cluster share structure positions, while tokens across clusters have distinct ones.

## Key Experimental Results

### ImageNet 256×256 Class-Conditional Generation

| Type | Model | FID(↓) | IS(↑) | Prec(↑) | Rec(↑) | Params |
|------|-------|--------|-------|---------|--------|--------|
| X-AR | VAR-d16 | 3.30 | 274.4 | 0.84 | 0.51 | 310M |
| X-AR | VAR-d20 | 2.57 | 302.6 | 0.83 | 0.56 | 600M |
| X-AR | VAR-d24 | 2.09 | 312.9 | 0.82 | 0.59 | 1.0B |
| **X-AR** | **NVG-d16** | **3.03** | **291.6** | - | - | - |
| **X-AR** | **NVG-d20** | **2.44** | **305.0** | - | - | - |
| **X-AR** | **NVG-d24** | **2.06** | **323.0** | - | - | - |
| Mask | MAR-H | 1.55 | 303.7 | 0.81 | 0.62 | 943M |
| Diff | SiT-X | 2.06 | 270.3 | 0.82 | 0.59 | 675M |

### Ablation Study: Granularity Decomposition vs. Spatial Decomposition

| Method | rFID(↓) | IS(↑) | Note |
|--------|--------|-------|------|
| NVG (granularity decomposition) | **better** | **better** | Semantically cleaner per-token representation |
| VAR (spatial decomposition) | baseline | baseline | Early tokens mix diverse semantics |

### Key Findings

1. NVG consistently outperforms VAR across all model scales (FID: 3.30→3.03, 2.57→2.44, 2.09→2.06)
2. Clear scaling laws: larger models yield continuous performance gains
3. Generated images align closely with structure maps, validating the effectiveness of structural control
4. Structure maps from reference images can be reused to enable structure transfer across content

## Highlights & Insights

1. **Elegant problem reformulation**: Shifts autoregressive generation from "next token" to "next granularity level"
2. **Resolves VAR's representational ambiguity**: Granularity-based decomposition yields semantically cleaner per-token representations
3. **Explicit structural control**: Built directly into the generation process without requiring additional conditioning modules
4. **Mitigates exposure bias**: Residual modeling combined with progressive canvas refinement avoids error accumulation in autoregressive generation

## Limitations & Future Work

- The greedy clustering strategy may not yield optimal structure construction
- The dual-model design (structure + content) increases overall system complexity
- Validation is currently limited to class-conditional generation; text-to-image generation remains unexplored
- The "cold start" problem of the structure generator requires unified cross-stage training to mitigate

## Related Work & Insights

- **Visual Autoregression**: VAR, LlamaGen, Open-MAGVIT2
- **Diffusion Models**: DiT, SiT, LDM
- **Masked Models**: MaskGIT, MAR, TiTok

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The concept of visual granularity sequences is distinctive and intuitive
- Technical Depth: ⭐⭐⭐⭐ — Structure embeddings and Structure-Aware RoPE are elegantly designed
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparisons with clear scaling analysis
- Value: ⭐⭐⭐⭐ — Introduces a novel image generation paradigm with built-in structural control

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Pyramidal Patchification Flow for Visual Generation](pyramidal_patchification_flow_for_visual_generation.md)
- [\[CVPR 2026\] AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](../../CVPR2026/image_generation/as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)
- [\[ICLR 2026\] SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation](ssg_scaled_spatial_guidance_for_multi-scale_visual_autoregressive_generation.md)
- [\[ICCV 2025\] Randomized Autoregressive Visual Generation](../../ICCV2025/image_generation/randomized_autoregressive_visual_generation.md)
- [\[ICLR 2026\] Generate Any Scene: Scene Graph Driven Data Synthesis for Visual Generation Training](generate_any_scene_scene_graph_driven_data_synthesis_for_visual_generation_train.md)

</div>

<!-- RELATED:END -->
