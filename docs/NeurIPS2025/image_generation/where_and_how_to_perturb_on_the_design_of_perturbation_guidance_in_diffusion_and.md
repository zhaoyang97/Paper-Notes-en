---
title: >-
  [Paper Note] Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models
description: >-
  [NeurIPS 2025][Image Generation][Attention Perturbation Guidance] This paper proposes the HeadHunter framework and SoftPAG method, refining the granularity of attention perturbation in diffusion models from the layer level down to individual attention heads. It is the first work to reveal that different attention heads govern distinct visual concepts (structure, style, texture, etc.), enabling more precise and composable generation guidance.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Attention Perturbation Guidance"
  - "Attention Heads"
  - "Diffusion Transformer"
  - "Fine-Grained Control"
  - "Style Transfer"
date: 2026-05-08
content_hash: f4a5cfd05e4e735f
---

# Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.10978](https://arxiv.org/abs/2506.10978)  
**Code**: [Project Page](https://cvlab-kaist.github.io/HeadHunter/)  
**Area**: Diffusion Models / Image Generation
**Keywords**: Attention Perturbation Guidance, Attention Heads, Diffusion Transformer, Fine-Grained Control, Style Transfer

## TL;DR

This paper proposes the HeadHunter framework and SoftPAG method, refining the granularity of attention perturbation in diffusion models from the layer level down to individual attention heads. It is the first work to reveal that different attention heads govern distinct visual concepts (structure, style, texture, etc.), enabling more precise and composable generation guidance.

## Background & Motivation

- **Background**: Classifier-Free Guidance (CFG) is central to generation quality in diffusion models, yet it suffers from two major limitations: (1) it is only applicable to conditional generation and cannot be used in unconditional settings (e.g., inverse problems); and (2) it tends to reduce diversity and cause oversaturation.

- **Limitations of Prior Work**: Attention perturbation guidance methods (e.g., PAG) offer an alternative by constructing an implicit weak model through perturbing attention layers. However, existing approaches lack a principled answer to a fundamental question — **where perturbation should be applied**.

- **Key Challenge**: This challenge stems from architectural differences. **U-Net** has a well-defined bottleneck (the middle block) responsible for global semantics, making the perturbation location straightforward. **Diffusion Transformers (DiT)**, by contrast, lack a coarse-to-fine hierarchical structure, and semantic processing is distributed more uniformly across all layers.

- **Core Idea**: The key insight arises from a simple yet profound experiment: applying PAG perturbation to individual attention heads in DiT yields visually distinct effects — some heads enhance dark tones, others alter geometric structure, and still others affect color palette. This demonstrates that attention heads are a more meaningful unit of perturbation than entire layers.

## Method

### Overall Architecture

The method comprises two complementary components: (1) **HeadHunter** — an iterative attention head selection framework; and (2) **SoftPAG** — a continuously adjustable mechanism for controlling attention perturbation intensity.

### Key Designs

1. **Head-Level Perturbation Guidance**: Unlike layer-level perturbation, which operates uniformly over all heads, head-level guidance selectively perturbs a chosen subset of attention heads. Given a selected head set $\mathcal{S} = \{(l_1,h_1), \ldots, (l_m,h_m)\}$, the corresponding attention maps are replaced with the identity matrix:
    $$\mathbf{A}_{l,h}^{(\text{PAG})} = \mathbf{I} \quad \text{for } (l,h) \in \mathcal{S}$$
   This fine-grained operation avoids the quality degradation caused by within-layer polysemanticity in layer-level perturbation.

2. **HeadHunter Iterative Head Selection Framework**: This component addresses the problem of automatically selecting attention heads aligned with user-specified objectives. Each round consists of three stages:

    - **Generation Stage**: For each candidate head $(l,h)$, perturbation is applied and samples are generated across multiple prompt–seed pairs.
    - **Evaluation Stage**: A user-specified objective function $\mathcal{O}$ (e.g., PickScore) computes the average score $s_{(l,h)} = \frac{1}{M}\sum_{j=1}^M \mathcal{O}(\hat{x}_j, p_j)$.
    - **Expansion Stage**: The top-$k$ heads are added to the final selected set $\mathcal{S}_{\text{final}}$.

   A key advantage of the iterative design is that certain heads produce poor results in isolation but effectively enhance specific styles (e.g., warm tones) when combined with already-selected structural heads. Such heads would never be identified in a one-shot evaluation, but their combinatorial value emerges through the iterative process.

3. **SoftPAG (Soft Perturbation Attention Guidance)**: SoftPAG provides continuously adjustable perturbation intensity via linear interpolation between the original attention map and the identity matrix:
    $$\mathbf{A}_{l,h}^{(\text{SoftPAG})} = (1-u)\mathbf{A}_{l,h} + u\mathbf{I}, \quad u \in [0,1]$$
   At $u=1$, this reduces to standard PAG; at $u=0$, no perturbation is applied. This formulation resolves the oversmoothing and oversimplification caused by excessively strong perturbation, identifying a sweet spot between quality enhancement and detail preservation.

### Concept Composability

Visual concepts governed by different heads can be superimposed by combining multiple heads. For example, composing a lighting head (L1,H10) with a shear-deformation head (L11,H15) yields a blended effect. However, this follows the law of diminishing returns — combining too many heads leads to oversaturation.

## Key Experimental Results

### Main Results: General Quality Improvement (SD3, MS-COCO 1K prompts)

| Method | PickScore ↑ | AES ↑ | HPS ↑ | ImReward ↑ |
|--------|-------------|-------|-------|------------|
| Baseline (no guidance) | 19.66 | 5.37 | 0.2147 | -0.591 |
| CFG (w=3.0) | 20.87 | 5.71 | 0.2924 | 0.844 |
| CFG (w=6.0) | 20.92 | 5.80 | 0.3046 | 1.063 |
| HeadHunter (w=3.0) | 20.70 | **5.92** | 0.2901 | 0.470 |
| CFG (3.0) + HeadHunter (3.0) | **20.92** | **5.93** | **0.3036** | 0.845 |

### Ablation Study: Number of Heads vs. Quality

| Configuration | Key Finding | Note |
|---------------|-------------|------|
| k=6 (only 25% of heads) | FID already outperforms full layer-level perturbation | A compact head set suffices to surpass heuristic layer selection |
| k=6→24 (increasing) | FID improves consistently (at w=3.0 and 4.0) | Supports composability of head effects |
| k>12 at w=6.0 | FID begins to degrade | Too many heads under high guidance strength causes oversaturation |
| SoftPAG u<1.0 | Optimal on most metrics at u<1.0 | Full replacement (PAG) is generally suboptimal |

### Key Findings

- **Within-layer diversity is the root cause of layer-level perturbation inefficiency**: Even layers considered "poor" (e.g., L13) contain individual heads capable of producing high-quality outputs.
- **Interpretable head specialization**: Specific heads consistently control visual attributes such as shear, brightness, and blue tones.
- **HeadHunter as a plug-and-play module**: Stacking HeadHunter on top of an existing CFG pipeline further improves aesthetic scores (AES from 5.80 to 5.93).
- **Style-directed search**: Over 5 iterative rounds (3 heads selected per round), generated images progressively reinforce target styles (e.g., golden warm lighting, line-art style).

## Highlights & Insights

- **First head-level analysis of attention perturbation guidance**: Reveals functional specialization among attention heads in diffusion models, offering significant interpretability value.
- **Combinatorial emergence from weak to strong**: Heads that individually perform poorly may contribute critical stylistic elements in combination — an effect analogous to weak learners in ensemble learning.
- **Simplicity and elegance of SoftPAG**: A linear interpolation requiring only a few lines of code provides continuously controllable perturbation intensity, making it highly practical.
- **Orthogonal and composable with CFG**: Requires no additional training, operates purely at inference time, and works synergistically with CFG.

## Limitations & Future Work

- The HeadHunter search process requires multiple generation and evaluation passes, incurring non-trivial computational cost (though the search need only be performed once).
- The identified head set may be coupled to a specific model architecture and version; switching models necessitates re-running the search.
- The optimal value of $u$ in SoftPAG may vary across prompts, whereas a fixed value is currently used.
- Validation is primarily conducted on SD3 and FLUX.1; generalizability to broader architectures (e.g., DiT-XL, PixArt) remains to be confirmed.

## Related Work & Insights

- PAG first introduced attention perturbation guidance into diffusion models but operated only on the middle block of U-Net.
- SEG (Smoothed Energy Guidance) interprets perturbation guidance from an energy perspective but likewise operates at the layer level.
- Research on attention head specialization in large language models (e.g., Transformer Circuits) provides theoretical inspiration for this work.
- The [CLS] token design in ViT and the analytical methodology for attention heads in diffusion models offer mutual insights.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First head-level analysis of perturbation guidance; findings carry far-reaching implications.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Rich quantitative and qualitative analysis, though broader architectural validation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation, in-depth analysis, and well-crafted visualizations.
- **Value**: ⭐⭐⭐⭐⭐ Plug-and-play, training-free, and composable — extremely practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)
- [\[NeurIPS 2025\] Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models](perturb_a_model_not_an_image_towards_robust_privacy_protection_via_anti-personal.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)
- [\[NeurIPS 2025\] Diffusion-Based Electromagnetic Inverse Design of Scattering Structured Media](diffusion-based_electromagnetic_inverse_design_of_scattering_structured_media.md)

</div>

<!-- RELATED:END -->
