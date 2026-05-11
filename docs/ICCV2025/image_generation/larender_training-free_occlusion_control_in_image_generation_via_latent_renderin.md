---
title: >-
  [Paper Note] LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering
description: >-
  [ICCV 2025][Image Generation][Occlusion Control] This paper proposes LaRender, a training-free image generation method grounded in volume rendering principles. It precisely controls inter-object occlusion relationships b…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Occlusion Control"
  - "Volume Rendering"
  - "Latent Rendering"
  - "Training-Free"
  - "Diffusion Models"
date: 2026-05-08
content_hash: 92fc8ae17a7868c1
---

# LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering

**Conference**: ICCV 2025
**arXiv**: [2508.07647](https://arxiv.org/abs/2508.07647)
**Code**: [https://xiaohangzhan.github.io/projects/larender/](https://xiaohangzhan.github.io/projects/larender/)
**Area**: Image Generation
**Keywords**: Occlusion Control, Volume Rendering, Latent Rendering, Training-Free, Diffusion Models, Image Generation

## TL;DR

This paper proposes LaRender, a training-free image generation method grounded in volume rendering principles. It precisely controls inter-object occlusion relationships by "rendering" object features in latent space. The method replaces only the cross-attention layers of a pretrained diffusion model without introducing any learnable parameters, significantly outperforming existing SOTA methods in occlusion accuracy while enabling rich effects such as semantic transparency control.

## Background & Motivation

**Why is occlusion control important?** In applications such as creative advertising, concept design, and complex scene generation, the spatial arrangement and occlusion relationships between objects must be accurately expressed. For example, "a cat in front of a dog" and "a dog in front of a cat" should yield visually distinct images.

**Why do existing methods fail to address occlusion control?**

**Text-to-image methods** (SDXL, FLUX): Occlusion relationships can only be described via prompts (e.g., "a dog behind a cat"), yet even the most advanced FLUX model fails to accurately control occlusion in complex scenes.

**Layout-to-image methods** (MIGC, 3DIS): These methods can control object placement but cannot explicitly handle occlusion. The same bounding box layout is identical regardless of occlusion order, making layout information insufficient to determine occlusion.

**Trained conditional models**: Require image–occlusion annotation pairs, which are prohibitively costly to collect.

**Core Insight**: Occlusion fundamentally shares the same physical mechanism as 3D rendering — what is perceived by the human eye or a camera is the volumetric integral along the line of sight.

## Method

### Overall Architecture

LaRender replaces all cross-attention layers in a pretrained diffusion model (SDXL) with Latent Rendering layers. Given an occlusion graph and bounding boxes, the system executes the following steps:

1. **Topological Sorting**: Objects are ordered from bottom to top according to the occlusion graph.
2. **Latent Feature Extraction**: At each cross-attention layer, input features attend to each object's prompt separately, yielding per-object latent features $\mathbf{R}_i^{(l)}$.
3. **Transmittance Estimation**: Per-object transmittance maps are estimated by combining bounding box masks and attention maps.
4. **Latent Space Rendering**: An orthographic virtual camera is placed, and object features are composited using the volume rendering formula.

### Latent Rendering Formulation

Borrowing from volume rendering principles but operating in latent space rather than pixel space:

$$\mathbf{R}^{(l+1)} = \frac{1}{\mathbf{S}} \sum_{i=1}^{N} \mathbf{T}_i (1 - \exp(-\sigma_i)) \mathbf{M}_i \mathbf{R}_i^{(l)}$$

where:
- $\mathbf{S} = \sum_{i=1}^{N} \mathbf{T}_i (1 - \exp(-\sigma_i)) \mathbf{M}_i$ is the normalization term, preventing the output from deviating from the original distribution.
- $\mathbf{T}_i = \exp(-\sum_{j=1}^{i-1} \mathbf{M}_j \sigma_j)$ is the accumulated transmittance.
- $\mathbf{M}_i$ is the transmittance map of object $i$, obtained by multiplying the bounding box mask by the normalized attention map.
- $\sigma_i > 0$ is the semantic density scalar.

### Transmittance Map Estimation

Two sources of information are combined:
- **Bounding box masks**: Object locations provided by the user or parsed by an LLM.
- **Cross-attention maps**: Attention maps of subject tokens extracted via dependency parsing, normalized and element-wise multiplied with bounding box masks to obtain more precise object contours.

### Density Scheduling

The semantic density $\sigma_i$ varies with denoising step $t$ via an inverse proportional function:

$$\sigma_i(t) = \frac{D_i T}{T + 1 - t}$$

where $D_i \geq 0$ is the user-specified target density and $T$ is the total number of steps. Design motivation:
- **Early stage** ($t = T$): $\sigma_i(T) = D_i T$ is large, degenerating to "opaque mode" to prevent concept confusion.
- **Late stage** ($t \to 1$): $\sigma_i \to D_i$, converging to the target density.

This respects the diffusion model property that concepts form rapidly in early steps and are refined in quality during later steps.

### Semantic Transparency Control

Semantic opacity is defined as $\alpha_i = 1 - \exp(-D_i)$, $\alpha_i \in [0, 1)$:
- $\alpha_i = 0$: The object is fully transparent (absent).
- $\alpha_i \to 1$: The object is opaque.
- $\alpha_i \in (0, 1)$: A high-level "semi-transparent" effect — not simple pixel blending, but conceptual intensity control (e.g., forest density, fog concentration, lighting intensity, lens effect strength).

## Key Experimental Results

### Main Results: Occlusion Control Accuracy

| Method | Control Type | UniDet↑ | User AUR↑ | User HPSR↑ | CLIP↑ | Inference Time (s) |
|--------|-------------|---------|-----------|------------|-------|-------------------|
| SDXL | text | 0.357 | 2.36±0.49 | 0.322±0.051 | 31.12 | 7.44 |
| FLUX | text | 0.401 | 1.94±0.42 | 0.293±0.023 | 30.86 | 122 |
| MIGC | layout | 0.373 | 3.56±0.52 | 0.448±0.068 | 30.73 | 11.1 |
| 3DIS | layout | 0.337 | 2.19±0.33 | 0.311±0.051 | 30.11 | 104 |
| **LaRender** | **occlusion** | **0.416** | **4.94±0.11** | **0.767±0.070** | 30.98 | 7.46 |

LaRender improves UniDet by 3.7% over the strongest baseline FLUX, substantially outperforms all methods in user study scores AUR/HPSR (4.94 vs. 3.56), and introduces virtually no additional inference overhead (7.46s vs. 7.44s for SDXL). The CLIP score is only marginally below SDXL, indicating well-preserved generation quality.

### Ablation Study: Density Scheduling Strategies

| Scheduling Strategy | Formula | UniDet↑ | CLIP Score↑ |
|--------------------|---------|---------|-------------|
| Fixed opaque mode | $\sigma_i(t) = D_i T$ | 0.240 | 28.32 |
| Fixed density | $\sigma_i(t) = D_i$ | 0.393 | 30.44 |
| **Inverse proportional** | $\sigma_i(t) = \frac{D_i T}{T+1-t}$ | **0.416** | **30.98** |

Fixed opaque mode severely degrades performance (UniDet 0.240); fixed density disrupts early-stage concept formation; the inverse proportional schedule achieves the best results on both metrics.

### Effect of Attention Maps on Transmittance Estimation

| Setting | UniDet↑ | CLIP Score↑ |
|---------|---------|-------------|
| LaRender w/o attn. map | 0.395 | 31.00 |
| LaRender w/ attn. map | **0.416** | 30.98 |

Refining transmittance contours with attention maps improves occlusion accuracy (+0.021) with negligible impact on generation quality.

## Highlights & Insights

1. **Elegant physics-inspired design**: Framing occlusion control as volume rendering yields clear physical intuition without requiring training data or additional parameters.
2. **Practical training-free deployment**: Only the cross-attention layers are replaced without modifying model weights, enabling direct application to pretrained models such as SDXL and FLUX.
3. **Unexpected benefit of semantic transparency**: By adjusting semantic density, the method controls not only occlusion but also fog concentration, forest density, lighting intensity, and other effects — these "free" capabilities enhance its practical value.
4. **LLM-assisted input parsing**: Users need only provide a natural language prompt; an LLM (e.g., DeepSeek R1) automatically parses the occlusion graph and bounding boxes, lowering the barrier to use.

## Limitations & Future Work

- Occlusion results may be incorrect when the specified layout is unreasonable.
- Concepts may occasionally be lost or confused, as latent features can be blended or erased during compositing to satisfy generative priors.
- Bounding boxes provide coarse positional control; precise bounding box alignment is not pursued.
- Primary evaluation is conducted on SDXL only; results for the FLUX variant are reported in supplementary material.

## Related Work & Insights

- **Layout-to-image methods** (MIGC, 3DIS, GLIGEN): Focus on positional control without addressing occlusion.
- **3DIS**: Generates a depth map first and then renders textures, but depth order ≠ occlusion order.
- **MULAN / LayerFusion**: Multi-layer image generation, but not designed for complex multi-object occlusion.
- **Insight**: Incorporating physical rendering principles into latent space is a promising direction that could generalize to additional physical phenomena such as depth ordering and lighting simulation.

## Rating ⭐⭐⭐⭐

- Novelty: ⭐⭐⭐⭐⭐ — First to introduce volume rendering principles into diffusion model latent space for occlusion control; the approach is original and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two datasets, user study, and complete ablations are provided, though RealOcc contains only 70 samples.
- Value: ⭐⭐⭐⭐ — Training-free, no efficiency degradation, LLM-assisted parsing; immediately deployable.
- Writing Quality: ⭐⭐⭐⭐ — Method is clearly articulated with strong physical intuition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](matchdiffusion_training-free_generation_of_match-cuts.md)
- [\[ICCV 2025\] Anchor Token Matching: Implicit Structure Locking for Training-free AR Image Editing](anchor_token_matching_implicit_structure_locking_for_training-free_ar_image_edit.md)
- [\[ICLR 2026\] Training-Free Reward-Guided Image Editing via Trajectory Optimal Control](../../ICLR2026/image_generation/training-free_reward-guided_image_editing_via_trajectory_optimal_control.md)
- [\[ICCV 2025\] Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention](fair_generation_without_unfair_distortions_debiasing_text-to-image_generation_wi.md)
- [\[ICCV 2025\] IntroStyle: Training-Free Introspective Style Attribution using Diffusion Features](introstyle_training-free_introspective_style_attribution_using_diffusion_feature.md)

</div>

<!-- RELATED:END -->
