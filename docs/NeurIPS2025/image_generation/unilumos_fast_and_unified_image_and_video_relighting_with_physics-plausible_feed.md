---
title: >-
  [Paper Note] UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback
description: >-
  [NeurIPS 2025][Image Generation][Relighting] This paper proposes UniLumos, a unified image and video relighting framework that enhances physical plausibility by incorporating RGB-space depth and normal geometry feedback…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Relighting"
  - "flow matching"
  - "physics-plausible feedback"
  - "video generation"
  - "geometry supervision"
date: 2026-05-08
content_hash: bc12f2ca3e891505
---

# UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback

**Conference**: NeurIPS 2025
**arXiv**: [2511.01678](https://arxiv.org/abs/2511.01678)  
**Code**: [GitHub](https://github.com/alibaba-damo-academy/Lumos-Custom)  
**Area**: Diffusion Models / Image Generation
**Keywords**: Relighting, flow matching, physics-plausible feedback, video generation, geometry supervision

## TL;DR

This paper proposes UniLumos, a unified image and video relighting framework that enhances physical plausibility by incorporating RGB-space depth and normal geometry feedback into a flow matching backbone, while achieving 20× inference speedup through path consistency learning.

## Background & Motivation

Relighting is a fundamental task in computer vision and graphics, requiring modification of scene illumination while preserving intrinsic scene properties. Existing approaches face the following core challenges:

**Limitations of Traditional Methods**: Inverse rendering-based methods require complex inputs (e.g., HDR images, spherical harmonic coefficients) and are constrained to specific scene types, making them ill-suited for practical scenarios involving only a single image or a brief text description.

**Fundamental Weakness of Diffusion Models**: Existing diffusion models operate in semantic latent spaces, where latent-space similarity does not guarantee physical correctness in visual space. Methods such as IC-Light and SynthLight lack explicit physical supervision, frequently producing artifacts including shadow misalignment, overexposed highlights, and incorrect lighting directions.

**Additional Challenges in Video Relighting**: Light-A-Video adopts a training-free framework but incurs substantial inference overhead, while RelightVid employs a joint training strategy yet still lacks physical supervision, resulting in inaccurate light-scene interactions.

**Absence of Evaluation Protocols**: Generic metrics (FID, LPIPS) fail to capture illumination-specific errors, and structured lighting description and evaluation protocols are lacking.

These issues motivate the authors to design a unified framework that bridges the gap between generative flexibility and physical correctness.

## Method

### Overall Architecture

UniLumos is built upon the Wan 2.1 flow matching video generation model. Inputs consist of a degraded video $\mathbf{V}_{\text{deg}}$, a background video $\mathbf{V}_{\text{bg}}$, and lighting conditions $\mathbf{C}$. Latent representations are obtained via a Wan-VAE encoder, and noisy inputs are concatenated with conditional signals along the channel dimension before being injected into the DiT module. The core contributions include a physics-plausible feedback mechanism and a structured lighting annotation protocol.

### Key Designs

1. **Physics-Plausible Feedback Mechanism**: In contrast to purely latent-space operations, UniLumos extracts depth maps $\hat{\mathbf{D}}$ and normal maps $\hat{\mathbf{N}}$ from the generated RGB outputs using frozen geometry estimators (e.g., Lotus), and compares them against precise geometric signals from the input reference image. The geometry feedback loss is defined as:

$$\mathcal{L}_{\text{phy}} = \mathbb{E}\left[\mathbf{M} \odot \left(\frac{\|\hat{\mathbf{D}} - \mathbf{D}\|_2}{\|\mathbf{D}\|_2} + \frac{\|\hat{\mathbf{N}} - \mathbf{N}\|_2}{\|\mathbf{N}\|_2}\right)\right]$$

where $\mathbf{M}$ denotes the foreground subject mask. Depth and surface normals, as illumination-invariant quantities, provide explicit alignment between lighting effects and scene geometry, significantly improving shadow alignment and shading consistency.

2. **Path Consistency Learning**: Physics-plausible feedback requires supervision in the RGB domain, but standard multi-step denoising is computationally prohibitive. Path consistency learning is therefore introduced to support few-step training by enforcing consistency among velocity predictions across different step sizes:

$$\mathcal{L}_{\text{fast}} = \mathbb{E}\left\|v_\theta(x_t, t, 2d) - \frac{1}{2}\left[v_\theta(x_t, t, d) + v_\theta(x_{t+d}, t+d, d)\right]\right\|_2^2$$

This enables the model to generate high-quality results under arbitrary step budgets without requiring a separate teacher–student distillation stage.

3. **LumosData Pipeline and Six-Dimensional Lighting Annotation**: A structured six-dimensional annotation protocol is designed, covering direction, light source type, intensity, color temperature, temporal dynamics, and optical phenomena. Relighting pairs are extracted from real videos, foreground subjects are segmented using BiRefNet, and IC-Light is used to synthesize relighted versions under varied lighting conditions. Annotations are automatically generated by VLMs such as Qwen2.5-VL, providing fine-grained conditional control for training.

### Loss & Training

The joint optimization objective integrates three complementary losses:

$$\mathcal{L} = \lambda_0 \mathcal{L}_0 + \lambda_1 \mathcal{L}_{\text{fast}} + \lambda_2 \mathcal{L}_{\text{phy}}$$

where $\lambda_0 = 1.0$ and $\lambda_1 = \lambda_2 = 0.1$. Training adopts a selective optimization strategy: 20% of each batch is allocated to path consistency loss (3 forward passes + 1 backward pass), while the remaining 80% uses the standard flow matching loss, of which 50% is further supervised with RGB-space geometry feedback. The model is initialized from Wan2.1-T2V-1.3B-480P and trained with the AdamW optimizer (lr=1e-5), batch size 8, for 5,000 iterations on 8 H20 GPUs.

## Key Experimental Results

### Main Results

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Avg. Score↑ | Dense L2↓ |
|--------|-------|-------|--------|-------------|-----------|
| IC-Light | 24.316 | 0.884 | 0.108 | 0.703 | 0.447 |
| SynthLight | 25.572 | 0.905 | 0.102 | 0.791 | 0.214 |
| **UniLumos (Image)** | **26.719** | **0.913** | **0.089** | **0.912** | **0.103** |
| IC-Light (per-frame) | 20.132 | 0.851 | 0.133 | 0.672 | 0.432 |
| Light-A-Video+Wan2.1 | 20.784 | 0.876 | 0.129 | 0.682 | 0.371 |
| **UniLumos (Video)** | **25.031** | **0.891** | **0.109** | **0.871** | **0.147** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | Dense L2↓ | Note |
|---------------|-------|-------|-----------|------|
| w/o All Feedback | 21.433 | 0.862 | 0.297 | Removing all feedback leads to substantial degradation |
| w/o Normal Feedback | 22.115 | 0.874 | 0.173 | Surface normals are more critical than depth |
| w/o Depth Feedback | 23.472 | 0.883 | 0.265 | Depth feedback also contributes significantly |
| w/o Path Consistency | 25.317 | 0.902 | 0.153 | Minimal quality impact but large efficiency gain |
| Only Video | 22.487 | 0.863 | 0.173 | Quality degrades without image training |
| Only Image | 24.471 | 0.872 | 0.182 | Temporal consistency degrades without video training |
| **UniLumos** | **25.031** | **0.891** | **0.147** | Unified training achieves optimal balance |

### Key Findings

- Normal supervision is more critical than depth supervision: removing normal feedback causes larger performance drops, indicating that surface orientation plays a greater role in shaping light–shadow interactions than distance information.
- Path consistency learning incurs negligible quality loss while providing substantial efficiency gains, particularly under few-step inference budgets.
- Unified image–video training outperforms single-modality training, achieving the best balance between quality and temporal consistency.
- Inference speed exceeds existing methods by more than 20×.

## Highlights & Insights

- **RGB-space geometry feedback is the key innovation**: It circumvents the fundamental limitation that latent-space operations cannot guarantee physical correctness, establishing explicit alignment between generated outputs and scene geometry via depth and surface normals as illumination-invariant quantities.
- **Path consistency learning resolves the feedback–efficiency trade-off**: RGB-domain supervision demands high-quality outputs, yet multi-step denoising is computationally expensive; path consistency learning elegantly unifies both requirements.
- **LumosBench introduces a new paradigm for lighting evaluation**: Attribute-level assessment based on VLMs captures fine-grained lighting control performance more effectively than pixel-level metrics.

## Limitations & Future Work

- The approach relies on the quality of pretrained geometry estimators (e.g., Lotus), which may produce inaccurate estimates in extreme scenarios.
- The six-dimensional lighting annotations are automatically generated by VLMs and may introduce annotation bias.
- Validation is currently limited to the 1.3B model; the effect of scaling to larger models remains unexplored.
- The handling of complex materials (e.g., highly reflective or translucent surfaces) has not been thoroughly analyzed.

## Related Work & Insights

- Compared to latent-space methods such as IC-Light and SynthLight, the physics-plausible feedback paradigm of UniLumos is generalizable to other generation tasks requiring physical constraints.
- The application of path consistency learning offers a new direction for accelerating few-step training of diffusion models.
- The attribute-level evaluation methodology of LumosBench can be applied to fine-grained quality assessment in other generative tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The RGB-space geometry feedback mechanism is a novel design for relighting, though individual components have precedents in other domains.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers both image and video modalities, multiple baselines, comprehensive ablations, efficiency analysis, and introduces a new evaluation benchmark.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear and problem formulation is well-articulated, though some formula details could be presented more concisely.
- **Value**: ⭐⭐⭐⭐ The 20× speedup and substantial quality improvements are practically significant, and the new benchmark has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[NeurIPS 2025\] GenIR: Generative Visual Feedback for Mental Image Retrieval](genir_generative_visual_feedback_for_mental_image_retrieval.md)
- [\[ICLR 2026\] PI-Light: Physics-Inspired Diffusion for Full-Image Relighting](../../ICLR2026/image_generation/pi-light_physics-inspired_diffusion_for_full-image_relighting.md)
- [\[NeurIPS 2025\] Fast Data Attribution for Text-to-Image Models](fast_data_attribution_for_text-to-image_models.md)
- [\[NeurIPS 2025\] Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)

</div>

<!-- RELATED:END -->
