---
title: >-
  [Paper Note] DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes
description: >-
  [Video Generation] DynamicScaler proposes a training-free unified framework that synthesizes panoramic dynamic scenes with arbitrary resolutions and aspect ratios via an Offset Shifting Denoiser (OSD) and Global Motion Guidance (GMG). It supports both conventional panorama and 360° field-of-view (FoV) video generation while maintaining a constant VRAM footprint.
tags:
  - "Video Generation"
date: 2026-05-08
content_hash: 6e431e18dfa3939a
---

# DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes

## TL;DR

DynamicScaler proposes a training-free unified framework that synthesizes panoramic dynamic scenes with arbitrary resolutions and aspect ratios via an Offset Shifting Denoiser (OSD) and Global Motion Guidance (GMG). It supports both conventional panorama and 360° field-of-view (FoV) video generation while maintaining a constant VRAM footprint.

## Background & Motivation

Immersive AR/VR applications have an increasing demand for scene-level and 360° panoramic videos. However, existing video diffusion models are limited by fixed resolutions and aspect ratios:

1. **Resolution Limits**: Most video diffusion models can only generate short videos of fixed resolutions (e.g., 512×512) and cannot directly generate ultra-wide or ultra-high resolution panoramas.
2. **Motion Consistency Challenges**: Splicing-based methods (such as MultiDiffusion and SyncDiffusion) employ overlapping windows, which incur heavy computational overhead and suffer from motion inconsistency.
3. **Challenges in 360° Panorama**: Equirectangular projection (ERP) introduces distortion, curved motion patterns, and the requirement of seamless stitching between the left and right boundaries.
4. **Memory Constraints**: The VRAM consumption of high-resolution video generation grows with resolution, restricting practical applications.
5. **Limitations of Prior 360° Methods**: 360DVD requires fine-tuning and has low resolution; 4K4DGen relies on an optimization process and has a restricted motion range.

**Core Problem**: How to generate arbitrary resolution/aspect-ratio panoramic videos without fine-tuning, using a pre-trained video diffusion model of fixed resolution, while ensuring motion consistency and spatial coherence?

## Method

### Overall Architecture

DynamicScaler adopts a two-stage generation strategy: (1) a low-resolution stage to establish coarse motion structures (initialized using panorama projection denoising for 360° scenes); (2) an upsampling stage that guides high-resolution panoramic generation from the low-resolution input via GMG. The core lies in the OSD mechanism, which shifts window positions at each denoising step to create "strided overlap" for global synchronization.

### Key Designs

#### 1. Offset Shifting Denoiser (OSD)

- **Function**: Splits the panoramic video latent into multiple windows, shifting the window positions at each step to achieve seamless denoising.
- **Mechanism**: In each denoising step, the panoramic latent of size $W_p \times H_p$ is split into $n_W \times n_H$ windows and fed into a fixed-resolution diffusion model for denoising. The key innovation is **shifting window positions horizontally and vertically at each step**, so that window boundaries from one step are covered and smoothed in the next step. Horizontally, the panorama is treated circularly—connecting the left and right boundaries, allowing windows to cross the boundary.
- **Design Motivation**: Traditional block-wise denoising creates seams and inconsistencies at window boundaries. Explicit overlap (e.g., MultiDiffusion) requires more windows, which doubles the computation. OSD achieves implicit synchronization through "strided overlap" (window shifts across different steps) without increasing the computation per step, resolving boundary artifacts in subsequent steps.

#### 2. Global Motion Guidance (GMG)

- **Function**: Ensures global motion consistency during high-resolution generation.
- **Mechanism**: Decomposes the generation into two phases: global layout and local content. It first generates a low-resolution video to capture high-level motion structure, then upsamples and re-noises it as the initialization for high-resolution generation, guiding the content layout and motion patterns.
- **Design Motivation**: The synchronization effect of OSD requires a sufficient accumulation of denoising steps and has limited influence during the early steps (the key phase determining the overall layout). This can lead to isolated motion patterns in different regions. GMG addresses this via a hierarchical approach that first determines global motion and subsequently refines local details at high resolution.

#### 3. Projection-based Panorama Denoiser + Temporal Extension

- **Function**: Extends OSD to 360° spherical panoramas and the temporal dimension.
- **Mechanism**: For 360° panoramas, the equirectangular projection is mapped back to multiple perspective viewport windows for input into Denver and then back-projected to ERP. The viewports' viewing angles are shifted at each step to achieve OSD in the spherical space. For long videos, frame windows are similarly segmented and shifted along the temporal dimension; treating the frame sequence circularly enables seamless looping videos. A mask $M_d$ is used to track the denoised regions and perform noise re-balancing on overlapping areas.
- **Design Motivation**: The distortion of ERP degrades the denoising performance of conventional diffusion models directly applied to the ERP space. Projecting to perspective viewports and utilizing standard models avoids distortion issues. Temporal extension breaks the limitation of short videos (from 16 frames to arbitrary lengths), and the circular mechanism allows for looping playback.

### Loss & Training

DynamicScaler is a training-free method and does not involve training losses. The core formula outlines the OSD denoising process:

$$Z_t = Con|_{1:n_W, 1:n_H}\left(\Phi_\theta(t, c, Split(Z_{t-1}, i, j, t, n_W, n_H))\right)$$

GMG hierarchical generation:

$$Z_{HR^0} = \Phi_\theta^{OSD}(noise(inter(\Phi_\theta^{OSD}(Z_{LR^T}))))$$

## Key Experimental Results

### Main Results Table

**Quantitative comparison with 360DVD (Tab. 1)**:

| Metric | 360DVD | DynamicScaler |
|------|--------|---------------|
| CLIP-Score↑ | 0.293 | **0.302** |
| Image Quality↑ | 0.436 | **0.583** |
| Dynamic Degree↑ | 0.412 | **0.783** |
| Motion Smoothness↑ | 0.917 | **0.963** |
| Temporal Flickering↑ | 0.964 | **0.982** |
| Scene↑ | 0.417 | **0.499** |
| Q-Align(I)↑ | 0.485 | **0.632** |
| Q-Align(V)↑ | 0.532 | **0.613** |

### Functional Comparison

| Feature | 360DVD | 4K4DGen | ScaleCrafter | VividDream | **DynamicScaler** |
|------|--------|---------|-------------|------------|-------------------|
| Training-free | ✗ | ✗ | ✓ | ✗ | **✓** |
| Arbitrary Size | ✓ | ✗ | ✓ | ✓ | **✓** |
| 360° FoV | ✓ | ✓ | ✗ | ✗ | **✓** |
| Text-conditioned | ✓ | ✗ | ✓ | ✗ | **✓** |
| Image-conditioned | ✗ | ✓ | ✗ | ✓ | **✓** |
| Infinite Video | ✗ | ✗ | ✗ | ✗ | **✓** |
| Looping Generation | ✗ | ✗ | ✗ | ✗ | **✓** |

### Key Findings

1. **Comprehensive Outperformance over 360DVD**: Outperforms 360DVD across all 8 metrics, with particularly significant margins in dynamic degree (0.783 vs. 0.412) and image quality (0.583 vs. 0.436).
2. **Constant VRAM**: Regardless of the output resolution, VRAM consumption remains constant (as only one fixed window is processed at a time).
3. **Most Versatile Capabilities**: It is the only method that simultaneously supports being training-free, having arbitrary sizes, 360° views, text/image conditioning, long videos, and loop generation.
4. Video length is extended from 16 frames to over 80+ frames while maintaining a consistent quality.

## Highlights & Insights

1. **Core Insight of Offset Shifting**: Rather than using overlapping windows that increase computational cost to eliminate seams, shifting the window positions between different denoising steps allows the "seams" to be covered in the next step. This is a clean yet highly effective design.
2. **Circular Connection for Seamlessness**: Treating the panoramic latent horizontally as a circle allows windows to cross the left and right boundaries, naturally supporting the continuity requirements of 360° panoramas and elegantly supporting temporal loops.
3. **Advantages of the Training-Free Pipeline**: It is built entirely on pre-trained video diffusion models, allowing it to directly benefit from updates to foundation models (e.g., upgrading from SVD to better base models) without retraining.
4. **Reasonable Hierarchical GMG Design**: Initializing the motion structure at low resolution before refining it at high resolution aligns nicely with the "coarse-to-fine" generation mechanism of diffusion models.

## Limitations & Future Work

1. **Limited Motion Complexity**: It relies heavily on the motion generation capability of the base model, which might struggle with complex scene-level motion (e.g., multi-object interaction).
2. **Distortion in High Polar Regions**: 360° projection results in heavy window overlapping near the poles, and the required noise re-balancing may introduce extra artifacts.
3. **Lack of Extensive Quantitative Evaluation**: It only provides quantitative comparisons with 360DVD, lacking comparison with other SOTA methods.
4. **Textual Semantic Control Precision**: Different regions of a wide-aspect panorama might require different textual controls; a single text description may not suffice.
5. It does not evaluate user interaction requirements, such as local editing of specific areas in the panorama.

## Related Work & Insights

- **MultiDiffusion / SyncDiffusion**: Panorama image-stitching methods that use overlapping windows; DynamicScaler replaces this with offset shifting.
- **360DVD**: The first 360° video diffusion model, but it requires fine-tuning and generates low-resolution outputs.
- **ScaleCrafter**: A spatially scalable diffusion model that, however, does not support 360°.
- **Insight**: The concept of offset shifting can be extended to other diffusion tasks that require spatial/temporal scalability, such as super-resolution and video extrapolation.

## Rating: ⭐⭐⭐⭐

The core OSD mechanism design is simple and efficient, covering a comprehensive set of functions (supporting 7 capabilities simultaneously), and the training-free pipeline offers strong practicality. One star is deducted due to the lack of sufficient quantitative comparisons and the limitation of motion complexity by the base model.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](../../ICCV2025/video_generation/prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](../../ICCV2025/video_generation/dive_taming_dino_for_subject-driven_video_editing.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](../../ICCV2025/video_generation/vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](../../ICCV2025/video_generation/leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
