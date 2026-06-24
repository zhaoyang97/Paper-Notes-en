---
title: >-
  [Paper Note] DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes
description: >-
  [CVPR 2025][Video Generation][Panoramic Video Generation] DynamicScaler is proposed as a training-free unified framework that achieves panoramic dynamic scene generation with arbitrary resolutions and aspect ratios through an offset-shifting denoiser and global motion guidance, supporting a 360° field of view, long durations, and loopable videos.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Panoramic Video Generation"
  - "Offset Shifting Denoising"
  - "360-degree Panorama"
  - "Training-free"
  - "Scalable"
date: 2026-05-08
content_hash: babdf8120d161f60
---

# DynamicScaler: Seamless and Scalable Video Generation for Panoramic Scenes

**Conference**: CVPR 2025  
**arXiv**: [2412.11100](https://arxiv.org/abs/2412.11100)  
**Code**: [https://dynamic-scaler.pages.dev/new](https://dynamic-scaler.pages.dev/new)  
**Area**: Video Generation / Panoramic Generation  
**Keywords**: Panoramic Video Generation, Offset Shifting Denoising, 360-degree Panorama, Training-free, Scalable

## TL;DR

DynamicScaler is proposed as a training-free unified framework that achieves panoramic dynamic scene generation with arbitrary resolutions and aspect ratios through an offset-shifting denoiser and global motion guidance, supporting a 360° field of view, long durations, and loopable videos.

## Background & Motivation

**Background**: Immersive AR/VR applications require high-quality panoramic scene synthesis, but video diffusion models are constrained by fixed resolutions and aspect ratios.

**Limitations of Prior Work**: Methods like MultiDiffusion generate panoramas using overlapping windows but incur heavy computational costs; 360DVD fine-tunes in the equirectangular projection space, leading to low resolution and interpolation artifacts; the fixed windows of 4K4DGen restrict the motion range.

**Key Challenge**: Achieving spatial scalability while maintaining motion coherence, without causing seams at window boundaries.

**Goal**: To generate panoramic dynamic scenes with arbitrary resolutions and aspect ratios without fine-tuning, while ensuring spatial and temporal coherence.

**Core Idea**: By shifting the denoising window positions at each step, "overlapping" regions are created between steps to synchronize the denoising process across the entire panorama.

## Method

### Overall Architecture

A two-stage pipeline: a low-resolution stage establishes the coarse motion structure (OSD + optional panoramic projection), and an upsampling stage generates high-resolution, detailed panoramas using more shifted windows combined with GMG.

### Key Designs

1. **Offset Shifting Denoiser (OSD)**:

    - **Function**: Achieves seamless panoramic video denoising.
    - **Mechanism**: Shifts the denoising windows vertically and horizontally at each denoising step, where the "overlap" between steps synchronizes content and motion. Horizontally, the panorama is treated as cyclic (connecting left and right boundaries) to ensure a seamless 360° transition.
    - **Design Motivation**: To avoid the high computational cost of overlapping windows by implicitly creating an overlapping effect through inter-step offsets.

2. **Global Motion Guidance (GMG)**:

    - **Function**: Ensures global motion consistency during high-resolution generation.
    - **Mechanism**: Generates a low-resolution video first to capture the overall motion structure, which is then upsampled and noised to serve as the initialization for high-resolution generation, guiding the high-resolution stage to refine local details while preserving global motion.
    - **Design Motivation**: At early denoising steps where the layout is built, the synchronization effect of OSD has not accumulated sufficiently, necessitating a global prior for guidance.

3. **Temporal Offset Shifting and Loop Generation**:

    - **Function**: Generates ultra-long and loopable videos.
    - **Mechanism**: Extends the OSD mechanism to the temporal dimension—splitting the long video into frame segment windows and shifting the segment window positions between steps. In loop mode, the starting and ending frames are treated as connected, allowing windows to wrap around boundaries.
    - **Design Motivation**: To break through the frame-limit constraints of video diffusion models (typically 16 frames) and achieve continuous motion.

### Loss & Training

A completely training-free method implemented as an inference-time modification based on existing video diffusion models (e.g., I2V, T2V). The VRAM consumption remains constant and does not increase with output resolution.

## Key Experimental Results

### Main Results

| Metric | DynamicScaler | 360DVD |
|------|--------------|--------|
| CLIP-Score | **0.302** | 0.293 |
| Image Quality | **0.583** | 0.436 |
| Dynamic Degree | **0.783** | 0.412 |
| Motion Smoothness | **0.963** | 0.917 |
| Q-Align(V) | **0.613** | 0.532 |

### Key Findings

- Constant VRAM consumption enables the generation of arbitrarily high resolutions.
- Can be scaled from 16 frames to 80+ frames while maintaining consistent visual quality.
- Cyclic boundary handling ensures seamless 360° views.

## Highlights & Insights

- The core idea of OSD is extremely simple—just shifting windows between steps.
- A unified framework that covers normal panoramas, 360° panoramas, long videos, and loopable videos.
- Completely training-free and plug-and-play for existing video diffusion models.

## Limitations & Future Work

- Complex motion patterns may lack coordination due to insufficient information propagation between windows.
- The 360° mode introduces additional complexity in handling overlaps in polar regions.
- Generation quality remains constrained by the capabilities of the underlying video diffusion model.

## Rating

- Novelty: 8/10 — OSD is simple and elegant.
- Technical Depth: 7/10 — Simple yet effective method.
- Experimental Thoroughness: 7/10 — Limited quantitative metrics.
- Writing Quality: 7/10 — Clear structure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Optical-Flow Guided Prompt Optimization for Coherent Video Generation](optical-flow_guided_prompt_optimization_for_coherent_video_generation.md)
- [\[CVPR 2025\] Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling](spatiotemporal_skip_guidance_for_enhanced_video_diffusion_sampling.md)
- [\[CVPR 2025\] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing](fade_frequency-aware_diffusion_model_factorization_for_video_editing.md)
- [\[CVPR 2025\] VIRES: Video Instance Repainting via Sketch and Text Guided Generation](vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)
- [\[CVPR 2025\] VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step](videoscene_distilling_video_diffusion_model_to_generate_3d_scenes_in_one_step.md)

</div>

<!-- RELATED:END -->
