---
title: >-
  [Paper Note] Beyond the Frame: Generating 360° Panoramic Videos from Perspective Videos
description: >-
  [ICCV 2025][Video Understanding][360° video generation] This paper introduces Argus, the first model to generate complete 360° panoramic videos from standard perspective videos. Through three geometry- and motion-aware techniques—camera movement simulation, view-based frame alignment, and blended decoding—Argus achieves spatially consistent and temporally coherent panoramic video generation within a diffusion-based framework.
tags:
  - ICCV 2025
  - Video Understanding
  - 360° video generation
  - video outpainting
  - diffusion model
  - panoramic video
  - view synthesis
date: 2026-05-08
content_hash: 0b4d575293516d6b
---

# Beyond the Frame: Generating 360° Panoramic Videos from Perspective Videos

**Conference**: ICCV 2025
**arXiv**: [2504.07940](https://arxiv.org/abs/2504.07940)
**Code**: None (project page available)
**Area**: Video Understanding / Video Generation
**Keywords**: 360° video generation, video outpainting, diffusion model, panoramic video, view synthesis

## TL;DR

This paper introduces Argus, the first model to generate complete 360° panoramic videos from standard perspective videos. Through three geometry- and motion-aware techniques—camera movement simulation, view-based frame alignment, and blended decoding—Argus achieves spatially consistent and temporally coherent panoramic video generation within a diffusion-based framework.

## Background & Motivation

360° videos offer a boundless field of view compared to standard videos, providing a more complete representation of dynamic visual scenes. However, existing video models are limited to generating narrow-FOV standard videos and cannot produce panoramic content. The video-to-360° task presents three core challenges:

**Extreme FOV expansion**: The input video covers only a limited viewing angle, requiring the model to infer the spatial layout and object dynamics of the entire scene.

**Limitations of existing outpainting methods**: Models trained on narrow-FOV videos exhibit degrading quality as the generated content moves farther from the input viewpoint.

**Non-linear distortion in equirectangular projection (ERP)**: Objects and spatial structures are warped under projection, and boundary discontinuities arise.

The authors' key insight is that the large corpus of 360° videos available on the internet constitutes a relatively underexploited data source for learning panoramic priors.

## Method

### Overall Architecture

Argus is built upon the Stable Video Diffusion architecture and formulates the problem as video outpainting under a dynamic mask. Given a perspective video, the model first estimates camera poses to project it into equirectangular coordinates, then performs diffusion-based generation conditioned on this projection. The core components are: video-conditioned 360° diffusion, camera movement simulation, view-based frame alignment, and blended decoding.

### Key Designs

1. **Video-Conditioned 360° Diffusion**:

    - Input perspective video $X_{\text{pers}} \in \mathbb{R}^{T \times 3 \times H \times W}$; output 360° panoramic video $Y_{\text{equi}} \in \mathbb{R}^{T \times 3 \times H' \times W'}$.
    - The perspective video is converted to equirectangular format $X_{\text{equi}}$, with unmapped regions set to black.
    - An encoder $\mathcal{E}$ encodes both representations into latent space; noise is added to $\mathbf{y}_{\text{equi}}$, which is then concatenated with $\mathbf{x}_{\text{equi}}$ and fed into the denoising network $f_\theta$.
    - CLIP is used to extract image feature sequences for cross-attention conditioning.
    - The training loss incorporates a latitude-dependent reweighting function: $\lambda(h) = (\frac{1}{2} - |\frac{1}{2} - h|)^2 + \delta$, assigning higher weights to equatorial regions, since polar areas are disproportionately magnified under equirectangular projection.

2. **Camera Movement Simulation**:

    - Training data is constructed by sampling real viewpoints from 360° videos.
    - Natural human camera motion is simulated via linear drift, oscillation, and noise terms:
        - $\phi_{\text{yaw}}(k) = \mathcal{N}(0, \eta_y) + a_y \sin(\omega k + \tau_y) + d_y k + \phi_0$
    - Horizontal and vertical field-of-view angles are randomly sampled from $[30°, 120°]$.
    - Data augmentation: random circular translation (horizontal rotation preserving the 360° property).

3. **View-Based Frame Alignment**:

    - Core problem: how to project a perspective video into equirectangular format?
    - Naively centering each frame at the equirectangular map center requires the model to implicitly learn camera motion; scene content such as the sky appears at inconsistent positions across frames, increasing learning difficulty.
    - Solution: SLAM (MegaSaM) is used to estimate relative camera poses, computing Euler angles relative to the first frame and projecting each frame into a shared coordinate system.
    - Effect: ensures that each region of the equirectangular map corresponds to approximately the same scene area across frames (sky consistently at the top, road consistently at the bottom).

4. **Blended Decoding**:

    - Problem: the left and right boundaries of equirectangular images are spatially adjacent in the scene but distant in image space, causing boundary artifacts.
    - The model decodes both the original latent representation and its 180°-rotated counterpart, producing two outputs with identical content but artifacts at different locations.
    - Distance-based weighted averaging: $Y_{k,i,j} = h_W(i) \cdot Y_{k,i,j} + (1 - h_W(i)) \cdot Y'_{k,i,j}$
    - where $h_W(x) = 1 - 2|\frac{x}{W} - \frac{1}{2}|$, assigning higher weights to pixels farther from the boundary.

### Loss & Training

- Built on the EDM diffusion framework with a score matching objective.
- Initialized from Stable Video Diffusion-I2V-XL.
- Two-stage training: first trained at $384 \times 768$ resolution for 100K iterations, then fine-tuned on a high-quality subset at $512 \times 1024$ for 20K iterations with batch size 16.
- Long video generation: context-aware training alternating between standard and context-aware inputs.
- Data: approximately 283,863 video clips filtered from the 360-1M dataset.

## Key Experimental Results

### Main Results (Video-to-360° Generation)

| Method | PSNR↑ | LPIPS↓ | FVD↓ | Motion↑ | Line cons.↑ |
|--------|-------|--------|------|---------|-------------|
| PanoDiffusion | 16.44 | 0.4138 | 2649.0 | 0.9426 | 0.6504 |
| **Argus** | **21.83** | **0.2409** | **1228.6** | **0.9802** | **0.8506** |

Comparison with video outpainting methods (FoV=60°):

| Method | Imaging↑ | Aesthetic↑ | Motion↑ |
|--------|----------|------------|---------|
| Be-Your-Outpainter | 0.4014 | 0.3461 | 0.9683 |
| Follow-Your-Canvas | 0.4268 | 0.4750 | 0.9704 |
| **Argus** | **0.4760** | **0.4722** | **0.9816** |

### Ablation Study

| Variant | PSNR↑ | LPIPS↓ | FVD↓ | Imaging↑ |
|---------|-------|--------|------|----------|
| w/o frame alignment | 20.42 | 0.3194 | 1349.6 | 0.3816 |
| w/o blended decoding | 22.09 | 0.2675 | 1226.3 | 0.4574 |
| **Full model** | **21.83** | **0.2409** | **1228.6** | **0.4939** |
| VAE reconstruction upper bound | 24.54 | 0.1663 | 121.8 | 0.5272 |

### Key Findings

- View-based frame alignment is critical to overall performance (removing it drops Imaging from 0.4939 to 0.3816).
- Blended decoding substantially improves boundary consistency, though its effect on quantitative metrics is modest.
- Argus demonstrates an understanding of dynamic content in the input video (e.g., vehicle motion) and extrapolates it reasonably; predicted trajectories closely match ground truth.
- 3D reconstruction validation: the mean rotational deviation of generated panoramic videos is only $(0.22°, 0.30°, 0.34°)$.
- Video outpainting methods degrade with increasing angular distance from the input viewpoint, whereas Argus maintains consistent quality across the full 360° range.

## Highlights & Insights

- **360° videos as a prior data source**: This work breaks away from the convention of training exclusively on narrow-FOV videos, leveraging the abundance of 360° video on the internet to learn panoramic priors.
- **Geometry-aware design**: The latitude reweighting loss, view-based frame alignment, and blended decoding all reflect a deep understanding of the geometric properties of equirectangular projection.
- **Rich downstream applications**: Video stabilization (without FOV loss), free-viewpoint control, dynamic environment mapping, and interactive VQA demonstrate the broad application potential of 360° video generation.

## Limitations & Future Work

- The output resolution of $512 \times 1024$ is substantially below that of typical 4K panoramic video, and resolution degrades further when reprojecting back to perspective views.
- Object shape inconsistencies and physical artifacts persist, limitations shared with the underlying SVD foundation model.
- Camera pose estimation is required at inference time, increasing pipeline complexity.
- Computational constraints limit generation at higher resolutions and longer durations.

## Related Work & Insights

- Unlike video panorama stitching methods such as VidPanos, Argus can extrapolate content well beyond the input viewpoint.
- The blended decoding strategy (mixing in pixel space rather than latent space) may generalize to other generation tasks involving boundary discontinuities.
- The camera movement simulation approach provides a systematic data construction methodology for training on 360° video data.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic treatment of the video-to-360° task, with technically rich geometric insights.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers quantitative evaluation, qualitative results, ablation studies, and multiple downstream application demonstrations.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and method motivation is well articulated.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction in panoramic video generation with substantial downstream application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[ICCV 2025\] An Empirical Study of Autoregressive Pre-training from Videos](an_empirical_study_of_autoregressive_pre-training_from_videos.md)
- [\[ICCV 2025\] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers](vamba_understanding_hour-long_videos_with_hybrid_mamba-transformers.md)
- [\[CVPR 2026\] VRR-QA: Visual Relational Reasoning in Videos Beyond Explicit Cues](../../CVPR2026/video_understanding/vrr-qa_visual_relational_reasoning_in_videos_beyond_explicit_cues.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)

</div>

<!-- RELATED:END -->
