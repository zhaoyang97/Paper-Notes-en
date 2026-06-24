---
title: >-
  [Paper Note] Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion
description: >-
  [ECCV 2024][Video Generation] This paper proposes Videoshop—a training-free method for localized semantic video editing. Users can modify the first frame of a video using any image editing tool, and the system automatically propagates the edits to all subsequent frames based on noise-extrapolated diffusion inversion and latent space normalization techniques. While maintaining semantic, spatial, and temporal consistency, it outperforms six baseline methods across ten evaluatio…
tags:
  - "ECCV 2024"
  - "Video Generation"
date: 2026-05-08
content_hash: 1fe07d508c7fcafe
---

# Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion

**Conference**: ECCV 2024  
**arXiv**: [2403.14617](https://arxiv.org/abs/2403.14617)  
**Area**: Video Generation

## TL;DR

This paper proposes Videoshop—a training-free method for localized semantic video editing. Users can modify the first frame of a video using any image editing tool, and the system automatically propagates the edits to all subsequent frames based on noise-extrapolated diffusion inversion and latent space normalization techniques. While maintaining semantic, spatial, and temporal consistency, it outperforms six baseline methods across ten evaluation metrics.

## Background & Motivation

- Traditional video editing requires frame-by-frame manual processing, which is time-consuming and demands professional skills.
- Existing text-driven video editing methods rely on coarse text instructions and lack fine-grained control—making it impossible to precisely specify editing locations and appearances.
- Image editing has already achieved fine-grained control (e.g., Photoshop, ControlNet), but these methods fail to guarantee temporal consistency across video web frames.
- Directly applying DDIM inversion to Stable Video Diffusion (SVD) leads to severe issues—it only accurately reconstructs the first frame, while subsequent frames suffer from cumulative approximation errors.

**Key Insight**: (1) The latent trajectories during the denoising process of video diffusion models are approximately linear; (2) Unnormalized outputs from the VAE encoder result in high variance in latent magnitudes, which hurts generation quality.

## Method

### Overall Architecture

A four-stage pipeline:
1. **Encoding & Normalization**: The VAE encodes the video into latents, which are normalized channel-wise to unit standard deviation.
2. **Noise-Extrapolated Inversion**: In each inversion step, linear extrapolation is leveraged to provide a correction term for accurate mapping to the noise space.
3. **Diffusion Generation**: Conditioned on the edited first frame, the inverted noise is denoised to generate the edited video.
4. **Scaling & Decoding**: The latents are rescaled using the mean and standard deviation of the target image, and decoded into the video using the VAE.

### Key Designs

**Noise-Extrapolated Inversion**:
- In the inversion equation under the EDM framework, $F_\theta(c_{in}^{t+1}\hat{x}_{t+1}; c_{noise}^{t+1})$ depends on the unknown next-step latent. Naive approaches use the current latent as an approximation, which leads to cumulative errors.
- Leveraging the **near-linear property** of denoising trajectories (averaging a cosine similarity of 0.9919 between adjacent steps), a better approximation is achieved via linear extrapolation:
$$\bar{x}_{t+1} \approx \frac{\sigma_{t+1}}{\sigma_t}(\hat{x}_t - x_0) + x_0$$
- Set a noise threshold $\Sigma$: when $\sigma_t \le \Sigma$, random Gaussian noise is used instead, avoiding instability caused by division by small decimals.

**Latent Normalization and Scaling**:
- Before inversion, the VAE-encoded latents are normalized channel-wise to unit standard deviation.
- After denoising, an affine transformation is applied to the results using the mean and standard deviation of the edited target image, ensuring color and brightness matching.
- All normalizations are conducted channel-wise.

### Loss & Training

Videoshop is a training-free method and does not involve loss function design. Its core lies in the inversion and generation strategies during the inference stage.

## Key Experimental Results

### Main Results

Quantitative comparison on the MagicBrush dataset (10 metrics):

| Method | CLIP_tgt ↑ | CLIP_tgt+ ↑ | TIFA ↑ | CLIP_src ↑ | CLIP_src+ ↑ | Flow ↓ | Flow+ ↓ | FVD ↓ | SSIM ↑ | CLIP_TC ↑ |
|------|-----------|-------------|--------|-----------|-------------|--------|---------|-------|--------|----------|
| BDIA | 82.12 | 82.19 | 57.67 | 82.48 | 87.10 | 2.83 | 1.43 | 3482 | 49.67 | 94.36 |
| Pix2Video | 71.19 | 76.47 | 51.98 | 74.55 | 79.03 | 3.59 | 2.58 | 2993 | 59.08 | 94.48 |
| Fate/Zero | 84.87 | 79.10 | 55.41 | 92.41 | 86.94 | 4.42 | 3.11 | 2205 | 48.59 | 95.71 |
| Spacetime | 63.85 | 75.20 | 46.33 | 65.74 | 71.91 | 8.24 | 5.62 | 4815 | 41.61 | 96.58 |
| RAVE | 74.70 | 78.58 | 51.12 | 75.99 | 80.19 | 3.35 | 2.42 | 2354 | 62.21 | 96.59 |
| **Videoshop** | **90.05** | **87.15** | **63.49** | **93.15** | **93.75** | **1.20** | **0.72** | **1568** | **75.78** | 96.07 |

Editing type distribution (expert dataset):

| Edit Type | Proportion |
|---------|------|
| Add Object | 36% |
| Change Appearance | 20% |
| Delete Object | 18% |
| Replace Object | 16% |
| Change Motion | 6% |
| Change Color | 4% |

### Ablation Study

Quantitative verification of the linear extrapolation trajectory:
- Average cosine similarity between all pairs of steps: **0.9282**
- Average cosine similarity between adjacent steps: **0.9919**
- Minimum cosine similarity between adjacent steps: **0.9107**

These values strongly support the validity of the linear extrapolation assumption.

Speed comparison: Videoshop is on average **2.23 times faster** than baseline methods. Editing a 14-frame video takes only about 2 minutes.

### Key Findings

1. Videoshop comprehensively leads in editing fidelity and source video fidelity, while maintaining strong temporal consistency.
2. Naive DDIM inversion on SVD can only reconstruct the first frame, while downstream frames are severely distorted—verifying the necessity of noise extrapolation.
3. Image-editing-based (non-text) methods provide finer-grained control and support various editing types.
4. Although BDIA claims precise inversion, it introduces severe visual artifacts when applied to video diffusion models.
5. Although Fate/Zero achieves high source fidelity (CLIP_src=92.41), its editing fidelity is insufficient.

## Highlights & Insights

- **Paradigm Shift**: Transitioning from text-driven to image-driven video editing, allowing users to leverage the entire ecosystem of existing image editing tools.
- **Discovery of Near-Linear Trajectories**: Systematic analysis of the denoising process of video diffusion models, serving as the theoretical foundation for the noise-extrapolated inversion method.
- **Simple Fix of Latent Normalization**: Identified and resolved the unnormalized VAE encoder output issue; the solution is extremely simple yet highly effective.
- **Training-Free Design**: Requires no training or fine-tuning, operating directly on pretrained SVD.
- **Scalability**: As video diffusion models advance (to support longer videos), Videoshop's editing capabilities will automatically scale up.

## Limitations & Future Work

- Editing is constrained to 14 frames (due to SVD's current limitations), making it unable to handle long videos.
- Requires users to manually edit the first frame, which increases interaction steps.
- For edits that require global style changes (rather than localized semantic editing), performance might be inferior to text-driven methods.
- The linear extrapolation assumption does not hold during early low-noise steps (near $x_0$), requiring the noise threshold $\Sigma$ to handle it.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MagDiff: Multi-Alignment Diffusion for High-Fidelity Video Generation and Editing](magdiff_multi-alignment_diffusion_for_high-fidelity_video_generation_and_editing.md)
- [\[ECCV 2024\] VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models](vfusion3d_learning_scalable_3d_generative_models_from_video_diffusion_models.md)
- [\[ECCV 2024\] MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model](mofa-video_controllable_image_animation_via_generative_motion_field_adaptions_in.md)
- [\[CVPR 2025\] Visual Prompting for One-Shot Controllable Video Editing Without Inversion](../../CVPR2025/video_generation/visual_prompting_for_one-shot_controllable_video_editing_without_inversion.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](../../ICCV2025/video_generation/dive_taming_dino_for_subject-driven_video_editing.md)

</div>

<!-- RELATED:END -->
