---
title: >-
  [Paper Note] Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models
description: >-
  [ECCV 2024][Image Generation][Video Super-Resolution] StableVSR is proposed, marking the first application of diffusion models to video super-resolution. By introducing a Temporal Conditioning Module (TCM) and a frame-wise bidirectional sampling strategy, it significantly enhances perceptual quality while ensuring temporal consistency across frames.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Video Super-Resolution"
  - "Diffusion Models"
  - "Temporal Consistency"
  - "Perceptual Quality"
  - "Texture Guidance"
date: 2026-05-08
content_hash: 806f3b912aa54d0d
---

# Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2311.15908](https://arxiv.org/abs/2311.15908)  
**Code**: [GitHub](https://github.com/claudiom4sir/StableVSR)  
**Area**: Image Generation  
**Keywords**: Video Super-Resolution, Diffusion Models, Temporal Consistency, Perceptual Quality, Texture Guidance

## TL;DR

StableVSR is proposed, marking the first application of diffusion models to video super-resolution. By introducing a Temporal Conditioning Module (TCM) and a frame-wise bidirectional sampling strategy, it significantly enhances perceptual quality while ensuring temporal consistency across frames.

## Background & Motivation

Video Super-Resolution (VSR) aims to increase the spatial resolution of videos. Existing methods suffer from a core conflict:

**Low Perceptual Quality**: Current state-of-the-art (SOTA) VSR methods (e.g., BasicVSR++, RVRT) focus on reconstruction quality (PSNR/SSIM). Although their outputs are close in pixel distance, they are visually blurry and lack details.

**Perception-Distortion Trade-off**: According to the perception-distortion trade-off, under limited model capacity, improving reconstruction quality inevitably leads to a decline in perceptual quality.

Diffusion models (DMs) have demonstrated the capability to synthesize realistic textures and details in single-image super-resolution (SISR). However, directly applying SISR to videos introduces two issues:

**Temporal Inconsistency**: Processing each frame independently causes inconsistent detail generation across adjacent frames, resulting in flickering.

**Underutilized Information**: Spatiotemporal redundancy across video frames is not exploited.

StableVSR is designed to simultaneously achieve **high perceptual quality** and **temporal consistency** in diffusion-based video super-resolution.

## Method

### Overall Architecture

StableVSR is built upon a pre-trained single-image super-resolution Latent Diffusion Model (Stable Diffusion ×4 Upscaler), extended to VSR by adding a **Temporal Conditioning Module (TCM)**. The core components include:

1. **Temporal Conditioning Module (TCM)**: Injects temporal information from neighboring frames into the denoising UNet.
2. **Temporal Texture Guidance**: Provides spatially aligned, detail-rich texture guidance.
3. **Frame-wise Bidirectional Sampling**: A frame-wise bidirectional sampling strategy to balance information propagation.

Given a low-resolution frame sequence $\{LR\}_{i=1}^N$, the goal is to generate a high-resolution sequence $\{\overline{HR}\}_{i=1}^N$.

### Key Designs

#### Temporal Conditioning Module (TCM)

TCM is injected into the decoder of the denoising UNet using a ControlNet architecture, aiming to:
1. Leverage spatiotemporal information from multiple frames to enhance single-frame quality.
2. Enforce temporal consistency across frames.

The input to the TCM is the **Temporal Texture Guidance** from neighboring frames, which is the core innovation of this paper.

#### Temporal Texture Guidance

This is the core innovation of the method. The key idea is to utilize predictions of neighboring frames to guide the generation of the current frame at sampling step $t$.

**$\tilde{x}_0$-based Guidance instead of $x_t$**: Directly using noisy $x_t$ as guidance is suboptimal as it is heavily contaminated by noise in most sampling steps. The solution is to project it to the initial state to obtain a noise-free approximation:

$$\tilde{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t))$$

Advantages of $\tilde{x}_0$:
- Virtually noise-free, regardless of $t$.
- Contains rich texture detail information.
- Progressively refines as $t$ decreases.

**Spatial Alignment**: Due to video motion, textures of the previous frame must be spatially aligned with the current frame. The pipeline is as follows:

1. Decode $\tilde{x}_0^{i-1}$ to the pixel domain using the VAE decoder $\mathcal{D}$.
2. Estimate the optical flow between low-resolution frames $LR^{i-1}$ and $LR^i$ using RAFT.
3. Complete motion compensation on the decoded frame.

Why not perform motion compensation directly in the latent space? Doing so introduces severe artifacts (as shown in Fig. 4).

Full formula:
$$\widetilde{HR}^{i-1 \rightarrow i} = MC(ME(LR^{i-1}, LR^i), \mathcal{D}(\tilde{x}_0^{i-1}))$$

#### Frame-wise Bidirectional Sampling

Existing schemes usually finish all sampling steps frame-by-frame in an autoregressive manner, leading to two issues:
- **Error Accumulation**: Errors in former frames propagate to all subsequent frames.
- **Unidirectional Information Propagation**: Information flows only from past to future, neglecting future frame context.

Bi-directional sampling strategy of StableVSR:

```python
for t = T to 1:
    for i = 1 to N:  // Perform sampling step t on all frames
        Compute temporal texture guidance
        Execute one denoising step
    end
    Reverse frame sequence order  // Alternating forward/backward propagation
end
```

Key Features:
1. **Frame-wise rather than Sequence-wise**: A single sampling step is executed on all frames before advancing to the next step.
2. **Alternating Bidirectional**: The frame sequence is reversed after each sampling step, alternating between forward and backward propagation.
3. **Balanced Information Flow**: The current frame is guided by past frames during forward propagation and by future frames during backward propagation.

### Loss & Training

- **Training Objective**: Standard diffusion denoising loss $\mathbb{E}[\|\epsilon - \epsilon_\theta(x_t^i, t, LR^i, \widetilde{HR}^{i-1 \rightarrow i})\|_2]$
- **TCM-only Training**: Pre-trained SISR LDM weights are frozen; only the ControlNet part is trained.
- **Training Inputs**: Consecutive frame pairs $(LR^{i-1}, HR^{i-1})$ and $(LR^i, HR^i)$.
- **Training Details**: Adam optimizer, lr=$1e-5$, batch size=32, 20,000 steps.
- **Data Augmentation**: $256 \times 256$ random cropping + horizontal flipping.
- **Optical Flow Estimation**: RAFT.
- **Sampling**: DDPM T=1000 for training, T=50 for inference.
- **Hardware**: 4× NVIDIA Quadro RTX 6000.

## Key Experimental Results

### Main Results

**Vimeo-90K-T Dataset (x4 SR)**

| Method | LPIPS↓ | DISTS↓ | tLP↓ | tOF↓ | PSNR↑ |
|------|--------|--------|------|------|-------|
| BasicVSR++ | 0.092 | 0.105 | 4.35 | 1.75 | **35.69** |
| RVRT | 0.088 | 0.101 | 4.28 | 1.42 | 36.30 |
| **StableVSR** | **0.070** | **0.087** | **3.89** | **1.37** | 31.97 |

**REDS4 Dataset (x4 SR)**

| Method | LPIPS↓ | DISTS↓ | tLP↓ | tOF↓ | PSNR↑ |
|------|--------|--------|------|------|-------|
| BasicVSR++ | 0.131 | 0.068 | 9.02 | 2.75 | **32.38** |
| RVRT | 0.128 | 0.067 | 8.97 | 2.72 | 32.74 |
| RealBasicVSR | 0.134 | 0.060 | 6.44 | 4.74 | 27.07 |
| **StableVSR** | **0.097** | **0.045** | **5.57** | **2.68** | 27.97 |

### Ablation Study

The paper systematically validates the designs:

1. **$\tilde{x}_0$ vs $x_t$ Guidance**: $x_t$ is heavily degraded by noise at large $t$, making texture information unavailable; $\tilde{x}_0$ provides clean details at all steps.
2. **Pixel-domain vs Latent-space Motion Compensation**: Performing motion compensation in the latent space leads to severe artifacts.
3. **Bidirectional vs Unidirectional Propagation**: Bidirectional sampling significantly improves temporal consistency and perceptual quality.

### Key Findings

1. **Significant Lead in Perceptual Quality**: StableVSR significantly outperforms existing methods across all perceptual metrics (LPIPS, DISTS); LPIPS is 24% lower than RVRT on REDS4.
2. **Superior Temporal Consistency**: Temporal consistency metrics tLP and tOF are both optimal, showcasing cleaner temporal profiles.
3. **Perception-Distortion Trade-off**: PSNR/SSIM metrics drop, but still outperform bicubic interpolation and RealBasicVSR, aligning with theoretical expectations.
4. **Generative Capability**: Synthesizes semantically consistent details absent in low-resolution frames (e.g., text textures, facial features).

## Highlights & Insights

1. **First to Utilize DMs for VSR**: Pioneering video super-resolution within a generative paradigm, breaking free from the constraints of traditional reconstruction frameworks.
2. **$\tilde{x}_0$-based Guidance Technique**: Obtaining a noise-free guidance signal by projecting back to the initial state is elegant and versatile.
3. **Frame-wise Bidirectional Sampling**: This straightforward strategy effectively resolves error accumulation and imbalanced information propagation.
4. **Insight into Pixel-domain Motion Compensation**: The finding that the latent spatial structure is unsuitable for direct motion compensation provides valuable reference for future work.

## Limitations & Future Work

1. **Slow Inference Speed**: Generating each frame requires multiple VAE decodings and optical flow estimations, leading to significant computational overhead.
2. **Large Number of Sampling Steps**: Inference still requires T=50; acceleration (e.g., via consistency models) is an important future direction.
3. **Sensitivity to Optical Flow Errors**: Motion compensation relies on optical flow quality, which may introduce incorrect guidance in occluded areas.
4. **Limitation to x4 SR**: Performance at other upscaling factors has not been validated.
5. **Long Video Processing**: Frame-wise sampling poses larger memory and time overheads on long videos.

## Related Work & Insights

- **StableSR**: Uses a pre-trained text-to-image DM for single-image super-resolution; this work extends it to video.
- **BasicVSR/BasicVSR++**: Baseline methods for VSR, introducing the bidirectional propagation framework.
- **ControlNet**: The TCM in this work adopts the ControlNet architecture to inject temporal conditions.
- **Insight**: The $\tilde{x}_0$ guidance strategy can be generalized to other diffusion-based applications requiring temporal consistency, such as video editing and video generation.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Innovation | 4.5 |
| Theoretical Depth | 3.5 |
| Experimental Thoroughness | 4 |
| Practical Value | 4 |
| Writing Quality | 4.5 |
| Overall Rating| 4.1 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] STCDiT: Spatio-Temporally Consistent Diffusion Transformer for High-Quality Video Super-Resolution](../../CVPR2026/image_generation/stcdit_spatio-temporally_consistent_diffusion_transformer_for_high-quality_video.md)
- [\[ECCV 2024\] XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)
- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)
- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution](dcdm_diffusion-conditioned-diffusion_model_for_scene_text_image_super-resolution.md)

</div>

<!-- RELATED:END -->
