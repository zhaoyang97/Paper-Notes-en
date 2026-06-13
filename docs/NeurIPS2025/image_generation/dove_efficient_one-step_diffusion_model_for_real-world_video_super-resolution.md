---
title: >-
  [Paper Note] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution
description: >-
  [NeurIPS 2025][Image Generation][One-step diffusion] This paper presents DOVE, a video super-resolution model built upon the CogVideoX pretrained video generation model. Through a two-stage latent-pixel space training st…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "One-step diffusion"
  - "video super-resolution"
  - "CogVideoX"
  - "latent-pixel space training"
  - "video data pipeline"
date: 2026-05-08
content_hash: aefb9fa2be7476d6
---

# DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution

**Conference**: NeurIPS 2025
**arXiv**: [2505.16239](https://arxiv.org/abs/2505.16239)  
**Code**: [Available](https://github.com/zhengchen1999/DOVE)  
**Area**: Image Generation / Diffusion Models / Video Super-Resolution
**Keywords**: One-step diffusion, video super-resolution, CogVideoX, latent-pixel space training, video data pipeline

## TL;DR

This paper presents DOVE, a video super-resolution model built upon the CogVideoX pretrained video generation model. Through a two-stage latent-pixel space training strategy and a curated high-quality HQ-VSR dataset, DOVE achieves single-step inference for video super-resolution, delivering 28× speedup over multi-step diffusion methods while achieving comparable or superior performance.

## Background & Motivation

Existing diffusion models demonstrate strong performance in real-world video super-resolution (VSR), yet face two major bottlenecks:

**Inefficient multi-step sampling**: Typical methods require tens of sampling steps; processing a 33-frame 720p video on an A100 GPU takes 173–425 seconds.

**Overhead from auxiliary modules**: Components such as ControlNet and temporal layers further slow inference.

One-step inference has been successfully demonstrated in image SR, but has not yet been realized for VSR. The key challenges are:
- **Prohibitive training cost for video**: Methods such as DMD/VSD require joint optimization of multiple networks, which is infeasible in the video domain.
- **High fidelity requirements**: The instability of adversarial training introduces undesirable artifacts in VSR.

## Method

### Overall Architecture

DOVE fine-tunes CogVideoX1.5 (a T2V pretrained model) with the following core design decisions:
- **No additional modules** are introduced (no ControlNet, no optical flow modules, no temporal layers); only the Transformer is fine-tuned.
- LR video is bilinearly upsampled and encoded into latent space $z_{lr}$ via VAE.
- $z_{lr}$ is treated as a noisy latent at timestep $t=399$ (rather than $t=999$), since the LR input already contains sufficient structural information.
- Single-step v-prediction denoising: $z_{sr} = \sqrt{\bar{\alpha}_t} \cdot z_{lr} - \sqrt{1-\bar{\alpha}_t} \cdot v_\theta(z_{lr}, c, t)$
- An empty text prompt is used and pre-encoded to reduce inference overhead.
- The output video $x_{sr}$ is obtained by VAE decoding.

### Key Designs

#### Latent-Pixel Two-Stage Training Strategy

The core innovation is the use of regression losses (rather than distillation or adversarial losses) for efficient training:

**Stage-1: Adaptation (Latent Space)**
- Minimizes MSE between the predicted latent $z_{sr}$ and the HR latent $z_{hr}$ in latent space.
- Exploits the high compression ratio of VAE for computational efficiency, enabling training on longer frame sequences.
- Trained for 10,000 steps; learning rate $2\times10^{-5}$; video resolution $320\times640$; 25 frames.

**Stage-2: Refinement (Pixel Space)**
- After latent-space training, $z_{sr}$ is close to $z_{hr}$, but decoding through the VAE amplifies residual discrepancies.
- Pixel-space training directly optimizes the gap between $x_{sr}$ and $x_{hr}$.
- **Image-video mixed training** is introduced to address GPU memory bottlenecks in pixel-space video training:
    - Images (single-frame videos) constitute fraction $\varphi=0.8$; pixel-space training is tractable.
    - Videos are processed by VAE encoding/decoding frame-by-frame to avoid multi-frame memory peaks, while the Transformer operates on the full latent sequence.
- Trained for only 500 steps; learning rate $5\times10^{-6}$.

**Stage-2 Loss Functions:**
- Image: $\mathcal{L}_{\text{s2-image}} = \text{MSE}(\hat{x}_{sr}, \hat{x}_{hr}) + \lambda_1 \cdot \text{DISTS}(\hat{x}_{sr}, \hat{x}_{hr})$
- Video: $\mathcal{L}_{\text{s2-video}} = \text{MSE} + \lambda_1 \cdot \text{DISTS} + \lambda_2 \cdot \mathcal{L}_{\text{frame}}$
- Frame difference loss $\mathcal{L}_{\text{frame}}$ enforces temporal consistency by aligning adjacent-frame differences $\Delta x_{sr}$ with $\Delta x_{hr}$.
- $\lambda_1 = \lambda_2 = 1$

#### Video Processing Pipeline (HQ-VSR Dataset Construction)

A four-step pipeline is used to curate high-quality VSR training data from OpenVid-1M:
1. **Metadata filtering**: Short side > 720px; frame count > 50.
2. **Scene filtering**: Scene detection and splitting; clips with < 50 frames are discarded.
3. **Quality filtering**: Strict multi-metric screening (CLIP-IQA + FasterVQA + DOVER).
4. **Motion processing**: Optical-flow-based motion scoring + **motion region detection algorithm**.
    - A motion intensity map $M$ is generated and thresholded to obtain a motion mask.
    - A bounding box $B$ localizes high-motion regions; cropped clips below 720p are discarded.
    - This resolves the issue of globally high motion but locally static content.

The pipeline yields the HQ-VSR dataset comprising 2,055 high-quality videos.

### Loss & Training

- Stage-1: MSE loss for latent-space alignment.
- Stage-2: MSE + DISTS (perceptual quality) + frame difference loss (temporal consistency).
- Only the Transformer is fine-tuned; VAE weights are frozen.
- 4× A800-80G GPUs; total batch size 8.
- AdamW optimizer; Stage-1: 10K steps; Stage-2: 500 steps.
- Image data: DIV2K (900 images) with Real-ESRGAN degradation.
- Video data: HQ-VSR (2,055 videos) with RealBasicVSR degradation.

## Key Experimental Results

### Main Results

**UDM10 Synthetic Benchmark (×4 SR)**

| Method | Steps | PSNR↑ | LPIPS↓ | CLIP-IQA↑ | DOVER↑ | E*warp↓ |
|--------|-------|-------|--------|-----------|--------|---------|
| RealBasicVSR | — | 24.13 | 0.3908 | 0.3494 | 0.7564 | 3.10 |
| MGLD-VSR | Multi | 24.23 | 0.3272 | 0.4557 | 0.7264 | 3.59 |
| STAR | Multi | 23.47 | 0.4242 | 0.2417 | 0.4830 | 2.08 |
| **DOVE** | **1** | **26.48** | **0.2696** | **0.5107** | **0.7809** | **1.77** |

DOVE achieves comprehensive superiority in both fidelity (PSNR +2.25) and perceptual quality (CLIP-IQA/DOVER).

**SPMCS Synthetic Benchmark (×4 SR)**

| Method | PSNR↑ | LPIPS↓ | CLIP-IQA↑ |
|--------|-------|--------|-----------|
| MGLD-VSR | 22.39 | 0.3263 | 0.4348 |
| **DOVE** | **23.11** | **0.2888** | **0.5690** |

**Efficiency Comparison (33-frame 720p video, single A100)**

| Method | Inference Time |
|--------|---------------|
| MGLD-VSR | 425.23s |
| STAR | 173.07s |
| **DOVE** | **~15s (28× speedup)** |

### Ablation Study

**Training Strategy Ablation (UDM10)**

| Strategy | PSNR | LPIPS | CLIP-IQA | DOVER |
|----------|------|-------|----------|-------|
| S1 (latent only) | 27.20 | 0.3037 | 0.3236 | 0.6154 |
| S1+S2-I (+pixel image) | 26.39 | 0.2784 | 0.5085 | 0.7694 |
| S1+S2-I/V (+pixel mixed) | 26.48 | 0.2696 | 0.5107 | 0.7809 |

Stage-2 substantially improves perceptual quality (CLIP-IQA: 0.32→0.51); mixed training outperforms image-only training.

**Image Ratio $\varphi$ Ablation**

| $\varphi$ | LPIPS | CLIP-IQA | DOVER |
|-----------|-------|----------|-------|
| 0% (video only) | 0.2624 | 0.4800 | 0.7647 |
| 80% (optimal) | 0.2696 | 0.5107 | 0.7809 |
| 100% (image only) | 0.2784 | 0.5085 | 0.7694 |

**HQ-VSR Dataset Comparison (Stage-1)**

| Dataset | # Videos | PSNR | DOVER |
|---------|----------|------|-------|
| YouHQ | 38,576 | 26.88 | 0.3965 |
| OpenVid-1M | ~400K | 27.04 | 0.4363 |
| **HQ-VSR** | **2,055** | **27.20** | **0.6154** |

A dataset of merely 2K videos surpasses one of 400K, demonstrating that data quality vastly outweighs data quantity.

### Key Findings

- One-step inference is fully viable for VSR and can surpass multi-step methods in performance.
- The latent-to-pixel two-stage training strategy is the key to balancing efficiency and quality.
- Frame-by-frame VAE processing effectively resolves the GPU memory bottleneck of pixel-space video training.
- Motion region detection and cropping is better suited to VSR scenarios than global motion scoring.

## Highlights & Insights

1. **First one-step diffusion VSR model**: 28× speedup without sacrificing performance, offering significant practical value.
2. **Minimalist architecture philosophy**: No auxiliary modules are added; the method relies entirely on the priors of the pretrained T2V model, yielding a simple and efficient design.
3. **Extremely short training schedule**: Fine-tuning is completed in only 10K+500 steps, far fewer than comparable methods.
4. **Data quality >> data quantity**: 2K high-quality videos outperform 400K videos; motion region cropping is the critical factor.
5. **Choice of $t=399$**: Rather than starting from pure noise, the method exploits the structural information already present in LR inputs, reducing unnecessary reconstruction burden.

## Limitations & Future Work

- The CogVideoX-based model is large in scale and still requires a GPU for single-frame inference.
- Evaluation is conducted only for ×4 super-resolution; extension to other scale factors and degradation types remains to be explored.
- The frame difference loss uses simple L1 alignment of adjacent-frame differences; more sophisticated temporal consistency constraints (e.g., optical flow) may yield further improvements.
- HQ-VSR contains only 2K videos; larger-scale high-quality data could further boost performance.

## Related Work & Insights

- CogVideoX serves as the backbone model: 3D causal VAE + Transformer denoiser.
- OSEDiff first explored one-step diffusion for image SR; DOVE extends this paradigm to the video domain.
- DISTS perceptual loss is employed both as a training objective and as an evaluation metric.
- The frame difference loss is a simple yet effective temporal consistency solution that avoids the additional computational overhead of optical flow estimation.

## Rating

- Novelty: ⭐⭐⭐⭐ (first to extend one-step diffusion to VSR; training strategy is innovative)
- Technical Depth: ⭐⭐⭐⭐ (two-stage training + data pipeline + architecture choices form a coherent methodology)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (6 test sets, 8 baseline methods, multi-dimensional ablation)
- Practicality: ⭐⭐⭐⭐⭐ (28× speedup has significant application value)
- Writing Quality: ⭐⭐⭐⭐⭐ (well-structured, with rich illustrations)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](../../AAAI2026/image_generation/realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](../../AAAI2026/image_generation/mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)
- [\[CVPR 2026\] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](../../CVPR2026/image_generation/duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)
- [\[NeurIPS 2025\] ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion](scalediff_higher-resolution_image_synthesis_via_efficient_and_model-agnostic_dif.md)
- [\[ICCV 2025\] PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution](../../ICCV2025/image_generation/patchscaler_an_efficient_patch-independent_diffusion_model_for_image_super-resol.md)

</div>

<!-- RELATED:END -->
