---
title: >-
  [Paper Note] REDUCIO! Generating 1K Video within 16 Seconds using Extremely Compressed Motion Latents
description: >-
  [ICCV 2025][Image Generation][Video Generation] This paper proposes Reducio-VAE, a content-frame-conditioned 3D video autoencoder that compresses video into a motion latent space 64× smaller than a standard 2D VAE. Paire…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Video Generation"
  - "Motion Latents"
  - "Video VAE"
  - "Extreme Compression"
  - "DiT"
  - "Efficient Inference"
date: 2026-05-08
content_hash: 0f6c4b01defac921
---

# REDUCIO! Generating 1K Video within 16 Seconds using Extremely Compressed Motion Latents

**Conference**: ICCV 2025
**arXiv**: [2411.13552](https://arxiv.org/abs/2411.13552)
**Code**: [GitHub](https://github.com/microsoft/Reducio-VAE)
**Area**: Image Generation / Video Generation
**Keywords**: Video Generation, Motion Latents, Video VAE, Extreme Compression, DiT, Efficient Inference

## TL;DR

This paper proposes Reducio-VAE, a content-frame-conditioned 3D video autoencoder that compresses video into a motion latent space 64× smaller than a standard 2D VAE. Paired with Reducio-DiT, it generates 16-frame 1024×1024 videos in 15.5 seconds on a single A100 GPU, with training requiring only 3,200 A100 GPU hours.

## Background & Motivation

- **Core Problem**: Current video LDMs (e.g., Sora, Runway Gen-3) require thousands of GPUs and millions of GPU hours for training, and generating one second of video takes minutes at inference — prohibitive costs that severely hinder research and large-scale deployment.
- **Overlooked Essence**: Most video LDMs inherit the image diffusion paradigm and directly adopt SD's 2D VAE (spatial 8× downsampling, temporal 1×). However, video is far more redundant than images, with large overlapping regions across adjacent frames.
- **Existing Acceleration Directions**:
    - Efficient attention modules (e.g., Mamba replacing Transformers)
    - Diffusion training strategy optimization (fewer sampling steps, distillation)
    - None of these address the root cause — **the latent space itself is too large**
- **Key Insight**: Video can be decomposed into a "content frame + a minimal set of motion latents." The spatial dimension can be aggressively compressed to 32× and the temporal dimension to 4×, achieving an overall 4096× downsampling, while reconstruction quality actually surpasses that of the standard VAE.

## Method

### Overall Architecture: Two-Stage Generation

1. **Stage 1**: Generate a content image using an off-the-shelf T2I LDM (e.g., PixArt-alpha).
2. **Stage 2**: Conditioned on text and the content image, generate video via Reducio-DiT in the extremely compressed motion latent space.

### Key Designs 1: Reducio-VAE

**Design Motivation**: Since reconstruction is conditioned on the content frame, the video latent only needs to encode inter-frame motion information.

**Architecture Evolution**:
- **(a) Standard 2D VAE** (SDXL): $f_s=8, f_t=1$ → each frame encoded independently
- **(b) 3D VAE**: $f_s=16, f_t=4$ → joint spatiotemporal compression, 16× improvement
- **(c) Reducio-VAE**: $f_s=32, f_t=4$ → conditioned on the middle frame as content, total downsampling 4096×

**Implementation Details**:
- The 3D encoder compresses the input video $3 \times T \times H \times W$ to $|z| \times T/4 \times H/32 \times W/32$
- The middle frame $V_{T/2}$ is selected as the content guide
- The 3D decoder fuses multi-scale feature pyramids from the content frame (at $H/8 \times W/8$ and $H/4 \times W/4$) via cross-attention to inject spatial detail
- For high-resolution video, spatial tiled encoding/decoding is applied (tile size 256, overlap 64) with linear blending in overlapping regions

**Core Advantage**: Reducio-VAE achieves a PSNR gain of 5 dB over SDXL-VAE in a latent space that is 64× smaller.

### Key Designs 2: Reducio-DiT

Built on the DiT-XL architecture, following PixArt-alpha's AdaLN-single + T5 text cross-attention design.

**Image Conditioning Modules**:
- **Semantic Encoder**: Pretrained OpenCLIP ViT-H, extracting high-level semantic features from the image
- **Content Encoder**: Initialized from SD2.1-VAE, extracting spatial detail
- Both feature types are concatenated with T5 text tokens and injected into the denoising process via cross-attention

**Attention Scheme**: Full 3D attention is used by default (direct 2D→3D conversion, no additional parameters); ablations show this outperforms decomposed 2D+1D attention.

**Multi-Resolution Training Strategy**:
- Stage 1: 256² video, batch 512, 4×A100, ~900 GPU hours
- Stage 2: 512² fine-tuning, ~300 GPU hours
- Stage 3: 1024 + multi-aspect-ratio augmentation, 8×MI300, ~1000 GPU hours
- Total: only **3,200 A100 GPU hours**

**High-Resolution Efficiency Strategy**: In Stage 3, content encoder tokens are compressed from $H/16 \times W/16$ to $H/32 \times W/32$; inspired by Deepstack, tokens are divided into 4 grid groups and concatenated iteratively.

### Loss & Training

- VAE training: standard reconstruction loss + KL regularization
- DiT training: standard diffusion denoising loss $\mathcal{L} = \mathbb{E}_{z,\epsilon,t} \|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2$

## Key Experimental Results

### VAE Reconstruction Performance Comparison

| Model | Downsampling | Latent Channels | PSNR | SSIM | LPIPS | rFVD |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| SD2.1-VAE | 1×8×8 | 4 | 29.23 | 0.82 | 0.09 | 25.96 |
| SDXL-VAE | 1×8×8 | 4 | 30.54 | 0.85 | 0.08 | 19.87 |
| OpenSora-1.2 | 4×8×8 | 16 | 30.72 | 0.85 | 0.11 | 60.88 |
| Cosmos-VAE (8×8) | 8×8×8 | 16 | 30.84 | 0.74 | 0.12 | 29.44 |
| Cosmos-VAE (8×16) | 8×16×16 | 16 | 28.14 | 0.65 | 0.18 | 77.87 |
| **Reducio-VAE** | **4×32×32** | **16** | **35.88** | **0.94** | **0.05** | **17.88** |

Reducio-VAE achieves the highest PSNR (35.88) and lowest LPIPS (0.05) at a 64× compression ratio, outperforming all baselines across all metrics.

### Video Generation FVD Comparison

| Method | UCF-101 FVD |
|------|:---:|
| VideoComposer | 576.8 |
| I2VGen-XL | - |
| **Reducio-DiT** | **318.5** |

### Efficiency Comparison

| Method | Time to Generate 1024² 16-frame Video | Speedup |
|------|:---:|:---:|
| Lavie | ~258s | 1× |
| **Reducio-DiT** | **15.5s** | **16.6×** |

### Ablation Study

| Ablation | Result |
|--------|------|
| Full 3D attention vs. 2D+1D | Full 3D attention performs better |
| Content frame selection | Middle frame outperforms first/last frame |
| $f_s$=32 vs. 16 | 32 is optimal (with content frame conditioning) |
| With/without content encoder | Content encoder significantly improves detail preservation |

## Highlights & Insights

1. **Video = Content Frame + Motion Latents**: This decomposition is the central insight of the paper. The content frame captures the vast majority of visual information, while temporal changes can be described with a minimal set of motion latents.
2. **Compression Paradox**: A 64× compression actually improves reconstruction quality (PSNR +5 dB), because the content frame provides a rich spatial prior.
3. **Extreme Efficiency**: Total training requires only 3,200 A100 GPU hours — roughly three orders of magnitude less than commercial models such as Sora; inference achieves a 16.6× speedup over Lavie at 15.5 seconds per video.
4. **Engineering Friendly**: Tiled encoding/decoding supports arbitrary resolutions; multi-aspect-ratio training enables flexible deployment.
5. **From Microsoft Research**, with open-source code and publicly available datasets.

## Limitations & Future Work

- The two-stage pipeline (generate image then generate video) constrains video content to the quality of the first frame, leading to cascading errors.
- The middle frame used as the content frame is unavailable at inference time — in practice it is replaced by the T2I-generated first frame, which may not be fully consistent with the final video.
- The 16-frame limit is relatively short; additional temporal extension mechanisms are required for long video generation (tens of seconds or more).
- While Reducio-VAE excels on the Pexels rFVD benchmark, its UCF-101 rFVD (65.17) is higher than SDXL-VAE (23.68), indicating room for improvement in generalization to certain video distributions.
- Extreme compression of motion latents may discard complex motion details such as fast motion and occlusion changes.

## Related Work & Insights

- **Video Diffusion Models**: AnimateDiff, SVD, Open-Sora, CogVideo, and other UNet/DiT-based video generation methods.
- **Two-Stage Generation**: Image-to-video paradigms in I2VGen-XL, DynamiCrafter, and SparseCtrl.
- **Efficient Diffusion**: CausVid (distillation), DIM (Mamba backbone), few-step sampling, etc.
- **Video Latent Spaces**: PVDM/CMD (three 2D plane decomposition), LaMD (motion latent decomposition — the most direct predecessor of this work).
- **High-Compression Image VAEs**: DC-AE, TiTok, and other high-downsampling-factor VAEs.

## Rating

| Dimension | Score (1–5) |
|------|:---:|
| Novelty | 4.5 |
| Theoretical Depth | 3.5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Value | 5 |
| **Overall** | **4.2** |

The core insight of this paper — that video can be decomposed into a content frame and a minimal set of motion latents — is both concise and powerful, and the experiments validate the counterintuitive conclusion that greater compression yields higher quality. The 64× compression ratio is a landmark figure with milestone significance for video generation efficiency. The practical value is exceptional: 3,200 GPU hours of training combined with 15.5-second inference makes high-quality video generation broadly accessible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Video Motion Graphs](video_motion_graphs.md)
- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [\[ICCV 2025\] Learning to See in the Extremely Dark](learning_to_see_in_the_extremely_dark.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[ICCV 2025\] Generating Multi-Image Synthetic Data for Text-to-Image Customization](generating_multi-image_synthetic_data_for_text-to-image_customization.md)

</div>

<!-- RELATED:END -->
