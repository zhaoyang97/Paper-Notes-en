---
title: >-
  [Paper Note] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models
description: >-
  [Video Generation] > LeanVAE is proposed as an ultra-efficient video VAE built upon non-overlapping patch operations, a Neighborhood-Aware Feedforward (NAF) module, wavelet transforms, and compressed sensing. With only 40M parameters, it achieves a 50× reduction in FLOPs and a 44× speedup in inference while maintaining competitive reconstruction quality.
tags:
  - "Video Generation"
date: 2026-05-08
content_hash: e5667ca4203d86c2
---

# LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models

| Info | Content |
|------|------|
| Conference | ICCV 2025 |
| arXiv | [2503.14325](https://arxiv.org/abs/2503.14325) |
| Code | [GitHub](https://github.com/westlake-repl/LeanVAE) |
| Area | Video Generation · VAE · Diffusion Models |
| Keywords | Video VAE, wavelet transform, compressed sensing, lightweight architecture, video diffusion |

## TL;DR

> LeanVAE is proposed as an ultra-efficient video VAE built upon non-overlapping patch operations, a Neighborhood-Aware Feedforward (NAF) module, wavelet transforms, and compressed sensing. With only 40M parameters, it achieves a 50× reduction in FLOPs and a 44× speedup in inference while maintaining competitive reconstruction quality.

## Background & Motivation

### Video VAE as the Bottleneck of Video Diffusion Models

Latent video diffusion models (LVDMs) such as Open-Sora, CogVideoX, and HunyuanVideo rely on a Video VAE to compress high-dimensional videos into compact latent spaces. However:

**Computational bottleneck**: Existing Video VAEs (e.g., OD-VAE) require ~32 GB of VRAM to process a 5-frame 1080p video, making them the primary computational bottleneck in LVDM training.

**Inheritance issue**: Most methods are directly inflated from the image VAE of Stable Diffusion (2D→3D convolutions), resulting in severe architectural redundancy.

**Efficiency–quality trade-off**: Lightweight alternatives (e.g., ViT-based methods) have fewer parameters but suffer from quadratic complexity.

### Design Philosophy

The core idea of LeanVAE is not to prune a heavy architecture, but to design an extremely lightweight Video VAE from scratch, leveraging classical signal processing tools (wavelet transforms and compressed sensing) to compensate for the reduced model capacity.

## Method

### Overall Architecture

The input video $\mathbf{x} \in \mathbb{R}^{(T+1) \times H \times W \times 3}$ is compressed into a latent representation $\mathbf{z} \in \mathbb{R}^{(T'+1) \times H' \times W' \times d}$ with temporal compression ratio $c_t = 4$ and spatial compression ratio $c_s = 8$.

### 1. Patchify: Frequency-Domain Non-Overlapping Patching

Unlike standard ViT which patches directly in RGB space, LeanVAE first applies a **Haar wavelet transform** followed by non-overlapping patching:

- Subsequent frames $\mathbf{x_{1:T}}$ undergo 3D Haar DWT → low-frequency component LC ($T/2 \times H/2 \times W/2 \times 3$) + high-frequency component HC ($T/2 \times H/2 \times W/2 \times 21$)
- The first frame $\mathbf{x_0}$ undergoes 2D Haar DWT (supporting joint image–video encoding)
- Linear layers project these into low-frequency embeddings $\mathbf{p^L}$ (384-dim) and high-frequency embeddings $\mathbf{p^H}$ (128-dim)

Three distinctions from standard ViT: ① the first frame and subsequent frames are processed separately (enabling joint image + video modeling); ② operations are performed in the frequency domain rather than RGB space; ③ **no patch normalization** is applied (LayerNorm was found to cause blocky artifacts, reducing PSNR by 3.27 dB).

### 2. Encoder/Decoder: NAF Backbone

The core module is the **Neighborhood-Aware Feedforward with Residual Connection (ResNAF)**:

$$\text{ResNAF}(x) = x + \text{FFN}(\text{DWConv}_{3\times3\times3}(x))$$

- 3D depthwise separable convolutions aggregate neighborhood context
- A feedforward layer performs feature transformation
- Residual connections facilitate gradient propagation

The encoder adopts a separate-then-fuse structure:

$$\mathbf{p} = \xi_f(\text{cat}(\xi_l(\mathbf{p^L}), \xi_h(\mathbf{p^H})))$$

where $\xi_l, \xi_h$ each consist of 2 ResNAF layers and $\xi_f$ consists of 4 ResNAF layers. **Causal padding** is applied along the temporal dimension to ensure each frame only attends to previous frames.

### 3. Channel Compression Bottleneck: ISTA-Net+ Compressed Sensing

**This represents the first application of compressed sensing to the channel compression bottleneck in Video VAEs.** A sensing matrix $\Phi \in \mathbb{R}^{d \times D}$ compresses features from $D=512$ to $d \in \{4, 16\}$ dimensions. Recovery is performed via the ISTA-Net+ iterative unrolling algorithm:

$$\mathbf{r}^{(k)} = \mathbf{p}^{(k-1)} - \rho^{(k)} \Phi^\top(\Phi \mathbf{p}^{(k-1)} - \mathbf{z})$$

$$\mathbf{p}^{(k)} = \mathbf{r}^{(k)} + \tilde{\mathcal{F}}^{(k)}(\text{soft}(\mathcal{F}^{(k)}(\mathbf{r}^{(k)}), \theta^{(k)}))$$

The forward/backward networks $\mathcal{F}^{(k)}, \tilde{\mathcal{F}}^{(k)}$ each use 2 NAF layers, with $K=2$ iterations.

### 4. Training Objective

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{lpips}} \mathcal{L}_{\text{lpips}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}} + \lambda_{KL} \mathcal{L}_{KL}$$

RGB + frequency-domain L1 reconstruction loss + VGG perceptual loss + PatchGAN adversarial loss + KL regularization. Training proceeds in two stages: 600K steps without GAN, followed by 100K steps with GAN.

## Key Experimental Results

### Main Results: Video Reconstruction Performance

| Method | Params | Channels | DAVIS PSNR↑ | DAVIS LPIPS↓ | DAVIS rFVD↓ | TokenBench PSNR↑ | TokenBench LPIPS↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CV-VAE | 182M | 4 | 25.75 | 0.1464 | 598.55 | 30.37 | 0.0706 |
| OD-VAE | 239M | 4 | 26.16 | 0.1173 | 407.20 | 30.47 | 0.0618 |
| VidTok | 157M | 4 | 26.50 | 0.1098 | 358.28 | 31.38 | 0.0526 |
| **LeanVAE** | **40M** | 4 | 26.04 | **0.0899** | **322.46** | 31.12 | **0.0432** |
| WF-VAE | 316M | 16 | 29.62 | 0.0628 | 149.27 | 35.11 | 0.0222 |
| VidTok | 157M | 16 | **31.06** | **0.0436** | **103.79** | **36.12** | **0.0166** |
| **LeanVAE** | **40M** | 16 | 30.15 | 0.0461 | 119.48 | 35.71 | 0.0173 |

Key finding: With only 40M parameters, LeanVAE achieves the **best LPIPS and rFVD** in the 4-channel setting, and approaches VidTok in the 16-channel setting using only 1/4 of its parameters.

### Efficiency Comparison (vs. VidTok)

- **FLOPs**: reduced by **50×** (at 768² resolution)
- **Inference speed**: **8–44× faster** (VidTok requires 20.26 seconds for 17 frames at 768²; LeanVAE requires only 0.46 seconds)
- **Memory**: can process 17 frames of 1080p video on a single A40 GPU (~15 GB FP16)

### Video Generation Experiment

| Method | Throughput | SkyTimelapse FVD↓ | UCF101 FVD↓ |
|------|:---:|:---:|:---:|
| Latte baseline (4 chn) | 1.60 | 59.82 | 477.97 |
| **Latte+LeanVAE (4 chn)** | **6.64** | **49.59** | **164.45** |

Training throughput improves by **315%** (supporting 4× larger batches), with simultaneous improvements in generation quality.

### Ablation Study

| Ablation | PSNR | LPIPS | rFVD |
|--------|:---:|:---:|:---:|
| Variant 2 (separate then fuse) ✓ | **26.18** | 0.145 | 470.64 |
| AE bottleneck (vs. CS) | 25.79 | 0.163 | 535.18 |
| With patch normalization | 22.91 | 0.158 | 599.38 |

Compressed sensing outperforms the AE bottleneck by 0.39 dB PSNR; patch normalization causes a 3.27 dB drop.

## Highlights & Insights

1. **First application of compressed sensing in Video VAE**: replacing a simple AE bottleneck with ISTA-Net+ yields significant performance gains.
2. **Patch normalization causes blocky artifacts**: this finding may inform improvements to related models in low-level vision tasks.
3. **Extreme parameter efficiency**: 40M parameters match or exceed competitors in the 150M–300M range; the causal design supports joint image–video modeling.
4. **Practical significance**: VAE encoding speed directly affects training throughput in LVDM training; LeanVAE can substantially accelerate large-scale video generation model training.

## Limitations & Future Work

- The 16-channel latent space yields better reconstruction quality but leads to degraded diffusion generation (FVD increases by 45), indicating that diffusion training with high-channel latent spaces remains an open problem.
- Only the 4×8×8 compression ratio has been evaluated; higher compression ratios remain unexplored.
- End-to-end evaluation on text-to-video generation has not been conducted.

## Related Work & Insights

- **Standard Video VAEs**: OD-VAE, CV-VAE, CogVideoX VAE, WF-VAE, VidTok, Cosmos Tokenizer, etc.
- **Efficiency directions**: factorized 3D convolutions (Open-Sora), wavelet transforms (WF-VAE), ViT-based architectures (OmniTokenizer, ViTok)
- **Compressed sensing**: ISTA-Net+ algorithm unrolling framework

## Rating

| Dimension | Score |
|------|:----:|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models](efficientmt_efficient_temporal_adaptation_for_motion_transfer_in_text-to-video_d.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](dive_taming_dino_for_subject-driven_video_editing.md)

</div>

<!-- RELATED:END -->
