---
title: >-
  [Paper Note] TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation
description: >-
  [ICCV 2025][Image Generation][Video Frame Interpolation] This paper proposes TLB-VFI, an efficient video diffusion model for frame interpolation. It employs a temporal-aware autoencoder—comprising a latent-space temporal…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Video Frame Interpolation"
  - "Brownian Bridge Diffusion"
  - "Temporal-Aware Autoencoder"
  - "3D Wavelet"
  - "Optical Flow Guidance"
date: 2026-05-08
content_hash: a8adda6d41919c6f
---

# TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation

**Conference**: ICCV 2025
**arXiv**: [2507.04984](https://arxiv.org/abs/2507.04984)
**Code**: [Project Page](https://github.com/)
**Area**: Video Frame Interpolation / Diffusion Models
**Keywords**: Video Frame Interpolation, Brownian Bridge Diffusion, Temporal-Aware Autoencoder, 3D Wavelet, Optical Flow Guidance

## TL;DR

This paper proposes TLB-VFI, an efficient video diffusion model for frame interpolation. It employs a temporal-aware autoencoder—comprising a latent-space temporal block and a pixel-space 3D wavelet gating mechanism—to extract rich temporal information, combined with a redesigned Brownian bridge diffusion process. With only 46.7M parameters (3× fewer than image diffusion methods and 20× fewer than video diffusion methods), TLB-VFI achieves approximately 20% FID improvement on SNU-FILM extreme and Xiph-4K benchmarks.

## Background & Motivation

### Problem Definition

Video frame interpolation (VFI) aims to predict an intermediate frame $I_n$ given two adjacent video frames $I_0$ and $I_1$. The core challenges are: (1) handling large motion variations; (2) maintaining temporal consistency; and (3) generating high-quality, perceptually natural intermediate frames.

### Limitations of Prior Work

**Traditional methods and image diffusion methods (LDMVFI, Consec.BB)**: These methods extract only spatial information and lack explicit temporal feature extraction. Image diffusion methods are less efficient than traditional approaches and cannot model inter-frame temporal relationships.

**Video diffusion methods (VIDIM, Dreammover, ViBiDSampler)**: Although capable of extracting temporal information, they suffer from:
   - Requirements for over 10 million training videos (vs. the standard VFI training set of only 51K triplets)
   - Massive model sizes (>943M parameters)
   - Extremely slow inference (8.48–52.55 seconds per frame)
   - Lack of pixel-level guidance such as optical flow

**Consecutive Brownian Bridge**: Applying the Brownian bridge between adjacent frames causes the process to degenerate into an identity mapping due to high feature similarity between neighboring frames, rendering the diffusion process ineffective.

### Root Cause

A **video diffusion model** is needed to extract temporal information (where a 3D UNet progressively constructs temporal relationships during sampling), while keeping training scale, model size, and inference time within practical bounds. The key lies in: (1) redesigning the autoencoder to extract temporal information in both latent and pixel space; and (2) leveraging optical flow guidance to reduce training data requirements.

## Method

### Overall Architecture

The method consists of two components: (1) a **temporal-aware autoencoder**—comprising an image encoder, temporal block, image decoder, and 3D wavelet gating—that predicts a mask $M$ and residual $\Delta$; and (2) a **Brownian bridge diffusion model** that operates in the latent space to align the distributional gap between $\mathcal{E}(V)$ and $\mathcal{E}(\tilde{V})$. The final output is:
$$\hat{I}_n = M \odot \text{warp}(I_0) + (1-M) \odot \text{warp}(I_1) + \Delta$$

### Key Designs

#### 1. **Latent-Space Temporal Feature Extraction**

- **Function**: Temporal blocks are inserted between the encoder and decoder of the autoencoder to extract inter-frame temporal relationships in the latent space.
- **Mechanism**: A shared image encoder encodes each frame independently to obtain spatial features. 3D convolutions and spatiotemporal attention then extract temporal relationships in the latent space. On the decoder side, spatiotemporal cross-attention aggregation transforms video features ($F \in \mathbb{R}^{C \times T \times H \times W}$) into single-frame features:
  $$V_{out} = \text{softmax}\left(\frac{QK^T}{\sqrt{C}}\right)V$$
  where $F_Q = F[t].\text{flatten}(1)$ (intermediate frame features) and $F_{KV} = F.\text{flatten}(1)$ (all frame features). Multi-scale features from the $I_0$ and $I_1$ encoders are additionally incorporated via warp and cross-attention to guide the decoder.
- **Design Motivation**: The key reason for decoupling temporal extraction from spatial encoding is that during inference, $I_n$ is replaced by a zero matrix ($\tilde{V}=[I_0,0,I_1]$). If the encoder itself were temporally aware, its multi-scale features would contain incomplete information due to the zero substitution, degrading decoding performance. The shared image encoder ensures that the multi-scale features of $I_0$ and $I_1$ remain unaffected.

#### 2. **3D Wavelet Feature Gating**

- **Function**: A 3D wavelet transform extracts temporally high-frequency information (motion-varying regions) in pixel space, serving as a complement to latent-space temporal features.
- **Mechanism**: A 3D wavelet transform is applied to the input video $V=[I_0,I_n,I_1]$, using a low-pass filter $[\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}]$ and a high-pass filter $[-\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}]$ to extract 8 frequency maps across combinations of height, width, and time dimensions. Two wavelet decompositions capture: (1) temporal information between $I_0$–$I_n$ and $I_n$–$I_1$; and (2) global temporal information across all frames. The encoded frequency features $f_w$ are fused via a gating mechanism:
  $$f = \sigma(f_w) \odot f_i + f_i$$
  where $\sigma$ denotes sigmoid and $f_i$ denotes the latent features from the image encoder.
- **Design Motivation**: The latent space is highly compressed, making latent-space temporal extraction alone insufficient. Pixel-level temporal information indicates which regions undergo greater motion variation, and the gating mechanism directs the model to focus on those regions.

#### 3. **Redesigned Brownian Bridge Diffusion Process**

- **Function**: The Brownian bridge is applied between $\mathcal{E}(V)$ (encoding of the complete video including $I_n$) and $\mathcal{E}(\tilde{V})$ (encoding of the video with $I_n$ replaced by zero), thereby avoiding the identity mapping problem.
- **Mechanism**: The BBDM forward process is defined as:
  $$q(\mathbf{x}_t | \mathbf{x}_0, \mathbf{x}_T) = \mathcal{N}\left(\frac{t}{T}\mathbf{x}_0 + (1-\frac{t}{T})\mathbf{x}_T, \frac{t(T-t)}{T}\mathbf{I}\right)$$
  where $\mathbf{x}_0 = \mathcal{E}(V)$ and $\mathbf{x}_T = \mathcal{E}(\tilde{V})$. During sampling, the denoising network predicts $\mathbf{x}_t - \mathbf{x}_0$.

  **Proposition 1**: A sufficient condition for effective Brownian bridge diffusion is a significant distributional shift between $\mathbb{E}(\mathbf{x}_0)$ and $\mathbb{E}(\mathbf{x}_T)$, which can be verified by rejecting $H_0: \mathbb{E}(\mathbf{x}_0 - \mathbf{x}_T)=0$ via a t-test.

  In the proposed design, the MAPE reaches 40–50% (vs. <1% in Consec.BB), and the t-statistic exceeds 21 (far above the threshold of 3.291 at the 0.001 significance level).
- **Design Motivation**: Consec.BB applies the Brownian bridge between nearly identical adjacent frame features, causing it to degenerate into an identity mapping. By replacing $I_n$ with zeros, this work introduces a significant distributional difference, enabling the Brownian bridge to genuinely perform information recovery.

### Loss & Training

- Autoencoder: L1 reconstruction loss + perceptual loss
- Diffusion model: MSE loss on the prediction of $\mathbf{x}_t - \mathbf{x}_0$
- Training data: Vimeo 90K triplets (51K)
- Data augmentation: random flipping, cropping, rotation, and temporal order reversal
- Inference: 10-step diffusion sampling

## Key Experimental Results

### Main Results

**LPIPS↓ / FloLPIPS↓ / FID↓ Main Results**:

| Method | Params | Xiph-4K | SNU-FILM extreme | Runtime (s/frame) |
|--------|--------|---------|-------------------|-------------------|
| LDMVFI | 439.0M | OOM | 0.123/0.204/47.04 | 2.48 |
| Consec.BB | 146.4M | 0.097/0.135/24.42 | 0.104/0.184/36.63 | 1.62 |
| PerVFI* | — | 0.086/0.128/18.85 | 0.090/0.151/32.37 | 1.52 |
| **Ours** | **46.7M** | **0.077/0.113/19.11** | **0.095/0.151/29.87** | **0.69** |

*PerVFI is trained on approximately 2× the data of other methods (including high-resolution data) and is marked in gray to exclude it from ranking.

### Ablation Study

**Ablation on Temporal-Aware Design Components (FID↓)**:

| Configuration | Xiph-4K | Xiph-2K | SNU-FILM extreme |
|---------------|---------|---------|-------------------|
| Full model | **19.114** | **9.901** | **29.868** |
| − 3D Wavelet | 19.247 | 10.092 | 30.717 |
| − Cross-attn aggregation | 19.663 | 10.499 | 30.903 |
| − Temporal attention | 19.944 | 10.911 | 32.061 |
| − 3D Convolution (→2D) | 23.481 | 12.679 | 33.155 |
| Ours† (temporal encoder) | 22.731 | 13.410 | 34.982 |

### Key Findings

1. **3D convolution is the most critical component**: Its removal causes the largest FID degradation (19.11→23.48 on Xiph-4K), as it forms the foundation of latent-space temporal feature extraction.
2. **The shared image encoder is indispensable**: Replacing it with a temporal encoder (Ours†) leads to a substantial performance drop, validating the necessity of keeping multi-scale features unaffected by zero substitution.
3. **3D wavelet gating provides complementary pixel-level information**: Although its individual contribution is modest (FID degradation of only ~0.1), its effect is more pronounced on difficult examples.
4. **Distributional shift is critical for Brownian bridge effectiveness**: The t-statistic increases from ~0.0001 in Consec.BB to over 21 in the proposed method, validating Proposition 1.
5. **PSNR/SSIM are misaligned with perceptual quality**: The paper demonstrates cases where EMAVFI exhibits visible visual artifacts yet achieves higher PSNR/SSIM, underscoring the importance of using LPIPS/FloLPIPS/FID for evaluation.

## Highlights & Insights

1. **Exceptional training efficiency**: Only 51K triplets are used (vs. >10M videos for video diffusion methods), with 46.7M parameters (vs. >1B for VIDIM), reducing training costs by 3–4 orders of magnitude.
2. **Theoretical contribution**: Proposition 1 provides a sufficient condition for the effectiveness of Brownian bridge diffusion, explaining the degeneration observed in Consec.BB.
3. **2.3× inference speedup**: Faster than comparable diffusion methods at the same number of sampling steps (0.69s vs. 1.62s).
4. **Greater advantage on difficult samples**: Performance gains increase progressively from SNU-FILM easy to extreme, demonstrating that temporal information is especially critical for large-motion scenarios.

## Limitations & Future Work

1. **Inference still requires 10 diffusion steps**: Although faster than comparable methods, it remains slower than non-diffusion approaches (e.g., IFRNet at 0.10s).
2. **Only three-frame interpolation is evaluated**: Results for continuous multi-frame interpolation (e.g., 8× frame rate upsampling) are presented in supplementary materials but not analyzed in depth.
3. **Dependence on optical flow estimation quality**: Whether the residual term $\Delta$ can fully compensate when optical flow estimation fails (e.g., in occluded or transparent regions) remains unclear.
4. **No evaluation on non-natural video**: Performance on synthetic content such as animation and video game footage is unknown.
5. **Computational overhead of 3D wavelet**: Although lightweight, two wavelet decompositions still add processing cost in pixel space.

## Related Work & Insights

- **Vs. Consec.BB**: Consec.BB applies the Brownian bridge between adjacent frame features (≈ identity mapping), whereas this work applies it between complete and zero-substituted video encodings (large distributional shift).
- **Vs. LDMVFI**: LDMVFI uses a kernel-based method to condition the autoencoder and employs image-based diffusion; this work uses optical flow guidance and video-based diffusion.
- **Broader implication**: The design principle of decoupling spatial encoding from temporal extraction is generalizable to tasks such as video inpainting and video deblurring.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Redesigning both the Brownian bridge endpoints and the temporal-aware autoencoder is innovative, and Proposition 1 constitutes a meaningful theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across multiple datasets and metrics (LPIPS/FloLPIPS/FID), with runtime, training cost comparisons, and qualitative analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear, motivation is logically coherent, and figures are informative.
- **Value**: ⭐⭐⭐⭐⭐ — Achieves state-of-the-art perceptual quality while substantially reducing both training and inference costs, offering high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](../../NeurIPS2025/image_generation/towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)
- [\[ICCV 2025\] Video Motion Graphs](video_motion_graphs.md)
- [\[ICCV 2025\] MaskControl: Spatio-Temporal Control for Masked Motion Synthesis](maskcontrol_spatio-temporal_control_for_masked_motion_synthesis.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)

</div>

<!-- RELATED:END -->
