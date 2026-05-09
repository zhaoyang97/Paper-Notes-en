---
title: >-
  [Paper Note] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression
description: >-
  [CVPR 2026][Model Compression][Vector Quantization] RDVQ introduces a differentiable relaxation over the codebook distribution, enabling for the first time end-to-end rate-distortion joint optimization for VQ-based image compression. At extremely low bitrates, the method achieves superior or competitive perceptual quality with less than 20% of the parameters of prior approaches.
tags:
  - CVPR 2026
  - Model Compression
  - Vector Quantization
  - Rate-Distortion Optimization
  - Generative Image Compression
  - Entropy Model
  - Differentiable Relaxation
date: 2026-05-08
content_hash: 9533f9d4402e808e
---

# RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression

**Conference**: CVPR 2026
**arXiv**: [2604.10546](https://arxiv.org/abs/2604.10546)
**Code**: [https://github.com/CVL-UESTC/RDVQ](https://github.com/CVL-UESTC/RDVQ)
**Area**: Image Compression / Restoration
**Keywords**: Vector Quantization, Rate-Distortion Optimization, Generative Image Compression, Entropy Model, Differentiable Relaxation

## TL;DR
RDVQ introduces a differentiable relaxation over the codebook distribution, enabling for the first time end-to-end rate-distortion joint optimization for VQ-based image compression. At extremely low bitrates, the method achieves superior or competitive perceptual quality with less than 20% of the parameters of prior approaches.

## Background & Motivation

**Background**: Learned image compression predominantly relies on scalar quantization (SQ), where differentiable approximations (e.g., additive noise or STE) allow gradient backpropagation to the encoder, enabling end-to-end rate-distortion optimization. Vector quantization (VQ) preserves richer structural information and perceptual quality, making it particularly suitable for extremely low bitrates.

**Limitations of Prior Work**: The discrete nearest-neighbor assignment in VQ blocks gradient flow from the rate loss to the encoder. The encoder-induced implicit prior distribution cannot be directly optimized by the rate objective, leading to a fundamental decoupling between representation learning and the entropy model.

**Key Challenge**: While VQ offers advantages in reconstruction quality, it cannot support end-to-end rate-distortion joint optimization as SQ does; bitrate control must rely on heuristics such as codebook size adjustment and selective transmission.

**Goal**: Restore a differentiable gradient path from the rate objective to the encoder in VQ-based compression, achieving true end-to-end rate-distortion optimization.

**Key Insight**: Replace hard nearest-neighbor assignment with a distance-aware soft distribution, used exclusively in the rate estimation branch, while reconstruction continues to use standard hard quantization.

**Core Idea**: During training, a softmax-relaxed codebook distribution is used to estimate the rate, enabling rate gradients to flow to the encoder; at inference, the system reverts to standard hard VQ, maintaining full compatibility.

## Method

### Overall Architecture
The analysis transform $g_a$ extracts multi-scale features → flattened into a sequence → the VQ module produces hard-quantized embeddings (for reconstruction), discrete indices (for coding), and a relaxed distribution (used only for rate estimation during training) → the synthesis transform $g_s$ reconstructs the image. The entropy model is a Masked Transformer that autoregressively predicts conditional probabilities over the relaxed distribution.

### Key Designs

1. **Differentiable Soft Relaxation**:

    - **Function**: Restores the gradient path from the rate objective to the encoder.
    - **Mechanism**: The distance $d_{b,l,k}$ between the encoder output and each codeword is computed, and a temperature-scaled softmax yields the relaxed distribution $p_{\text{soft}}(b,l,k) = \text{softmax}_k(-d_{b,l,k}/\tau)$. During training, the rate objective is computed as a cross-entropy over this continuous distribution, while reconstruction still uses hard quantization.
    - **Design Motivation**: Introducing relaxation only in the rate estimation branch leaves the reconstruction and inference pipelines unchanged, ensuring training–inference consistency.

2. **Dependency-Aware Autoregressive Entropy Model**:

    - **Function**: Accurately models the conditional probability distribution over codebook indices.
    - **Mechanism**: Multi-scale features are organized spatially and hierarchically into a unified sequence; a dependency-aware ordering vector $o$ is constructed, and masked attention $M = (o > o^\top)$ enables autoregressive factorization under parallel training. Coarse scales are encoded first, and fine scales are conditioned on coarse scales.
    - **Design Motivation**: The multi-scale structure naturally exhibits hierarchical dependencies; dependency-aware ordering captures these relationships more effectively than simple raster-scan ordering.

3. **Test-Time Bitrate Adjustment**:

    - **Function**: Enables bitrate control within a limited range without retraining.
    - **Mechanism**: A prefix of the index sequence is transmitted, and the autoregressive entropy model completes the remaining indices. Joint rate-distortion optimization renders the latent space highly predictable, so quality degradation from prefix completion is smooth.
    - **Design Motivation**: Practical deployment requires flexible bitrate control; prefix transmission combined with autoregressive completion provides an elegant solution.

### Loss & Training
Three-stage training: (1) pretraining the autoencoder and codebook (reconstruction loss); (2) pretraining the entropy model (rate objective); (3) joint fine-tuning of the full model (rate + distortion), followed by adaptation on high-resolution data. The loss comprises a GAN loss, LPIPS perceptual loss, and a relaxed cross-entropy rate loss.

## Key Experimental Results

### Main Results

| Dataset | Metric | RDVQ | RDEIC | BD-Rate Savings |
|--------|------|------|-------|---------|
| DIV2K-val | DISTS | Best | 2nd | -75.71% |
| DIV2K-val | LPIPS | Best | 2nd | -37.63% |
| Kodak | DISTS | SOTA | - | - |
| CLIC2020 | CLIPIQA | SOTA | - | - |

### Ablation Study

| Configuration | bpp | DISTS | LPIPS | FID |
|------|-----|-------|-------|-----|
| RDVQ (full) | 0.0247 | 0.1005 | 0.2321 | 19.96 |
| w/o Relaxation | 0.0464 | 0.2147 | 0.5031 | 86.93 |
| K-means VQ | 0.0247 | 0.1253 | 0.2831 | 28.08 |

### Key Findings
- Removing the differentiable relaxation causes a sharp performance drop; even at higher bitrates, the degraded variant falls far short of the full model, confirming that relaxation is the cornerstone of end-to-end rate-distortion optimization.
- K-means-based bitrate control yields noticeably inferior quality at the same bitrate compared to RDVQ, demonstrating that heuristic methods cannot eliminate redundancy in the index distribution.
- As the bitrate decreases, encoder features become progressively smoother and codebook utilization more concentrated, indicating that the model autonomously learns an adaptive compression strategy.

## Highlights & Insights
- **Elegant Separation of Relaxation**: The relaxation is applied exclusively in the rate estimation branch during training; the reconstruction path always uses hard quantization, requiring no modification at inference. This dual-path design simultaneously resolves the gradient issue and maintains deployment compatibility.
- **Unified Perspective on Image Tokenization and Compression**: Existing VQ tokenizers can be converted into compression models by introducing entropy constraints, and conversely, compression objectives can improve tokenizer efficiency.

## Limitations & Future Work
- Test-time bitrate adjustment is limited in range (0.02–0.32 bpp); quality degrades noticeably outside this range.
- At 251.9M parameters, the model is significantly smaller than baselines but cannot be considered lightweight.
- Future work may explore applying this framework to entropy-aware training of visual tokenizers.

## Related Work & Insights
- **vs. OSCAR/RDEIC**: These diffusion- or large-model-prior-based methods require substantially more parameters; RDVQ is trained from scratch with less than 20% of their parameter count.
- **vs. DLF**: This dual-branch SQ+VQ hybrid approach fundamentally cannot perform rate-distortion optimization on the VQ branch.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to achieve end-to-end rate-distortion optimization for VQ, with a clear theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, multiple metrics, with comprehensive ablations and analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is precise and mathematical derivations are clearly presented.
- Value: ⭐⭐⭐⭐⭐ Significant implications for both VQ-based compression and image tokenization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness](../../AAAI2026/model_compression/reinforced_rate_control_for_neural_video_compression_via_inter-frame_rate-distor.md)
- [\[CVPR 2026\] Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression](parallax_to_align_them_all_an_omniparallax_attention_mechanism_for_distributed_m.md)
- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)
- [\[ICCV 2025\] DLF: Extreme Image Compression with Dual-generative Latent Fusion](../../ICCV2025/model_compression/dlf_extreme_image_compression_with_dual-generative_latent_fusion.md)
- [\[CVPR 2026\] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)

</div>

<!-- RELATED:END -->
