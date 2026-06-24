---
title: >-
  [Paper Note] LALIC: Linear Attention Modeling for Learned Image Compression
description: >-
  [CVPR 2025][Model Compression][learned image compression] This work introduces the RWKV linear attention mechanism to learned image compression for the first time. It designs a Bi-RWKV transform block to achieve global receptive field feature extraction with linear complexity. Combined with an RWKV spatial-channel-temporal context entropy model, it outperforms VTM-9.1 by 15.26% BD-rate with relatively low complexity.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "learned image compression"
  - "linear attention"
  - "RWKV"
  - "Bi-RWKV"
  - "entropy modeling"
  - "rate-distortion"
date: 2026-05-08
content_hash: f2bf3e4669f93668
---

# LALIC: Linear Attention Modeling for Learned Image Compression

**Conference**: CVPR 2025  
**arXiv**: [2502.05741](https://arxiv.org/abs/2502.05741)  
**Code**: [sjtu-medialab/RwkvCompress](https://github.com/sjtu-medialab/RwkvCompress)  
**Area**: Model Compression  
**Keywords**: learned image compression, linear attention, RWKV, Bi-RWKV, entropy modeling, rate-distortion

## TL;DR

This work introduces the RWKV linear attention mechanism to learned image compression for the first time. It designs a Bi-RWKV transform block to achieve global receptive field feature extraction with linear complexity. Combined with an RWKV spatial-channel-temporal context entropy model, it outperforms VTM-9.1 by 15.26% BD-rate with relatively low complexity.

## Background & Motivation

**Background**: Learned image compression (LIC) has outperformed traditional codecs (JPEG, VVC), mainly relying on non-linear transform networks and learnable entropy models. Transformers have become the mainstream backbone, but their quadratic complexity limits high-resolution image processing.

**Limitations of Prior Work**: Swin-Transformer approaches rely on window partitioning strategies to approximate global attention, which limits the receptive field. Linear attention models like Mamba have been used in NLP but remain insufficiently explored in image compression. Every gain in coding efficiency is accompanied by a significant increase in complexity.

**Key Challenge**: Efficient global dependency modeling vs. acceptable computational complexity.

**Key Insight**: Leveraging the linear attention characteristics of RWKV to achieve a true global receptive field while maintaining linear computational complexity.

**Core Idea**: Replacing Transformers/CNNs with Bi-RWKV (bidirectional WKV attention + Omni-Shift local convolution) as the fundamental building block for both transform and entropy models.

## Method

### Overall Architecture

Following the standard non-linear transform coding framework:
1. Analysis transform $g_a$ encodes image $x$ into latent representation $y$.
2. Hyper-prior encoder $h_a$ extracts hyper-latent representation $z$.
3. The RWKV-SCCTX entropy model estimates the conditional details of the Gaussian distribution parameters of $y$.
4. Synthesis transform $g_s$ reconstructs image $\hat{x}$ from the quantized latent representation $\hat{y}$.
5. Bi-RWKV blocks are utilized to replace traditional Transformer blocks in all transform networks.

### Key Designs

**1. Bi-RWKV Transform Block**
- **Function**: Serves as the fundamental feature extraction module in $g_a$, $g_s$, $h_a$, and $h_s$; each block contains two branches: Spatial Mix and Channel Mix.
- **Mechanism**: Spatial Mix captures global spatial dependencies with linear complexity via BiWKV attention; Channel Mix implicitly constructs an MLP using squared ReLU to achieve channel mixing. Omni-Shift (reparameterized 5×5 depthwise convolution) is used to capture 2D local context.
- **Design Motivation**: BiWKV introduces a channel-level decay parameter $w$ and a current-token boost parameter $u$ to automatically balance local and global dependencies based on distance. Effective Receptive Field (ERF) visualization confirms that the RWKV block achieves a true global receptive field (outperforming TCM's window pattern and FAT's local enhancement pattern).

**2. RWKV Spatial-Channel-Temporal Context Entropy Model (RWKV-SCCTX)**
- **Function**: Jointly models the redundancy of latent representation $y$ in spatial and channel dimensions.
- **Mechanism**: For the spatial dimension, a checkerboard masked convolution is employed to split the latents into anchor and non-anchor groups. For the channel dimension, the 320 channels are divided into 5 chunks (16, 16, 32, 64, and the remaining), using Bi-RWKV blocks to model the channel context between decoded chunks and the current chunk. The channel context uses the Channel Mix module without Omni-Shift to maintain a 1×1 receptive field to satisfy causal decoding.
- **Design Motivation**: The first few chunks have fewer channels but are frequently referenced by subsequent chunks, carrying most of the critical information. The global modeling capability of RWKV makes it superior to pure convolutional schemes in channel context modeling.

**3. BiWKV Attention Mechanism**
- **Function**: Adds bidirectional distance decay and current-token boosting on top of the KV linear attention of AFT (Attention-Free Transformer).
- **Mechanism**: $wkv_t = \frac{\sum_{i \neq t} e^{-(|t-i|-1)/T \cdot w + k_i} v_i + e^{u+k_t} v_t}{\sum_{i \neq t} e^{-(|t-i|-1)/T \cdot w + k_i} + e^{u+k_t}}$, and the final output is modulated by sigmoid-gated receptance.
- **Design Motivation**: Bidirectional processing adapts to the non-causal nature of 2D images. Distance decay assigns higher weights to neighboring tokens, balancing local precision and global modeling.

### Loss & Training

- Rate-distortion loss: $L = \lambda \|x - \hat{x}\|^2 + R(\hat{z}) + R(\hat{y})$
- $\lambda \in \{0.0025, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483\}$ (MSE optimized)
- Adam optimizer, initial learning rate $10^{-4}$ decaying to $10^{-5}$, fine-tuned on 512×512 crops
- Training set: First 400K images from OpenImages, RTX 4090 GPU

## Key Experimental Results

### Main Results — BD-rate (PSNR, anchored against VTM-9.1)

| Method | Decoding Time (s) | FLOPs (G) | Params (M) | Kodak | CLIC | Tecnick |
|---|---|---|---|---|---|---|
| ELIC | 0.120 | 332 | 33.3 | -7.02% | -1.19% | -7.64% |
| MambaVC | 0.222 | 393 | 47.9 | -9.73% | - | - |
| TCM-large | 0.151 | 701 | 75.9 | -11.73% | -9.41% | -10.93% |
| FAT | >10.0 | 245 | 69.8 | -14.56% | -10.79% | -14.40% |
| MLIC++ | 0.268 | 443 | 83.3 | -15.02% | -14.45% | -17.21% |
| **LALIC (Ours)** | **0.150** | **286** | **63.2** | **-15.26%** | **-15.41%** | **-17.63%** |

### Ablation Study

| Configuration | FLOPs (G) | Params (M) | BD-rate |
|---|---|---|---|
| 2,2,2,2 + Conv SCCTX | 164 | 27.6 | 0.00% |
| 2,4,6,6 + Conv SCCTX | 239 | 42.6 | -1.68% |
| 2,4,6,6 + Conv Plus SCCTX | 304 | 62.1 | -2.74% |
| 2,4,6,6 + **RWKV SCCTX** | 286 | 63.2 | **-3.50%** |

Ablation of attention mechanism:

| Attention | ΔFLOPs (G) | Loss |
|---|---|---|
| AFT | 0.60 | 0.5657 |
| AFT + Shift | 4.91 | 0.5604 |
| BiWKV + Shift | 6.80 | **0.5551** |

### Key Findings

1. **LALIC achieves the fastest decoding among methods with >10% savings using the fewest parameters**: 150ms decoding time + 63.2M parameters.
2. **RWKV-SCCTX outperforms Conv Plus SCCTX**: Additional 0.76% reduction in BD-rate with almost identical parameter count and lower FLOPs.
3. **Distinct advantage on high resolution**: Gains on CLIC (2K) and Tecnick (1K) are larger than on Kodak (768×512), validating the benefit of global modeling for high-resolution images.
4. **ERF analysis reveals that RWKV achieves a true global receptive field** and reduces local correlation in the latent representations.

## Highlights & Insights

- First work to successfully apply RWKV linear attention to learned image compression.
- ERF visualization intuitively illustrates the receptive field discrepancies among RWKV, Transformer, and CNN.
- Ingeniously removes Omni-Shift in the entropy model to maintain causality, demonstrating a deep understanding of the decoding process.
- Encoding/decoding latency is highly competitive among SOTA methods, indicating strong practical deployability.

## Limitations & Future Work

- The encoding time (274ms) is relatively long; more lightweight analysis transforms can be explored.
- Only RWKV linear attention is explored, lacking systematic comparison with other architectures like Mamba or RetNet.
- Consider replacing squared ReLU in Channel Mix with other activation functions.
- The potential application of RWKV in video compression remains unexplored.
- End-to-end MS-SSIM optimized results are not analyzed in depth.

## Related Work & Insights

- MambaVC first introduced SSMs to image compression, following a linear attention route; LALIC demonstrates that RWKV is superior for this task.
- TCM and FAT represent window-based Transformer and frequency-aware Transformer paradigms, respectively; RWKV outperforms both with lower complexity.
- Insights: The advantages of linear attention in low-level vision tasks could potentially generalize to pixel-level tasks such as super-resolution and denoising.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First application of RWKV to image compression; the designs of Bi-RWKV blocks and RWKV-SCCTX are sound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validation on three datasets + detailed ablation studies + ERF/correlation visualization.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich visualizations.
- **Value**: ⭐⭐⭐⭐ Achieves a new SOTA in the efficiency-performance trade-off, presenting strong practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](learned_image_compression_with_dictionary-based_entropy_model.md)
- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](../../ICCV2025/model_compression/learned_image_compression_with_hierarchical_progressive_context_modeling.md)
- [\[CVPR 2025\] MambaIC: State Space Models for High-Performance Learned Image Compression](mambaic_state_space_models_for_high-performance_learned_image_compression.md)
- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](../../NeurIPS2025/model_compression/linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [\[CVPR 2026\] What Matters in Practical Learned Image Compression](../../CVPR2026/model_compression/what_matters_in_practical_learned_image_compression.md)

</div>

<!-- RELATED:END -->
