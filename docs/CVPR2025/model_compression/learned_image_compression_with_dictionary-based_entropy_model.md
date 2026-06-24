---
title: >-
  [Paper Note] Learned Image Compression with Dictionary-based Entropy Model
description: >-
  [CVPR 2025][Model Compression][learned image compression] Propose a Dictionary-based Cross-Attention Entropy model (DCAE), introducing a learnable dictionary to extract typical texture structure priors of natural images from the training dataset. Through multi-scale feature aggregation and cross-attention, accurate probability distribution estimation is achieved. With an encoding/decoding speed of only 193ms, it achieves a BD-rate of -17.0%/-21.1%/-19.7% (Kodak/Tecnick/CLIC)…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "learned image compression"
  - "entropy model"
  - "dictionary learning"
  - "cross attention"
  - "rate-distortion"
date: 2026-05-08
content_hash: fb63878c9d18be61
---

# Learned Image Compression with Dictionary-based Entropy Model

**Conference**: CVPR 2025  
**arXiv**: [2504.00496](https://arxiv.org/abs/2504.00496)  
**Code**: [GitHub](https://github.com/LabShuHangGU/DCAE)  
**Area**: Model Compression  
**Keywords**: learned image compression, entropy model, dictionary learning, cross attention, rate-distortion

## TL;DR

Propose a Dictionary-based Cross-Attention Entropy model (DCAE), introducing a learnable dictionary to extract typical texture structure priors of natural images from the training dataset. Through multi-scale feature aggregation and cross-attention, accurate probability distribution estimation is achieved. With an encoding/decoding speed of only 193ms, it achieves a BD-rate of -17.0%/-21.1%/-19.7% (Kodak/Tecnick/CLIC), completely outperforming state-of-the-art (SOTA) methods.

## Background & Motivation

**Background**: Learned Image Compression (LIC) consists of two main components: nonlinear autoencoders and entropy models. The entropy model is responsible for estimating the probability distribution of latent representations to achieve efficient entropy coding, which is key to the compression ratio. Existing methods mainly exploit the **internal dependencies** of latent representations through hyper-prior and autoregressive architectures.

**Limitations of Prior Work**:
1. Existing entropy models only focus on spatial/channel dependencies within the latent representations, **ignoring the extraction of external priors** from the training data.
2. Training data contains rich natural image priors (typical textures and structures) that have been proven effective in the image restoration field but are underutilized in LIC.
3. Serial autoregressive context models are effective but suffer from high encoding/decoding latency (e.g., MLIC++ reaches 772ms).

**Design Motivation**: Natural images contain a large number of repeating local texture patterns (grids, stripes, edges, etc.). If a shared dictionary is used to store these patterns, the decoded partial information can be used to query the dictionary during the autoregressive prediction process to fill in the missing information, thereby significantly improving the accuracy of probability distribution estimation.

## Method

### Overall Architecture

Standard LIC pipeline: encoder $g_a$ $\rightarrow$ latent representation $\bm{y}$ $\rightarrow$ entropy model estimating distribution parameters $(\bm{\mu}, \bm{\sigma})$ $\rightarrow$ quantization + entropy coding $\rightarrow$ decoder $g_s$ reconstruction. The core contribution of this work is the introduction of a dictionary module into the entropy model.

### Key Designs

#### 1. Learnable Dictionary

- The dictionary $\bm{D} \in \mathbb{R}^{N \times C_d}$ ($N=128$ entries, $C_d=640$ dimensions) serves as learnable network parameters.
- It automatically learns to fit typical structures in natural images during training (analogous to traditional dictionary learning from image patches).
- **The encoder and decoder share the same dictionary**, requiring no extra bits for transmission.

#### 2. Multi-Scale Feature Aggregation Module (MSFA)

To make dictionary queries more precise, texture information needs to be extracted from different scales:
- Use multiple layers of efficient convolutions (two linear layers + 3×3 depthwise separable convolution) stacked in $m=3$ layers.
- Shallow layers capture fine-grained textures, while deep layers capture large-scale textures.
- After concatenation, weighting is applied through spatial attention: $\text{MSFA}(\bm{X}_i) = \text{SA}(\bm{X}_i^{merge}) \odot \bm{X}_i^{merge}$

#### 3. Dictionary-based Cross-Attention Module (DCA)

When autoregressively predicting the $i$-th slice:
- The decoded slices $\bar{\bm{y}}_{<i}$ and the hyper-prior feature $\mathcal{F}_z$ obtain query features $\bm{X}_{ms_i}$ via MSFA.
- **Query** = $\bm{X}_{ms_i} \bm{W}^Q$, **Key** = $\bm{D}\bm{W}^K$, **Value** = $\bm{D}$
- Cross-attention: $\bm{A}_i = \text{SoftMax}(\bm{Q}_i \bm{K}^T / \tau)$, $\mathcal{F}_{dict_i} = \bm{A}_i \bm{V}$
- $\tau$ is a learnable temperature parameter. The dictionary features $\mathcal{F}_{dict_i}$ are fed into the distribution estimation module along with internal features.

### Loss & Training

Standard Lagrangian R-D loss: $\mathcal{L} = \mathcal{R}(\hat{\bm{y}}) + \mathcal{R}(\hat{\bm{z}}) + \lambda \cdot \mathcal{D}(\bm{x}, \hat{\bm{x}})$

The distortion metric uses MSE, with $\lambda \in \{0.0018, ..., 0.0500\}$ corresponding to different compression rates.

## Key Experimental Results

### Main Results

**BD-rate (with VVC as the anchor, lower is better)**:

| Method | Kodak | Tecnick | CLIC | Latency (ms) |
|------|-------|---------|------|---------|
| ELIC (CVPR'22) | -7.1% | - | - | 210 |
| TCM (CVPR'23) | -11.8% | -12.0% | -12.0% | 293 |
| MLIC++ | -15.1% | -18.6% | -16.9% | **772** |
| FTIC (ICLR'24) | -14.6% | -15.1% | -13.6% | - |
| CCA (NeurIPS'24) | -13.7% | -15.3% | -14.5% | 223 |
| **Ours (DCAE)**| **-17.0%** | **-21.1%** | **-19.7%** | 193 |

The latency is only **1/4** of MLIC++, with superior BD-rate.

### Ablation Study

| Module | BD-rate | Latency (ms) |
|------|---------|---------|
| Baseline | -4.20% | 143 |
| + DCA | -7.28% | 153 |
| + DCA + MSFA | -8.50% | 160 |

**Ablation on Dictionary Size**:

| Number of Entries N | None | 64 | 128 | 192 | 256 |
|---------|-----|-----|------|------|------|
| BD-rate | -4.20% | -6.84% | **-7.28%** | -7.26% | -6.92% |

**Ablation on MSFA Convolutional Layers**:

| Number of Layers m | 0 | 1 | 2 | 3 | 4 |
|--------|-----|-----|-----|-----|-----|
| BD-rate | -7.28% | -7.62% | -8.04% | **-8.50%** | -8.36% |

### Key Findings

- DCA alone contributes a 3.08% improvement in BD-rate, with only a 10ms increase in latency.
- 128 dictionary entries are sufficient; further expansion leads to saturation or degradation.
- A 3-layer MSFA is optimal, indicating that multi-scale features are crucial for accurate dictionary queries.
- Visualization shows that the same dictionary entries are activated in **similar texture regions** across different images, validating that the dictionary indeed learns typical structural priors.

## Highlights & Insights

1. **Introduction of External Priors**: Systematically introduces external priors (dictionaries) of training data into the LIC entropy model for the first time, complementing the limitations of relying solely on internal dependencies.
2. **No Extra Bit Overhead**: The dictionary exists as shared network parameters at both the encoder and decoder ends, eliminating the need for per-image transmission like global tokens.
3. **Excellent Speed-Performance Trade-off**: 193ms latency + SOTA BD-rate, significantly outperforming MLIC++ (772ms).
4. **Comparison with Global Tokens** (Tab. 5): The dictionary with 128 entries performs better than 8 global tokens (-7.28% vs. -6.59%) because the dictionary can store common patterns across different images.

## Limitations & Future Work

1. The generalization of the dictionary to **non-natural images** (e.g., medical imaging, remote sensing images) has not been verified.
2. The current dictionary entries remain static; **online adaptive updating** might further increase compression efficiency in specific domains.
3. Only MSE is used as the distortion metric; optimization for **perceptual quality** (MS-SSIM, LPIPS) has not yet been explored.
4. The asymmetric design of the encoder/decoder (different channel numbers at different stages) is effective but lacks systematic search (e.g., NAS).

## Related Work & Insights

- **MLIC++** (Jiang et al.): Currently achieves the strongest BD-rate but suffers from excessive latency; this work surpasses its performance while maintaining speed advantages.
- **Kim et al. (global token)**: Capture global information using 8 learnable tokens, which require transmission; the proposed dictionary design is superior.
- **CompressAI**: A unified LIC evaluation framework.
- **Inspiration**: The paradigm of dictionary learning + cross-attention can be generalized to temporal entropy models for video compression, point cloud compression, and other fields.

## Rating

⭐⭐⭐⭐ — Clear methodology, extensive experiments, reaching SOTA in both performance and efficiency. The innovation lies in introducing dictionary priors to the entropy model, which is a novel and practical direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LALIC: Linear Attention Modeling for Learned Image Compression](linear_attention_modeling_for_learned_image_compression.md)
- [\[CVPR 2025\] MambaIC: State Space Models for High-Performance Learned Image Compression](mambaic_state_space_models_for_high-performance_learned_image_compression.md)
- [\[ICML 2026\] Efficient Learned Image Compression without Entropy Coding](../../ICML2026/model_compression/efficient_learned_image_compression_without_entropy_coding.md)
- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](../../ICCV2025/model_compression/learned_image_compression_with_hierarchical_progressive_context_modeling.md)
- [\[ECCV 2024\] Bidirectional Stereo Image Compression with Cross-Dimensional Entropy Model](../../ECCV2024/model_compression/bidirectional_stereo_image_compression_with_cross-dimensional_entropy_model.md)

</div>

<!-- RELATED:END -->
