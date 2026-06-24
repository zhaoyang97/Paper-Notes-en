---
title: >-
  [Paper Note] Sampling Innovation-Based Adaptive Compressive Sensing
description: >-
  [CVPR 2025][Model Compression][Adaptive Compressive Sensing] The SIB-ACS framework is proposed, which guides multi-stage adaptive sampling allocation through a "sampling innovation" criterion (measuring the reduction in reconstruction error brought by sampling increments) and designs a Principal Component Compressed Domain Network (PCCD-Net) for high-fidelity image reconstruction, significantly surpassing SOTA compressive sensing methods.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Adaptive Compressive Sensing"
  - "Sampling Innovation"
  - "Negative Feedback Mechanism"
  - "Deep Unfolding Network"
  - "Principal Component Gradient Descent"
date: 2026-05-08
content_hash: 70f2a55374b47403
---

# Sampling Innovation-Based Adaptive Compressive Sensing

**Conference**: CVPR 2025  
**arXiv**: [2503.13241](https://arxiv.org/abs/2503.13241)  
**Code**: [GitHub](https://github.com/giant-pandada/SIB-ACS_CVPR2025)  
**Area**: Model Compression  
**Keywords**: Adaptive Compressive Sensing, Sampling Innovation, Negative Feedback Mechanism, Deep Unfolding Network, Principal Component Gradient Descent

## TL;DR

The SIB-ACS framework is proposed, which guides multi-stage adaptive sampling allocation through a "sampling innovation" criterion (measuring the reduction in reconstruction error brought by sampling increments) and designs a Principal Component Compressed Domain Network (PCCD-Net) for high-fidelity image reconstruction, significantly surpassing SOTA compressive sensing methods.

## Background & Motivation

Compressive Sensing (CS) leverages signal sparsity for undersampled reconstruction and is widely applied in fields such as medical imaging and hyperspectral imaging. Uniform Compressive Sensing (UCS) uses the same sampling rate for all image blocks, failing to adapt to differences in regional complexity. Adaptive Compressive Sensing (ACS) dynamically allocates sampling resources based on image block content, but faces key challenges in unknown scenarios (without ground truth).

Existing ACS methods present two main issues: (1) Measurement-domain methods (based on measurement errors, cosine similarity, etc.)—the undersampled measurement data itself is ill-posed, leading to inaccurate estimation; (2) Image-domain methods (based on reconstruction complexity analysis)—the analysis metrics and sampling form a positive feedback loop, causing sampling to concentrate in initially allocated regions without error-correction capability.

This paper proposes a negative feedback mechanism based on "sampling innovation" to address these issues: by comparing the reconstruction differences before and after sampling increments, it determines which regions need extra sampling most. As the principal components are recovered, the innovation value naturally decreases, forming a negative feedback.

## Method

### Overall Architecture

SIB-ACS comprises two main modules: (1) Adaptive Sampling Module (ASM)—achieves adaptive sampling allocation via the innovation criterion and multi-stage negative feedback; (2) Image Reconstruction Module (PCCD-Net)—performs highly efficient reconstruction through dual-path proximal gradient descent in the principal component and compressed domains.

### Key Design 1: Sampling Innovation Criterion

**Function**: Accurately determine how much extra sampling resource each image block needs.

**Mechanism**: Define "innovation" as the variation in the reconstructed image brought by the sampling increment: $\alpha = \|\hat{\mathbf{x}}_{\text{IS}} - \hat{\mathbf{x}}_{\text{HM}}\|_2^2$, which represents the difference between reconstruction with incremental sampling and reconstruction with historical measurements. The adaptive sample size for each block is allocated proportionally to its innovation value: $M_n = M_{\text{ASR}} \cdot \frac{\|\alpha_n\|_2^2}{\sum_n \|\alpha_n\|_2^2}$.

**Design Motivation**: The innovation value directly estimates the reduction in reconstruction error, aligning with the objective of minimizing reconstruction error. More importantly, innovation is a relative metric—as the principal components of the image blocks are recovered, the innovation value naturally decreases, forming a negative feedback and preventing the sampling concentration problem caused by positive feedback.

### Key Design 2: Innovation-Guided Multi-Stage Adaptive Sampling

**Function**: Gradually eliminate residual innovation in each block through iterative negative feedback.

**Mechanism**: Each AS stage consists of three steps: (1) Innovation Sampling (IS)—performs uniform probing based on the distribution of the previous stage; (2) Innovation Estimation (IE)—reconstructs the images before and after IS using lightweight networks and calculates the difference; (3) Adaptive Sampling (AS)—allocates new samples according to innovation weights. The initial stage uses a uniform sampling rate $SR_{\text{init}}$, followed by $S$ iterative stages. The maximum sampling rate for each block is set to $s/S$.

**Design Motivation**: The multi-stage framework leverages negative feedback to progressively correct sampling allocation errors. The innovation estimation in each stage is independent of previous stage errors, leading to better convergence.

### Key Design 3: Principal Component Compressed Domain Network (PCCD-Net)

**Function**: Efficiently perform high-fidelity image reconstruction in adaptive compressive sensing scenarios.

**Mechanism**: A deep unfolding network is adopted, containing two parallel paths in each iteration stage. PCPGD Path: aggregates $C$-channel features into a single-channel principal component image, computes the gradient in the image domain, and then expands it back to the feature domain. CDPGD Path: compresses features to an $L$-dimensional compressed domain and computes the gradient along the channel dimension. Finally, the features of the two paths are complimentarily added: $\mathbf{X}^k = \mathbf{X}^k_p + \mathbf{X}^k_c$.

**Design Motivation**: In ACS, complex regions have higher sampling rates and larger sampling matrices, leading to a dramatic increase in the computational cost of feature-domain gradient descent. Reducing the dimensionality of GD operations through the principal component path controls computational costs while utilizing the compressed domain path to supplement detailed features.

### Loss & Training

A weighted combination of $\ell_1$ loss and SSIM loss is used: $\mathcal{L}(\Theta) = \mathcal{L}_{l_1}(\Theta) + \mu \mathcal{L}_{\text{SSIM}}(\Theta)$, to optimize both pixel accuracy and texture quality simultaneously.

## Key Experimental Results

### Main Results: PSNR (dB) Comparison on BSD68 Dataset

| Method | SR=0.10 | SR=0.25 | SR=0.50 | Avg |
|------|---------|---------|---------|------|
| CPP-Net (CVPR'24, UCS) | 28.41 | 32.25 | 37.30 | 33.33 |
| CASNet (TIP'22, ACS) | 28.41 | 32.31 | 37.49 | 33.41 |
| AMS-Net (TMM'22, ACS) | 29.36 | 33.53 | 39.20 | 34.73 |
| **SIB-ACS (Ours)** | **29.54** | **34.35** | **41.14** | **35.83** |

### PSNR (dB) Comparison on Urban100 Dataset

| Method | SR=0.10 | SR=0.25 | SR=0.50 | Avg |
|------|---------|---------|---------|------|
| CPP-Net (CVPR'24, UCS) | 28.48 | 33.37 | 38.29 | 34.24 |
| AMS-Net (TMM'22, ACS) | 28.04 | 33.22 | 38.33 | 34.06 |
| **SIB-ACS (Ours)** | **30.72** | **36.30** | **42.96** | **37.35** |

### Key Findings

- SIB-ACS achieves an average PSNR on BSD68 that is 2.5 dB higher than the strongest UCS method (CPP-Net) and 1.1 dB higher than the strongest ACS method (AMS-Net).
- On Urban100 (texture-rich scenes), the advantage is even more pronounced, with the average PSNR being 3.29 dB higher than AMS-Net, indicating that the negative feedback mechanism achieves more accurate adaptive allocation for complex regions.
- The advantage of SIB-ACS is most significant at low sampling rates (SR=0.10), as an accurate allocation strategy is more critical when resources are scarce.
- The dual-path design of PCCD-Net reduces computational costs while maintaining reconstruction quality comparable to full feature-domain GD.

## Highlights & Insights

1. **Ingeniously Designed Negative Feedback Mechanism**: Sampling innovation is inherently a self-limiting metric—the better the reconstruction, the lower the innovation—naturally avoiding positive feedback traps. This represents a fundamental improvement over existing ACS frameworks.
2. **Complementary Dual-Path in PCCD-Net**: The principal component path processes the main structure, while the compressed-domain path supplements detail. The complementarity of the two is akin to the low-frequency and high-frequency decomposition concept.
3. **Single Model for Multiple Sampling Rates**: Achieves adaptive sensing for arbitrary sampling rates with a single model through the multi-stage framework.

## Limitations & Future Work

- The multi-stage sampling increases total sampling and reconstruction time, which limits real-time performance.
- The accuracy of the lightweight reconstruction network (used for fast reconstruction in IE) may affect the precision of innovation estimation.
- Currently, adaptive allocation is performed at the block level; finer-grained pixel-level allocation could potentially further improve performance.
- Experiments are predominantly validated on natural images; applicability to specialized fields such as medical or remote sensing imaging remains to be explored.

## Related Work & Insights

- **OCTUF (CVPR'23)**, **CPP-Net (CVPR'24)**: Strong UCS baselines; the proposed ACS strategy achieves significant improvements on top of them through more intelligent sampling allocation.
- **Deep Unfolding Networks (DUNs)**: PCCD-Net inherits the advantage of physical-information injection from DUNs, while addressing the computational problem of GD dimensional explosion in ACS scenarios via dual-path compression.
- **AMS-Net (TMM'22)**: The previous strongest ACS method, which, however, lacks a negative feedback mechanism. Ours surpasses it across all sampling rates.

## Rating

⭐⭐⭐⭐ — The design concept of sampling innovation and negative feedback is elegant and effective, and the dual-path architecture of PCCD-Net is highly practical. The experiments are thorough, achieving significant PSNR improvements across multiple benchmarks, particularly showing outstanding performance in texture-rich scenes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling](../../ECCV2024/model_compression/adaptive_compressed_sensing_with_diffusionbased_posterior_sa.md)
- [\[ECCV 2024\] Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing](../../ECCV2024/model_compression/adaptive_selection_of_samplingreconstruction_in_fourier_comp.md)
- [\[CVPR 2025\] AutoSSVH: Exploring Automated Frame Sampling for Efficient Self-Supervised Video Hashing](autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_h.md)
- [\[CVPR 2025\] HyperLoRA: Parameter-Efficient Adaptive Generation for Portrait Synthesis](hyperlora_parameter-efficient_adaptive_generation_for_portrait_synthesis.md)
- [\[CVPR 2025\] WAVE: Weight Templates for Adaptive Initialization of Variable-sized Models](wave_weight_templates_for_adaptive_initialization_of_variable-sized_models.md)

</div>

<!-- RELATED:END -->
