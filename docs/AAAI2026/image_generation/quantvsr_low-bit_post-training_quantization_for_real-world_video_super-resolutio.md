---
title: >-
  [Paper Note] QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution
description: >-
  [AAAI 2026][Image Generation][Video Super-Resolution] This paper proposes QuantVSR, the first low-bit (4/6-bit) post-training quantization framework for diffusion-based video super-resolution (VSR). It introduces a Spati…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Video Super-Resolution"
  - "Low-Bit Quantization"
  - "Post-Training Quantization"
  - "Diffusion Model Compression"
  - "Spatiotemporal Complexity"
date: 2026-05-08
content_hash: 8870a9c44791dfdd
---

# QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution

**Conference**: AAAI 2026
**arXiv**: [2508.04485](https://arxiv.org/abs/2508.04485)  
**Code**: [https://github.com/bowenchai/QuantVSR](https://github.com/bowenchai/QuantVSR)  
**Area**: Image Generation
**Keywords**: Video Super-Resolution, Low-Bit Quantization, Post-Training Quantization, Diffusion Model Compression, Spatiotemporal Complexity

## TL;DR

This paper proposes QuantVSR, the first low-bit (4/6-bit) post-training quantization framework for diffusion-based video super-resolution (VSR). It introduces a Spatiotemporal Complexity-Aware (STCA) mechanism for layer-adaptive rank allocation and a Learnable Bias Alignment (LBA) module to mitigate quantization bias. Under the 4-bit setting, QuantVSR achieves 84.39% parameter compression and 82.56% computation reduction while maintaining performance comparable to the full-precision model.

## Background & Motivation

### State of the Field

Video super-resolution (VSR) aims to recover high-resolution details from low-resolution video. Diffusion-based VSR methods (e.g., MGLD-VSR) leverage strong generative priors to achieve significant visual quality improvements in real-world scenarios, far surpassing GAN-based approaches. However, diffusion model inference is extremely slow and resource-intensive, which severely limits deployment on edge devices.

### Limitations of Prior Work

Quantization is an effective model compression technique and has made progress in image/video generation (SVDQuant, ViDiT-Q) and image restoration (PassionSR). However, applying quantization to VSR models presents two unique challenges:

**Loss of Temporal Consistency**: Model quantization introduces inconsistent errors across frames, disrupting the temporal coherence of generated videos.

**Complex Data Distributions**: VSR models embed temporal dynamics into latent features, resulting in more complex activation distributions — both spatial and temporal dimensions must be considered to bridge the performance gap between full-precision and quantized models.

### Root Cause

Low-bit quantization (e.g., 4-bit) represents the wide dynamic range of floating-point weights and activations with extremely limited integer values, causing a sharp drop in reconstruction quality. Existing general-purpose quantization methods (e.g., SVDQuant) employ fixed-rank full-precision branches to mitigate this, but suffer from two issues: (1) the rank allocation strategy is suboptimal, incurring unnecessary computational overhead; and (2) the full-precision branch may degrade the performance of the low-bit branch, leading to an overall suboptimal solution.

### Starting Point

**Core Idea**: Exploit the spatiotemporal characteristics of VSR inputs to guide quantization — adaptively allocate the rank of the full-precision branch based on each layer's temporal complexity (inter-frame differences) and spatial complexity (spatial variance). Layers with higher complexity receive higher rank to preserve information, while simpler layers use minimum rank to save computation. A learnable bias alignment module is also introduced to correct systematic biases introduced by low-bit quantization.

## Method

### Overall Architecture

QuantVSR is built upon the UNet architecture of MGLD-VSR, replacing original layers (Linear, Conv2d, Conv3d) with custom quantization layers. Each quantization layer consists of:
- **Full-Precision (FP) Branch**: Low-rank matrices $L_1 L_2$ that bypass quantization
- **Low-Bit Branch**: Quantized computation after Hadamard transform smoothing
- **Learnable Bias Alignment (LBA)**: Corrects quantization bias

The quantization layer formula is:
$$\boldsymbol{XW} = \underbrace{\boldsymbol{XL_1L_2}}_{\text{FP, STCA}} + \underbrace{Q_A(\boldsymbol{XH})Q_W(\boldsymbol{H}^\top\boldsymbol{R})}_{\text{Low-Bit}} + \underbrace{\boldsymbol{A}_{\text{bias}}}_{\text{LBA}}$$

The calibration procedure consists of three stages: spatiotemporal complexity analysis → joint refinement of both branches → LBA module training.

### Key Designs

#### 1. **Spatiotemporal Complexity-Aware Mechanism (STCA)**

**Function**: Adaptively allocates rank to the full-precision branch based on each layer's temporal and spatial complexity, achieving a balance between performance and efficiency.

**Temporal complexity** is defined as inter-frame difference energy:
$$C_t = \frac{1}{T-1} \sum_{t=1}^{T-1} \frac{1}{CHW} \|\boldsymbol{X}_{t+1} - \boldsymbol{X}_t\|_2^2$$

Higher values indicate more intense inter-frame motion and greater reconstruction difficulty.

**Spatial complexity** is defined as the mean of spatial variances:
$$C_s = \frac{1}{TC} \sum_{t=1}^T \sum_{c=1}^C \sigma_{h,w}(\boldsymbol{X}_{t,c})$$

Higher spatial variance indicates richer texture and edge features with greater information density.

**Rank Allocation Strategy**:
- Compute per-layer spatiotemporal complexity distributions on the calibration set and set upper/lower thresholds (75th/25th percentiles)
- If both temporal and spatial complexity exceed the upper threshold: rank +1
- If both fall below the lower threshold: rank −1
- Otherwise: rank unchanged
- Constrained to $[r_{\min}, r_{\max}] = [16, 64]$ and rounded to multiples of 8

**Design Motivation**: The rank of the full-precision branch directly determines computational cost (linear growth at $r\frac{m+n}{mn}$). A uniformly high rank wastes computation, while a uniformly low rank loses information. Layer-adaptive allocation based on complexity preserves the most information with the least computation. In practice, the average rank is only 24, lower than SVDQuant's fixed rank of 32.

#### 2. **Dual-Branch Refinement**

**Function**: After layer-adaptive rank allocation, jointly optimizes the full-precision branch and low-bit branch to achieve an overall optimum.

**Mechanism**: The full-precision branch alters the data distribution of the low-bit branch (residual $R = W - L_1L_2$), potentially making quantization more difficult. Since both branches jointly contribute to the final output, they must be optimized together.

- $L_1$ and $L_2$ are initialized via SVD, providing a good starting point to avoid slow convergence from random initialization
- Trained for a small number of steps on the calibration set using MSE between FP and quantized model outputs as the objective

**Design Motivation**: Strengthening the FP branch (by increasing rank) may paradoxically degrade the low-bit branch performance; joint refinement ensures both branches reach an overall optimal balance.

#### 3. **Learnable Bias Alignment (LBA)**

**Function**: Corrects systematic bias introduced by low-bit quantization.

**Problem Analysis**: Quantization error is biased — the mean outputs of the full-precision and quantized models differ. When both weights and activations are quantized, the bias is:
$$\mathbb{E}(\hat{W}\hat{X}) - \mathbb{E}(WX) = \Delta W \mathbb{E}(\hat{X}) + W \mathbb{E}(\Delta X)$$

This bias is influenced by the activation quantization error $\Delta X$ and cannot be corrected by simple statistics.

**Core Idea**: A learnable bias $A_{\text{bias}}$ is appended after the quantization layer output, sharing the same dimensionality as the layer bias. Its parameter count is negligible relative to the full model, and it converges quickly. During inference, it can be directly absorbed into the layer bias with zero additional computational overhead.

**Training**: LBA is trained independently with all other parameters frozen.

### Loss & Training

- **Calibration Data**: Sampled at equal intervals during the denoising process (5 points from 50 steps) from the FP UNet of MGLD-VSR on REDS30, yielding 1,800 input-output pairs, each with shape $5 \times 4 \times 64 \times 64$
- **Training Objective**: MSE between FP and quantized model outputs + STE gradient approximation
- **Training Hardware**: NVIDIA RTX A6000, 2 epochs
- **Learning Rate**: $1 \times 10^{-3}$ for the first epoch, $2 \times 10^{-4}$ for the second epoch

## Key Experimental Results

### Main Results

Results on synthetic datasets (REDS4, SPMCS) and real-world dataset (MVSR4x):

**REDS4 W4A4 Quantization Results** (most challenging setting):

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DOVER ↑ | E*warp ↓ |
|--------|--------|--------|---------|---------|----------|
| MGLD-VSR (FP) | 23.27 | 0.6180 | 0.2117 | 0.6761 | 7.24 |
| MaxMin | 16.18 | 0.1995 | 0.6720 | 0.1451 | 52.27 |
| Q-Diffusion | 19.99 | 0.3176 | 0.5279 | 0.4936 | 19.63 |
| SVDQuant | 21.19 | 0.4138 | 0.4718 | 0.5865 | 12.46 |
| **QuantVSR** | **23.31** | **0.6143** | **0.2286** | **0.6822** | **6.88** |

**MVSR4x W4A4 (Real-World)**:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DOVER ↑ | E*warp ↓ |
|--------|--------|--------|---------|---------|----------|
| MGLD-VSR (FP) | 22.77 | 0.7422 | 0.3571 | 0.6321 | 1.54 |
| SVDQuant | 21.70 | 0.5021 | 0.5780 | 0.4727 | 3.30 |
| **QuantVSR** | **22.90** | **0.7367** | **0.3590** | **0.6219** | **1.40** |

### Ablation Study

**Per-Component Ablation on SPMCS W4A4**:

| Configuration | SC | LBA | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DOVER ↑ | E*warp ↓ |
|--------------|-----|-----|--------|--------|---------|---------|----------|
| No skip connection | ✗ | ✗ | 17.13 | 0.2595 | 0.6480 | 0.0802 | 10.00 |
| No skip connection | ✗ | ✓ | 21.38 | 0.4996 | 0.3666 | 0.6515 | 3.27 |
| SVDQuant SC | ✓ | ✗ | 18.94 | 0.2820 | 0.5921 | 0.4028 | 6.40 |
| SVDQuant SC | ✓ | ✓ | 22.58 | 0.5783 | 0.3296 | 0.6673 | 1.90 |
| **STCA** | ✓ | ✗ | 22.75 | 0.6071 | 0.2914 | 0.6886 | 1.74 |
| **STCA** | ✓ | ✓ | **22.76** | **0.6075** | **0.2857** | **0.6969** | **1.76** |

**Compression Ratios**:

| Quantization | Params (M) | Compression | FLOPs (G) | Compression |
|-------------|-----------|-------------|-----------|-------------|
| W32A32 (FP) | 935 | 0% | 1881 | 0% |
| W8A8 | 263 | 71.87% | 563 | 70.07% |
| W6A6 | 204 | 78.18% | 446 | 76.29% |
| W4A4 | 146 | 84.39% | 328 | 82.56% |

### Key Findings

1. **4-bit quantization is nearly lossless**: QuantVSR W4A4 achieves a PSNR of 23.31, marginally exceeding the FP model (23.27) — an outcome entirely unattainable by other methods (SVDQuant reaches only 21.19).
2. **STCA substantially outperforms SVDQuant's fixed-rank strategy**: PSNR improves from 22.58 to 22.75 with a lower average rank (24 vs. 32) and reduced computation.
3. **LBA is most effective when performance degradation is severe**: Without skip connections, PSNR jumps from 17.13 to 21.38.
4. **Temporal consistency metric E*warp is the most discriminative indicator**: QuantVSR achieves 6.88 on REDS4 W4A4, while SVDQuant reaches 12.46 (nearly twice as poor), demonstrating the necessity of spatiotemporal-aware design.
5. Some methods score higher than the FP model on no-reference IQA metrics (e.g., CLIP-IQA) despite poor structural metrics (PSNR/SSIM) — noisy images can also receive high no-reference scores.

## Highlights & Insights

1. **First systematic application of low-bit quantization to diffusion-based VSR**: Fills a gap in the field with a generalizable methodology.
2. **Spatiotemporal complexity-driven rank allocation**: Incorporates domain-specific knowledge (temporal dynamics + spatial texture of video) into the quantization strategy, offering greater efficiency than general-purpose approaches.
3. **Zero inference cost for LBA**: The learnable bias is absorbed into the layer bias at inference time, incurring no additional overhead whatsoever.
4. **Systematic evaluation design**: Simultaneously assesses image quality (IQA), video quality (VQA), and temporal consistency (E*warp), providing a comprehensive metric coverage.

## Limitations & Future Work

1. Validation is limited to MGLD-VSR (U-Net-based); applicability to DiT-architecture VSR models (e.g., VSR built upon video generation models) remains unknown.
2. Calibration data is drawn from REDS30; cross-domain generalization warrants further investigation.
3. The STCA rank allocation thresholds (25th/75th percentiles) are fixed empirical values; finer search may yield further gains.
4. A minor SSIM drop remains on the real-world MVSR4x dataset under 4-bit (0.7367 vs. FP 0.7422); scenarios with extreme precision requirements may necessitate 6-bit quantization.
5. Mixed-precision quantization strategies — assigning different bit-widths to different layers — are not explored and may offer further improvements.

## Related Work & Insights

- **SVDQuant** (Li et al.): A general 4-bit quantization method using a 16-bit parallel low-rank branch. QuantVSR's STCA is a targeted improvement over its fixed-rank strategy.
- **ViDiT-Q** (Zhao et al.): Quantization for image/video generation, but without consideration of the spatiotemporal characteristics of VSR.
- **PassionSR** (Zhu et al.): Quantization for single-step diffusion super-resolution, but limited to image SR.
- **EfficientDM** (He et al.): A pioneer in low-rank quantization fine-tuning strategies.
- The spatiotemporal complexity measurement concept is transferable to quantization in other tasks requiring inter-frame consistency, such as video editing and video inpainting.

## Rating

- Novelty: ⭐⭐⭐⭐ — STCA spatiotemporal-aware rank allocation and LBA are innovative; first systematic treatment of diffusion VSR quantization
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Synthetic and real-world datasets; IQA + VQA + temporal consistency; thorough ablation
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear; method description is detailed
- Value: ⭐⭐⭐⭐ — 84% compression with near-lossless quality; clear practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [\[AAAI 2026\] Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution](continuous_degradation_modeling_via_latent_flow_matching_for_real-world_super-re.md)
- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)
- [\[ICML 2026\] Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution](../../ICML2026/image_generation/q-dit4sr_exploration_of_detail-preserving_diffusion_transformer_quantization_for.md)
- [\[NeurIPS 2025\] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](../../NeurIPS2025/image_generation/dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)

</div>

<!-- RELATED:END -->
