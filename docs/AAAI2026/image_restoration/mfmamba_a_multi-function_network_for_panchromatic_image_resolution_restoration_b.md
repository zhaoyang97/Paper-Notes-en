---
title: >-
  [Paper Note] MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model
description: >-
  [AAAI 2026][Image Restoration][Remote sensing imagery] This paper proposes MFmamba, a multi-function network built upon a UNet++ backbone that integrates a Mamba Upsampling Block (MUB), Dual Pooling Attention (DPA)…
tags:
  - "AAAI 2026"
  - "Image Restoration"
  - "Remote sensing imagery"
  - "panchromatic image"
  - "super-resolution"
  - "spectral restoration"
  - "Mamba"
  - "state-space model"
  - "UNet++"
date: 2026-05-08
content_hash: 5a32e9bcab1ddf42
---

# MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model

**Conference**: AAAI 2026
**arXiv**: [2511.18888](https://arxiv.org/abs/2511.18888)
**Authors**: Qian Jiang, Qianqian Wang, Xin Jin, Michal Wozniak, Shaowen Yao, Wei Zhou
**Code**: [GitHub](https://github.com/QianqianWang1325/MFmamba)
**Area**: Image Restoration
**Keywords**: Remote sensing imagery, panchromatic image, super-resolution, spectral restoration, Mamba, state-space model, UNet++

## TL;DR

This paper proposes MFmamba, a multi-function network built upon a UNet++ backbone that integrates a Mamba Upsampling Block (MUB), Dual Pooling Attention (DPA), and a Multi-scale Hybrid Cross Block (MHCB). Using only panchromatic (PAN) images as input, the unified framework simultaneously supports three tasks: super-resolution, spectral restoration, and joint SR with colorization.

## Background & Motivation

### State of the Field
Remote sensing imagery is widely used in object detection, urban planning, and environmental monitoring. Due to physical sensor constraints, panchromatic (PAN) images offer high spatial resolution but are grayscale, while multispectral (MS) images provide high spectral resolution at the cost of spatial resolution. A practical need exists to simultaneously enhance both spatial and spectral resolution from PAN-only input, yielding high-resolution color remote sensing imagery.

### Limitations of Prior Work
- **Super-resolution (SR) methods**: Can only enhance spatial resolution; unable to recover spectral information.
- **Colorization methods**: Can only recover spectral information; unable to enhance spatial resolution.
- **Pansharpening methods**: Require co-registered PAN and MS inputs, and do not support super-resolution.
- **Cascaded pipelines** (e.g., SR→colorization or colorization→SR): Suffer from severe error accumulation, leading to substantially degraded PSNR (e.g., CIR+HAT achieves only 27.777 dB on Potsdam).
- Existing methods tend to generate erroneous textures in complex regions, and color distortion remains an unsolved challenge.

### Root Cause
There is no unified multi-function framework capable of handling SR, spectral restoration, and joint restoration within a single model using only PAN input, without the cascading error propagation inherent in multi-stage pipelines.

### Paper Goals
To design a unified multi-function network that supports SR, spectral restoration, and joint restoration through different input configurations under a single framework, thereby eliminating error accumulation from multi-stage cascading.

## Method

### Overall Architecture
MFmamba adopts UNet++ (depth 4) as the backbone and integrates three core modules:
1. **MHCB**: Used for initial feature extraction (two MHCBs in series).
2. **DPA**: Replaces the original skip connections in UNet++ for same-level feature propagation.
3. **MUB**: Combines the Mamba state-space model for upsampling and resolution restoration.

The three tasks are realized through different input configurations: the SR task takes a low-resolution PAN image; the spectral restoration task takes the original grayscale PAN image; the joint task also takes a low-resolution PAN image.

### Multi-scale Hybrid Cross Block (MHCB)
MHCB extracts multi-scale features via parallel convolutional branches:
- $3\times3$ convolution branch: Focuses on local detail features.
- $5\times5$ convolution branch: Captures larger-range global features.
- Outputs from both branches and the input are combined via residual connections and fused through a $1\times1$ convolution.
- The second stage repeats this operation; the final output is again fused with a residual connection added.
- A dense residual grouping strategy preserves gradient flow and strengthens the persistence of key features across network layers.

### Dual Pooling Attention (DPA)
DPA employs a dual-stream architecture for channel-level feature recalibration:
- **Adaptive global average pooling**: Compresses global spatial information into a channel descriptor $A_{sq1} = \frac{1}{H \times W}\sum_{i,j} X_{i,j,c}$
- **Max pooling**: Captures salient features $A_{sq2} = \max_{p,q} X_{p,q,c}$
- Each stream passes through a Sigmoid function to generate a $1\times1\times C$ weight map.
- The final output is the element-wise sum of the two re-weighted features: $DPA_{out} = (X \odot A_{ex1}) \oplus (X \odot A_{ex2})$
- The combination of both pooling operations enables the model to attend to both salient details and global semantic information simultaneously.

### Mamba Upsampling Block (MUB)
MUB is built on the Selective State Space Model (S6), with a 2D Selective Scan Mechanism (2D-SSM) at its core:
- 2D feature maps are flattened into 1D sequences, and long-range dependencies are computed via discretized state-space equations.
- State-space equations: $h_\tau = \bar{\mathbf{A}} h_{\tau-1} + \bar{\mathbf{B}} x_\tau$, $y_\tau = \mathbf{C} h_\tau + \mathbf{D} x_\tau$
- **Key improvement**: The original 4 horizontal scanning directions are extended to **6 scanning directions** by adding 2 diagonal directions, enabling more comprehensive spatial information capture.
- A $2\times2$ patch configuration is used (outperforming $3\times3$ patches).
- Compared to Transformer-based modules: faster inference, fewer parameters, and lower GPU memory consumption.

### Loss & Training
The L1 loss is used: $L_1(y, \hat{y}) = \sum_{i=1}^{n} |y_i - \hat{y}_i|$

## Key Experimental Results

### Experimental Setup
- Datasets: Potsdam, NWPU, QuickBird, GF2, RSSCN7
- Metrics: PSNR, SSIM, MSE, MAE, SAM, LPIPS
- UNet++ depth 4, growth rate 32, batch size 1, trained for 32 epochs
- Adam optimizer, learning rate $1\times10^{-4}$, StepLR decay every 10 epochs

### Table 1: Joint SR and Spectral Restoration Comparison (Potsdam & NWPU)

| Method | Potsdam PSNR↑ | Potsdam SSIM↑ | Potsdam MSE↓ | NWPU PSNR↑ | NWPU SSIM↑ | NWPU MSE↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| MBPRR | 34.953 | 0.943 | 28.653 | 28.504 | 0.913 | 101.887 |
| CASR | 32.685 | 0.946 | 51.054 | 28.200 | 0.906 | 109.594 |
| RSI | 34.857 | 0.910 | 25.760 | 32.354 | 0.913 | 59.534 |
| HAT+CIR | 32.853 | 0.958 | 34.774 | 22.760 | 0.533 | 393.884 |
| CIR+HAT | 27.777 | 0.720 | 375.405 | 24.519 | 0.672 | 257.197 |
| SwinIR+CIR | 32.838 | 0.957 | 34.907 | 22.789 | 0.536 | 390.958 |
| **MFmamba** | **40.148** | **0.966** | **7.096** | **33.183** | **0.927** | **46.606** |

MFmamba achieves 40.148 dB PSNR on Potsdam, surpassing the best baseline MBPRR by **+5.2 dB**; MSE decreases from 25.760 to 7.096, a reduction of 72%. The cascaded pipeline (CIR+HAT) degrades severely due to error accumulation, achieving only 27.777 dB PSNR.

### Table 2: Spectral Restoration (Colorization) Comparison (Potsdam)

| Method | PSNR↑ | SSIM↑ | MSE↓ | MAE↓ | SAM↓ | LPIPS↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| RSI | 33.135 | 0.983 | 47.805 | 103.962 | 0.068 | 0.052 |
| CIR | 31.990 | 0.957 | 42.428 | 95.134 | 0.069 | 0.077 |
| SEGAN | 32.513 | 0.632 | 396.840 | 122.054 | 0.083 | 0.640 |
| Huang | 32.025 | 0.968 | 72.866 | 105.335 | 0.066 | 0.175 |
| **MFmamba** | **35.569** | **0.984** | **28.477** | **88.036** | **0.052** | **0.048** |

MFmamba achieves 35.569 dB PSNR on the colorization task, surpassing RSI by **+2.4 dB**; SAM decreases from 0.066 to 0.052, yielding the lowest spectral angle error.

### Ablation Study
- Removing DPA: PSNR drops from 40.148 to 39.980 (−0.17 dB).
- Removing MUB: PSNR drops to 39.794 (−0.35 dB), the largest individual impact.
- Removing MHCB: PSNR drops to 40.061 (−0.09 dB).
- Number of MHCBs: 2 is optimal (40.148); 1 (40.072) and 3 (40.080) are both slightly inferior.
- UNet++ depth: 4 layers is optimal (40.148); 2 layers yields only 39.548.
- 6-direction scanning outperforms single-direction variants with 5 or 6 directions.

## Highlights & Insights

- **Unified multi-function framework**: A single model handles SR, colorization, and joint restoration through different input configurations, avoiding the error accumulation inherent in multi-stage cascading pipelines.
- **Mamba for remote sensing**: This work is among the first to introduce the Mamba state-space model to PAN image resolution restoration. The 6-direction 2D-SSM scanning enhances the global receptive field while achieving faster inference and fewer parameters than Transformer-based counterparts.
- **DPA attention design**: The dual-stream channel attention combining global average pooling and max pooling attends to both global semantics and salient features simultaneously.
- **Substantial performance gains**: MFmamba leads the second-best method by over 5 dB PSNR on the joint task and achieves state-of-the-art or near-state-of-the-art results across all three individual tasks.
- **Open source**: Full code and experimental reproduction support are provided.

## Limitations & Future Work

- **Training efficiency**: Batch size of 1 and 32 training epochs with an 80 GB GPU requirement are unfriendly to resource-constrained settings.
- **Limited dataset scale**: All five remote sensing datasets are small to medium in scale; validation on large-scale high-resolution satellite data is absent.
- **Fixed upscaling factors only**: The SR task supports only ×2 and ×4 fixed scales; arbitrary-scale super-resolution is not supported.
- **Simple loss function**: Only L1 loss is used; perceptual loss, adversarial loss, and spectral consistency constraints are not incorporated.
- **No downstream task evaluation**: The utility of restored images for downstream tasks such as object detection and semantic segmentation is not validated.
- **Domain-specific**: Generalization to natural image SR or colorization benchmarks is not evaluated.

## Related Work & Insights

- **MBPRR (Jin et al. 2024)**: A comparable PAN image joint restoration method, but achieves only 34.953 dB PSNR; MFmamba surpasses it by 5.2 dB, primarily attributed to Mamba's long-range modeling capability.
- **HAT (Chen et al. 2023)**: A Transformer-based SR method with moderate standalone SR performance (34.963 dB), but suffers severe degradation when cascaded with colorization.
- **SwinIR (Liang et al. 2021)**: Similar issues to HAT; SR performance of 34.921 dB degrades significantly when cascaded with colorization.
- **MambaIR (Guo et al. 2024)**: A representative work introducing Mamba to image restoration; MFmamba extends this to a unified multi-function framework and adds 6-direction scanning.
- **RSI (Feng et al. 2022)**: An end-to-end CNN method with moderate capability on both colorization and SR, but inferior to MFmamba across all metrics.
- **CSRDNN (Feng et al. 2021)**: A CNN-based method with competitive MAE but substantially worse PSNR and MSE compared to MFmamba.

## Rating

- Novelty: ⭐⭐⭐⭐ — The unified multi-function framework is conceptually novel, and introducing Mamba to PAN image restoration is a meaningful contribution; however, individual modules (attention, multi-scale convolution) are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Five datasets, three tasks, and detailed ablation studies are provided; however, large-scale data validation and downstream task evaluation are lacking.
- Writing Quality: ⭐⭐⭐ — The paper is structurally complete, but contains minor formula errors (e.g., repeated $A_{ex1}$ in the DPA output expression) and requires further refinement of writing details.
- Value: ⭐⭐⭐⭐ — Practically meaningful for remote sensing PAN image processing; the unified framework paradigm is generalizable to other multi-task scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining](sd-psfnet_sequential_and_dynamic_point_spread_function_netwo.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](../../ICCV2025/image_restoration/eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [\[AAAI 2026\] Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration](clear_nights_ahead_towards_multi-weather_nighttime_image_res.md)
- [\[ICCV 2025\] PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining](../../ICCV2025/image_restoration/pre-mamba_a_4d_state_space_model_for_ultra-high-frequent_event_camera_deraining.md)
- [\[AAAI 2026\] SpatioTemporal Difference Network for Video Depth Super-Resolution](spatiotemporal_difference_network_for_video_depth_super-resolution.md)

</div>

<!-- RELATED:END -->
