---
title: >-
  [Paper Note] DnLUT: Ultra-Efficient Color Image Denoising via Channel-Aware Lookup Tables
description: >-
  [CVPR 2025][Image Restoration][Lookup Table] This work proposes DnLUT, an ultra-efficient color image denoising framework based on lookup tables (LUTs). By employing a Pairwise Channel Mixer (PCM) to capture inter-channel correlation and an L-shaped convolution kernel to expand the receptive field, DnLUT achieves state-of-the-art LUT denoising performance with only 500KB of storage and 0.1% of the energy consumption of DnCNN.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Lookup Table"
  - "Color Image Denoising"
  - "Edge Device Deployment"
  - "Channel Correlation"
  - "Lightweight"
date: 2026-05-08
content_hash: 02e73a00180ec3a9
---

# DnLUT: Ultra-Efficient Color Image Denoising via Channel-Aware Lookup Tables

**Conference**: CVPR 2025  
**arXiv**: [2503.15931](https://arxiv.org/abs/2503.15931)  
**Code**: [GitHub](https://github.com/Stephen0808/DnLUT)  
**Area**: Image Restoration  
**Keywords**: Lookup Table, Color Image Denoising, Edge Device Deployment, Channel Correlation, Lightweight

## TL;DR

This work proposes DnLUT, an ultra-efficient color image denoising framework based on lookup tables (LUTs). By employing a Pairwise Channel Mixer (PCM) to capture inter-channel correlation and an L-shaped convolution kernel to expand the receptive field, DnLUT achieves state-of-the-art LUT denoising performance with only 500KB of storage and 0.1% of the energy consumption of DnCNN.

## Background & Motivation

Color image denoising is an important and challenging task. Although deep neural networks have significantly improved denoising quality, their high computational complexity and memory footprint make deployment on edge devices difficult. Lookup Table (LUT) methods offer a solution by replacing runtime computations with simple array indexing operations, but they face the following key bottlenecks:

- **Exponential Storage Growth**: LUT storage requirements grow exponentially with the input dimension, e.g., a $2 \times 2$ kernel with a depth of 3 requires approximately 582TB of storage.
- **Lack of Channel Information**: Existing LUT methods typically process each channel independently (spatial LUT), ignoring the strong correlation between RGB channels in color images.
- **Channel-Spatial Trade-off**: Channel-only LUTs ($1 \times 1$ with a depth of 3) ignore spatial relations, while spatial-only LUTs ($2 \times 2$ with a depth of 1) ignore channel correlations.
- **Rotational Redundancy**: Traditional rotation ensemble strategies repeatedly access about half of the pixels, introducing design redundancy.

## Method

### Overall Architecture

DnLUT consists of two phases: DnNet during the training phase and DnLUT during the inference phase. DnNet is structured in two stages: the first stage utilizes multiple groups of LUTs to generate multi-channel features which are then concatenated in a fusion module, while the second stage integrates two core modules, PCM and L-shaped convolution. After training, all components are converted into 3D or 4D LUTs for highly efficient inference.

### Key Design 1: Pairwise Channel Mixer (PCM) — Simultaneously Capturing Spatial and Channel Information

**Function**: Simultaneously processes spatial and channel information within the manageable index dimensions of LUTs, overcoming the channel-spatial trade-off.

**Mechanism**: Reorganizes the three RGB channels into three pairwise combinations (RG, GB, BR), with each pair processed in parallel using a convolution kernel of $1 \times 2$ spatial dimension and a depth of 2. Each convolution processes 4 pixel values to produce a 1-channel output, which can be efficiently converted into a 4D LUT. The output formulation is:

$$
(V_R, V_G, V_B) = \text{Cat}(LUT_{RG}[I_{R}][I_{R'}][I_{G}][I_{G'}], LUT_{GB}[\cdot], LUT_{BR}[\cdot])
$$

**Design Motivation**: Directly using a $2 \times 2$ convolution kernel with a depth of 3 would require a 12D LUT (approx. 582TB), whereas pairwise combination restricts the dimension to 4D (approx. 83.5KB), enabling joint channel-spatial modeling within a feasible storage budget. As a plug-and-play module, PCM brings over 1dB of improvement to existing LUT methods with only a 12% increase in runtime and an 8% increase in storage.

### Key Design 2: L-Shaped Rotation-Non-Overlapping Convolutional Kernel — Maximizing Pixel Utilization

**Function**: Expands the receptive field while preventing repetitive pixel access, and degrades the 4D LUT into a more efficient 3D LUT.

**Mechanism**: An L-shaped convolution kernel is designed such that each rotation only processes 2 additional pixels other than the center pixel, without any overlap. After 4 rotations, each surrounding pixel contributes exactly once, achieving equivalent coverage of a $3 \times 3$ receptive field.

**Design Motivation**: Traditional SR-LUT uses a $2 \times 2$ kernel with 4 rotations to extend the receptive field to $3 \times 3$, but approximately half of the pixels are searched repeatedly. The L-shaped kernel eliminates this redundancy, and since each rotation only requires 3 pixel indexes, it can be converted into a 3D LUT instead of a 4D LUT, reducing storage by 17×.

### Key Design 3: Multi-Scale Fusion Architecture — Stepwise Receptive Field Expansion

**Function**: Achieves large-range feature aggregation through hierarchical LUT combinations.

**Mechanism**: The first stage feeds features into multiple groups of LUTs with different patterns to extract multi-channel features for concatenation and fusion, and the second stage further refines the features by applying PCM and L-shaped convolutions on top of the fused representation.

**Design Motivation**: A single LUT has a limited receptive field. Hierarchical cascading allows information to propagate across a larger scope while maintaining low-dimensional indexing for each individual LUT.

### Loss & Training

Training employs the standard $L_1$ loss function, directly optimizing the pixel-level difference between the denoised and clean images.

## Key Experimental Results

### Main Results: Gaussian Color Image Denoising (CPSNR/dB)

| Method | CBSD68 σ=15 | CBSD68 σ=25 | CBSD68 σ=50 | Urban100 σ=25 |
|------|------------|------------|------------|--------------|
| SR-LUT | 29.76 | 26.71 | 22.41 | 26.04 |
| MuLUT | 30.52 | 28.11 | 24.85 | 27.67 |
| SPF-LUT | 30.97 | 28.56 | 25.33 | 28.26 |
| **DnLUT** | **32.41** | **29.88** | **26.03** | **28.87** |
| DnCNN (DNN) | 33.90 | 31.24 | 27.95 | 30.81 |

### Real-World Denoising

| Dataset | SPF-LUT | DnLUT | DnCNN |
|--------|---------|-------|-------|
| SIDD (CPSNR) | 34.91 | **35.44** | 36.45 |
| DnD (PSNR) | 36.22 | **36.67** | 37.11 |

### Efficiency Comparison

| Metric | DnLUT | DnCNN |
|------|-------|-------|
| Storage | **500KB** | ~500× larger |
| Energy | **0.1%** | 100% |
| Inference Speed | **20× faster** | Baseline |

### Key Findings

- DnLUT achieves the best performance among all LUT methods, surpassing SPF-LUT by up to **1.3dB**.
- As a plug-and-play module, PCM consistently provides over **1dB** improvement to existing LUT methods.
- The L-shaped convolution kernel reduces storage by **17×** while maintaining the receptive field.
- In real-world denoising tasks, DnLUT outperforms the classic CBM3D method by nearly **5dB**.

## Highlights & Insights

1. **Pairwise Channel Grouping is the Key Insight**: Decomposing the three-channel RGB problem into three two-channel sub-problems, thereby converting an unviable 12D LUT into manageable 4D LUTs, is the most elegant aspect of this work.
2. **Rotation-Non-Overlapping Design Possesses Mathematical Elegance**: The L-shaped kernel leverages geometric symmetry to ensure that each pixel is accessed exactly once, presenting a theoretically optimal solution to the storage-performance trade-off in LUT methods.
3. **Plug-and-Play Design of PCM Demonstrates Generalizability**: Significant improvements are achieved without changing existing architectures, thereby lowering the barrier for adoption.

## Limitations & Future Work

- A significant gap still remains between this approach and DNN-based methods (e.g., SwinIR 34.42dB vs. DnLUT 32.41dB). The representational capacity of LUT methods is inherently limited by their index dimensions.
- Currently, the method is primarily designed for Gaussian and real-world noise scenarios; its applicability to other degradation types (such as blur or JPEG compression) remains to be validated.
- The depth and combination strategies of cascaded LUTs warrant further investigation.
- Future work could explore hybrid architectures combining neural networks and LUTs, enabling flexible switching under varying computational budgets.

## Related Work & Insights

- **SR-LUT / MuLUT / RC-LUT**: LUT-based super-resolution methods that focus solely on spatial information.
- **SPF-LUT**: Key contribution includes shift aggregation and multi-LUT cascading, but channel modeling is still missing.
- **DnCNN**: A classic CNN denoising method, highly effective but computationally heavy.
- **CBM3D**: A classic color image denoising method that operates in YCbCr color space to exploit channel correlations.

## Rating

⭐⭐⭐⭐ — This work makes a highly meaningful contribution to the field of LUT-based denoising, featuring ingenious designs for both PCM and the L-shaped kernel. Although a conspicuous gap still exists compared to DNN methods, it offers exceptional practical value for edge device deployment scenarios. The methodology is clear, the evaluation is thorough, and the plug-and-play capability is strong.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](../../ICCV2025/image_restoration/lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)
- [\[CVPR 2025\] HVI: A New Color Space for Low-light Image Enhancement](hvi_a_new_color_space_for_low-light_image_enhancement.md)
- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](efficient_visual_state_space_model_for_image_deblurring.md)
- [\[CVPR 2025\] Rotation-Equivariant Self-Supervised Method in Image Denoising](rotation-equivariant_self-supervised_method_in_image_denoising.md)
- [\[CVPR 2025\] Degradation-Aware Feature Perturbation for All-in-One Image Restoration](degradation-aware_feature_perturbation_for_all-in-one_image_restoration.md)

</div>

<!-- RELATED:END -->
