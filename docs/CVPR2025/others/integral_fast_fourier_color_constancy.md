---
title: >-
  [Paper Note] Integral Fast Fourier Color Constancy
description: >-
  [CVPR 2025][Automatic White Balance] This paper proposes IFFCC, which extends the FFCC algorithm to multi-illuminant scenes. By using an integral UV histogram to accelerate regional histogram computation and parallelize Fourier convolution operations, it achieves accuracy comparable to pixel-level neural networks while achieving real-time multi-illuminant automatic white balance with 400x fewer parameters and 20-100x speedups.
tags:
  - "CVPR 2025"
  - "Automatic White Balance"
  - "Multi-Illuminant Scenes"
  - "Integral Histogram"
  - "Fast Fourier Transform"
  - "Real-Time Processing"
date: 2026-05-08
content_hash: 4b1710a0e5442375
---

# Integral Fast Fourier Color Constancy

**Conference**: CVPR 2025  
**arXiv**: [2502.03494](https://arxiv.org/abs/2502.03494)  
**Code**: None  
**Area**: Computational Photography / Color Constancy  
**Keywords**: Automatic White Balance, Multi-Illuminant Scenes, Integral Histogram, Fast Fourier Transform, Real-Time Processing

## TL;DR
This paper proposes IFFCC, which extends the FFCC algorithm to multi-illuminant scenes. By using an integral UV histogram to accelerate regional histogram computation and parallelize Fourier convolution operations, it achieves accuracy comparable to pixel-level neural networks while achieving real-time multi-illuminant automatic white balance with 400x fewer parameters and 20-100x speedups.

## Background & Motivation

1. **Background**: Automatic White Balance (AWB) is a core component of ISP. Traditional methods (Gray World, White Patch, etc.) assume a single global illuminant, yielding limited effectiveness. Deep network methods (such as U-Net architectures) achieve high accuracy in multi-illuminant scenes but require large parameter sizes and high computational overhead.
2. **Limitations of Prior Work**: The FFCC algorithm is efficient and accurate in single-illuminant scenes but cannot be directly applied to multi-illuminant scenes—extracting histograms repeatedly for each target region leads to a dramatic increase in computational complexity; additionally, the lack of a spatial smoothing mechanism causes abrupt transitions in white balance results. Deep network methods have millions of parameters and inference times far exceeding 10ms, failing to meet the real-time requirements of resource-constrained devices like smartphones.
3. **Key Challenge**: Multi-illuminant AWB requires estimating illumination separately for different regions of the image. However, the overhead of calculating histograms region by region is proportional to the number of regions, creating a conflict between accuracy and speed.
4. **Goal**: (a) How to efficiently compute chromaticity histograms for arbitrary sub-regions? (b) How to achieve spatial illumination smoothing while keeping edges sharp? (c) How to implement real-time multi-illuminant AWB running on the CPU?
5. **Key Insight**: Drawing inspiration from the concept of Integral Images, the integral histogram is applied to the UV log-chromaticity space, enabling $O(1)$ histogram queries for arbitrary regions.
6. **Core Idea**: Utilize a one-time precomputated integral UV histogram to obtain the chromaticity histogram of any region with just a few addition and subtraction operations, then achieve real-time multi-illuminant white balance via parallel FFT convolution and spatial smoothing.

## Method

### Overall Architecture
Given an input thumbnail image (e.g., 64×48), it is first converted to the log-chroma space to compute the integral UV histogram. Then, an overlapping sliding window is used to efficiently extract local histograms of various regions from the integral histogram. FFT convolutions are executed in parallel for each region to predict the illumination chromaticity. Finally, linear interpolation and guided filtering are applied to generate a smooth pixel-level illumination map, which is combined with the original image to output the white-balanced image.

### Key Designs

1. **Integral UV Histogram**:

    - **Function**: Efficiently compute log-chroma histograms of arbitrary rectangular regions in an image.
    - **Mechanism**: Convert the RGB image to the log-chroma space $u^{(i)} = \log(I_g/I_r)$, $v^{(i)} = \log(I_g/I_b)$, then propagate the integral histogram from the top-left corner in a wavefront scanning order: $H_{\text{Integral}}(u,v,b) = H(u-1,v,b) + H(u,v-1,b) - H(u-1,v-1,b) + B(u,v)$. Once precomputation is completed, obtaining the histogram of any rectangular region requires only 3 additions/subtractions. Under a configuration of a 128×128 window and 64 bins, the computational efficiency ratio compared to traditional methods is 19.71x.
    - **Design Motivation**: When dealing with multiple illuminants, FFCC needs to extract histograms for each region individually. Here, the one-time precomputation avoids redundant calculations, reducing the complexity from $O(N \cdot M)$ to $O(N+M)$.

2. **Overlapping Grid + Parallel Prediction**:

    - **Function**: Efficiently obtain illumination estimates for multiple local regions.
    - **Mechanism**: Use overlapping sliding windows on the integral histogram to extract a set of histograms $X(k,B,B,N)$ for $k$ sub-regions, then execute parallel convolutions in the frequency domain $H = \text{IFFT}(\sum_k \text{FFT}(X) \cdot \text{FFT}(F)) + B$. Through a parallelized Bivariate von Mises distribution mean estimation, the illumination of all regions is predicted simultaneously in a single inference step. Overlapping regions preserve continuity between adjacent windows.
    - **Design Motivation**: FFCC originally predicted only a single global illumination value. Here, parallelization compresses the prediction time for a batch of regions to near that of a single prediction.

3. **Spatial Smoothing**:

    - **Function**: Reconstruct discrete patch-level illumination estimates into a pixel-level smooth illumination map.
    - **Mechanism**: First, linear interpolation is used to generate a smooth transition between the $k$ estimated regional values; then, guided filtering (using the original image as the guidance image) is applied to preserve structural features at illumination edges: $O_i = a_k G_i + b_k$, where coefficients $a_k, b_k$ are determined by local statistics and the regularization parameter $\epsilon$. The final output is a weighted average within the window.
    - **Design Motivation**: Pure interpolation blurs illumination boundaries (e.g., interfaces between different color temperatures), whereas guided filtering leverages the edge information of the original image to maintain these crucial illumination boundaries.

### Loss & Training
The training employs the frequency domain optimization method of FFCC for 64 iterations, with a histogram size of 64×64. Training images are randomly cropped to 128×128, and testing is conducted on 256×256 images. Two training strategies are used: B (mixed illumination GT) and M (dominant illuminant GT), with strategy B performing better. The entire process requires only CPU, with no GPU needed.

## Key Experimental Results

### Main Results

**Shadow Dataset (by camera):**

| Method | Type | Canon 5d mean | Canon 5d median | All mean | All median |
|------|------|--------------|----------------|-----------|------------|
| CRF(White-Patch) | Traditional | 7.66 | 5.96 | 7.19 | 5.44 |
| Patch-based | Traditional | 4.85 | 3.11 | 4.30 | 2.89 |
| Domislovic et al. | Network | 2.63 | 2.18 | 2.28 | 1.60 |
| **IFFCC** | **Traditional** | **2.06** | **1.54** | **2.19** | **1.56** |

**LSMI Dataset (multi-illuminant):**

| Method | Type | Galaxy mean | Nikon mean | Sony mean | Parameters (M) | CPU Time (s) |
|------|------|------------|-----------|----------|----------|-----------|
| Bianoco | Patch | 5.56 | 4.65 | 4.38 | 0.16 | - |
| AID | Pixel | 2.03 | 2.26 | 2.16 | 6.4 | >>1 |
| **IFFCC** | **Patch** | **2.48** | **2.30** | **2.48** | **0.012** | **0.03** |

### Ablation Study

| Training Strategy | Window / Overlap | Mean ↓ | Median ↓ | Description |
|---------|----------|--------|----------|------|
| B | [32, 16] | 2.65 | 2.17 | Insufficient information in small windows |
| M | [128, 64] | 2.42 | 1.95 | Dominant illuminant GT |
| B | [128, 64] | 2.06 | 1.54 | Mixed GT, default configuration |
| B | [192, 128] | 1.98 | 1.34 | Large window has highest accuracy but blurs boundaries |

**FFCC vs IFFCC Speed Comparison (Shadow Dataset):**

| Window / Overlap | FFCC | IFFCC | Speedup |
|----------|------|-------|--------|
| [128, 64] | 88ms | 27ms | 3.3× |
| [128, 96] | 234ms | 33ms | 7.1× |
| [64, 48] | 689ms | 40ms | 17.2× |

### Key Findings
- IFFCC achieves accuracy comparable to pixel-level network methods like AID with only 0.012M parameters (only 1/533 of AID).
- Inference for a 256×256 image takes only 5.8ms on a CPU (using a 64×48 thumbnail), which easily satisfies the 10ms real-time requirement for AWB.
- Window size is a key hyperparameter: small windows (32×32) retain fine textural details but have lower accuracy, while large windows (192×128) yield high accuracy but blur illumination boundaries.
- The mixed illumination GT (strategy B) consistently outperforms the single dominant illuminant GT (strategy M), as it better reflects the actual illumination distribution within the regions.
- Performs exceptionally well on multi-camera datasets, as the log-chroma histogram method does not depend on camera-specific spectral sensitivity.

## Highlights & Insights
- The **combination of integral histograms and log-chromaticity space** is highly elegant: applying the classic computer vision integral image technique to chromaticity histograms realizes $O(1)$ region queries, resolving the efficiency bottleneck of FFCC in multi-illuminant scenes.
- **Extreme engineering practicality**: 0.012M parameters, CPU-only, < 10ms inference time; this is a rare academic work that can be directly deployed on smartphone ISPs.
- The **extension paradigm from global to local** is highly generalizable: the strategy of extending any global estimation method to a locally adaptive one via integral histograms can be transferred to other tasks requiring region-level statistics.

## Limitations & Future Work
- Window sizes must be manually selected, and different scenes may require different window configurations.
- The local smoothing from guided filtering may still be insufficiently accurate in regions with extreme illumination changes.
- Testing has been conducted only at a 256×256 resolution; performance under higher resolutions remains to be validated.
- Adaptive window sizing strategies could be explored to dynamically adjust windows based on local chromaticity distribution complexity.

## Related Work & Insights
- **vs FFCC**: FFCC is a single-illuminant global method. IFFCC extends it to a multi-illuminant method via integral histograms, parallel prediction, and spatial smoothing, with only a marginal increase in computation time.
- **vs AID (pixel-level)**: AID utilizes slot attention for pixel-level prediction; though slightly more accurate, its parameters are 533x larger and speed is over 33x slower, making it unsuitable for real-time deployment.
- **vs Domislovic et al.**: Relative to other state-of-the-art non-deep-learning methods, IFFCC comprehensively outperforms it on the Shadow dataset.

## Rating
- Novelty: ⭐⭐⭐⭐ The application of integral histograms in chromaticity space is clever, though the overall work is an incremental extension of FFCC.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducts comprehensive comparison with multiple methods on two standard datasets, featuring thorough ablation and speed comparisons.
- Writing Quality: ⭐⭐⭐⭐ Direct and clear mathematical derivations, though some symbol definitions are somewhat dense.
- Value: ⭐⭐⭐⭐ High practical value for industrial deployment; it is one of the few AWB algorithms that can be directly applied to smartphone ISPs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Latent Fourier Transform](../../ICLR2026/others/latent_fourier_transform.md)
- [\[ICML 2026\] A Hypertoroidal Covering for Perfect Color Equivariance](../../ICML2026/others/a_hypertoroidal_covering_for_perfect_color_equivariance.md)
- [\[ICLR 2026\] Probabilistic Kernel Function for Fast Angle Testing](../../ICLR2026/others/probabilistic_kernel_function_for_fast_angle_testing.md)
- [\[ECCV 2024\] Real-Data-Driven 2000 FPS Color Video from Mosaicked Chromatic Spikes](../../ECCV2024/others/real-data-driven_2000_fps_color_video_from_mosaicked_chromatic_spikes.md)
- [\[ICML 2026\] New Bounds for Kernel Sums via Fast Spherical Embeddings](../../ICML2026/others/new_bounds_for_kernel_sums_via_fast_spherical_embeddings.md)

</div>

<!-- RELATED:END -->
