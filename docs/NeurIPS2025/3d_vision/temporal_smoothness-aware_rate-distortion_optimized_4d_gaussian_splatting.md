---
title: >-
  [Paper Note] Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting
description: >-
  [NeurIPS 2025][3D Vision][4D Gaussian Splatting] This paper proposes the first end-to-end rate-distortion (RD) optimized compression framework for 4D Gaussian Splatting. By exploiting the temporal smoothness prior of dynamic point trajectories via Haar wavelet transforms, the method achieves up to 91× compression over Ex4DGS (average model size ~1.1% of the original) while maintaining reasonable rendering quality and flexible rate-quality trade-off control.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 4D Gaussian Splatting
  - Rate-Distortion Optimization
  - Wavelet Transform
  - Temporal Compression
  - Dynamic Scenes
date: 2026-05-08
content_hash: f56700131e071423
---

# Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting

**Conference**: NeurIPS 2025  
**arXiv**: [2507.17336](https://arxiv.org/abs/2507.17336)  
**Code**: [https://github.com/HyeongminLEE/RD4DGS](https://github.com/HyeongminLEE/RD4DGS)  
**Area**: 3D Vision / Neural Rendering  
**Keywords**: 4D Gaussian Splatting, Rate-Distortion Optimization, Wavelet Transform, Temporal Compression, Dynamic Scenes

## TL;DR
This paper proposes the first end-to-end rate-distortion (RD) optimized compression framework for 4D Gaussian Splatting. By exploiting the temporal smoothness prior of dynamic point trajectories via Haar wavelet transforms, the method achieves up to 91× compression over Ex4DGS (average model size ~1.1% of the original) while maintaining reasonable rendering quality and flexible rate-quality trade-off control.

## Background & Motivation

4D Gaussian Splatting (4DGS) extends the high-speed rendering capabilities of 3DGS to dynamic scenes, but suffers from severe storage overhead: (1) a large number of Gaussian primitives must store parameters such as positions, covariances, and color (spherical harmonic coefficients); (2) dynamic points are stored independently per time frame, introducing significant **temporal redundancy**; (3) existing 4DGS compression methods lack an **entropy-aware** bit-level compression framework that enables flexible rate-distortion optimization.

Although compression of 3DGS has seen notable progress (e.g., Compact3DGS, HAC++, RD3DGS), 4DGS faces additional challenges: the redundancy introduced by the temporal dimension requires dedicated compression strategies. Existing 4DGS compression works (e.g., Light4GS, QUEEN) either do not explicitly optimize entropy or target different objectives (e.g., per-frame streaming compression), leaving end-to-end RD optimization for 4DGS unexplored.

The core idea of this paper is to exploit the **smoothness prior of dynamic point motion trajectories**—real-world object motion is typically smooth and can be efficiently represented via wavelet transforms. By discarding high-frequency detail coefficients through Haar wavelet decomposition, the storage cost of dynamic positions is substantially reduced, complemented by standard techniques such as mask pruning and vector quantization for comprehensive compression.

## Method

### Overall Architecture

Using Ex4DGS (fully explicit 4D Gaussian Splatting) as the baseline, the scene is decomposed into **static** and **dynamic** components. Parameters shared by both (spherical harmonic coefficients, rotations, scales) are compressed using standard 3DGS compression methods. For the position trajectories unique to dynamic Gaussians, wavelet transforms are introduced for temporal compression. Opacity components are handled with a differentiated quantization strategy. The entire pipeline is jointly optimized via end-to-end RD training.

### Key Designs

1. **Gaussian Primitive and Spherical Harmonic Pruning**:

    - Learnable masks $\phi_i$ produce soft masks via sigmoid and hard masks via STE binarization.
    - Primitive pruning cost: $L_{\text{GSprune}} = (1/N)\sum \phi_i^{\text{soft}}$
    - Spherical harmonic coefficients are pruned in a degree-stratified manner: $\theta_i^{(l)}$ corresponds to $l \geq 1$ order coefficients, with weighting factor $(2l+1)/((k+1)^2-1)$ encouraging higher-order coefficients to be pruned more aggressively.
    - Applied **uniformly** to both static and dynamic Gaussians.

2. **Wavelet Transform for Dynamic Position Compression (Core Contribution)**:

    - A single-level Haar wavelet transform is applied along the time axis to the position trajectory $\mu_d = [p_1, p_2, \ldots, p_T]^T \in \mathbb{R}^{T \times 3}$ of each dynamic point.
    - The transform decomposes the trajectory into approximation coefficients $F_a \in \mathbb{R}^{T/2 \times 3}$ (low-frequency / coarse motion) and detail coefficients $F_d \in \mathbb{R}^{T/2 \times 3}$ (high-frequency).
    - $F_d$ is **explicitly discarded** (set to zero); only $F_a$ is retained.
    - Reconstruction: $\hat{\mu}_d$ is recovered via the inverse Haar transform (transpose of the orthogonal matrix).
    - Intuition: motion trajectories are typically smooth; the low-frequency component captures the dominant information, while high-frequency details can be sacrificed.
    - Storage is reduced from $T \times 3$ to $T/2 \times 3$, and experiments show rendering quality is slightly improved (PSNR +0.19 dB at Level 1).

3. **Differentiated Opacity Quantization**:

    - Ex4DGS parameterizes dynamic opacity using two Gaussian mixture model parameters: center parameters $(a_s^o, a_f^o)$ and variance parameters $(b_s^o, b_f^o)$.
    - Ablation studies reveal that static/dynamic base opacities and center parameters are insensitive to quantization and can be safely quantized.
    - **Variance parameters $(b_s^o, b_f^o)$ are extremely sensitive to quantization**—quantization degrades PSNR from 29.57 to 28.52 (Level 6), while the additional compression gain is only 1.57%.
    - The adopted strategy is therefore to quantize $\alpha_s$, $\alpha_d$, and $a_*^o$, while **skipping** $b_*^o$.

4. **End-to-End Rate-Distortion Optimization**:

    - Total loss: $L_{\text{total}} = L_{\text{dist}} + \lambda_R \cdot L_{\text{rate}} + \lambda_{\text{reg}} \cdot L_{\text{reg}}$
    - $L_{\text{dist}} = (1 - \lambda_{\text{dssim}}) \cdot L_1 + \lambda_{\text{dssim}} \cdot (1 - \text{SSIM})$
    - $L_{\text{rate}}$ aggregates all bitrate costs: VQ index entropy, mask pruning penalties, etc.
    - $L_{\text{reg}}$ denotes the regularization terms from Ex4DGS (static displacement penalty, temporal smoothness).
    - Six compression levels are defined by adjusting $\lambda_{\text{GSprune}}$ and $\lambda_{\text{SHprune}}$.

### Loss & Training

Training follows a two-stage procedure: the base model is first trained using the standard Ex4DGS pipeline (~1 hour), followed by a second stage incorporating RD optimization components (pruning, ECVQ, wavelet transform) for an additional ~1 hour. Six compression levels are realized by adjusting pruning hyperparameters: $\lambda_{\text{GSprune}} \in [0.05, 0.0005]$ and $\lambda_{\text{SHprune}} \in [0.5, 0.005]$.

## Key Experimental Results

### Main Results

| Method | PSNR (dB) ↑ | Size (MB) ↓ | FPS ↑ | Compression Ratio |
|--------|------------|------------|-------|-------------------|
| Ex4DGS (N3V) | 32.11 | 115 | 72.3 | 1× |
| Ours Level 6 | 29.66 | 11.06 | 100.9 | **10.4×** |
| Ours Level 1 | 27.04 | 1.26 | 163.0 | **91.3×** |
| Ex4DGS (Technicolor) | 33.62 | 140.2 | 72.3 | 1× |
| Ours Level 6 | 32.20 | 19.6 | 113.1 | **7.2×** |
| Ours Level 1 | 28.60 | 2.1 | 213.9 | **66.8×** |

### Ablation Study

| Configuration | PSNR (L1) | Size (L1) | PSNR (L6) | Size (L6) | Notes |
|---------------|----------|----------|----------|----------|-------|
| w/o wavelet | 27.20 | 2.01 | 30.17 | 19.87 | Baseline RD compression |
| **+Wavelet Transform** | **27.39** | **1.63** | **30.26** | **14.67** | +0.19 dB / −19% (L1) |
| Wavelet Level 2 (1/4) | 26.89 | 1.36 | 28.47 | 11.35 | More aggressive but worse RD curve |
| Wavelet Level 3 (1/8) | 26.43 | 1.16 | 27.74 | 9.80 | Over-compression |

### Key Findings
- **The wavelet transform simultaneously improves both quality and compression**: counterintuitively, discarding high-frequency components not only reduces model size but also improves PSNR (0.09–0.19 dB), suggesting that removing high-frequency noise actually enhances trajectory modeling accuracy.
- Deeper wavelet decompositions (Level 2/3) further reduce model size but yield inferior RD curves compared to single-level decomposition, indicating that one level constitutes the optimal trade-off.
- Level 6 (11 MB) is smaller than 4DGaussians (34 MB) while achieving higher PSNR (29.66 vs. 28.63).
- Level 1 achieves real-time rendering at 163 FPS with only 1.26 MB, making it suitable for edge deployment.

## Highlights & Insights
- **First bit-level RD optimization framework for 4DGS**: fills the gap in end-to-end compression of dynamic Gaussian Splatting.
- **Principled application of wavelet transforms**: leverages the physical prior of motion trajectories (smoothness) to transfer signal processing tools into 3D representation compression.
- **Differentiated quantization strategy**: rather than treating all parameters uniformly, ablation studies reveal large differences in quantization sensitivity across parameters—quantizing variance parameters yields an extremely unfavorable cost-benefit ratio.
- Provides six levels of flexible compression control, allowing users to select appropriate rate-quality trade-off points according to the deployment platform.

## Limitations & Future Work
- The strategy of discarding high-frequency wavelet components may introduce motion blur artifacts for fast-moving objects.
- Dynamic points still account for a significant portion of storage (Appendix G/H), leaving substantial room for improvement in dynamic component compression.
- Separate pruning weights for static and dynamic points could potentially further improve performance.
- The framework is built upon Ex4DGS, though the core RD optimization principles are generalizable to other 4DGS models.
- The high-fidelity end of the RD curve (near original quality) still has room for improvement.

## Related Work & Insights
- **vs. RD3DGS (Wang et al.)**: directly extends RD optimization from 3DGS to 4D; this paper additionally introduces wavelet transforms and differentiated quantization to address temporal challenges.
- **vs. QUEEN**: QUEEN performs per-frame compressed streaming without explicitly optimizing entropy; this paper targets holistic model compression for efficient storage of complete dynamic scenes.
- **vs. Light4GS**: employs spatio-temporal pruning with entropy coding but does not perform end-to-end RD optimization.
- **vs. Video Codecs (H.264/H.265)**: wavelet transforms are widely used in traditional video coding; this paper transfers that idea to 4D Gaussian representations.

## Rating
- Novelty: ⭐⭐⭐⭐ First RD optimization framework for 4DGS; the introduction of wavelet transforms is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two datasets (N3V and Technicolor), six compression levels, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; ablation analysis is convincing.
- Value: ⭐⭐⭐⭐ Makes 4DGS practically deployable on edge devices and fills an important technical gap.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes](../../ICCV2025/3d_vision/mega_memory-efficient_4d_gaussian_splatting_for_dynamic_scenes.md)
- [\[ICCV 2025\] Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs](../../ICCV2025/3d_vision/compression_of_3d_gaussian_splatting_with_optimized_feature_planes_and_standard_.md)
- [\[NeurIPS 2025\] Orientation-anchored Hyper-Gaussian for 4D Reconstruction from Casual Videos](orientation-anchored_hyper-gaussian_for_4d_reconstruction_from_casual_videos.md)
- [\[ICCV 2025\] 4D Gaussian Splatting SLAM](../../ICCV2025/3d_vision/4d_gaussian_splatting_slam.md)
- [\[ICCV 2025\] 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting](../../ICCV2025/3d_vision/7dgs_unified_spatialtemporalangular_gaussian_splatting.md)

<!-- RELATED:END -->
