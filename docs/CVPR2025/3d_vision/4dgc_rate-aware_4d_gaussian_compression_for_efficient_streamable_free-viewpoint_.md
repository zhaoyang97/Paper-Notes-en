---
title: >-
  [Paper Note] 4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint Video
description: >-
  [CVPR 2025][3D Vision][4D Gaussian Splatting] This paper proposes 4DGC, a rate-distortion-aware 4D Gaussian compression framework. By adopting motion-aware dynamic Gaussian modeling (multi-resolution motion grids + sparse compensatory Gaussians) and end-to-end compression (differentiable quantization + implicit entropy model), 4DGC achieves 16× compression over 3DGStream without sacrificing rendering quality.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "4D Gaussian Splatting"
  - "video compression"
  - "rate-distortion"
  - "free-viewpoint video"
  - "streamable"
date: 2026-05-08
content_hash: 85626c57761cc290
---

# 4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint Video

**Conference**: CVPR 2025  
**arXiv**: [2503.18421](https://arxiv.org/abs/2503.18421)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 4D Gaussian Splatting, video compression, rate-distortion, free-viewpoint video, streamable

## TL;DR
This paper proposes 4DGC, a rate-distortion-aware 4D Gaussian compression framework. By adopting motion-aware dynamic Gaussian modeling (multi-resolution motion grids + sparse compensatory Gaussians) and end-to-end compression (differentiable quantization + implicit entropy model), 4DGC achieves 16× compression over 3DGStream without sacrificing rendering quality.

## Background & Motivation

### Background

1. **Background**: 3D Gaussian Splatting (3DGS) enables high-quality free-viewpoint video (FVV) rendering. However, it requires storing a vast amount of Gaussian attributes (position, color, covariance, etc.) for each frame, resulting in extremely high storage and transmission costs.

2. **Limitations of Prior Work**: (1) Existing methods handle Gaussian representation and compression separately, ignoring the rate-distortion trade-off. (2) Inter-frame redundancy is underutilized—Gaussian attributes of adjacent frames are highly similar. (3) Static 3DGS compression methods cannot be directly extended to dynamic scenes.

3. **Key Challenge**: High-quality FVV rendering requires massive Gaussian parameters, whereas streaming demands extremely low bitrates. It is necessary to consider compression efficiency during the representation design stage.

4. **Goal**: Design an end-to-end 4D Gaussian compression scheme that simultaneously optimizes rate-distortion performance at both representation and compression levels.

5. **Key Insight**: Utilize motion grids to capture inter-frame rigid motion (which covers most scene dynamics), and only use sparse compensatory Gaussians to represent residuals, drastically reducing the amount of information to be encoded.

6. **Core Idea**: Inter-frame motion modeling via motion grids + new region handling via sparse compensation + end-to-end rate-distortion optimized compression.

### Proposed Approach

**Goal**: ### Overall Architecture
4DGC comprises two core modules: (1) motion-aware dynamic Gaussian modeling, which estimates inter-frame motion via multi-resolution motion grids and handles newly appeared regions with sparse compensatory Gaussians; (2) end-to-end compression, which performs differentiable quantization on attributes and estimates bitrates with an implicit entropy model to jointly optimize rendering quality and bitrate.


## Method

### Overall Architecture
4DGC comprises two core modules: (1) motion-aware dynamic Gaussian modeling, which estimates inter-frame motion via multi-resolution motion grids and handles newly appeared regions with sparse compensatory Gaussians; (2) end-to-end compression, which performs differentiable quantization on attributes and estimates bitrates with an implicit entropy model to jointly optimize rendering quality and bitrate.

### Key Designs

1. **Multi-Resolution Motion Grid**
    - **Function**: Estimate inter-frame rigid motion (translation + rotation)
    - **Mechanism**: $\Delta\boldsymbol{\mu}_t = \Phi_{\mu}(\bigcup_{l=1}^L \text{interp}(\mathbf{P}_{t-1}^l, \mathbf{M}_t^l))$
    - **Design Motivation**: Motion grids provide continuous, low-dimensional representations, which are far more efficient than storing motion vectors per Gaussian.

2. **Sparse Compensatory Gaussians**
    - **Function**: Introduce additional Gaussians for newly appeared or rapidly changing regions.
    - **Mechanism**: Two triggering conditions: gradient change ($|\nabla| > \tau_g$) and rapid displacement ($|\Delta\mu| > \tau_\mu$).
    - **Final Representation**: $\hat{\mathbf{G}}_t = \hat{\mathbf{G}}_{t-1}(\cdot) + \Delta\hat{\mathbf{G}}_t$

3. **End-to-End Rate-Distortion Compression**
    - **Differentiable Quantization**: Quantize Gaussian attributes directly during training.
    - **Implicit Entropy Model**: Estimate the bitrate for each attribute.
    - **RD Loss**: $\mathcal{L} = \mathcal{L}_{render} + \lambda \cdot R$

### Loss & Training
- Rendering loss + $\lambda \times$ bitrate, where $\lambda$ controls the compression rate.

## Key Experimental Results

### Main Results

| Method | Compression Ratio | PSNR | Storage |
|------|--------|------|------|
| 3DGStream | 1× | Baseline | Large |
| 4DGC | **16×** | ≈ Baseline | 1/16 |

### Ablation Study

| Component | Effect |
|------|------|
| w/o Motion Grid | Significant drop in compression efficiency |
| w/o Compensatory Gaussians | Poor rendering quality in newly appeared regions |
| w/o End-to-End Training | Suboptimal RD performance |
| Full | Best RD trade-off |

### Key Findings
- The motion grid captures the vast majority of inter-frame changes, with compensatory Gaussians accounting for only a small portion.
- Implicit entropy models fit the distribution of Gaussian attributes better than traditional entropy coding.
- Rendering quality is nearly lossless under 16× compression.

## Highlights & Insights
- **Joint Optimization of Representation and Compression**: High-quality representations are designed to be inherently compression-friendly, rather than compressing after modeling.
- The motion grid design leverages the prior knowledge of dynamic scenes where "most regions exhibit rigid motion".
- The end-to-end differentiable framework enables rate-distortion (RD) optimization.

## Limitations & Future Work
- Non-rigid motion (e.g., cloth, liquids) may require more complex motion modeling.
- Streaming latency has not been fully evaluated.
- Combining with other compression/detection methods may yield better results.
- Evaluation on larger-scale datasets (e.g., longer video sequences, more diverse scenes) remains to be conducted.
- Deployment optimization for different application scenarios (mobile, server) is worth exploring.
- Theoretical analysis of the method can be further deepened.

## Related Work & Insights
- **vs 3DGStream**: Lacks compression design; its storage cost is 16× that of 4DGC.
- **vs Compact3D**: Only handles compression for static scenes.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Strong idea of joint representation and compression design
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive RD curve comparisons
- **Writing Quality**: ⭐⭐⭐⭐ Clear technical descriptions
- **Value**: ⭐⭐⭐⭐ Addresses a core problem in FVV streaming

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction](../../NeurIPS2025/3d_vision/motion_matters_compact_gaussian_streaming_for_free-viewpoint_video_reconstructio.md)
- [\[NeurIPS 2025\] Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](../../NeurIPS2025/3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)
- [\[CVPR 2025\] GIFStream: 4D Gaussian-based Immersive Video with Feature Stream](gifstream_4d_gaussian-based_immersive_video_with_feature_stream.md)
- [\[AAAI 2026\] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video](../../AAAI2026/3d_vision/streamstgs_streaming_spatial_and_temporal_gaussian_grids_for_real-time_free-view.md)
- [\[CVPR 2025\] Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation](efficient_dynamic_scene_editing_via_4d_gaussian-based_static-dynamic_separation.md)

</div>

<!-- RELATED:END -->
