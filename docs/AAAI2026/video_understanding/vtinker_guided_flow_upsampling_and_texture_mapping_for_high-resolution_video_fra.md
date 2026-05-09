---
title: >-
  [Paper Note] VTinker: Guided Flow Upsampling and Texture Mapping for High-Resolution Video Frame Interpolation
description: >-
  [AAAI 2026][Video Understanding][Video Frame Interpolation] VTinker is a pipeline that addresses blurry optical flow boundaries via Guided Flow Upsampling (GFU) and eliminates ghosting and discontinuities by replacing conventional per-pixel blending with texture mapping, achieving state-of-the-art performance in high-resolution video frame interpolation.
tags:
  - AAAI 2026
  - Video Understanding
  - Video Frame Interpolation
  - Optical Flow Upsampling
  - Texture Mapping
  - High-Resolution Video
  - Motion Estimation
date: 2026-05-08
content_hash: 704f687f21f62bbf
---

# VTinker: Guided Flow Upsampling and Texture Mapping for High-Resolution Video Frame Interpolation

**Conference**: AAAI 2026  
**arXiv**: [2511.16124](https://arxiv.org/abs/2511.16124)  
**Code**: [https://github.com/Wucy0519/VTinker](https://github.com/Wucy0519/VTinker)  
**Area**: Video Understanding / Video Frame Interpolation  
**Keywords**: Video Frame Interpolation, Optical Flow Upsampling, Texture Mapping, High-Resolution Video, Motion Estimation

## TL;DR

VTinker is a pipeline that addresses blurry optical flow boundaries via Guided Flow Upsampling (GFU) and eliminates ghosting and discontinuities by replacing conventional per-pixel blending with texture mapping, achieving state-of-the-art performance in high-resolution video frame interpolation.

## Background & Motivation

**Background**: Flow-based video frame interpolation (VFI) is the dominant paradigm, typically comprising three stages: motion estimation (performed at low resolution to reduce computation) → optical flow upsampling (from low to high resolution) → frame synthesis (warping two frames using the upsampled flow and fusing them).

**Limitations of Prior Work**: Existing methods suffer from deficiencies at three levels. (1) **Flow upsampling**: Bilinear upsampling blurs flow boundaries, while adaptive filter upsampling (AFU) tends to produce blocky artifacts when applied to task-oriented flows trained end-to-end without ground-truth supervision. (2) **Motion accuracy**: Low-resolution motion estimation fails to capture fine-grained pixel-level motion in high-resolution frames, resulting in misaligned upsampled flows. (3) **Frame synthesis**: The conventional Mask&Res mechanism blends two warped frames per-pixel, producing ghosting, blurring, and discontinuities when flows are inaccurate.

**Key Challenge**: In high-resolution video, motion displacements can exceed 100 pixels. Motion estimation must be performed at low resolution for efficiency, but the subsequent low-to-high-resolution flow mapping and per-pixel dual-source texture blending both amplify estimation errors.

**Goal**: (1) How to align upsampled flow boundaries with image boundaries. (2) How to avoid ghosting and discontinuities caused by per-pixel dual-source blending.

**Key Insight**: Inspired by UPFlow, the paper leverages high-resolution input frames as guidance to refine flow upsampling, and replaces per-pixel blending with texture mapping that selects coherent texture patches from a single source frame.

**Core Idea**: Use input frames to guide optical flow upsampling for sharp boundaries; replace pixel-level dual-source blending with patch-level texture mapping to eliminate ghosting.

## Method

### Overall Architecture

Given two frames $I_0, I_1 \in \mathbb{R}^{H \times W \times 3}$, VTinker first estimates bidirectional flows $F_{0\to1}, F_{1\to0}$ at low resolution, upsamples them to high resolution via GFU, warps both frames using the upsampled flows to generate intermediate proxies, extracts texture patches from the input frames, selects the best-matching patches through flow-guided search and local matching, maps them onto the proxy, and finally generates the interpolated frame via a reconstruction module.

### Key Designs

1. **Guided Flow Upsampling (GFU)**:

    - **Function**: Upsample low-resolution optical flow to high resolution while preserving sharp motion boundaries.
    - **Mechanism**: The low-resolution flow is first upsampled via bilinear interpolation, then convolutional layers extract guidance features from the input frames; these features are used to correct the blurry boundaries introduced by bilinear upsampling. Since flow boundaries should align with image boundaries of the corresponding frame (e.g., $F_{0\to1}^{up}$ aligns with $I_0$), guidance naturally comes from the corresponding input frame.
    - **Design Motivation**: Bilinear upsampling smooths flow values across motion boundaries, causing blurring; AFU produces discontinuous boundaries when applied to task-oriented flows without GT supervision. GFU introduces high-resolution image structure to guide boundary alignment.

2. **Texture Mapping**:

    - **Function**: Replace per-pixel blending with coherent patch-level texture selection to eliminate ghosting and discontinuities caused by flow misalignment.
    - **Mechanism**: The process consists of three steps. (a) **Proxy generation**: The two warped frames $I_t^0, I_t^1$ are processed by multi-layer convolutions to produce an intermediate proxy $\mathcal{Q} \in \mathbb{R}^{H/2 \times W/2 \times C}$. (b) **Texture extraction and patch partitioning**: Texture features $\mathcal{T}_0, \mathcal{T}_1$ are extracted from $I_0, I_1$ and divided into overlapping texture patches $\mathcal{B}_0^{x,y}, \mathcal{B}_1^{x,y}$ of block size $s$. (c) **Patch selection**: A coarse search is first performed via flow guidance (downsampling flows to patch level and indexing texture patches via nearest-mode GridSample), followed by local matching for fine-grained alignment—proxy patches and texture patches are compressed into index vectors via convolution, and the highest-correlation patch within an $N \times N$ neighborhood is selected.
    - **Design Motivation**: The output of the conventional Mask&Res approach is a pixel-level mixture from both $I_0$ and $I_1$; when warps are inaccurate, blending two misaligned sources introduces ghosting. Texture mapping ensures that each spatial location receives texture exclusively from a single source frame ($I_0$ or $I_1$ only), guaranteeing intra-patch continuity.

3. **Reconstruction Module and Texture Quality Assurance**:

    - **Function**: Reconstruct the final interpolated frame from the texture-mapped proxy, while ensuring texture quality.
    - **Mechanism**: A UNet-like network transforms latent-space features into image space. A key design is the use of a single weight-shared reconstruction module to independently reconstruct $\mathcal{T}_0 \to \hat{I}_0$ and $\mathcal{T}_1 \to \hat{I}_1$, supervised by the input frames $I_0, I_1$ to ensure high-quality texture extraction. At inference time, only the interpolated frame needs to be reconstructed; $\hat{I}_0$ and $\hat{I}_1$ are not required.
    - **Design Motivation**: If the extracted texture patches are of poor quality, even a correct mapping strategy cannot yield good results. An auxiliary texture reconstruction loss constrains the quality of the texture extractor.

### Loss & Training

The Style loss proposed in FILM is adopted: $\mathcal{L}_S = w_l \mathcal{L}_1 + w_{VGG} \mathcal{L}_{VGG} + w_{Gram} \mathcal{L}_{Gram}$. The total loss is a weighted sum of three Style losses: $\mathcal{L}_S^{all} = w_t \times \mathcal{L}_S^t + w_0 \times \mathcal{L}_S^0 + w_1 \times \mathcal{L}_S^1$, corresponding to the interpolated frame and the two reconstructed input frames, respectively. The motion estimator is redesigned based on UPR-Net, adopting a PWC-Net-style structure for feature alignment.

## Key Experimental Results

### Main Results

| Dataset | Metric | VTinker | Prev. SOTA | Gain |
|--------|------|---------|---------|------|
| DAVIS (1080p) | PSNR↑ | 26.778 | 26.927 (SGM-1/2) | −0.15 |
| DAVIS (1080p) | LPIPS↓ | 0.108 | 0.114 (PerVFI) | +0.006 |
| DAVIS (1080p) | DISTS↓ | 0.039 | 0.042 (PerVFI) | +0.003 |
| DAVIS (4K) | PSNR↑ | 26.610 | 26.798 (SGM-1/2) | −0.19 |
| DAVIS (4K) | LPIPS↓ | 0.115 | — | Best |
| Xiph-4K | LPIPS↓ | 0.066 | 0.084 (RIFE) | +0.018 |
| Xiph-4K | DISTS↓ | 0.025 | 0.035 (RIFE) | +0.010 |

### Ablation Study

| Configuration | Key Metric | Remarks |
|------|---------|------|
| Bilinear upsampling vs. GFU | GFU outperforms bilinear and AFU | Sharper flow boundaries |
| Mask&Res vs. Texture Mapping | Texture mapping reduces ghosting and discontinuities | Significant visual quality improvement |
| Without vs. with texture supervision | Supervision improves texture quality | Weight-shared reconstruction module is effective |

### Key Findings

- VTinker consistently leads on perceptual metrics (LPIPS, DISTS), with larger margins at 4K resolution.
- PSNR is slightly below SGM-1/2, but perceptual quality is superior.
- GFU gains are most pronounced at motion boundaries; texture mapping gains are most evident in occluded and large-motion regions.
- Many competing methods run out of memory at 4K resolution; VTinker remains operable.

## Highlights & Insights

- The paper decomposes the VFI problem into two orthogonal dimensions—"flow quality" and "synthesis strategy"—and addresses each with GFU and texture mapping respectively, yielding a clean and well-motivated design.
- The "single-source patch selection" idea in texture mapping is conceptually simple yet highly effective: guaranteeing that each location's texture originates from a single frame eliminates dual-source blending artifacts.
- The weight-shared reconstruction module simultaneously serves texture quality assurance and final frame reconstruction, constituting an economical design.
- VTinker demonstrates substantial advantages over competing methods at 4K resolution, offering strong practical utility in real-world scenarios.

## Limitations & Future Work

- Computational cost is high (121 ms @ 1080p, 765G FLOPs), precluding real-time applications.
- The patch size $s$ is fixed; adaptive patch sizes could yield better results across regions with varying motion magnitudes.
- Texture patch matching relies on flow-guided coarse search and may fail when flows are severely erroneous.
- PSNR does not reach the top, indicating room for improvement in pixel-level accuracy.

## Related Work & Insights

- **vs. SGM**: SGM employs AFU for flow upsampling, which produces blocky artifacts under end-to-end training; VTinker's GFU avoids this via image-guided boundary refinement.
- **vs. UPR-Net**: VTinker redesigns UPR-Net's motion estimator to perform alignment at the original timestep rather than the interpolated timestep, improving flow estimation efficiency.
- **vs. FILM / PerVFI**: These perceptual-quality-oriented methods are surpassed by VTinker on perceptual metrics, while VTinker additionally avoids out-of-memory failures at 4K resolution.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The patch-level single-source selection in texture mapping is novel; GFU, though inspired by UPFlow, introduces meaningful adaptations for VFI.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple resolutions (720p/1080p/2K/4K) and datasets, with thorough ablations and clear visual comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Problem analysis is rigorous, method diagrams are clear, and ablation experiments are well-designed.
- **Value**: ⭐⭐⭐⭐ Significant practical impact for high-resolution VFI applications; open-source code enhances reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](../../ICCV2025/video_understanding/memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](../../ICCV2025/video_understanding/q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)
- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](../../ICCV2025/video_understanding/alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [\[AAAI 2026\] KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals](kinest_a_kinematics-guided_spatiotemporal_state_space_model_for_human_motion_tra.md)
- [\[NeurIPS 2025\] Cloud4D: Estimating Cloud Properties at a High Spatial and Temporal Resolution](../../NeurIPS2025/video_understanding/cloud4d_estimating_cloud_properties_at_a_high_spatial_and_temporal_resolution.md)

</div>

<!-- RELATED:END -->
