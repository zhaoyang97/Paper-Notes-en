---
title: >-
  [Paper Note] 4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation
description: >-
  [AAAI 2026][3D Vision][4D Gaussian Splatting] The 4DSTR framework is proposed, which significantly enhances the spatial-temporal consistency of 4D Gaussian generation and its adaptability to rapid temporal changes through Mamba-based temporal correlation rectification (correcting the scale and rotation of Gaussian points) and a per-frame adaptive densification and pruning strategy.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "4D Gaussian Splatting"
  - "Spatial-Temporal Consistency"
  - "Video-to-4D"
  - "Mamba"
  - "Adaptive Densification"
date: 2026-05-08
content_hash: 36d1a801110174f6
---

# 4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation

**Conference**: AAAI 2026  
**arXiv**: [2511.07241](https://arxiv.org/abs/2511.07241)  
**Code**: None  
**Area**: 3D/4D Vision, 4D Content Generation  
**Keywords**: 4D Gaussian Splatting, Spatial-Temporal Consistency, Video-to-4D, Mamba, Adaptive Densification  

## TL;DR

The 4DSTR framework is proposed, which significantly enhances the spatial-temporal consistency of 4D Gaussian generation and its adaptability to rapid temporal changes through Mamba-based temporal correlation rectification (correcting the scale and rotation of Gaussian points) and a per-frame adaptive densification and pruning strategy.

## Background & Motivation

Significant progress in 2D image and 3D shape generation has naturally driven research into dynamic 4D content generation. Existing 4D generation methods mainly follow two pathways: text-to-4D (e.g., MAV3D, AYG, TC4D) and video-to-4D (e.g., Consistent4D, DreamGaussian4D, SC4D, STAG4D). The video-to-4D pipeline typically employs deformable 4D Gaussian Splatting as an intermediate representation, yet confronts two core challenges:

1. **Spatial-Temporal Inconsistency**: Existing methods process Gaussian attributes of each time frame independently, lacking explicit cross-frame temporal correlations, which leads to incoherence between frames in the generated 4D sequences.
2. **Poor Adaptability to Rapid Changes**: Using the same number of Gaussian points across all frames fails to handle drastic appearance changes in the scene (e.g., when a Minion's mouth suddenly opens, more Gaussian points are needed to represent details).

Methods represented by STAG4D introduce temporal anchors, but still lack an effective temporal correlation mechanism, and their densification strategy applies the same gradient threshold to all frames, failing to adaptively adjust.

## Core Problem

How to establish effective spatial-temporal modeling in the 4D Gaussian generation process such that: (1) cross-frame Gaussian attributes (especially scale and rotation) maintain temporal consistency; (2) the number of Gaussian points per frame can be dynamically adjusted to adapt to rapid spatial changes.

## Method

### Overall Architecture

Given an input video, 4DSTR first utilizes Zero123++ to generate multi-view frames and initializes the 3D Gaussians of the first frame. Then, a lightweight multi-head decoder maps voxel features to per-frame 4D Gaussian parameters. The core innovations lie in: (1) a temporal correlation module that rectifies the residuals of Gaussian scale and rotation; (2) per-frame adaptive densification and pruning that dynamically adjusts the number of Gaussian points. Training incorporates multi-view SDS loss combined with reconstruction loss and foreground mask loss.

### Key Designs

1. **Mamba-based Temporal Correlation and Rectification**:

    - A temporal buffer is designed to store the history of Gaussian attributes for a length of $T$ frames. After the Gaussian attributes of the current frame and the historical attributes in the buffer are concatenated via a sliding window mechanism, they are input into a Mamba state space model for temporal correlation encoding.
    - After temporal correlation, the features are fused with the scale/rotation of the current and previous frames (via dynamic weighting) to regress the scale residual $\Delta s_t$ and rotation residual $\Delta r_t$, which are used to rectify the Gaussian attributes of the current frame: $\hat{s}_t = s_t + \Delta s_t$.
    - Mamba is chosen over GRU or Attention due to its linear complexity, enabling efficient modeling of long-range temporal dependencies. Experiments demonstrate that Mamba outperforms GRU and Attention across all metrics and achieves the fastest speed (80 FPS).

2. **Per-Frame Adaptive Gaussian Densification and Pruning**:

    - **Densification**: The cumulative gradient $G(p)$ of each Gaussian point during training is analyzed, which follows a log-normal distribution. A densification threshold $\tau_t = \text{Quantile}_{(1-\lambda)}$ is calculated independently for each frame, and only the top $\lambda = 2.5\%$ of Gaussian points whose gradients exceed the threshold are densified.
    - **Pruning**: Invalid Gaussian points are pruned based on opacity, screen-space size, and world-space scaling constraints. Specifically, a point is deleted when its opacity $\alpha(p) < \tau_o$ or its scale falls outside the range of $[s_{\min}, s_{\max}]$.
    - This allows each frame to have a different number of Gaussian points, increasing the point count in areas with sudden texture changes and reducing it in smooth regions.

3. **Gaussian Correspondence Alignment**:

    - Per-frame densification and pruning disrupt the correspondence of Gaussian points between frames, which the temporal rectification module relies on. Therefore, a per-frame index is designed to explicitly mark the correlation of each densified/pruned Gaussian point with its corresponding frame, ensuring that Gaussian points in the temporal buffer still maintain correct temporal alignment after densification/pruning.

### Loss & Training

- **Multi-view SDS Loss**: $\mathcal{L}_{\text{MVSDS}} = \lambda_1 \cdot \mathcal{L}_{\text{SDS}}(\phi, I_t^i) + \lambda_2 \cdot \mathcal{L}_{\text{SDS}}(\phi, I_t^{\text{ref}})$, utilizing supervision from 6 anchor views generated by Zero123++ plus the reference view.
- **Reconstruction Loss** $\mathcal{L}_{\text{rec}}$ and **Foreground Mask Loss** $\mathcal{L}_{\text{mask}}$.
- **Total Loss**: $\mathcal{L} = \mathcal{L}_{\text{MVSDS}} + \lambda_3 \mathcal{L}_{\text{rec}} + \lambda_4 \mathcal{L}_{\text{mask}}$.
- **Collective Average Loss (CAL)**: Inspired by MOTR, the loss is aggregated over a sub-clip of $T_s$ frames: $\mathcal{L}_{\text{CAL}} = \frac{1}{T_s} \sum \mathcal{L}_t$, enabling the model to learn temporal changes across frames.
- **Training Strategy**: The canonical 3D Gaussians are first trained using static frames, and then dynamic 4D Gaussians are learned using anchor views and the reference view. The learning rate decays from $1.6 \times 10^{-4}$ to $1.6 \times 10^{-6}$.

## Key Experimental Results

| Dataset | Metric | Ours (4DSTR) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| Consistent4D Test Set | FID-VID ↓ | **45.31** | STAG4D: 53 | 15.1% |
| Consistent4D Test Set | FVD ↓ | **795.21** | STAG4D: 992 | 19.9% |
| Consistent4D Test Set | CLIP ↑ | **0.92** | STAG4D/MVTokenFlow: 0.91 | +0.01 |
| Consistent4D Test Set | LPIPS ↓ | **0.12** | MVTokenFlow: 0.12 | Comparable |
| 60-frame Extended Sequence | FID-VID ↓ | **43.72** | STAG4D: 76.00 | 42.5% |
| 60-frame Extended Sequence | FVD ↓ | **733.24** | STAG4D: 1035.00 | 29.2% |
| text-to-4D User Study | Visual Quality Preference ↑ | **53.3%** | STAG4D: 33.3% | +20.0pp |
| text-to-4D User Study | Temporal Consistency Preference ↑ | **50.0%** | STAG4D: 30.0% | +20.0pp |
| text-to-4D User Study | Text Alignment Preference ↑ | **46.7%** | STAG4D: 36.7% | +10.0pp |

- The temporal module only increases parameters by about 0.1M and GPU memory by 0.23 GiB, achieving a rendering speed of 80 FPS.
- All experiments were conducted on a single RTX 4090 GPU.
- In the text-to-4D user study, Ours achieved the highest ratings in visual quality, consistency, and text alignment (53.3% / 50.0% / 46.7% respectively), significantly outperforming STAG4D.

### Ablation Study

1. **Both Temporal and Spatial Rectifications are Indispensable**: Removing temporal rectification increases FID-VID from 45.31 to 55.32 (+22.1%), and removing spatial rectification increases it to 52.21. The combination of both yields the best performance.
2. **Mamba is Optimal**: Compared with GRU (50.32) and Attention (54.23), Mamba achieves the lowest FID-VID (45.31) and the fastest speed (80 vs. 68/72 FPS).
3. **Temporal Window $T=10$ is Sufficient**: Increasing $T$ from 2 to 10 decreases FID-VID from 57.32 to 45.31; at $T=15$, there is only a marginal improvement and FVD slightly bounces back (804.32 vs. 795.21), suggesting that a 10-frame window is sufficient to capture temporal dependencies.
4. **Robustness to Long Sequences**: In the 60-frame test, the performance of STAG4D drops sharply, whereas the FID-VID and FVD of 4DSTR decrease further, demonstrating the superior scalability of the spatial-temporal rectification mechanism on long sequences.

## Highlights & Insights

1. **Novel Temporal Rectification Concept**: Instead of simply predicting Gaussian attributes independently for each frame, it regresses scale and rotation residuals by encoding cross-frame temporal correlations through Mamba. This residual rectification manner preserves the basic structure of the initial predictions while introducing temporal consistency constraints.
2. **Per-Frame Adaptive Densification Hits the Mark**: It captures the key insight that "different frames require different numbers of Gaussian points"—more points are needed when dynamic region textures change drastically, while static regions can be pruned. This is a fundamental improvement over the global threshold strategy of STAG4D.
3. **Lightweight and Efficient Modules**: The temporal module only adds 0.1M parameters and 0.23 GiB of VRAM without affecting real-time rendering (80 FPS), offering strong engineering practicality.
4. **Long Sequence Scalability**: It not only maintains but further improves performance on 60-frame sequences, indicating that the designed spatial-temporal mechanism possesses excellent generalization capability.

## Limitations & Future Work

2. **Dependency on Multi-view Generation Quality of Zero123++**: The input multi-view frames originate from Zero123++, whose generation quality directly determines the upper bound of 4D reconstruction.
3. **Fixed Temporal Window $T=10$**: Although ablation shows $T=10$ is sufficient, a fixed window might not be optimal for longer or more complex dynamic sequences—adaptive window lengths are worth exploring.
4. **Static Bias in the Global Percentage of Densification Threshold $\lambda=2.5\%$**: Different types of dynamic variations might require different densification ratios; the current strategy does not distinguish between motion types.
5. **Lack of Complete Comparison with Recent Methods like CAT4D**: CAT4D does not provide data for all metrics in the quantitative comparison.
6. **Evaluation Limited to Specific Benchmark**: Quantitative evaluation is only conducted on 7 dynamic objects from Consistent4D, offering limited scene diversity.

## Related Work & Insights

| Method | Representation | Temporal Modeling | Densification Strategy | Core Difference |
|------|------|---------|-----------|---------|
| **Consistent4D** | DyNeRF | Interpolation Consistency Loss | None | Implicit representation, slow optimization |
| **DreamGaussian4D** | Deformable 4DGS | No explicit temporal correlation | Fixed threshold | Lack of spatial-temporal consistency |
| **SC4D** | Deformable 4DGS | No explicit temporal correlation | Fixed threshold | Same as above |
| **STAG4D** | Deformable 4DGS | Temporal anchors | Adaptive but uniform across all frames | Lack of temporal correlation, densification does not differentiate frames |
| **4DSTR (Ours)** | Deformable 4DGS | Mamba temporal encoding + residual rectification | Per-frame adaptive | Explicit temporal correlation + per-frame densification |

The core advantages of Ours compared to the strongest baseline STAG4D are: (1) establishing true cross-frame feature correlation using Mamba instead of relying solely on anchors; (2) calculating the densification threshold independently per frame instead of globally.

## Inspirations & Connections

1. **Increasingly Widespread Application of Mamba in 3D/4D Tasks**: From Mamba4D to the temporal encoding in Ours, the linear complexity of Mamba makes it an ideal choice for processing long-sequence 3D/4D data. Its expansion to other 3D tasks (e.g., point cloud sequence understanding, dynamic scene reconstruction) is worth monitoring.
2. **Residual Rectification Paradigm**: Instead of directly predicting the final attributes, predicting the correction amounts to the initial predictions is a concept that can be generalized to other generative 3D/4D tasks.
3. **Adaptive Point Cloud Density Control**: The concept of adjusting the number of Gaussian points per frame can be extended to generic scene reconstruction (non-generative) of 3DGS—increasing density in dynamic regions and reducing density in static regions.
4. **Potential for Integration with Video Diffusion Models**: Currently, Zero123++ is used for multi-view generation. Replacing it with more advanced video generation models (such as the Sora series) could further elevate the upper bound.

## Rating

- Novelty: ⭐⭐⭐⭐ The ideas of temporal residual rectification and per-frame adaptive densification are clear and somewhat novel, but individual components (Mamba, residual learning, adaptive thresholds) are not entirely new concepts. The core contribution lies in their ingenious combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation studies are detailed (temporal/spatial rectification, encoding methods, window size, long sequences, user study), but the scale of the quantitative evaluation dataset is relatively small (7 objects), displaying limited scene diversity.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, illustrations are intuitive, and the motivation is fully-articulated. However, equation notations and method descriptions are slightly redundant in some details.
- Value: ⭐⭐⭐⭐ Achieved significant SOTA improvements on the video-to-4D task (FVD reduced by 19.9%), with a lightweight design that offers strong engineering practicality. However, the 4D generation field is still evolving rapidly, and the long-term impact of this method remains to be seen.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ShapeGen4D: Towards High Quality 4D Shape Generation from Videos](../../ICLR2026/3d_vision/shapegen4d_towards_high_quality_4d_shape_generation_from_videos.md)
- [\[AAAI 2026\] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion](simba_towards_high-fidelity_and_geometrically-consistent_point_cloud_completion_.md)
- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [\[CVPR 2026\] Mark4D: Temporally-Consistent Watermarking for 4D Gaussian Splatting](../../CVPR2026/3d_vision/mark4d_temporally-consistent_watermarking_for_4d_gaussian_splatting.md)
- [\[CVPR 2026\] ConsisVLA-4D: Advancing Spatiotemporal Consistency in Efficient 3D-Perception and 4D-Reasoning for Robotic Manipulation](../../CVPR2026/3d_vision/consisvla-4d_advancing_spatiotemporal_consistency_in_efficient_3d-perception_and.md)

</div>

<!-- RELATED:END -->
