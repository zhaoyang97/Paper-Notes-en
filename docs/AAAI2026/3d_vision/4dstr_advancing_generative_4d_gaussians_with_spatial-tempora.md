---
title: >-
  [Paper Note] 4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation
description: >-
  [AAAI 2026][3D Vision][4D Gaussian Splatting] This paper proposes the 4DSTR framework, which significantly improves the spatial-temporal consistency of 4D Gaussian generation and its adaptability to rapid temporal change…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "4D Gaussian Splatting"
  - "Spatial-Temporal Consistency"
  - "Video-to-4D"
  - "Mamba"
  - "Adaptive Densification"
date: 2026-05-08
content_hash: 46457b6eedf34cb3
---

# 4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation

**Conference**: AAAI 2026
**arXiv**: [2511.07241](https://arxiv.org/abs/2511.07241)  
**Code**: None  
**Area**: 3D/4D Vision, 4D Content Generation
**Keywords**: 4D Gaussian Splatting, Spatial-Temporal Consistency, Video-to-4D, Mamba, Adaptive Densification

## TL;DR

This paper proposes the 4DSTR framework, which significantly improves the spatial-temporal consistency of 4D Gaussian generation and its adaptability to rapid temporal changes through a Mamba-based temporal correlation rectification module (correcting Gaussian scale and rotation residuals) and a per-frame adaptive densification and pruning strategy.

## Background & Motivation

Recent advances in 2D image and 3D shape generation have naturally motivated research into dynamic 4D content generation. Existing 4D generation methods follow two main lines: text-to-4D (e.g., MAV3D, AYG, TC4D) and video-to-4D (e.g., Consistent4D, DreamGaussian4D, SC4D, STAG4D). Video-to-4D approaches commonly adopt deformable 4D Gaussian splatting as an intermediate representation, yet face two core challenges:

1. **Spatial-temporal inconsistency**: Existing methods process Gaussian attributes for each frame independently, lacking explicit cross-frame temporal correlation, which leads to inter-frame incoherence in the generated 4D sequences.
2. **Poor adaptability to rapid changes**: Using the same number of Gaussian primitives for all frames makes it difficult to handle drastic appearance changes in the scene (e.g., a Minion's mouth suddenly opening requires more Gaussians to capture fine details).

Methods such as STAG4D introduce temporal anchors but still lack effective temporal correlation mechanisms; moreover, their densification strategies apply a uniform gradient threshold across all frames, without per-frame adaptation.

## Core Problem

How to establish effective spatial-temporal modeling during 4D Gaussian generation such that: (1) cross-frame Gaussian attributes (especially scale and rotation) maintain temporal consistency; and (2) the number of Gaussian primitives per frame can be dynamically adjusted to accommodate rapid spatial changes.

## Method

### Overall Architecture

Given an input video, 4DSTR first generates multi-view frames using Zero123++ and initializes 3D Gaussians from the first frame. A lightweight multi-head decoder then maps voxel features to per-frame 4D Gaussian parameters. The core innovations are: (1) a temporal correlation module that rectifies residuals of Gaussian scale and rotation; and (2) per-frame adaptive densification and pruning that dynamically adjusts the number of Gaussian primitives. Training employs a multi-view SDS loss combined with a reconstruction loss and a foreground mask loss.

### Key Designs

1. **Mamba-based Temporal Correlation and Rectification**:

    - A temporal buffer of length $T$ stores the history of Gaussian attributes across frames. The current frame's Gaussian attributes are concatenated with historical attributes from the buffer via a sliding window mechanism and fed into a Mamba state-space model for temporal correlation encoding.
    - The temporally correlated features are fused with the scale/rotation of the current and previous frames (via dynamic weighting) to regress scale residuals $\Delta s_t$ and rotation residuals $\Delta r_t$, which rectify the current frame's Gaussian attributes: $\hat{s}_t = s_t + \Delta s_t$.
    - Mamba is chosen over GRU or Attention due to its linear complexity, which enables efficient modeling of long-range temporal dependencies. Experiments show that Mamba outperforms GRU and Attention on all metrics while achieving the highest speed (80 FPS).

2. **Per-Frame Adaptive Gaussian Densification and Pruning**:

    - **Densification**: The accumulated gradient $G(p)$ of each Gaussian primitive during training follows a log-normal distribution. A per-frame densification threshold $\tau_t = \text{Quantile}_{(1-\lambda)}$ is computed independently for each frame, and only the top $\lambda = 2.5\%$ of Gaussians with gradients exceeding the threshold are densified.
    - **Pruning**: Invalid Gaussian primitives are pruned based on opacity, screen-space size, and world-space scale constraints. Specifically, a primitive is removed when its opacity $\alpha(p) < \tau_o$ or its scale falls outside $[s_{\min}, s_{\max}]$.
    - This allows each frame to maintain a different number of Gaussian primitives, increasing point count in regions with abrupt texture changes and reducing it in smooth regions.

3. **Gaussian Correspondence Alignment**:

    - Per-frame densification and pruning disrupt inter-frame Gaussian correspondence, upon which the temporal rectification module relies. To address this, per-frame indices are designed to explicitly track the association between each densified/pruned Gaussian primitive and its corresponding frame, ensuring that primitives in the temporal buffer remain correctly temporally aligned after densification and pruning.

### Loss & Training

- **Multi-View SDS Loss**: $\mathcal{L}_{\text{MVSDS}} = \lambda_1 \cdot \mathcal{L}_{\text{SDS}}(\phi, I_t^i) + \lambda_2 \cdot \mathcal{L}_{\text{SDS}}(\phi, I_t^{\text{ref}})$, supervised using 6 anchor views generated by Zero123++ plus a reference view.
- **Reconstruction Loss** $\mathcal{L}_{\text{rec}}$ and **Foreground Mask Loss** $\mathcal{L}_{\text{mask}}$.
- **Total Loss**: $\mathcal{L} = \mathcal{L}_{\text{MVSDS}} + \lambda_3 \cdot \mathcal{L}_{\text{rec}} + \lambda_4 \cdot \mathcal{L}_{\text{mask}}$.
- **Collective Averaging Loss (CAL)**: Inspired by MOTR, losses are aggregated over sub-clips of $T_s$ frames: $\mathcal{L}_{\text{CAL}} = \frac{1}{T_s} \sum \mathcal{L}_t$, enabling the model to learn cross-frame temporal variations.
- **Training Strategy**: Static frames are first used to train canonical 3D Gaussians; anchor views and reference views are then used to learn dynamic 4D Gaussians. The learning rate decays from $1.6 \times 10^{-4}$ to $1.6 \times 10^{-6}$.

## Key Experimental Results

| Dataset | Metric | Ours (4DSTR) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| Consistent4D test set | FID-VID↓ | **45.31** | STAG4D: 53 | 15.1% |
| Consistent4D test set | FVD↓ | **795.21** | STAG4D: 992 | 19.9% |
| Consistent4D test set | CLIP↑ | **0.92** | STAG4D/MVTokenFlow: 0.91 | +0.01 |
| Consistent4D test set | LPIPS↓ | **0.12** | MVTokenFlow: 0.12 | On par |
| 60-frame extended sequences | FID-VID↓ | **43.72** | STAG4D: 76.00 | 42.5% |
| 60-frame extended sequences | FVD↓ | **733.24** | STAG4D: 1035.00 | 29.2% |
| Text-to-4D user study | Visual quality preference↑ | **53.3%** | STAG4D: 33.3% | +20.0pp |
| Text-to-4D user study | Temporal consistency preference↑ | **50.0%** | STAG4D: 30.0% | +20.0pp |
| Text-to-4D user study | Text alignment preference↑ | **46.7%** | STAG4D: 36.7% | +10.0pp |

- The temporal module adds only ~0.1M parameters and 0.23 GiB VRAM, with no impact on real-time rendering (80 FPS).
- All experiments are conducted on a single RTX 4090.
- In the text-to-4D user study, 4DSTR achieves the highest scores across visual quality, temporal consistency, and text alignment (53.3% / 50.0% / 46.7%), substantially outperforming STAG4D.

### Ablation Study

1. **Both temporal and spatial rectification are necessary**: Removing temporal rectification increases FID-VID from 45.31 to 55.32 (+22.1%); removing spatial rectification increases it to 52.21. The combination yields the best results.
2. **Mamba is optimal**: Compared to GRU (50.32) and Attention (54.23), Mamba achieves the lowest FID-VID (45.31) and the highest speed (80 vs. 68/72 FPS).
3. **Temporal window $T=10$ is sufficient**: As $T$ increases from 2 to 10, FID-VID drops from 57.32 to 45.31; at $T=15$ there is only a marginal improvement while FVD slightly increases (804.32 vs. 795.21), indicating that a 10-frame window is sufficient to capture temporal dependencies.
4. **Robustness on long sequences**: STAG4D degrades sharply on 60-frame sequences, whereas 4DSTR's FID-VID and FVD further decrease, demonstrating the superior scalability of the spatial-temporal rectification mechanism on long sequences.

## Highlights & Insights

1. **Novel temporal rectification paradigm**: Rather than independently predicting Gaussian attributes for each frame, the method employs Mamba to encode cross-frame temporal correlations and regresses scale/rotation residuals. This residual rectification preserves the basic structure of the original predictions while introducing temporal consistency constraints.
2. **Per-frame adaptive densification addresses the core issue**: The key insight that different frames require different numbers of Gaussian primitives is directly exploited—dynamically varying regions with dramatic texture changes require more primitives, while static regions can be pruned. This represents a fundamental improvement over STAG4D's global threshold strategy.
3. **Lightweight and efficient modules**: The temporal module adds only 0.1M parameters and 0.23 GiB VRAM without affecting real-time rendering (80 FPS), demonstrating strong practical engineering value.
4. **Long-sequence scalability**: Performance is not only maintained but further improved on 60-frame sequences, indicating that the designed spatial-temporal mechanisms generalize well.

## Limitations & Future Work

1. **Dependence on Zero123++ multi-view generation quality**: The multi-view frames used as input are generated by Zero123++, whose quality directly determines the upper bound of 4D reconstruction quality.
2. **Fixed temporal window $T=10$**: Although ablation studies suggest $T=10$ is sufficient, a fixed window may be suboptimal for longer or more complex dynamic sequences—adaptive window lengths deserve exploration.
3. **Globally static densification ratio $\lambda=2.5\%$**: Different types of dynamic changes may require different densification ratios; the current strategy does not distinguish between motion types.
4. **Incomplete comparison with recent methods such as CAT4D**: CAT4D does not provide data for all metrics in the quantitative comparison.
5. **Evaluation limited to a specific benchmark**: Quantitative evaluation is conducted on only 7 dynamic objects from Consistent4D, limiting scene diversity.

## Related Work & Insights

| Method | Representation | Temporal Modeling | Densification Strategy | Core Difference |
|------|------|---------|-----------|---------|
| **Consistent4D** | DyNeRF | Interpolation consistency loss | None | Implicit representation; slow optimization |
| **DreamGaussian4D** | Deformable 4DGS | No explicit temporal correlation | Fixed threshold | Lacks spatial-temporal consistency |
| **SC4D** | Deformable 4DGS | No explicit temporal correlation | Fixed threshold | Same as above |
| **STAG4D** | Deformable 4DGS | Temporal anchors | Adaptive but frame-unified | Lacks temporal correlation; densification not frame-differentiated |
| **4DSTR (Ours)** | Deformable 4DGS | Mamba temporal encoding + residual rectification | Per-frame adaptive | Explicit temporal correlation + per-frame densification |

The core advantages of the proposed method over the strongest baseline STAG4D are: (1) Mamba establishes genuine cross-frame feature correlation rather than relying solely on anchors; and (2) densification thresholds are computed independently per frame rather than globally unified.

The following additional insights are notable:

1. **Mamba is increasingly applied in 3D/4D tasks**: From Mamba4D to the temporal encoding in this work, Mamba's linear complexity makes it an ideal choice for modeling long-sequence 3D/4D data, and its extension to other 3D tasks (e.g., point cloud sequence understanding, dynamic scene reconstruction) warrants attention.
2. **Residual rectification paradigm**: Rather than directly predicting final attributes, predicting corrections to initial predictions is a transferable paradigm applicable to other generative 3D/4D tasks.
3. **Adaptive point cloud density control**: The idea of per-frame Gaussian count adjustment can be extended to general (non-generative) 3DGS scene reconstruction—increasing density in dynamic regions and reducing it in static regions.
4. **Potential for integration with video diffusion models**: The current pipeline uses Zero123++ for multi-view generation; replacing it with more advanced video generation models (e.g., Sora-series) could further raise the performance ceiling.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The temporal residual rectification and per-frame adaptive densification ideas are clear and reasonably novel; however, the individual components (Mamba, residual learning, adaptive thresholding) are not entirely new concepts—the core contribution lies in their effective combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies are comprehensive (temporal/spatial rectification, encoder type, window size, long sequences, user study), but the quantitative evaluation dataset is small (7 objects) with limited scene diversity.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is well-structured with intuitive figures and well-motivated design choices, though certain notational conventions and method descriptions are slightly redundant in places.
- **Value**: ⭐⭐⭐⭐ Achieves significant state-of-the-art improvements on the video-to-4D task (FVD reduced by 19.9%), with a lightweight design that is practically engineering-friendly; however, the 4D generation field is evolving rapidly and the lasting impact of this method remains to be seen.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion](simba_towards_high-fidelity_and_geometrically-consistent_point_cloud_completion_.md)
- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [\[AAAI 2026\] AnchorHOI: Zero-shot Generation of 4D Human-Object Interaction via Anchor-based Prior Distillation](anchorhoi_zero-shot_generation_of_4d_human-object_interactio.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](../../ICLR2026/3d_vision/ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)
- [\[NeurIPS 2025\] Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](../../NeurIPS2025/3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
