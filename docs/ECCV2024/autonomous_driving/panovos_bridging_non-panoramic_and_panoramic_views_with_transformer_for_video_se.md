---
title: >-
  [Paper Note] PanoVOS: Bridging Non-panoramic and Panoramic Views with Transformer for Video Segmentation
description: >-
  [ECCV 2024][Autonomous Driving][Panoramic Video Segmentation] Proposes the first panoramic video object segmentation dataset PanoVOS (150 videos, 19K instance annotations), revealing that existing VOS models fail to handle pixel discontinuity and severe distortion in panoramic videos, and designs PSCFormer to address left-right boundary continuity using panoramic spatial consistency attention.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Panoramic Video Segmentation"
  - "Video Object Segmentation"
  - "Dataset"
  - "Transformer"
  - "Spatial Consistency"
date: 2026-05-08
content_hash: b65a1ed11c8a27bd
---

# PanoVOS: Bridging Non-panoramic and Panoramic Views with Transformer for Video Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2309.12303](https://arxiv.org/abs/2309.12303)  
**Code**: [https://github.com/shilinyan99/PanoVOS](https://github.com/shilinyan99/PanoVOS)  
**Area**: Autonomous Driving  
**Keywords**: Panoramic Video Segmentation, Video Object Segmentation, Dataset, Transformer, Spatial Consistency

## TL;DR

Proposes the first panoramic video object segmentation dataset PanoVOS (150 videos, 19K instance annotations), revealing that existing VOS models fail to handle pixel discontinuity and severe distortion in panoramic videos, and designs PSCFormer to address left-right boundary continuity using panoramic spatial consistency attention.

## Background & Motivation

Panoramic videos ($360^\circ \times 180^\circ$ FoV) are widely used in autonomous driving and VR, providing richer spatial information than standard videos. However, existing VOS datasets (DAVIS, YouTube-VOS) only focus on planar videos captured by traditional pinhole cameras, suffering from the following issues:

**Dataset Absence**: There is no pixel-level instance-annotated dataset specifically for panoramic videos to evaluate video segmentation.

**Content Discontinuity**: After equirectangular projection of panoramic videos, the left and right boundaries are actually continuous, but are severed in the planar image. For example, a penguin crossing the left and right boundaries of an image will be split into two parts.

**Severe Distortion**: Panoramic projection causes severe object shape distortion, making the feature matching mechanisms of existing methods fail in this scenario.

Existing panoramic video datasets (SHD360, SOD360, Wild360) are either only used for saliency detection, lack instance-level annotations, or exhibit small motion, making them unsuitable for VOS evaluation.

## Method

### Overall Architecture

This work consists of two parts: (1) constructing the PanoVOS dataset; (2) proposing the PSCFormer model. PSCFormer is based on the AOT architecture, introducing a Panoramic Spatial Consistency (PSC) module to replace standard attention. Given query and reference frames, features are extracted via a memory encoder and a query encoder, followed by multiple stacked PSC blocks for spatio-temporal matching, and finally, a decoder generates the segmentation mask.

### Key Designs

1. **PanoVOS Dataset Construction**:

    - **Scale**: 150 videos, 13,995 frames, 19,145 instance annotations, 35 categories.
    - **Characteristics**: Average video length of 20s (4 times that of DAVIS/YouTube-VOS), nearly half of the videos are in 4K resolution, featuring large motion.
    - **Categories**: Covers humans (parkour, skateboard), animals (elephant, monkey), and common objects (basketball, hot air balloon).
    - **Split**: 80 training, 35 validation, and 35 testing videos, with the validation/test sets containing unseen categories for generalization evaluation.
    - **Annotation Pipeline**: Semi-automatic human-in-the-loop — manually annotating keyframes at 1fps, propagating them to all frames at 6fps using the AOT model, and then manually refining them (especially for panoramic distortion and discontinuous areas).

2. **Panoramic Space Consistency Block (PSC Block)**:

    - **Structure**: Self-Attention $\rightarrow$ Cross-Attention + PSC-Attention $\rightarrow$ FFN (GELU)
    - Self-Attention aggregates target correlation information within the query frame.
    - Cross-Attention learns target information from reference frames.
    - PSC-Attention specifically handles left-right boundary consistency (core innovation).

3. **PSC-Attention (Panoramic Space Consistency Attention)**:

    - **Mechanism**: Simulates the continuity of left and right boundaries in panoramic images through feature concatenation.
    - Shifts the rightmost $W/p$ columns of the reference frame features $\mathbf{f}(\mathbf{x}) \in \mathbb{R}^{H \times W \times C}$ to the far left, and the leftmost $W/p$ columns to the far right, keeping the middle unchanged, yielding the rearranged features $\mathbf{f}(\mathbf{x})'$.
    - Applies windowed attention on the rearranged features, where each query token only computes attention with keys in a window of size $(2s+1)^2$:

    $$\text{PSCAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T \mathbf{R}}{\sqrt{C}}\right) V$$

   where $\mathbf{R}$ is a window mask matrix, meaning a query at position $(x,y)$ only attends to keys at $(i,j)$ satisfying $(x-i)^2 \leq s^2$ and $(y-j)^2 \leq s^2$.
    - Complexity is reduced from $(HW)^2$ to $(2s+1)^2$, which is both efficient and effective.
    - **Design Motivation**: Directly concatenating the entire image doubles the length, causing an explosion in computation. Only concatenating boundary regions and using window attention is sufficient to connect originally severed objects in panoramic videos.
    - Hyperparameters: $p=2$ (concatenation ratio), $s=7$ (window size), 8 attention heads.

4. **Two Model Variants**:

    - **Ours-Base**: Uses only the first frame and the previous frame as references ($\mathcal{R} = \{1, t-1\}$), allowing fast inference.
    - **Ours-Large**: Uses multiple historical frames as references, achieving better performance.

### Loss & Training

Uses standard VOS loss (cross-entropy + dice loss). It is pre-trained on static image datasets (COCO, ECSSD) and then main-trained on the PanoVOS training set. The frame sampling intervals $\delta$ during training and testing are 2 and 5, respectively.

## Key Experimental Results

### Main Results

**Domain Transfer Results (testing existing methods on PanoVOS, trained only on traditional datasets)**:

| Method | YouTube-VOS $\mathcal{J\&F}$ | PanoVOS Val $\mathcal{J\&F}$ | Drop |
|------|-----|-----|-----|
| XMem | 85.7 | 66.1 | ↓19.6 |
| AOTL | 83.8 | 71.9 | ↓11.9 |
| AOTB | 83.5 | 70.5 | ↓13.0 |
| STCN | 83.0 | 61.8 | ↓21.2 |
| AFB-URR | 79.6 | 55.1 | ↓24.5 |

**Results after training on PanoVOS**:

| Method | Val $\mathcal{J\&F}$ | Test $\mathcal{J\&F}$ |
|------|-----|------|
| XMem | 55.7 | 53.5 |
| AOTL | 66.6 | 53.8 |
| AOTB | 67.6 | 55.4 |
| **Ours-Base** | **74.0** | **56.8** |
| **Ours-Large** | **77.9** | **59.9** |

### Ablation Study

**Effect of PSC-Attention**:

| Model | PSCAttn | Val $\mathcal{J\&F}$ | Test $\mathcal{J\&F}$ | Description |
|------|---------|------|------|------|
| Ours-Base | ✗ | 72.8 | 55.4 | Baseline |
| Ours-Base | ✓ | **74.0** | **56.8** | +1.2 / +1.4 |
| Ours-Large | ✗ | 74.8 | 59.5 | Baseline |
| Ours-Large | ✓ | **77.9** | **59.9** | +3.1 / +0.4 |

**PSCAttn vs. Standard Cross-Attention**:

| Model | Attention Type | Val $\mathcal{J\&F}$ | Test $\mathcal{J\&F}$ |
|------|-----------|------|------|
| Ours-Base | CrossAttn | 72.5 | 54.8 |
| Ours-Base | PSCAttn | **74.0** | **56.8** |
| Ours-Large | CrossAttn | 76.8 | 59.1 |
| Ours-Large | PSCAttn | **77.9** | **59.9** |

**Analysis of concatenation ratio $p$** (Ours-Large): $p=2$ is optimal ($\mathcal{J\&F}$=77.9); without concatenation, it is only 73.7 ($\downarrow$ 4.2), $p=3$ yields 76.3, and $p=5/10/15$ are all inferior to $p=2$.

### Key Findings

- **Huge Domain Transfer Gap**: All 15 existing VOS models suffer a severe performance drop on PanoVOS (average drop $\geq 20$), confirming that panoramic video is a unique and unresolved challenge.
- **Poor Performance of SAM Series**: PerSAM achieves only 19.1 $\mathcal{J\&F}$ and SAM-PT only 47.5, suggesting that visual foundation models still have substantial room for improvement in panoramic scenarios.
- **Consistently Effective PSCAttn**: It brings improvements in both Base and Large configurations, outperforming simple additional Cross-Attention.
- Training on larger scale data (YouTube-VOS, BL30K) can partially mitigate the domain gap but cannot fully resolve it.
- **Remaining Issue**: Severe distortion is still unhandled (current methods lack deformable designs).

## Highlights & Insights

- **Dataset Contribution as Core**: PanoVOS fills the gap in panoramic VOS evaluation. With 150 high-quality videos, an average length of 20s, and 4K resolution, it is more challenging than existing panoramic video datasets.
- **Ingenious PSC-Attention Design**: It models panoramic continuity via simple boundary region swapping + window attention, avoiding the computational explosion of full-image concatenation. The pipeline is simple and effective.
- The comprehensive benchmark evaluation of 15 models provides a valuable reference for future panoramic video understanding research.

## Limitations & Future Work

- **Unaddressed Distortion**: PSCFormer lacks special designs (such as deformable convolution) targeting the severe distortion in panoramic projections, and still fails on extremely deformed objects.
- **Limited Generality**: PSC-Attention is designed specifically for panoramic videos and may introduce unnecessary computation for normal videos.
- The dataset scale is relatively small (150 videos) and can be further expanded.
- Combining with panoramic depth estimation and panoramic 3D perception is a promising direction to explore.

## Related Work & Insights

- Unlike SHD360 (small motion/human scenarios) and SOD360/Wild360 (no instance annotations), PanoVOS provides a panoramic VOS benchmark combining large motion and instance annotations.
- The AOT series serves as the main baseline, upon which PSCFormer incorporates panoramic-specific spatial consistency design.
- Similar boundary continuity issues also exist in panoramic semantic segmentation (DensePASS) and panoramic 3D detection, allowing mutual inspiration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Dual contribution of dataset + method, being the first to systematically study the panoramic VOS problem, with a highly creative PSC-Attention design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Very comprehensive, featuring a benchmark of 15 models, SAM evaluations, ablation studies, hyperparameter analysis, and comparisons of multiple training strategies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, complete dataset construction process, and structured answers to 5 RQs.
- **Value**: ⭐⭐⭐⭐ — Opens up the new direction of panoramic VOS, with the dataset offering long-term value to the community, although the method itself may have a relatively narrow range of applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](../../CVPR2025/autonomous_driving/panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)
- [\[CVPR 2026\] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera](../../CVPR2026/autonomous_driving/oneocc_semantic_occupancy_prediction_for_legged_robots_with_a_single_panoramic_c.md)
- [\[ECCV 2024\] Reliability in Semantic Segmentation: Can We Use Synthetic Data?](reliability_in_semantic_segmentation_can_we_use_synthetic_data.md)
- [\[ECCV 2024\] Random Walk on Pixel Manifolds for Anomaly Segmentation of Complex Driving Scenes](random_walk_on_pixel_manifolds_for_anomaly_segmentation_of_complex_driving_scene.md)
- [\[ECCV 2024\] ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation](ittakestwo_leveraging_peer_representations_for_semi-supervised_lidar_semantic_se.md)

</div>

<!-- RELATED:END -->
