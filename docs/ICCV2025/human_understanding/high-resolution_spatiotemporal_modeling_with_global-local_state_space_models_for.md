---
title: >-
  [Paper Note] High-Resolution Spatiotemporal Modeling with Global-Local State Space Models for Video-Based Human Pose Estimation
description: >-
  [ICCV 2025][Human Understanding][Video Human Pose Estimation] This paper proposes GLSMamba, the first pure-Mamba framework for video-based human pose estimation (VHPE). It models global dynamic context via a Global Spati…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "Video Human Pose Estimation"
  - "Mamba"
  - "State Space Model"
  - "Spatiotemporal Modeling"
  - "High-Resolution"
date: 2026-05-08
content_hash: f8976ca200011175
---

# High-Resolution Spatiotemporal Modeling with Global-Local State Space Models for Video-Based Human Pose Estimation

**Conference**: ICCV 2025
**arXiv**: [2510.11017](https://arxiv.org/abs/2510.11017)  
**Code**: N/A  
**Area**: Human Understanding
**Keywords**: Video Human Pose Estimation, Mamba, State Space Model, Spatiotemporal Modeling, High-Resolution

## TL;DR

This paper proposes GLSMamba, the first pure-Mamba framework for video-based human pose estimation (VHPE). It models global dynamic context via a Global Spatiotemporal Mamba (GSM) module—featuring 6D selective space-time scanning and spatiotemporal-modulated scan merging—and captures local keypoint details via a Local Refinement Mamba (LRM) with windowed spatiotemporal scanning. The method achieves state-of-the-art performance on four benchmarks with linear computational complexity.

## Background & Motivation

Video human pose estimation (VHPE) requires dense spatiotemporal analysis, with the key challenge being the simultaneous capture of:
- **Global dynamic context**: Overall body motion patterns and trends
- **Local motion details**: High-frequency variations at individual keypoints

Inherent limitations of existing approaches:

**CNN-based methods** (e.g., TDMI): Fixed receptive fields limit global reasoning, leading to large errors under occlusion and motion blur.

**Transformer-based methods** (e.g., DiffPose): Capable of capturing global dependencies but neglect local high-frequency details, and incur quadratic complexity on high-resolution sequences—directly operating at $\frac{1}{4}$ resolution $\times T$ frames (15,360 tokens) results in OOM.

**Existing video Mamba methods** (e.g., VideoMamba): Perform only per-frame bidirectional scanning by flattening spatial tokens, which increases the distance between temporally adjacent tokens and lacks dedicated designs for local details.

Core observation: There is a need for an architecture that (1) performs global modeling on high-resolution spatiotemporal sequences with linear complexity, and (2) simultaneously enhances local keypoint motion details.

## Method

### Overall Architecture

Input video sequence → Visual encoder (ViTPose, frozen) extracts high-resolution features → Global Spatiotemporal Mamba (GSM, 4 blocks) → Local Refinement Mamba (LRM, 2 blocks) → Detection head → Pose heatmaps.

Feature resolution is $\frac{1}{4} \times H \times W \times T$. For a 5-frame sequence ($\delta=2$, two preceding and two following frames plus the current frame), the total token count reaches 15,360.

### Key Designs

1. **Global Spatiotemporal Mamba (GSM)**:

    - **Sequential Channel Attention**: Concatenates the feature sequence and applies GAP → MLPs → sigmoid to produce per-frame channel attention weights, adaptively activating important spatiotemporal information.
    - **6D Selective Space-Time Scan (STS6D)**: Flattens the feature sequence into 1D along six spatiotemporal scanning paths and feeds each into an S6 block. Specifically, multi-frame features are stacked into a panoramic spatiotemporal representation; horizontal/vertical traversals yield $\tilde{\mathbf{y}}_1, \tilde{\mathbf{y}}_4$ (unified scan, capturing high-level spatiotemporal representations); per-frame spatial traversals yield $\tilde{\mathbf{y}}_2, \tilde{\mathbf{y}}_5$ (spatial scan, capturing complete body spatial context); pixel-wise temporal traversals yield $\tilde{\mathbf{y}}_3, \tilde{\mathbf{y}}_6$ (temporal scan, capturing dense motion trends).
    - **Spatial- and Temporal-Modulated scan Merging (STMM)**: First merges bidirectional scan results by type ($\tilde{\mathbf{y}}_u, \tilde{\mathbf{y}}_s, \tilde{\mathbf{y}}_t$), then applies spatial and temporal modulation compensation via Deformable Convolution to adaptively aggregate scan knowledge from different semantic streams.

   Design Motivation: Adapts 1D Mamba to video spatiotemporal modeling; the 6-direction scanning fully exploits information across all dimensions, and DCN-based adaptive fusion avoids the information loss caused by simple summation.

2. **Local Refinement Mamba (LRM)**:

    - **Windowed Space-Time Scan (WSTS)**: Partitions the feature sequence into non-overlapping 3D temporal tube windows (e.g., $8\times6\times T$), performing forward and backward scanning within each window and feeding the results into S6 blocks.
    - Enhances local details while maintaining a sequence-length-scale receptive field.
    - Removes Sequential Channel Attention and replaces STS6D/STMM with WSTS.

   Design Motivation: While GSM focuses on global understanding, it lacks the local high-frequency details of individual keypoints. LRM complements this with fine-grained motion information via dense scanning within local windows.

3. **Dual-stream gating design**: In each GSM block, the main stream passes through STS6D+STMM to produce global features $\tilde{\mathcal{F}}$; a parallel stream applies depthwise convolution + LayerNorm + SiLU to produce gating attention $\bar{\mathcal{A}}$. The two streams are multiplied element-wise before passing through an FFN.

### Loss & Training

- Standard heatmap estimation loss: $\mathcal{L}_H = \|\hat{\mathbf{H}}^i_t - \mathbf{H}^i_t\|_2^2$
- Initialized with ViTPose pretrained weights (on COCO); backbone is frozen during inference.
- AdamW optimizer; initial lr $1\times10^{-4}$, decayed to $1\times10^{-5}$ at epoch 6, and $1\times10^{-6}$ at epoch 12.
- Data augmentation: random rotation/scaling, cropping, and flipping.
- Temporal span $\delta=2$ (5 frames total); trained for 20 epochs on a single TITAN RTX GPU.

## Key Experimental Results

### Main Results (Tables)

**PoseTrack2017 Validation Set (mAP):**

| Method | Backbone | Mean mAP |
|--------|----------|----------|
| PoseWarper | HRNet-W48 | 81.2 |
| DCPose | HRNet-W48 | 82.8 |
| FAMI-Pose | HRNet-W48 | 84.8 |
| TDMI | HRNet-W48 | 85.7 |
| DiffPose | ViT-B | 86.4 |
| DSTA | ViT-H | 85.6 |
| **GLSMamba-B** | **ViT-B** | **86.9** |
| **GLSMamba-H** | **ViT-H** | **88.0** |

**PoseTrack2018 / PoseTrack21 / Sub-JHMDB:**

| Dataset | GLSMamba-B | GLSMamba-H | Prev. SOTA |
|---------|------------|------------|------------|
| PoseTrack2018 | 84.2 | **84.9** | 83.5 (TDMI/DSTA) |
| PoseTrack21 | 84.1 | **84.7** | 83.5 (TDMI/DSTA) |
| Sub-JHMDB | **97.9** | - | 96.0 (FAMI-Pose) |

### Ablation Study (Tables)

**Component Ablation (PoseTrack2017):**

| Setting | GSM | LRM | mAP |
|---------|-----|-----|-----|
| Backbone only | - | - | 74.2 |
| + GSM | ✓ | - | 86.0 (+11.8) |
| + GSM + LRM (full) | ✓ | ✓ | **86.9** (+0.9) |

**STS6D Scanning Direction Ablation:**

| Scanning Directions | #Params | GFLOPs | mAP |
|--------------------|---------|--------|-----|
| Unified scan | 9.1M | 137.4 | 85.8 |
| + Spatial scan | 9.4M | 138.1 | 86.5 |
| + Spatial + Temporal scan (full STS6D) | 9.8M | 138.9 | **86.9** |
| Full STS6D w/o STMM | 9.1M | 137.4 | 86.2 |

**Resolution Impact and Computational Efficiency:**

| Method | Resolution | #Tokens | #Params | GFLOPs | mAP |
|--------|-----------|---------|---------|--------|-----|
| GLSMamba-B | $\frac{1}{4}\times T$ | 15,360 | 9.8M | 138.9 | **86.9** |
| GLSMamba-BLR | $\frac{1}{16}\times T$ | 960 | 9.8M | 85.1 | 85.7 |
| TransLR | $\frac{1}{16}\times T$ | 960 | 46.3M | 125.7 | 84.2 |
| TransNR | $\frac{1}{8}\times T$ | 3,840 | 47M | 315.2 | 84.8 |
| TransHR | $\frac{1}{4}\times T$ | 15,360 | - | - | OOM |

### Key Findings

1. **GSM contributes most**: Introducing GSM alone raises mAP from 74.2 to 86.0 (+11.8), demonstrating the critical importance of global spatiotemporal modeling for VHPE.
2. **6-direction scanning yields incremental gains**: Unified → +Spatial → +Temporal scanning improves mAP from 85.8 → 86.5 → 86.9 with negligible additional computation.
3. **STMM outperforms simple summation by 0.7 mAP**: Adaptive fusion of semantically distinct scan results is important.
4. **High resolution is significantly beneficial**: $\frac{1}{4}$ resolution outperforms $\frac{1}{16}$ by 1.2 mAP, while Transformer-based architectures OOM at the same resolution.
5. **Extremely parameter-efficient**: Only 9.8M trainable parameters (86.2% fewer than methods that fine-tune the backbone), with 138.9 GFLOPs (66% of PoseWarper).

## Highlights & Insights

- **First pure-Mamba VHPE framework**: Demonstrates the substantial potential of SSMs for dense prediction tasks in computer vision.
- **Design philosophy of decoupled global-local modeling**: GSM and LRM serve distinct roles, proving more effective than a unified architecture.
- **Linear complexity for high-resolution sequences**: While Transformers OOM at 15,360 tokens, Mamba operates at only 138.9 GFLOPs.
- **Elegant multi-direction design of STS6D**: The three scan types—unified, spatial, and temporal—each capture distinct semantics and are strongly complementary.
- **Extremely low training cost**: Frozen backbone with only 9.8M trainable parameters, trainable on a single TITAN RTX GPU.

## Limitations & Future Work

1. The backbone weights are fully frozen, which may limit adaptability in domain-specific settings.
2. The temporal span is fixed at $\delta=2$ (5 frames); a longer temporal range may further improve performance.
3. On Sub-JHMDB, there remains a notable gap compared to the post-processing method DeciWatch (98.8 vs. 97.9), as post-processing methods operate in pose coordinate space and are qualitatively different.
4. The local window size ($8\times6\times T$) is fixed; adaptive window sizes may be more effective.
5. Other dense spatiotemporal tasks, such as 3D human pose estimation and video segmentation, remain unexplored.

## Related Work & Insights

- **SSM/Mamba family**: The evolution from S4 to Mamba; this work extends 1D Mamba to high-resolution video spatiotemporal modeling.
- **Limitations of CNN vs. Transformer**: CNNs have limited receptive fields; Transformers suffer from quadratic complexity; Mamba achieves the best trade-off at high resolution.
- **Comparison with VideoMamba**: VideoMamba performs only simple per-frame flattening, whereas the proposed 6D scanning combined with windowed local scanning provides a more comprehensive treatment.
- The proposed approach provides a reference for adapting Mamba to other spatiotemporal tasks such as tracking and action recognition.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First pure-Mamba VHPE framework; STS6D multi-direction scanning and STMM fusion are novel designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive validation on four benchmarks with detailed ablations covering components, scanning directions, resolution, and computational efficiency.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, complete mathematical derivations, and rich visualizations (activation maps, comparisons).
- **Value**: ⭐⭐⭐⭐ Opens a new direction for Mamba in dense spatiotemporal prediction tasks, with a prominent computational efficiency advantage.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals](../../AAAI2026/human_understanding/kinest_a_kinematics-guided_spatiotemporal_state_space_model_for_human_motion_tra.md)
- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](../../CVPR2026/human_understanding/e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)
- [\[CVPR 2026\] GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion](../../CVPR2026/human_understanding/gazeonce360_fisheye-based_360_multi-person_gaze_estimation_with_global-local_fea.md)
- [\[ICCV 2025\] ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling](imhead_a_large-scale_implicit_morphable_model_for_localized_head_modeling.md)
- [\[ICCV 2025\] MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)

</div>

<!-- RELATED:END -->
