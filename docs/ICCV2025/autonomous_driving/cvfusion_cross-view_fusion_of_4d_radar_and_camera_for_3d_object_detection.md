---
title: >-
  [Paper Note] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection
description: >-
  [ICCV 2025][Autonomous Driving][4D millimeter-wave radar] This paper proposes CVFusion — the first two-stage 4D radar-camera fusion network for 3D object detection. Stage 1 generates high-recall proposals via a Radar-Gui…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "4D millimeter-wave radar"
  - "camera fusion"
  - "3D object detection"
  - "two-stage detection"
  - "BEV fusion"
date: 2026-05-08
content_hash: dfe479773938fefc
---

# CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection

**Conference**: ICCV 2025
**arXiv**: [2507.04587](https://arxiv.org/abs/2507.04587)  
**Code**: [https://github.com/zhzhzhzhzhz/CVFusion](https://github.com/zhzhzhzhzhz/CVFusion)  
**Area**: Autonomous Driving
**Keywords**: 4D millimeter-wave radar, camera fusion, 3D object detection, two-stage detection, BEV fusion

## TL;DR

This paper proposes CVFusion — the first two-stage 4D radar-camera fusion network for 3D object detection. Stage 1 generates high-recall proposals via a Radar-Guided Iterative (RGIter) BEV fusion module, while Stage 2 refines each proposal by aggregating heterogeneous multi-view features through Point-Guided Fusion (PGF) and Grid-Guided Fusion (GGF). CVFusion achieves mAP improvements of 9.10% and 3.68% on VoD and TJ4DRadSet, respectively.

## Background & Motivation

- **Background**: 4D millimeter-wave radar has attracted broad attention in autonomous driving due to its robustness in adverse weather conditions (rain, fog, snow). Compared to conventional 3D radar, 4D radar additionally provides elevation measurements, yielding LiDAR-like 3D point clouds. However, 4D radar suffers from two inherent limitations: (1) **extreme sparsity** — only 0.1k–2k points per frame (vs. 10k–100k for LiDAR); and (2) **high noise** — multipath effects in millimeter-wave sensing lead to inaccurate measurements. These characteristics make radar-only detection insufficient, motivating fusion with cameras.
- **Limitations of Prior Work**: Existing 4D radar-camera fusion methods (e.g., RCFusion, LXL) all adopt **single-stage pipelines** that project both modalities into BEV space and fuse them directly. This leads to a **feature misalignment** problem: depth ambiguity in monocular images causes projection errors from perspective view to BEV, while sparse and noisy radar points cannot adequately resolve image depth uncertainty. Naively concatenating the two BEV feature maps causes misalignment, which particularly degrades performance on small objects and introduces localization errors and false positives.
- **Key Challenge**: Detection should not be completed in a single step; instead, it should proceed through progressive refinement stages. The two-stage paradigm has proven effective in LiDAR-based detection (e.g., PV-RCNN), yet has never been applied to 4D radar-camera fusion — primarily because radar sparsity makes effective feature aggregation during proposal refinement highly non-trivial.
- **Goal**: Introduce the two-stage paradigm into 4D radar-camera fusion and design effective multi-view, multi-granularity feature aggregation strategies tailored to the sparse and noisy nature of 4D radar.

## Method

### Overall Architecture

CVFusion consists of two stages:
- **Stage 1**: Independent 4D radar and image branches extract features and project them to the BEV plane. An RGIter-BEV fusion module combines these features, which are then fed into an RPN to generate proposal boxes.
- **Stage 2**: For each proposal, two branches — PGF (Point-Guided Fusion) and GGF (Grid-Guided Fusion) — aggregate features from three views (radar points, image front view, and BEV) to perform refined prediction.

### Key Designs

1. **Radar-Guided Iterative BEV Fusion (RGIter-BEV)**: To address depth uncertainty in image BEV features, multi-scale radar BEV features $F_{B,k}^{Rad}$ ($k=0,1,2$) are used to generate occupancy probability weight maps $W_k = \text{Sigmoid}(\text{Conv2d}(F_{B,k}^{Rad}))$, which are applied to reweight camera BEV features: $F_{B,k}^{Cam'} = F_{B,k}^{Cam} \odot W_k$. This weighting enhances image features at radar-supported locations while suppressing depth-uncertain regions. The key innovation lies in **iterative multi-scale processing**: the weighted image features are passed through a strided convolution to produce the input for the next scale, $F_{B,k+1}^{Cam} = \text{Conv2d}(F_{B,k}^{Cam'}, \text{stride}=2)$, propagating radar-guided spatial information across scales. The fused features from all three scales are unified via up/downsampling and concatenated.

2. **Point-Guided Fusion (PGF)**: Handles proposals containing radar points. For each radar point $p$, its voxel feature $f_p$ is projected onto the 2D image plane to obtain position $r_p$, and **Cross-Modal Deformable Attention (CMDA)** is applied to query nearby image features: $f_I^* = \text{CMDA}(f_p, r_p, F_I)$. CMDA uses point features as queries and image features as keys/values, aggregating local image information via learnable sampling offsets and attention weights. The queried features are concatenated with point features and fused via MLP: $f_p^* = \text{MLP}([f_p, f_I^*])$. A **Kernel Density Estimation (KDE)** feature is additionally incorporated to distinguish isolated noise points. Finally, RoI-Pooling over a $U^3$ grid aggregates features into a proposal-level representation $f_b^{pt}$.

3. **Grid-Guided Fusion (GGF)**: Addresses the large proportion of proposals that contain no radar points. Each proposal is divided into $U^3$ uniform grid cells (independent of radar points). A **Grid Position Encoder (GPE)** generates initial features for each grid cell: $f_{g_j}^{pos} = \text{MLP}(\delta_j, c_b, \log(|N_{g_j}| + \epsilon))$, encoding the relative offset from the grid to the box center and the number of points within the grid. Features are then aggregated through **two cascaded CMDA modules**: the first projects each grid cell onto the image plane to query front-view features $f_{g_j}^{fv}$; the second projects $f_{g_j}^{fv}$ onto the BEV plane to query fused BEV features. The **spatial orthogonality** between the front view and BEV provides constraints along two independent directions for bounding box refinement.

4. **PGF + GGF Fusion**: The outputs $f_b^{pt}$ and $f_b^{gd}$ from the two branches are deeply fused via Transformer self-attention: $f_b = \text{SelfAttn}(f_b^{pt} + f_b^{gd})$, and subsequently fed into a Refine Head for confidence and box offset prediction.

### Loss & Training

The total loss is $L = L_{RPN} + L_{refine}$, where $L_{refine}$ comprises confidence loss and regression loss. Training is conducted using the OpenPCDet framework with batch size 2, learning rate 0.01, 80 epochs, and 4 GTX 3090 GPUs. The 2D backbone uses an ImageNet-pretrained Swin-Tiny with frozen weights; all other modules are trained from scratch. Post-processing applies a confidence threshold of 0.1 and an NMS threshold of 0.01.

## Key Experimental Results

### Main Results

| Method | Modality | VoD Entire mAP↑ | VoD Corridor mAP↑ | TJ4DRad 3D mAP↑ |
|--------|----------|-----------------|-------------------|-----------------|
| PointPillars | LiDAR | 61.16 | 79.93 | 43.76 (BEV) |
| LXL | R+C | 56.31 | 72.93 | 41.20 (BEV) |
| RCFusion | R+C | 49.65 | 69.23 | — |
| RCBEVDet | R+C | 49.99 | 69.80 | — |
| CVFusion (Stage 1) | R+C | 59.70 | 76.66 | — |
| **CVFusion** | **R+C** | **65.41** | **82.42** | **44.07 (BEV)** |

CVFusion advances the state of the art on VoD from 56.31 to 65.41 (+9.10%), and in the Corridor region even surpasses the LiDAR-based method PointPillars (82.42 vs. 79.93).

### Ablation Study

| Configuration | VoD Entire mAP↑ | VoD Corridor mAP↑ | Note |
|---------------|-----------------|-------------------|------|
| Camera only | 18.8 | 35.3 | Single-modality baseline |
| Radar only | 46.7 | 67.8 | Radar-only detection |
| Radar+Cam (w/o RGIter) | 57.1 | 76.3 | Simple BEV concatenation |
| +RGIter | 59.7 | 76.7 | Iterative fusion +2.6% |
| +RGIter+PGF | 61.7 | 81.5 | Point-level refinement +2.0% |
| +RGIter+GGF | 63.0 | 81.9 | Grid-level refinement +3.3% |
| **+RGIter+PGF+GGF** | **65.4** | **82.4** | PGF+GGF complementary |

### Key Findings

- **Two-stage design yields substantial gains**: Stage 1 alone significantly outperforms existing single-stage methods (59.70 vs. 56.31 for LXL), and Stage 2 further improves this to 65.41.
- **GGF contributes more than PGF** (+3.3% vs. +2.0%): Since a large fraction of proposals contain no radar points, GGF's grid-based strategy ensures feature availability for all proposals.
- **PGF and GGF are complementary**: Using both branches together yields larger gains (+5.7%) than either branch alone (+2.0% / +3.3%), demonstrating that point-level precision and grid-level coverage are mutually beneficial.
- **Surpasses LiDAR in the VoD Corridor region**: 82.42 vs. 79.93, demonstrating the strong potential of 4D radar-camera fusion for close-range detection.
- Stage 1–only FPS is 6.9; the full model achieves 5.4 FPS, representing an acceptable computational overhead.

## Highlights & Insights

- This work is the first to introduce the two-stage paradigm into 4D radar-camera fusion, with a clear motivation and significant empirical gains.
- The RGIter design — radar occupancy probability weighting combined with iterative downsampling — is simple yet effective, leveraging radar spatial priors to correct depth ambiguity in image BEV features.
- The dual-branch PGF+GGF design elegantly addresses 4D radar sparsity: points are used when available; grids are used otherwise — the two strategies are independent yet mutually reinforcing.
- The exploitation of FV-BEV spatial orthogonality provides constraints along two independent directions during proposal refinement.

## Limitations & Future Work

- The two-stage design introduces additional computational overhead (FPS drops from 6.9 to 5.4), posing a challenge for latency-sensitive autonomous driving scenarios.
- Evaluation is limited to two medium-scale datasets (VoD and TJ4DRadSet); assessment on larger-scale benchmarks (e.g., nuScenes 4D radar subset) is absent.
- Performance under adverse weather conditions — the primary advantage scenario for 4D radar — is not explicitly analyzed.
- KDE-based denoising is a post-hoc processing step; more principled radar point cloud preprocessing may yield greater benefits.

## Related Work & Insights

- The two-stage design is inspired by LiDAR-based methods such as PV-RCNN and LoGoNet, with key design adaptations to accommodate 4D radar sparsity.
- CMDA (Cross-Modal Deformable Attention) is derived from Deformable DETR, applied here for cross-modal queries between points/grids and image features.
- Compared to the contemporaneous RCBEVDet (which uses CAMF alignment), CVFusion addresses the alignment problem from a more systematic "multi-view, multi-granularity" perspective.

## Rating

- Novelty: ⭐⭐⭐⭐ First two-stage 4D radar-camera fusion; the PGF+GGF dual-branch design is creative
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablations with comparisons across multiple modality configurations
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear; method descriptions are detailed
- Value: ⭐⭐⭐⭐⭐ 4D radar-camera fusion surpassing LiDAR baselines represents a significant advancement for autonomous driving perception

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](../../CVPR2026/autonomous_driving/r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [\[ICCV 2025\] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)
- [\[AAAI 2026\] Exploring Surround-View Fisheye Camera 3D Object Detection](../../AAAI2026/autonomous_driving/exploring_surround-view_fisheye_camera_3d_object_detection.md)
- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [\[ICCV 2025\] Robust 3D Object Detection using Probabilistic Point Clouds from Single-Photon LiDARs](robust_3d_object_detection_using_probabilistic_point_clouds_from_single-photon_l.md)

</div>

<!-- RELATED:END -->
