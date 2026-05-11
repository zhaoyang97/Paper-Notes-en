---
title: >-
  [Paper Note] AllTracker: Efficient Dense Point Tracking at High Resolution
description: >-
  [ICCV 2025][3D Vision][dense point tracking] This paper proposes AllTracker, which reformulates point tracking as multi-frame long-range optical flow estimation. Through low-resolution iterative inference (2D convolution…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "dense point tracking"
  - "optical flow"
  - "long-range correspondence"
  - "high-resolution tracking"
  - "recurrent architecture"
date: 2026-05-08
content_hash: 1ab8ed77f7eac6e9
---

# AllTracker: Efficient Dense Point Tracking at High Resolution

**Conference**: ICCV 2025
**arXiv**: N/A
**Code**: [Project Page](https://alltracker.github.io)
**Area**: 3D Vision / Point Tracking
**Keywords**: dense point tracking, optical flow, long-range correspondence, high-resolution tracking, recurrent architecture

## TL;DR

This paper proposes AllTracker, which reformulates point tracking as multi-frame long-range optical flow estimation. Through low-resolution iterative inference (2D convolutions + temporal attention) followed by high-resolution upsampling, AllTracker achieves state-of-the-art accuracy for high-resolution (768–1024 px) dense point tracking across all pixels with only 16M parameters.

## Background & Motivation

Long-range point tracking in video is a fundamental problem in computer vision. Optical flow methods capture only instantaneous motion between adjacent frames and accumulate drift when chained; existing point trackers (e.g., CoTracker3) address drift and occlusion via multi-frame temporal priors, but at the cost of spatial coverage—they operate on sparse point sets and cannot produce dense outputs at high resolution. Recent "dense" tracking attempts (DTF, DELTA) are less accurate than state-of-the-art sparse trackers and run out of memory on high-resolution inputs. The core insight is that **learnable multi-frame temporal priors and high-resolution spatial awareness can be jointly achieved** by reformulating point tracking as a multi-frame long-range optical flow problem.

## Method

### Overall Architecture

AllTracker estimates high-resolution flow fields from a designated query frame to every frame in the video, and employs a sliding-window strategy for long videos. The architectural core is a recurrent module that iteratively refines correspondence estimates on a low-resolution grid, propagating spatial information via 2D convolutions and temporal information via pixel-aligned attention layers. An upsampling layer then recovers full resolution.

### Key Designs

1. **Long-Range Optical Flow as Point Tracking**: Flow fields are computed directly from the query frame to each target frame (rather than only between adjacent frames); sampling these fields yields long-range trajectories for arbitrary points. Multiple flow problems are solved simultaneously within a 16-frame sliding window, with information shared both within and across windows, enabling accurate correspondence across occlusions and large temporal gaps. This renders separate sparse trackers unnecessary.

2. **Low-Resolution Processing + High-Resolution Upsampling**: Inspired by SEA-RAFT, the main computation is performed on a $1/8$-resolution grid—low-resolution flows are estimated directly, features are extracted with ResNet-34, a multi-scale cost volume pyramid is constructed, and iterative refinement is applied via 2D convolutions. A pixel-shuffle layer then rapidly upsamples to full resolution. With only 16M parameters, the model handles 768–1024 px inputs on a 40 GB GPU.

3. **Joint Training on Optical Flow and Point Tracking Datasets**: A unified loss function is designed to support joint training on optical flow datasets (FlyingThings3D, Spring, etc.) and point tracking datasets (Kubric, PointOdyssey, etc.). Optical flow data provides dense and accurate two-frame supervision, while point tracking data provides sparse but temporally extended supervision. Joint training is critical to final performance.

### Loss & Training

The model is jointly trained on multiple synthetic datasets with uniform sampling. A long training schedule is essential for performance. No pseudo-label self-supervision is used; strong results are obtained solely by incorporating more synthetic data.

## Key Experimental Results

### Main Results

| Method | Parameters | Resolution | TAP-Vid Eval | Dense/Sparse |
|--------|-----------|-----------|-------------|-------------|
| CoTracker3 | — | Standard | SOTA (sparse) | Sparse |
| DELTA | — | Low | Second-best | Dense |
| DTF | — | Low | Non-competitive | Dense |
| **AllTracker** | **16M** | **768–1024** | **SOTA** | **Dense** |

AllTracker achieves SOTA on high-resolution dense tracking while also surpassing the latest sparse trackers on sparse evaluation benchmarks.

### Ablation Study

- Joint optical flow training vs. Kubric only: joint training yields significant gains.
- Temporal attention layers: removal degrades cross-occlusion tracking.
- Low-resolution processing ratio: $1/8$ provides the best trade-off.
- Training schedule length: longer training leads to better convergence.

### Key Findings

- Optical flow and point tracking can be addressed synergistically within a unified framework.
- 2D convolutions on low-resolution grids are sufficient for spatial message propagation without complex Transformers.
- Jointly using optical flow and tracking datasets outperforms using either alone.
- High-resolution upsampling is an underappreciated yet critical component.

## Highlights & Insights

- Minimalist and elegant design—reducing point tracking to an optical flow problem unifies two research directions.
- SOTA performance with only 16M parameters, demonstrating exceptional parameter efficiency.
- First method to simultaneously achieve dense coverage, high resolution, and SOTA accuracy.
- The joint multi-dataset training strategy is straightforward yet highly effective.

## Limitations & Future Work

- Training is conducted exclusively on synthetic data; real-world generalization relies on the robustness of feature matching.
- The sliding-window strategy may introduce boundary inconsistencies on extremely long videos.
- Occlusion handling depends on the learning capacity of the temporal attention layers, with no explicit occlusion reasoning.
- Computational cost still scales linearly with the number of frames.

## Related Work & Insights

- SEA-RAFT provides architectural inspiration for low-resolution processing with upsampling.
- CoTracker3's virtual-point design is accurate but restricts the model to sparse tracking.
- RAFT's iterative refinement strategy has been widely adopted.
- The idea of unifying optical flow and tracking is extensible to 3D scene flow.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Reformulating point tracking as optical flow offers a distinctive perspective.
- **Technical Depth**: ⭐⭐⭐⭐ — Architecture design is sophisticated with well-motivated component choices.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive ablations, multiple benchmarks, and multi-resolution evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated with thorough analysis of related work.
- **Value**: ⭐⭐⭐⭐⭐ — 16M parameters, high-resolution dense tracking, and open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Multi-View 3D Point Tracking](multi-view_3d_point_tracking.md)
- [\[ICCV 2025\] ArgMatch: Adaptive Refinement Gathering for Efficient Dense Matching](argmatch_adaptive_refinement_gathering_for_efficient_dense_matching.md)
- [\[ICCV 2025\] TAPNext: Tracking Any Point (TAP) as Next Token Prediction](tapnext_tracking_any_point_tap_as_next_token_prediction.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[NeurIPS 2025\] EUGens: Efficient, Unified, and General Dense Layers](../../NeurIPS2025/3d_vision/eugens_efficient_unified_and_general_dense_layers.md)

</div>

<!-- RELATED:END -->
