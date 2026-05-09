---
title: >-
  [Paper Note] GSOT3D: Towards Generic 3D Single Object Tracking in the Wild
description: >-
  [ICCV 2025][3D Vision][3D single object tracking] This paper presents GSOT3D, the largest generic 3D single object tracking benchmark to date, comprising 620 multimodal sequences (point cloud + RGB + depth) spanning 54 object categories. It supports three 3D tracking tasks (PC / RGB-PC / RGB-D) and introduces PROT3D, a progressive spatiotemporal tracker that achieves state-of-the-art performance via 9DoF bounding box estimation.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D single object tracking
  - generic tracking
  - point cloud
  - multimodal
  - benchmark dataset
  - 9DoF
date: 2026-05-08
content_hash: 9b79844e94e6c77d
---

# GSOT3D: Towards Generic 3D Single Object Tracking in the Wild

**Conference**: ICCV 2025
**arXiv**: [2412.02129](https://arxiv.org/abs/2412.02129)
**Code**: [ailovejinx/GSOT3D](https://github.com/ailovejinx/GSOT3D)
**Area**: 3D Vision
**Keywords**: 3D single object tracking, generic tracking, point cloud, multimodal, benchmark dataset, 9DoF

## TL;DR

This paper presents GSOT3D, the largest generic 3D single object tracking benchmark to date, comprising 620 multimodal sequences (point cloud + RGB + depth) spanning 54 object categories. It supports three 3D tracking tasks (PC / RGB-PC / RGB-D) and introduces PROT3D, a progressive spatiotemporal tracker that achieves state-of-the-art performance via 9DoF bounding box estimation.

## Background & Motivation

3D single object tracking (SOT) is critical for autonomous driving, mobile robotics, and navigation, yet existing benchmarks severely constrain the development of generic 3D trackers:

**Extremely limited object categories**: KITTI covers only 8 categories and NuScenes 23, both confined to vehicles and pedestrians in autonomous driving scenes, leaving trackers unable to generalize to everyday objects.

**Homogeneous scenes**: Existing datasets are drawn exclusively from traffic scenarios, failing to cover diverse indoor/outdoor environments (offices, parks, homes, etc.), and are thus ill-suited for training and evaluating generic trackers.

**Restricted degrees of freedom**: KITTI and NuScenes employ 7DoF bounding boxes (4D pose + 3D size), omitting full 6D pose (3D translation + 3D rotation), and therefore cannot accurately describe objects in arbitrary orientations.

**Insufficient RGB-D 3D tracking data**: Track-it-in-3D provides 9DoF annotations but contains only 300 sequences (36K frames), which is insufficient for training deep 3D trackers.

**Core insight**: Advancing generic 3D tracking requires a large-scale benchmark with rich category diversity, varied scenes, multiple modalities, and precise 9DoF annotations. GSOT3D fills this gap with 620 sequences, 54 categories, and 123K frames, and is the first benchmark to simultaneously support PC, RGB-PC, and RGB-D 3D tracking tasks.

## Method

### GSOT3D Benchmark Construction

**Data collection platform**: A Clearpath Husky A200 mobile robot equipped with a 64-beam LiDAR (Ouster OS-64), a depth camera (OAK D-Pro), and an RGB camera (FLIR). All sensors are calibrated and synchronized, with a 20 fps output rate.

**Category design**: 54 subcategories organized under 10 meta-categories (furniture, humans, vehicles, household items, office supplies, food, animals, sports equipment, toys, and miscellaneous), covering everyday objects suitable for 3D tracking.

**Annotation quality assurance**:
- Each frame is manually annotated with the tightest 9DoF 3D bounding box (3D translation + 3D rotation + 3D size).
- An iterative pipeline of multi-round expert review and annotator correction ensures annotation accuracy.
- Seven sequence attributes are labeled: invisibility (INV), deformation (DEF), fast motion (FM), rotation (ROT), scale variation (SV), similar distractors (SD), and sparsity (SPA).

**Dataset scale comparison**:

| Benchmark | Sequences | Frames | Categories | Scenes | Supported Tasks |
|-----------|-----------|--------|------------|--------|-----------------|
| KITTI | 21 | 15K | 8 | Outdoor | PC, RGB-PC |
| NuScenes | 1000 | 40K | 23 | Outdoor | PC, RGB-PC |
| Track-it-in-3D | 300 | 36K | 44 | Indoor/Outdoor | RGB-D |
| **GSOT3D** | **620** | **123K** | **54** | **Indoor/Outdoor** | **PC, RGB-PC, RGB-D** |

**Evaluation protocol**: Mean average overlap (mAO) and mean success rate (mSR@0.5/0.75) computed via 3D IoU. Precision metrics are excluded because center-point distance cannot assess the size and orientation accuracy of 9DoF bounding boxes.

### PROT3D Tracker

PROT3D is a category-agnostic tracker for 3D-SOT_PC built around a progressive spatiotemporal network.

**Overall pipeline**:
1. A shared backbone $\Phi(\cdot)$ extracts the current-frame feature $\mathbf{x}^1_t$ and features from the past $K$ frames, which are concatenated into a memory feature $\mathbf{H}_{t-1}$.
2. A multi-stage progressive architecture iteratively refines the search region features.

**Per-stage processing** (stage $i$):
- Spatiotemporal Transformer fusion: $\mathbf{F}^i_t = \text{SPT}(\mathbf{x}^i_t, \mathbf{H}_{t-1})$, involving cross-attention and self-attention.
- MLP localization: $R^i_t = [C^i_t, M^i_t, S^i_t]$ (target center, objectness mask, proposal score).
- FPS sampling + feature transformation: $\mathbf{x}^{i+1}_t = \text{FTB}(\bar{C}^i_t, M^i_t) + \text{Conv1D}(S^i_t)$.
- Refined features are passed to the next stage for further refinement.

**Final localization**: The last stage feeds into an MLP that predicts 9DoF bounding box parameters and objectness score:

$$\mathcal{R}_t = \text{MLP}(\mathbf{x}^{N+1}_t), \quad b_t = \mathcal{B}_t(h), \quad h = \arg\max_d \mathcal{S}(d)$$

**Key designs**:
- Progressive feature refinement: search region features become increasingly discriminative by encoding target information at each stage.
- 9DoF bounding box prediction: predicts offsets for center translation, orientation, and size.
- Multi-frame memory: temporal information from the past $K$ frames enhances robustness.

## Key Experimental Results

### Main Results: 3D-SOT_PC Overall Performance

| Tracker | mAO (%) | mSR50 (%) | mSR75 (%) |
|---------|---------|-----------|-----------|
| P2B | 9.79 | 8.59 | 1.75 |
| BAT | 6.56 | 3.54 | 0.88 |
| M2-Track | 20.26 | 14.34 | 1.88 |
| MBPTrack | 20.54 | 16.55 | 2.57 |
| M3SOT | 17.40 | 12.47 | 1.74 |
| **PROT3D** | **21.97** | **19.76** | **5.22** |

**Key Findings**: PROT3D leads on all metrics, with the most pronounced gain on mSR75 (+2.65% vs. MBPTrack), demonstrating that progressive refinement is particularly effective for high-precision localization. The substantial performance drop of all existing trackers on GSOT3D confirms the significant challenge of generic 3D tracking.

### Ablation Study

| Configuration | mAO (%) | mSR50 (%) | mSR75 (%) |
|---------------|---------|-----------|-----------|
| 1-stage + 7DoF | 19.86 | 15.16 | 2.36 |
| 1-stage + 9DoF | 20.03 | 15.46 | 3.29 |
| **2-stage + 9DoF (PROT3D)** | **21.97** | **19.76** | **5.22** |
| 3-stage + 9DoF | 21.58 | 19.61 | 5.19 |
| Memory K=2 | 21.37 | 19.52 | 5.32 |
| **Memory K=3** | **21.97** | **19.76** | **5.22** |
| Memory K=4 | 21.84 | 19.69 | 5.17 |

**Key Findings**:
- 9DoF vs. 7DoF: mSR75 improves from 2.36% to 3.29% (+39.4%), indicating that more precise pose estimation significantly boosts success at high IoU thresholds.
- Progressive architecture (2-stage) raises mSR50 from 15.46% to 19.76% (+27.8%), while 3 stages show slight overfitting.
- Memory size K=3 is optimal; including too many historical frames may introduce noise.

### Cross-Benchmark Comparison with KITTI

| Tracker | KITTI mAO (%) | GSOT3D mAO (%) | Drop |
|---------|--------------|----------------|------|
| MBPTrack | 71.95 | 20.54 | −71.4% |
| M2-Track | 67.71 | 20.26 | −70.1% |
| CXTrack | 70.18 | 14.29 | −79.6% |

All trackers suffer a 70–80% performance collapse on GSOT3D, highlighting the enormous difficulty of transferring from few-category traffic scenes to diverse generic scenarios.

## Highlights & Insights

1. **Comprehensive and forward-looking benchmark**: GSOT3D is the first benchmark to simultaneously support PC, RGB-PC, and RGB-D 3D tracking tasks within a single dataset, with 54 categories and precise 9DoF annotations.
2. **Simplicity and effectiveness of progressive refinement**: PROT3D's multi-stage cascade progressively encodes target information, rendering search region features increasingly discriminative in a straightforward yet effective manner.
3. **Exposing the gap in generic 3D tracking**: All state-of-the-art trackers perform poorly on GSOT3D (mAO only 6–22%), far below their performance on KITTI (60–72%), underscoring the importance of diverse training data for generic 3D tracking.
4. **Evidence for data-driven improvements**: Retraining on GSOT3D yields substantial gains (e.g., P2B mAO improves from 2.81% to 9.79%), confirming the critical role of diverse training data.

## Limitations & Future Work

- Experiments focus primarily on the PC modality; RGB-PC and RGB-D multimodal tracking remain underexplored.
- Sequences are relatively short (198 frames on average), limiting applicability to long-term tracking research.
- Although larger than existing 3D SOT benchmarks, the dataset scale (620 sequences) is far smaller than 2D tracking benchmarks (thousands to tens of thousands of videos).
- PROT3D is restricted to the point cloud modality and does not exploit the complementarity of RGB and depth information.

## Related Work & Insights

- **3D SOT benchmarks**: KITTI (8 categories, traffic scenes), NuScenes (23 categories, traffic scenes), Track-it-in-3D (44 categories, RGB-D, 300 sequences).
- **Generic 2D tracking benchmarks**: Large-scale multi-category datasets such as GOT-10K, LaSOT, and TrackingNet.
- **3D tracking algorithms**: Point-cloud-based Siamese/Transformer trackers including P2B, BAT, M2-Track, CXTrack, and MBPTrack.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The primary contribution lies in the comprehensiveness of the benchmark; the progressive architecture of PROT3D is notable but not highly original.
- **Technical Quality**: ⭐⭐⭐⭐ — Data collection and annotation pipelines are rigorous, though evaluating only the PC modality is a limitation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Eight baselines, detailed ablations, attribute analysis, and cross-benchmark comparisons are collectively convincing.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with rich illustrations.
- **Overall Score**: 7.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Multi-View 3D Point Tracking](multi-view_3d_point_tracking.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [\[ICCV 2025\] Learning Robust Stereo Matching in the Wild with Selective Mixture-of-Experts](learning_robust_stereo_matching_in_the_wild_with_selective_mixture-of-experts.md)

</div>

<!-- RELATED:END -->
