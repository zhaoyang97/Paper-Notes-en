---
title: >-
  [Paper Note] TAPVid-360: Tracking Any Point in 360 from Narrow Field of View Video
description: >-
  [NeurIPS 2025][Video Understanding][point tracking] This paper introduces the TAPVid-360 task and dataset, requiring models to track the 3D direction of query points (including those outside the field of view) in narrow…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "point tracking"
  - "360° video"
  - "out-of-field-of-view tracking"
  - "spatial representation"
  - "panoramic understanding"
date: 2026-05-08
content_hash: 0a018a0530fe2a92
---

# TAPVid-360: Tracking Any Point in 360 from Narrow Field of View Video

**Conference**: NeurIPS 2025
**arXiv**: [2511.21946](https://arxiv.org/abs/2511.21946)  
**Code**: [https://finlay-hudson.github.io/tapvid360](https://finlay-hudson.github.io/tapvid360) (project page)  
**Area**: Video Understanding
**Keywords**: point tracking, 360° video, out-of-field-of-view tracking, spatial representation, panoramic understanding

## TL;DR

This paper introduces the TAPVid-360 task and dataset, requiring models to track the 3D direction of query points (including those outside the field of view) in narrow field-of-view video. By leveraging 360° video to generate training data and fine-tuning CoTracker3 for directional prediction, the proposed approach substantially outperforms existing methods on out-of-field-of-view tracking.

## Background & Motivation

Humans possess a remarkable ability to construct panoramic mental models of their surroundings—even when only a fraction of the scene is visible, they can maintain a coherent spatial representation and continuously track occluded objects (e.g., a chair behind them). Current computer vision systems, however, are severely limited in this kind of persistent, panoramic scene understanding.

In the Track Any Point (TAP) task, existing methods can only track 2D points within the field of view. Once a target point leaves the frame, tracking largely fails and re-entry is treated as a re-identification problem. TAPVid-3D introduced 3D trajectory prediction, but its training data requires complete 4D scene models, making data acquisition prohibitively expensive; moreover, most implementations rely on 2.5D representations (pixel + depth) that cannot address out-of-field-of-view tracking.

The paper's core insight is that **the proliferation of consumer-grade 360° cameras provides abundant panoramic video data that can serve as supervision for training models to learn object permanence and out-of-field-of-view reasoning.** The key innovation is reformulating the tracking problem from predicting 2D coordinates to predicting 3D directional unit vectors—analogous to how a human can roughly point toward a chair behind them without knowing its exact distance.

## Method

### Overall Architecture

The pipeline consists of three stages:
1. **Dataset Construction**: Generating narrow field-of-view perspective videos with directional tracking annotations from 360° footage.
2. **Baseline Model**: Modifying CoTracker3 to predict rotation matrices instead of displacements.
3. **Evaluation**: Benchmarking on the custom TAPVid360-10k dataset.

### Key Designs

1. **TAPVid-360 Task Definition**: Given a narrow field-of-view perspective video and a query pixel in the first frame, the goal is to predict a 3D directional unit vector for each query point in each frame, expressed in the camera coordinate system. This directional representation offers two advantages: (a) it can represent arbitrary directions, including points more than 180° behind the camera; (b) depth estimation is not required, reducing task difficulty.

2. **Data Generation Pipeline**: This is the paper's most significant contribution. Starting from the 360-1M dataset:

    - **Coarse Filtering**: Videos with more than 15 likes (~100K) are retained; non-360, static, 180° format, and videos with discontinuous edge seams are removed.
    - **Fine Filtering**: Videos are split into 10-second clips; optical flow is used to filter static content; PySceneDetect identifies scene cuts; a watermark detection network removes watermarked clips; approximately 130K high-quality clips are retained.
    - **Object Detection and Tracking**: Lang-SAM detects dynamic objects (people, vehicles, animals, etc.) in the first frame; SAM2 propagates the segmentation masks throughout the video.
    - **Perspective Projection and Point Tracking**: Perspective videos are generated that follow each object; CoTracker3 performs 2D point tracking on these perspective clips; camera intrinsics are used to convert results to 3D directional unit vectors.
    - **Novel View Sampling**: Multiple camera motion strategies (spiral, random, simulated human motion, etc.) are used to sample new perspective videos from the original 360° footage, with the previously obtained 3D directions serving as ground truth for each viewpoint.

3. **CoTracker360 Baseline Model**: Built upon CoTracker3 with the following modifications:

    - The final decoder layer is replaced by a linear layer outputting a 9-dimensional vector (3×3 matrix), projected to a rotation matrix via Procrustes orthogonalization.
    - Query pixel coordinates are converted to directional unit vectors, which are then updated frame-by-frame using the predicted rotation matrices.
    - A Huber loss supervises the directional predictions.
    - The model is fine-tuned for 120 epochs on 5K training samples.

### Evaluation Metrics

- **$<\delta_{\text{avg}}^x$**: Fraction of predicted directions within an angular threshold (based on the per-pixel angular resolution of 0.2755°).
- **$\text{AD}_{\text{avg}}^x$**: Mean angular distance between predicted and ground-truth directions.
- Each metric is reported separately for in-field-of-view (IF), out-of-field-of-view (OOF), and all points.

## Key Experimental Results

### Main Results: TAPVid360-10k Tracking Performance

| Method | $<\delta_{\text{avg}}$ (all) | $<\delta_{\text{avg}}$ (IF) | $<\delta_{\text{avg}}$ (OOF) | $\text{AD}_{\text{avg}}$ (all)↓ | $\text{AD}_{\text{avg}}$ (IF)↓ | $\text{AD}_{\text{avg}}$ (OOF)↓ |
|---|---|---|---|---|---|---|
| TAPNext | 0.008 | 0.019 | 0.0004 | 51.98° | 36.59° | 62.46° |
| TAPIR | 0.011 | 0.025 | 0.0003 | 49.81° | 33.88° | 60.52° |
| BootsTAPIR | 0.013 | 0.029 | 0.0005 | 48.36° | 33.32° | 58.38° |
| TAPIP3D | 0.248 | 0.470 | 0.085 | 36.44° | 23.32° | 45.80° |
| SpatialTracker | 0.224 | 0.489 | 0.030 | 38.88° | 22.16° | 50.44° |
| CoTracker3 | 0.244 | 0.559 | 0.016 | 37.43° | 17.64° | 50.98° |
| **CoTracker360 (Ours)** | **0.239** | 0.406 | **0.116** | **8.27°** | **3.95°** | **10.98°** |

### Dataset Statistics Comparison

| Dataset | Videos | Clips | Tracks/Clip | Real/Synthetic | Data Type |
|---|---|---|---|---|---|
| TAPVid-Kinetics | 1,189 | — | 26.3 | Real | 2D |
| TAPVid-3D | 2,828 | 4,569 | 50–1024 | Real | 3D |
| **TAPVid360-10k** | **4,772** | **4,772** | **256** | **Real** | **360°** |

### Key Findings

- CoTracker360 achieves a breakthrough improvement in out-of-field-of-view tracking: OOF angular distance is reduced from 45.80° (TAPIP3D) to 10.98°, a **more than 4× reduction**.
- Existing 2D TAP methods (e.g., TAPIR) fail entirely on OOF points ($<\delta_{\text{avg}}$ OOF near zero).
- The dataset contains 36.28M in-field-of-view points and 45.64M out-of-field-of-view points, providing comprehensive coverage of varying visibility conditions.
- CoTracker3 achieves the highest in-field-of-view accuracy (0.559) but is nearly ineffective out-of-field-of-view (0.016).

## Highlights & Insights

- The problem is elegantly reformulated from 2D/3D coordinate prediction to 3D directional prediction, circumventing the difficulties of depth estimation and full 3D reconstruction.
- 360° video is cleverly exploited as a freely available supervision signal—directional annotations can be obtained from 2D tracking and camera intrinsics alone, without requiring 3D ground truth.
- The data generation pipeline is rigorously designed: multi-stage filtering (coarse + fine) combined with automated annotation (Lang-SAM → SAM2 → CoTracker3) yields a highly scalable framework.
- The model modification is remarkably minimal—only the final layer of CoTracker3 is replaced and the model is fine-tuned—yet yields dramatic performance gains, demonstrating that directional representations are better suited to panoramic tracking than coordinate-based ones.
- Object permanence is incorporated as a hard constraint within the task definition itself.

## Limitations & Future Work

- The dataset uses a fixed field-of-view angle and does not evaluate model generalization to varying FOV or zoom levels.
- The baseline model uses fixed positional encodings and cannot handle dynamically changing fields of view.
- No uncertainty modeling is provided—the model cannot express confidence in the predicted location of out-of-field-of-view points.
- Fine-tuning is performed on only 5K training samples, leaving the full scaling potential of the data pipeline unexplored.
- No comparison with traditional methods such as SLAM is conducted.

## Related Work & Insights

- This work innovatively bridges the TAP point tracking and 360° video understanding communities, pioneering a new direction in panoramic point tracking.
- The use of 360° video as a supervision signal is broadly transferable: other tasks requiring global spatial understanding (e.g., navigation, spatial reasoning) could similarly benefit from 360° video training data.
- The directional representation paradigm (as opposed to coordinate-based representations) may offer insights for SLAM, visual localization, and related tasks.
- Potential downstream applications include robotic active vision, re-identification priors, and temporal consistency constraints for video generation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Novel task formulation and innovative data generation methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive baseline comparisons, detailed dataset statistics, and both qualitative and quantitative evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous logic, clear exposition, and effective motivation through analogy with human cognition.
- Value: ⭐⭐⭐⭐⭐ — Opens a new research direction; the dataset and task design will drive future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Beyond the Frame: Generating 360° Panoramic Videos from Perspective Videos](../../ICCV2025/video_understanding/beyond_the_frame_generating_360deg_panoramic_videos_from_perspective_videos.md)
- [\[NeurIPS 2025\] Fixed-Point RNNs: Interpolating from Diagonal to Dense](fixed-point_rnns_interpolating_from_diagonal_to_dense.md)
- [\[ICCV 2025\] Online Dense Point Tracking with Streaming Memory](../../ICCV2025/video_understanding/online_dense_point_tracking_with_streaming_memory.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](../../ICCV2025/video_understanding/alltracker_efficient_dense_point_tracking_at_high_resolution.md)

</div>

<!-- RELATED:END -->
