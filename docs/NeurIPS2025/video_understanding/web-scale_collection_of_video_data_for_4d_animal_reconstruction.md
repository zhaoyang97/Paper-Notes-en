---
title: >-
  [Paper Note] Web-Scale Collection of Video Data for 4D Animal Reconstruction
description: >-
  [NeurIPS 2025][Video Understanding][4D animal reconstruction] This paper proposes a fully automated large-scale video data collection pipeline that mines and processes 30K animal videos (2M frames) from YouTube…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "4D animal reconstruction"
  - "data pipeline"
  - "YouTube video mining"
  - "benchmark dataset"
  - "single-view reconstruction"
date: 2026-05-08
content_hash: 114d986ab811ef97
---

# Web-Scale Collection of Video Data for 4D Animal Reconstruction

**Conference**: NeurIPS 2025
**arXiv**: [2511.01169](https://arxiv.org/abs/2511.01169)
**Code**: [https://github.com/briannlongzhao/Animal-in-Motion](https://github.com/briannlongzhao/Animal-in-Motion)
**Area**: Video Understanding / 3D Vision
**Keywords**: 4D animal reconstruction, data pipeline, YouTube video mining, benchmark dataset, single-view reconstruction

## TL;DR
This paper proposes a fully automated large-scale video data collection pipeline that mines and processes 30K animal videos (2M frames) from YouTube, establishes the first 4D quadruped animal reconstruction benchmark Animal-in-Motion (230 sequences / 11K frames), and introduces a baseline method 4D-Fauna that achieves model-free 4D reconstruction via sequence-level optimization.

## Background & Motivation

Visual analysis of animal morphology and motion has important applications in wildlife conservation, biomechanics, and robotics. Traditional approaches rely on expensive multi-view controlled environments or marker-based systems. Recent single-view methods (pose estimation, tracking, 3D/4D reconstruction) have made progress, but are **severely constrained by data scale**.

Existing animal video datasets suffer from three critical issues: (1) **extremely small scale**—the largest, APT-36K, contains only 2.4K clips of 15 frames each; (2) **lack of object-centric crops**—raw videos may contain multiple overlapping animals with no segmentation masks; (3) **lack of essential preprocessing**—no auxiliary annotations (keypoints, optical flow, depth, etc.) prepared for 3D/4D reconstruction tasks. The only dataset genuinely suitable for 4D animal reconstruction, BADJA, contains merely 11 videos.

The root cause lies in the tension between the data demands of data-driven methods and the prohibitive cost of collecting and annotating animal videos. The paper resolves this by leveraging the vast video resources on YouTube to build a fully automated collection–processing–annotation pipeline.

## Method

### Overall Architecture

The pipeline consists of four stages: (1) searching and downloading raw videos from YouTube; (2) video preprocessing (shot segmentation, CLIP filtering); (3) animal detection and tracking (Grounded-SAM-2) to generate object-centric crops; and (4) feature extraction (keypoints, DINO features, optical flow, depth maps, occlusion boundaries). The entire pipeline is coordinated through a central database and supports multi-process parallelism.

### Key Designs

1. **Intelligent Search Query Generation**:

    - Given an animal category (e.g., "horse"), GPT is used to generate sub-breeds (Clydesdale, Mustang) and contextual phrases (racing competition, in a farm).
    - These are randomly combined into diverse search queries to maximize video diversity.
    - Selenium and pytube are used for search and download.

2. **Multi-Level Filtering and Tracking**:

    - **Shot segmentation**: PySceneDetect is used to detect scene transitions via pixel-level changes, preventing cross-shot tracking confusion.
    - **CLIP filtering**: CLIPScore between frames and category text is computed; low-scoring clips are discarded.
    - **Grounded-SAM-2 tracking**: Iterative grounding-tracking for long-term tracking.
    - **Multi-level filtering**:
        - Overlapping instance filtering (IoU threshold to remove frames with multiple overlapping animals)
        - Low-resolution filtering (bounding box area < 1/4 of crop size)
        - Truncated instance filtering (bounding box near frame boundary)
        - Inconsistent trajectory filtering (abrupt IoU changes between adjacent frames to detect identity switches)
    - **Object-centric crops**: Square crops centered on the bounding box, smoothed via moving average.
    - GPT-based final visual verification to remove false detections and severely occluded instances.

3. **Comprehensive Feature Extraction Module**:

    - ViTPose++: animal keypoint estimation
    - DINOv2: image features
    - SEA-RAFT: optical flow estimation
    - Depth Anything V2: depth estimation
    - Occlusion boundaries: computed from depth discontinuities at mask boundaries

4. **4D-Fauna Baseline Method**:

    - Built upon 3D-Fauna (a model-free approach) with sequence-level optimization.
    - **Keypoint supervision**: 2D keypoints introduced as part-level constraints to resolve leg ordering ambiguity.
    - **Temporal smoothing loss**: Regularization applied to changes in camera pose parameters and joint velocity of animal pose.
    - **Efficient per-sequence overfitting**: Camera pose and joint parameters are directly optimized per frame, initialized from pretrained network outputs.

### Loss & Training

4D-Fauna uses the original inverse rendering losses from 3D-Fauna (mask IoU + DINO feature matching), augmented with a keypoint reprojection loss and temporal smoothing regularization terms on camera pose and joint velocity. Optimization is performed per sequence on top of the pretrained model.

## Key Experimental Results

### Main Results

| Method | IoU↑ | PCK@0.1↑ | PCK@0.05↑ | KT-PCK@0.1↑ | MPJVE↓ | Type |
|--------|------|----------|-----------|-------------|--------|------|
| SMALify | **0.867** | **0.954** | **0.787** | **0.623** | **0.023** | Model-based |
| AniMer | 0.677 | 0.537 | 0.199 | 0.566 | 0.038 | Model-based |
| 3D-Fauna | 0.670 | 0.470 | 0.177 | 0.329 | 0.058 | Model-free |
| 4D-Fauna | 0.814 | 0.664 | 0.317 | 0.418 | 0.044 | Model-free |

### Ablation Study

| Configuration | IoU | PCK@0.1 | Notes |
|---------------|-----|---------|-------|
| 3D-Fauna (direct inference) | 0.670 | 0.470 | Baseline model-free method |
| + Sequence optimization + Keypoints | 0.814 | 0.664 | Keypoints resolve leg ordering |
| + Temporal smoothing | ↑ | ↑ | Reduces inter-frame jitter |

### Key Findings
- **Misleading 2D metrics**: SMALify achieves the best quantitative scores across all metrics, yet qualitative inspection reveals frequent unnatural 3D shapes (depth-elongated bodies, abnormal leg bending, distorted frontal-view geometry), exposing the limitations of 2D projection metrics.
- **Advantages of model-free methods**: 3D-Fauna and 4D-Fauna produce more naturally plausible 3D shapes and poses, despite yielding lower 2D metric scores.
- **Necessity of sequence optimization**: Feed-forward inference with 3D-Fauna causes abrupt leg switching between frames; 4D-Fauna effectively addresses this via keypoint constraints and temporal smoothing.
- The data pipeline successfully collects 30K videos / 2M frames spanning 23 animal categories.

## Highlights & Insights
- **End-to-end automation**: The pipeline is fully automatic from search query generation to final feature extraction; only benchmark verification requires limited human effort.
- **Revealing metric deficiencies**: The paper clearly demonstrates the inconsistency between 2D projection metrics and 3D reconstruction quality, underscoring the need for 3D-aware evaluation metrics.
- **Elegant model adaptation**: Rather than retraining, 4D-Fauna uses feed-forward model outputs as optimization initialization, combining the generalization of model-free methods with the precision of model-based ones.

## Limitations & Future Work
- Annotations from the automated pipeline are not entirely clean; benchmark evaluation still requires manual verification.
- The benchmark relies solely on 2D projection metrics, lacking true 3D ground truth.
- 4D-Fauna offers limited modeling of temporal consistency—future work could explore autoregressive models to capture inter-frame dynamics.
- Original RGB video frames are not released due to copyright concerns; only derived data are provided.
- The pipeline currently focuses on quadruped animals; extension to birds, fish, and other categories requires adaptation.

## Related Work & Insights
- **vs. APT-36K**: Scale increased by 12.5× (30K vs. 2.4K), with comprehensive preprocessed features included.
- **vs. BADJA**: Expanded from 11 videos to 230 benchmark sequences, establishing the first benchmark genuinely targeting 4D reconstruction.
- **vs. 3D-Fauna**: 4D-Fauna adds keypoint and temporal constraints, yielding consistent improvements across all metrics.
- **vs. SMALify**: Quantitatively superior yet questionable in 3D quality, highlighting fundamental issues with the current evaluation paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐ — Significant engineering contribution in the pipeline; 4D-Fauna method is incrementally innovative but practically effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Impressive data scale and comprehensive benchmark, though lacking 3D quantitative evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Pipeline description is thorough and clear, with insightful analysis.
- Value: ⭐⭐⭐⭐⭐ — The dataset and pipeline offer substantial value to the community and are poised to advance the field of 4D animal reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unleashing Hour-Scale Video Training for Long Video-Language Understanding](unleashing_hour-scale_video_training_for_long_video-language_understanding.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](../../ICCV2025/video_understanding/4d_bench_benchmarking_multimodal_llms_for_4d_object_understanding.md)
- [\[CVPR 2026\] Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning](../../CVPR2026/video_understanding/reconstruction-guided_slot_curriculum_addressing_object_over-fragmentation_in_vi.md)
- [\[NeurIPS 2025\] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)

</div>

<!-- RELATED:END -->
