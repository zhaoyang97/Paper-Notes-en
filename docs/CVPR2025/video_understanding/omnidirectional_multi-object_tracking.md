---
title: >-
  [Paper Note] OmniTrack: Omnidirectional Multi-Object Tracking
description: >-
  [CVPR 2025][Video Understanding][Omnidirectional Tracking] This paper proposes OmniTrack, the first multi-object tracking framework for 360° panoramic images, which unifies the Tracking-by-Detection (TBD) and end-to-end (E2E) tracking paradigms. It mitigates panoramic distortion through the CircularStatE module, introduces temporal priors via FlexiTrack instances, and provides trajectory feedback via Tracklet Management, alongside constructing the QuadTrack dataset for quadru…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Omnidirectional Tracking"
  - "Panoramic Images"
  - "Quadruped Robots"
  - "Geometric Distortion"
  - "Multi-Object Tracking"
date: 2026-05-08
content_hash: ee0fa5f9dc08c058
---

# OmniTrack: Omnidirectional Multi-Object Tracking

**Conference**: CVPR 2025  
**arXiv**: [2503.04565](https://arxiv.org/abs/2503.04565)  
**Code**: [GitHub](https://github.com/xifen523/OmniTrack)  
**Area**: Video Understanding/Multi-Object Tracking  
**Keywords**: Omnidirectional Tracking, Panoramic Images, Quadruped Robots, Geometric Distortion, Multi-Object Tracking

## TL;DR

This paper proposes OmniTrack, the first multi-object tracking framework for 360° panoramic images, which unifies the Tracking-by-Detection (TBD) and end-to-end (E2E) tracking paradigms. It mitigates panoramic distortion through the CircularStatE module, introduces temporal priors via FlexiTrack instances, and provides trajectory feedback via Tracklet Management, alongside constructing the QuadTrack dataset for quadruped robot panoramic MOT.

## Background & Motivation

Panoramic cameras provide comprehensive environmental information with a 360° field of view (FoV), which is highly valuable for applications such as autonomous driving and robot navigation. However, panoramic MOT faces unique challenges:

- **Geometric Distortion**: Serious geometric deformation exists in high-latitude areas after panoramic images are projected (equirectangular projection).
- **Resolution Loss**: Compressing 360° information into a single-frame image reduces the effective resolution.
- **Uneven Lighting/Color**: Differences in lighting conditions across different directions cause inconsistent image features.
- **Severe Motion**: The biomimetic gait of quadruped robots introduces complex non-linear camera motion.

Existing MOT algorithms (such as ByteTrack, OC-SORT) are designed for pinhole cameras, and their performance drops significantly when directly applied to panoramic images. Previously, there was a lack of dedicated panoramic MOT frameworks and panoramic datasets containing severe motion.

## Method

### Overall Architecture

OmniTrack is designed with a feedback mechanism: FlexiTrack instances utilize historical trajectory information to guide the detector to focus on key areas, the CircularStatE module handles panoramic distortion, and Tracklet Management manages trajectory data and provides prior knowledge. Under a unified setup, disabling data association yields the E2E tracker OmniTrackE2E, while enabling it yields the TBD tracker OmniTrackDA.

### Key Designs

#### Key Design 1: CircularStatE Module

- **Function**: Mitigate wide-angle distortion, resolution loss, and lighting inconsistency in panoramic images.
- **Mechanism**: Leverage the circular characteristic of panoramic images (where the left and right edges are actually connected) to perform circular padding and statistical feature enhancement at the feature level, improving detection consistency across different regions.
- **Design Motivation**: Zero-padding in standard CNNs introduces discontinuity at the boundaries of panoramic images, disrupting the circular geometry of the panorama. Circular padding makes use of the natural topology of panoramas.

#### Key Design 2: FlexiTrack Instance

- **Function**: Utilize temporal priors to guide target detection and association in the current frame.
- **Mechanism**: Retrieve the trajectory information from the previous frame $\mathcal{I}_F$ from Tracklet Management, and input it into the Decoder along with the feature map $\mathcal{I}_L$ processed by CircularStatE, respectively outputting FlexiTrack detection $\mathcal{D}_k^F$ (based on trajectory prior) and learned detection $\mathcal{D}_k^L$ (based on current frame features).
- **Design Motivation**: Panoramic images have a wide field of view, resulting in a vast search space. Utilizing historical trajectories as priors can significantly narrow down the search space and improve tracking stability in fast-motion scenarios.

#### Key Design 3: QuadTrack Dataset

- **Function**: Provide a panoramic MOT benchmark that encapsulates complex motion dynamics.
- **Mechanism**: Collect 19,200 images using a $360° \times 70°$ panoramic camera mounted on a quadruped robot across five campuses in two cities. The biomimetic gait of the quadruped robot introduces realistic and complex non-linear motion features.
- **Design Motivation**: Existing MOT datasets utilize static or linear-motion platforms, which fail to evaluate tracking performance under the combination of a panoramic field of view and violent motion.

### Loss & Training

Standard detection loss (classification + regression) and matching loss are used for end-to-end training. Data association is performed using the Hungarian algorithm.

## Key Experimental Results

### Main Results: JRDB Dataset

| Method | HOTA ↑ | MOTA ↑ | IDF1 ↑ |
|------|--------|--------|--------|
| ByteTrack | 18.40 | - | - |
| OC-SORT | 23.49 | - | - |
| **OmniTrackDA** | **26.92** | - | - |

### QuadTrack Dataset

| Method | HOTA ↑ | Gain |
|------|--------|------|
| Baseline | 16.64 | - |
| **OmniTrackDA** | **23.45** | +6.81% |

### Ablation Study

| Configuration | HOTA |
|------|------|
| Baseline (Standard Detector) | Lowest |
| + CircularStatE | +Gain |
| + FlexiTrack | +Significant Gain |
| + Full Framework | **Best** |

### Key Findings

- Achieved a HOTA of 26.92% on JRDB, outperforming OC-SORT by 3.43 percentage points.
- Outperformed the baseline on QuadTrack by 6.81%, with a more pronounced advantage in violent motion scenarios.
- OmniTrackDA (TBD paradigm) consistently outperformed OmniTrackE2E (E2E paradigm), indicating that data association remains crucial.
- CircularStatE delivered the most significant improvement in target detection within high-latitude areas.

## Highlights & Insights

1. **First Panoramic MOT Framework**: Systematically addresses the key challenges of MOT in panoramic images.
2. **Unified TBD and E2E**: Achieves a unified framework for both paradigms through a toggle design, facilitating fair comparisons.
3. **QuadTrack Dataset**: The intense, non-linear motion of quadruped robots provides a highly challenging new benchmark.

## Limitations & Future Work

- The absolute HOTA remains relatively low (26.92%), reflecting that panoramic MOT is an extremely difficult task.
- The motion pattern of the quadruped robot is highly specific, and its generalizability remains to be verified.
- The integration of 3D MOT or depth information has not been explored.
- The scale of the QuadTrack dataset is relatively small.

## Related Work & Insights

- **ByteTrack**: A confidence-based multi-stage association strategy.
- **OC-SORT**: An optimized motion estimation module.
- **360VOT**: An omnidirectional single-object tracking benchmark.
- The strategy for handling circularity in panorama perception can be generalized to other panoramic vision tasks.

## Rating

⭐⭐⭐⭐ — First to systematically address the panoramic MOT problem with a well-designed framework. The QuadTrack dataset fills an important gap. However, there is still substantial room for absolute performance improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[CVPR 2026\] Hypergraph-State Collaborative Reasoning for Multi-Object Tracking](../../CVPR2026/video_understanding/hypergraph-state_collaborative_reasoning_for_multi-object_tracking.md)
- [\[AAAI 2026\] PlugTrack: Multi-Perceptive Motion Analysis for Adaptive Fusion in Multi-Object Tracking](../../AAAI2026/video_understanding/plugtrack_multi-perceptive_motion_analysis_for_adaptive_fusion_in_multi-object_t.md)
- [\[CVPR 2026\] ProgTrack: A Multi-Object Tracking Algorithm with Progressive Matching Strategy](../../CVPR2026/video_understanding/progtrack_a_multi-object_tracking_algorithm_with_progressive_matching_strategy.md)
- [\[ICLR 2026\] UniTrack: Differentiable Graph Representation Learning for Multi-Object Tracking](../../ICLR2026/video_understanding/unitrack_differentiable_graph_representation_learning_for_multi-object_tracking.md)

</div>

<!-- RELATED:END -->
