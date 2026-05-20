---
title: >-
  [Paper Note] Temporally Consistent Long-Term Memory for 3D Single Object Tracking
description: >-
  [CVPR 2026][Video Understanding][3D single object tracking] This paper proposes ChronoTrack, a robust long-term 3D single object tracking framework built upon compact learnable memory tokens and two complementary objecti…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "3D single object tracking"
  - "long-term memory"
  - "temporal consistency"
  - "point cloud"
  - "memory tokens"
date: 2026-05-08
content_hash: 6b5c47c388c62b4f
---

# Temporally Consistent Long-Term Memory for 3D Single Object Tracking

**Conference**: CVPR 2026
**arXiv**: [2604.13789](https://arxiv.org/abs/2604.13789)  
**Code**: [github.com/ujaejoon/ChronoTrack](https://github.com/ujaejoon/ChronoTrack)  
**Area**: Video Understanding
**Keywords**: 3D single object tracking, long-term memory, temporal consistency, point cloud, memory tokens

## TL;DR

This paper proposes ChronoTrack, a robust long-term 3D single object tracking framework built upon compact learnable memory tokens and two complementary objectives — a temporal consistency loss and a memory cycle-consistency loss — achieving state-of-the-art performance on multiple benchmarks while running in real time at 42 FPS.

## Background & Motivation

Memory-based methods for 3D single object tracking (3D-SOT) leverage point-level feature representations of the target from historical frames, yet are typically limited to short-term contexts spanning only 2–3 frames. Naively extending the memory length encounters two fundamental challenges: (1) temporal consistency of target features degrades sharply as the inter-frame gap grows, rendering features from distant frames ineffective or even detrimental; and (2) storage and computational overhead of point-level memory scales linearly with memory length. The authors identify temporal feature inconsistency as the core bottleneck limiting the effectiveness of long-term memory.

## Method

### Overall Architecture

Given the current point cloud, a backbone network extracts point features, which are then fed into a Memory-augmented Feature Refiner (MFR) to interact with long-term foreground memory and short-term background memory, producing target-aware features. A decoder predicts 3D bounding boxes and objectness masks, which in turn drive memory updates. Two complementary losses ensure memory reliability and diversity during training.

### Key Designs

1. **Compact token-level memory**: $K$ learnable memory tokens are defined and recurrently updated at each timestep via cross-attention, fusing current predicted foreground features with accumulated historical context. The fixed-size design ensures that the overhead of long-term memory does not grow over time. Short-term background memory retains only the background point features from the most recent frame.

2. **Temporal Consistency Loss $\mathcal{L}_{TC}$**: Foreground points from different frames are transformed into a canonical coordinate system (defined by the center and orientation of the GT bounding box), and cross-frame point correspondences are established via nearest-neighbor matching. The loss enforces high cosine similarity between features of corresponding points, mitigating feature drift caused by appearance variation and enabling effective utilization of features from distant frames.

3. **Memory Cycle-Consistency Loss $\mathcal{L}_{MCC}$**: Each memory token performs a two-step cycle walk (memory → points → memory), and the optimization objective is to maximize the probability of each token returning to itself and passing through foreground points. This encourages different tokens to encode distinct semantic parts of the target, promoting memory diversity.

### Loss & Training

The total loss comprises a tracking loss (bounding box regression + objectness classification), the temporal consistency loss, and the memory cycle-consistency loss. Temporal consistency is established in the canonical coordinate system without relying on optical flow.

## Key Experimental Results

### Main Results

ChronoTrack achieves new state-of-the-art results on 3D-SOT benchmarks including KITTI, NuScenes, and Waymo:

| Benchmark | Metric | MBPTrack | ChronoTrack | Gain |
|-----------|--------|----------|-------------|------|
| Multiple  | Success | Prev. SOTA | **New SOTA** | Significant |
| Multiple  | Precision | Prev. SOTA | **New SOTA** | Significant |

The method runs in real time at 42 FPS on a single RTX 4090 GPU.

### Ablation Study

- The temporal consistency loss maintains high feature similarity across distant frames, whereas MBPTrack exhibits rapid decay.
- ChronoTrack consistently benefits from increasing memory length, while MBPTrack's performance degrades under the same condition.
- The improvement in token diversity attributed to the memory cycle-consistency loss is clearly demonstrated through visualization.

### Key Findings

- Temporal feature consistency is strongly correlated with tracking performance.
- Compact token-level memory is one to two orders of magnitude more efficient than point-level memory for long-range modeling.
- The two loss functions act synergistically: the consistency loss ensures feature quality, while the cycle-consistency loss ensures feature diversity.

## Highlights & Insights

- The paper clearly identifies temporal feature inconsistency as a previously overlooked core problem.
- Establishing point correspondences in the canonical coordinate system elegantly avoids dependence on optical flow.
- The cycle-walk mechanism is effectively transferred from graph networks and NLP to memory design in 3D tracking.

## Limitations & Future Work

- Supervision for temporal consistency relies on GT bounding boxes to construct the canonical coordinate system.
- The selection of the number of memory tokens $K$ lacks an adaptive mechanism.
- Performance under extreme occlusion and long-term disappearance–reappearance scenarios is not thoroughly analyzed.

## Related Work & Insights

- The paradigm of replacing point-level memory with token-level memory is generalizable to domains such as video understanding.
- The design of the temporal consistency loss offers useful reference for other sequence modeling tasks.
- Using cycle-consistency to promote representational diversity constitutes a general regularization strategy.

## Rating

8/10 — The paper offers deep problem insight, elegant method design, and thorough experimentation, making it a high-quality contribution to the 3D tracking community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UETrack: A Unified and Efficient Framework for Single Object Tracking](uetrack_a_unified_and_efficient_framework_for_single_object_tracking.md)
- [\[CVPR 2026\] Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)
- [\[NeurIPS 2025\] MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments](../../NeurIPS2025/video_understanding/memtrack_evaluating_long-term_memory_and_state_tracking_in_multi-platform_dynami.md)
- [\[CVPR 2026\] VidTAG: Temporally Aligned Video to GPS Geolocalization](vidtag_video_gps_geolocalization.md)
- [\[CVPR 2026\] Event6D: Event-based Novel Object 6D Pose Tracking](event6d_event-based_novel_object_6d_pose_tracking.md)

</div>

<!-- RELATED:END -->
