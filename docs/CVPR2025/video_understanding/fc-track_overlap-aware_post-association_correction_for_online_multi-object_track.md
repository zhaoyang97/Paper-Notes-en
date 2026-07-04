---
title: >-
  [Paper Note] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking
description: >-
  [CVPR 2025][Video Understanding][Multi-Object Tracking] Proposes FC-Track, a lightweight post-association correction framework. By using IoA (Intersection over Area)-based appearance feature filtering and similarity comparisons within overlapping tracklet pairs, it online corrects detection-tracklet mismatch errors caused by target overlap. This reduces the ratio of long-term identity switches from 36.86% to 29.55% while maintaining SOTA performance on MOT17/MOT20.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Multi-Object Tracking"
  - "ID Switch Correction"
  - "Overlap-Aware"
  - "Post-Association Correction"
  - "Online Tracking"
  - "IoA"
date: 2026-05-08
content_hash: 6f80cf540b383b86
---

# FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking

**Conference**: CVPR 2025  
**arXiv**: [2603.12758](https://arxiv.org/abs/2603.12758)  
**Code**: To be confirmed  
**Area**: Video Understanding / Multi-Object Tracking  
**Keywords**: Multi-Object Tracking, ID Switch Correction, Overlap-Aware, Post-Association Correction, Online Tracking, IoA

## TL;DR

Proposes FC-Track, a lightweight post-association correction framework. By using IoA (Intersection over Area)-based appearance feature filtering and similarity comparisons within overlapping tracklet pairs, it online corrects detection-tracklet mismatch errors caused by target overlap. This reduces the ratio of long-term identity switches from 36.86% to 29.55% while maintaining SOTA performance on MOT17/MOT20.

## Background & Motivation

### 1. Background
Multi-Object Tracking (MOT) is a core perceptual component of robotic systems and video analysis. Mainstream methods adopt the tracking-by-detection paradigm: detecting targets frame-by-frame first, and then associating detection results with existing trajectories using motion or appearance cues.

### 2. Limitations of Prior Work
- **Irreversible Association Errors**: Once a mismatch occurs, identity errors propagate to subsequent frames, leading to long-term identity switches and severe tracklet degradation.
- **Prior Methods Address Symptoms Rather Than Root Causes**: Most methods attempt to reduce the error rate through better appearance representations, motion models, or global optimization, but treat association decisions as unalterable once made.
- **Global Optimization Inapplicable to Online Scenarios**: Offline/batch optimization methods can correct historical associations but are incompatible with the constraints of real-time applications like robotics.
- **Overlap Scenarios as the Main Failure Source**: Frequent occlusions and target overlaps in crowded scenes make both motion cues and appearance similarity unreliable.

### 3. Key Challenge
How to detect and correct association errors caused by target overlaps while maintaining online real-time performance, preventing short-term errors from propagating into long-term identity drift?

### 4. Key Observation
- Most detection-tracklet mismatches occur between overlapping bounding boxes.
- During the overlap, both motion association and appearance similarity are unreliable, but targets become more distinguishable after the overlap ends.
- The key is to preserve reliable (non-overlapping) appearance features during overlap, and use them for correction after the overlap ends.

### 5. Core Idea
Instead of redesigning the association module, a lightweight correction module is added after the association stage: (1) using IoA to detect the overlap state and filter out unreliable appearance updates, and (2) re-assigning detection results for overlapping tracklet pairs through appearance similarity comparison.

### 6. Motivation Summary
Robust multi-object tracking should not only strive to avoid association errors but also possess the ability to recognize and correct errors online, serving as an orthogonal complement to existing pipelines.

## Method

### Overall Architecture
FC-Track is integrated into a standard online MOT pipeline (based on TrackTrack) as a post-association correction module. The per-frame pipeline: perform standard detection and association first $\to$ divide matched results into overlapping and non-overlapping groups $\to$ directly accept non-overlapping ones, while re-evaluating overlapping groups through appearance similarity comparison $\to$ merge to output final tracking results $\to$ update IoA status and appearance features.

### Key Designs

#### Key Design 1: Overlap-Aware Appearance Feature Filtering

- **Function**: Selectively suppress appearance feature updates based on the IoA relationship matrix, and identify overlapping tracklet pairs.
- **Mechanism**: Compute IoA (Intersection over Area of the reference box) for all tracklet pairs, distinguishing two threshold levels:
    - $IoA \ge \tau_{update}$ (0.3): Suspend appearance feature updates and retain features from the latest non-overlapping frame to avoid cross-contamination of appearance.
    - $IoA \ge \tau_{overlap}$ (0.8): Form overlapping tracklet pairs (prime-auxiliary) to prepare for the correction pipeline.
- **Design Motivation**: Appearance features observed during overlap blend information from multiple targets, which would pollute identity representations if used for updates. The asymmetry of IoA ($IoA(A, B) \neq IoA(B, A)$) provides a natural role assignment (the tracklet with its area as the denominator is designated as prime).
- **prime-auxiliary Assignment**: The tracklet whose bounding box area is used as the IoA denominator is designated as prime (usually the smaller, occluded target), and the other as auxiliary. This ensures consistent and deterministic role assignment.

#### Key Design 2: Mismatch Re-allocation

- **Function**: Perform local correction of detection-tracklet matches within overlapping tracklet pairs.
- **Mechanism**: Use the prime tracklet as an index to find its matched detection in the current frame, compute the similarity between this detection and the saved features of both the prime and auxiliary tracklets, and re-allocate if conditions are met.
- **Correction Conditions** (must be satisfied simultaneously):
    1. Prime similarity distance $S_{pri} \ge \tau_{min}$ (0.8): The match with the prime tracklet is indeed poor.
    2. $S_{pri} - S_{aux} \ge \tau_{dif}$ (0.4): The auxiliary tracklet is significantly more matching.
- **Design Motivation**: Strict double-threshold conditions ensure correction occurs only at high confidence, preventing new errors; performing local correction (strictly within the overlapping pair) instead of global re-association maintains computational efficiency.

### Loss & Training
FC-Track is a post-processing module utilized purely during inference, with no training loss. The core metric is cosine distance for similarity comparison of appearances.

## Key Experimental Results

### Main Results: MOT17 Test Set

| Tracker | HOTA↑ | MOTA↑ | IDF1↑ | AssA↑ | IDs↓ | FPS↑ |
|--------|-------|-------|-------|-------|------|------|
| ByteTrack | 63.05 | 80.25 | 77.30 | 61.98 | 2196 | 29.6 |
| OC-SORT | 63.16 | 78.00 | 77.50 | 63.40 | 1950 | 29.0 |
| BoT-SORT | 65.05 | 80.55 | 80.23 | 65.49 | 1212 | 6.8 |
| Deep OC-SORT | 64.88 | 79.37 | 80.58 | 65.93 | 1023 | 28.1 |
| TrackTrack (baseline) | 66.94 | 81.71 | 82.78 | 66.80 | 837 | 5.9 |
| **FC-Track (Ours)** | **66.95** | **81.73** | **82.81** | **67.81** | **837** | 5.7 |

### Main Results: MOT20 Test Set

| Tracker | HOTA↑ | MOTA↑ | IDF1↑ | AssA↑ | IDs↓ | FPS↑ |
|--------|-------|-------|-------|-------|------|------|
| ByteTrack | 61.34 | 77.76 | 75.21 | 59.56 | 1223 | 17.5 |
| OC-SORT | 62.36 | 75.67 | 76.32 | 62.47 | 942 | 5.1 |
| SUSHI | 64.33 | 74.29 | 79.80 | 67.47 | 706 | 5.3 |
| TrackTrack (baseline) | 65.61 | 77.52 | 80.82 | 67.35 | 719 | 0.7 |
| **FC-Track (Ours)** | **65.67** | **77.52** | **80.90** | **67.48** | **719** | 0.6 |

### ID Switch Duration Analysis (MOT17 Val Set)

| Tracker | Count↓ | Mean↓ | Med.↓ | Long Ratio↓ | IDTP↑ | IDFP↓ | IDFN↓ |
|--------|-------|-------|-------|-------------|-------|-------|-------|
| ByteTrack | 201 | 33.04 | 11 | 50.25% | 40434 | 13456 | 6951 |
| BoT-SORT | 199 | 32.89 | 5 | 38.69% | 41757 | 12133 | 6137 |
| TrackTrack | 236 | 22.88 | 5 | 36.86% | 42144 | 11746 | 6927 |
| **FC-Track** | 308 | **18.33** | **3** | **29.55%** | **42305** | **11585** | 6843 |

- The ratio of long-term identity switches decreases from 36.86% to **29.55%**, significantly outperforming compared methods.
- The mean duration drops from 22.88 frames to **18.33 frames**, and the median drops from 5 to **3**.
- IDTP increases (42305 vs 42144), while IDFP decreases (11585 vs 11746).

### Ablation Study: Similarity Metrics (MOT17 Val Set)

| Similarity | HOTA↑ | MOTA↑ | IDF1↑ | AssA↑ | IDs↓ |
|--------|-------|-------|-------|-------|------|
| Baseline | 69.40 | 76.57 | 81.86 | 73.57 | 400 |
| Euclidean Distance | 69.48 | 76.49 | 81.90 | 73.71 | 400 |
| **Cosine Distance** | **69.67** | **76.60** | **82.12** | **74.08** | **398** |

### Ablation Study: Matching Stages

| Stage 1 | Stage 2 | HOTA↑ | IDF1↑ | AssA↑ |
|---------|---------|-------|-------|-------|
| ✕ | ✕ | 69.40 | 81.86 | 73.57 |
| ✓ | ✕ | 69.67 | 82.12 | 74.08 |
| ✕ | ✓ | 69.40 | 81.86 | 73.57 |
| ✓ | ✓ | 69.67 | 82.12 | 74.08 |

- The correction module is effective in Stage 1 (high-confidence association), but shows no obvious effect in Stage 2 (low-confidence/ambiguous matching).

### Key Findings
1. The core value of FC-Track does not lie in reducing the number of ID switches (Count even increases slightly), but in significantly shortening the duration of each ID switch.
2. Cosine distance outperforms Euclidean distance as the appearance similarity metric.
3. The correction module is only effective in Stage 1 matching—early correction of high-confidence association can effectively prevent identity drift.
4. The four threshold parameters remain robust over a wide range, consistently outperforming the baseline.
5. Minimal computational overhead: FPS only decreases from 5.9 to 5.7 on MOT17.

## Highlights & Insights

1. **New Perspective of "Correction Over Prevention"**: Instead of pursuing perfect associations, it acknowledges that errors are inevitable and designs a correction mechanism, representing an orthogonal improvement to MOT pipelines.
2. **ID Switch Duration Analysis**: For the first time, a systematic analysis of the temporal characteristics of ID switches is proposed instead of merely counting them—making it more aligned with practical application requirements.
3. **Minimalist Design**: Requires only 4 threshold parameters without training, global optimization, or re-ID networks, achieving a plug-and-play capability.
4. **Ingenious Use of IoA**: Employs IoA instead of IoU to model the asymmetrical relationship of "who is occluded by whom," providing a natural basis for prime-auxiliary role assignment.
5. **Strong Practicality**: As a lightweight post-processing module, it can be integrated into any Tracking-by-Detection (TBD)-based tracker, making it suitable for real-time robotic applications.

## Limitations & Future Work

1. **Limited Improvement in Overall Metrics**: HOTA only increases from 66.94 to 66.95 on MOT17, and IDF1 from 82.78 to 82.81, showing extremely small absolute improvements.
2. **No Reductions in Total ID Switches**: Count increases from 236 to 308; although the duration of each is shorter, the total frequency increases instead.
3. **Only Handles Pairwise Overlaps**: Scenarios with simultaneous overlap of three or more targets are not addressed.
4. **Reliance on the Quality of Appearance Features**: Correction may fail when targets share highly similar appearances (e.g., wearing the same uniform).
5. **Manual Tuning of Thresholds Required**: The 4 thresholds may require different settings for various scenarios.
6. **Low FPS on MOT20**: 0.6 FPS can hardly be called "real-time," where the primary bottleneck lies in the TrackTrack baseline.
7. **No Comparison with Offline/Global Optimization Methods**: Lacks comparative analysis with post-processing global optimization methods.

## Related Work & Insights

- **Comparison with ByteTrack**: ByteTrack focuses on full bounding box association (utilizing both high and low confidence), whereas FC-Track focuses on post-association error correction. The two are orthogonal.
- **Comparison with OC-SORT**: OC-SORT utilizes virtual trajectories to maintain state estimation during occlusions, whereas FC-Track uses saved non-overlapping appearance features to correct matches after overlaps.
- **Comparison with TrackTrack**: FC-Track is directly built on top of TrackTrack, demonstrating that post-association correction still yields benefits for already strong baselines.
- **Insights**: The concept of "post-correction" can be transferred to other scenarios requiring online decision making (e.g., multi-sensor fusion, error recovery in dialog systems). Indeed, "admitting errors and fixing them quickly" may be more practical than "striving for zero errors."

## Rating
- Novelty: ⭐⭐⭐⭐ (Post-association correction is a valuable new perspective, but the technique itself is relatively simple)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Standard MOT17/20 benchmarks, comprehensive ablation studies, threshold sensitivity, and ID switch duration analysis)
- Writing Quality: ⭐⭐⭐⭐ (Well-structured but slightly wordy, with some repetition in the motivation section)
- Value: ⭐⭐⭐⭐ (Practical improvements are limited, but the concept is novel, making it a good complementary module for MOT pipelines)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OmniTrack: Omnidirectional Multi-Object Tracking](omnidirectional_multi-object_tracking.md)
- [\[CVPR 2026\] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](../../CVPR2026/video_understanding/occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)
- [\[CVPR 2026\] Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation](../../CVPR2026/video_understanding/out_of_sight_out_of_track_adversarial_attacks_on_propagation-based_multi-object_.md)
- [\[CVPR 2026\] Hypergraph-State Collaborative Reasoning for Multi-Object Tracking](../../CVPR2026/video_understanding/hypergraph-state_collaborative_reasoning_for_multi-object_tracking.md)
- [\[AAAI 2026\] PlugTrack: Multi-Perceptive Motion Analysis for Adaptive Fusion in Multi-Object Tracking](../../AAAI2026/video_understanding/plugtrack_multi-perceptive_motion_analysis_for_adaptive_fusion_in_multi-object_t.md)

</div>

<!-- RELATED:END -->
