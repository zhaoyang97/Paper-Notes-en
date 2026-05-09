---
title: >-
  [Paper Note] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking
description: >-
  [CVPR 2026][Video Understanding][multi-object tracking] This paper proposes FC-Track, a lightweight post-association correction framework that suppresses appearance updates via IoA triggering and reassigns locally mismatched detection–tracklet pairs, reducing the proportion of long-term identity switches from 36.86% to 29.55% while maintaining state-of-the-art performance on MOT17/MOT20.
tags:
  - CVPR 2026
  - Video Understanding
  - multi-object tracking
  - occlusion handling
  - post-association correction
  - identity switch
  - IoA filtering
  - online tracking
date: 2026-05-08
content_hash: 5ef800c958eedc0e
---

# FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking

**Conference**: CVPR 2026
**arXiv**: [2603.12758](https://arxiv.org/abs/2603.12758)
**Code**: To be confirmed
**Area**: Multi-Object Tracking / Robot Vision
**Keywords**: multi-object tracking, occlusion handling, post-association correction, identity switch, IoA filtering, online tracking

## TL;DR
This paper proposes FC-Track, a lightweight post-association correction framework that suppresses appearance updates via IoA triggering and reassigns locally mismatched detection–tracklet pairs, reducing the proportion of long-term identity switches from 36.86% to 29.55% while maintaining state-of-the-art performance on MOT17/MOT20.

## Background & Motivation
As robots move from controlled environments into unstructured scenarios such as logistics, healthcare, and agriculture, reliable multi-object tracking (MOT) becomes a core perceptual component. While mainstream tracking-by-detection methods have steadily improved detection and association accuracy, they remain prone to erroneous detection–tracklet associations caused by overlapping bounding boxes in crowded, occluded scenes. More critically, once a short-term mismatch occurs, identity errors propagate along the temporal axis and deteriorate into long-term identity switches — a failure mode that existing methods are almost entirely unable to correct online. Offline global optimization or re-identification approaches can retrospectively fix such errors, but they do not satisfy the latency requirements of real-time robot applications.

## Core Problem
In online MOT, overlapping occlusions simultaneously invalidate motion cues and appearance similarity, producing detection–tracklet mismatches that persist across subsequent frames and manifest as long-term identity switches. Existing methods treat association decisions as irreversible and lack any online error-correction mechanism.

## Method

### Overall Architecture
FC-Track is a general-purpose post-association correction module inserted after the association stage of a standard online MOT pipeline. The workflow proceeds as follows: (1) after each frame's tracking is complete, already-matched (detection, tracklet) pairs are partitioned into overlapping and non-overlapping groups based on the overlap state from the previous frame; (2) non-overlapping pairs are accepted directly, while overlapping pairs are re-evaluated via appearance similarity; (3) corrected and uncorrected matches are merged to produce the final result; (4) IoA values among all tracklets in the current frame are computed and saved as the overlap state for the next frame, after which appearance features are updated. The module does not modify the detector or the motion model. In experiments, it is integrated into the high-performance online tracker TrackTrack.

### Key Designs
1. **Overlap-aware appearance feature filtering**: At the end of each frame, the IoA (Intersection over Area) matrix is computed for all tracklet pairs. When a tracklet's IoA exceeds the update threshold $\tau_{\text{update}}=0.3$, its appearance feature update is suspended and the most recent feature from a non-overlapping frame is retained, preventing cross-contamination of appearance representations during occlusion. When one tracklet's IoA relative to another exceeds $\tau_{\text{overlap}}=0.8$, the pair is designated as an overlapping pair (prime/auxiliary), where the prime tracklet corresponds to the denominator in the IoA calculation, ensuring consistent role assignment.
2. **Mismatch reassignment strategy**: For each overlapping tracklet pair from the previous frame, the detection matched to the prime tracklet in the current frame is retrieved using the prime tracklet as an index key. The cosine distance between this detection and the prime tracklet (prime distance) and the cosine distance between it and the auxiliary tracklet (auxiliary distance) are then computed. When prime distance $> \tau_{\text{min}}=0.8$, auxiliary distance $<$ prime distance, and their difference $> \tau_{\text{dif}}=0.4$, a mismatch is declared: the detection is reassigned to the auxiliary tracklet and the prime tracklet is moved to the unmatched set. This strategy is applied within each stage of the two-stage matching procedure.
3. **Online operation with negligible computational overhead**: No global optimization or re-identification network is required. The module relies solely on IoA computation and cosine similarity comparisons against stored features, incurring minimal computational cost.

### Loss & Training
FC-Track is a purely inference-time module and involves no additional training or loss functions. All hyperparameters ($\tau_{\text{update}}=0.3$, $\tau_{\text{overlap}}=0.8$, $\tau_{\text{min}}=0.8$, $\tau_{\text{dif}}=0.4$) are set manually, and cosine distance is used for appearance similarity.

## Key Experimental Results

| Dataset | Method | HOTA↑ | MOTA↑ | IDF1↑ | AssA↑ | IDs↓ | FPS |
|---------|--------|-------|-------|-------|-------|------|-----|
| MOT17 | TrackTrack | 66.94 | 81.71 | 82.78 | 66.80 | 837 | 5.9 |
| MOT17 | **FC-Track** | **66.95** | **81.73** | **82.81** | **67.81** | 837 | 5.7 |
| MOT20 | TrackTrack | 65.61 | 77.52 | 80.82 | 67.35 | 719 | 0.7 |
| MOT20 | **FC-Track** | **65.67** | **77.52** | **80.90** | **67.48** | 719 | 0.6 |

**Identity switch duration analysis (MOT17 val)**:

| Method | # Switches | Mean Frames↓ | Median↓ | Long-term Switch Ratio↓ | IDTP↑ | IDFP↓ |
|--------|-----------|-------------|---------|------------------------|-------|-------|
| ByteTrack | 201 | 33.04 | 11 | 50.25% | 40434 | 13456 |
| BoT-SORT | 199 | 32.89 | 5 | 38.69% | 41757 | 12133 |
| TrackTrack | 236 | 22.88 | 5 | 36.86% | 42144 | 11746 |
| **FC-Track** | 308 | **18.33** | **3** | **29.55%** | **42305** | **11585** |

### Ablation Study
- **Similarity metric**: Cosine distance (HOTA=69.67) outperforms Euclidean distance (69.48), and both exceed the baseline (69.40).
- **Matching stage**: Insertion into only the first stage is effective (HOTA=69.67); insertion into the second stage yields no gain — the first stage handles high-confidence matches where correction potential is greater.
- **Threshold robustness**: All four hyperparameters outperform the baseline across a broad range of values, indicating that the method is insensitive to threshold selection.

## Highlights & Insights
- The paradigm of "post-association correction" is thought-provoking: rather than improving the association algorithm itself, an error-correction layer is added after association, making the approach orthogonal to the underlying association module.
- IoA (rather than IoU) is more appropriate for characterizing asymmetric occlusion relationships: the occluded tracklet exhibits high IoA, while the occluding tracklet exhibits low IoA.
- Analyzing identity switch duration is a more informative evaluation dimension than simply counting IDs.
- Suspending appearance updates during overlapping periods is a simple yet overlooked but important technique.

## Limitations & Future Work
- Improvements in overall HOTA/MOTA are marginal (+0.01–0.06); the primary contribution lies in identity switch duration rather than standard benchmarks.
- The total number of switches actually increases (236→308): switches are shorter but more frequent.
- Only geometric IoA is used to trigger correction; motion predictions and semantic information are not exploited.
- Validation is limited to pedestrian tracking datasets; generalization to vehicles, multi-class targets, and other scenarios remains unknown.
- FPS is low (5.7/0.6), with the primary bottleneck being the baseline TrackTrack detector rather than the correction module itself.

## Related Work & Insights
- vs. **TrackTrack (CVPR 2025)**: FC-Track adds post-association correction on top of TrackTrack without altering detection results, primarily improving AssA and reducing the long-term identity switch ratio.
- vs. **OC-SORT (CVPR 2023)**: OC-SORT corrects accumulated Kalman filter errors during occlusion using virtual tracklets, focusing on motion correction; FC-Track targets appearance-level detection–tracklet mismatch correction.
- vs. **ByteTrack**: ByteTrack reduces missed detections by leveraging low-confidence detections but does not address identity errors caused by overlap; FC-Track is orthogonal to detection-level improvements.

## Association with My Research
- Possible connection: `20260317_freq_fusion_small_target.md`
- Possible connection: `20260317_active_freq_detection.md`

## Rating
- Novelty: 5/10 — The idea of IoA filtering combined with appearance-based reassignment is straightforward and intuitive, lacking deeper technical contributions.
- Experimental Thoroughness: 7/10 — Complete metrics are reported on MOT17/MOT20 test sets, and the identity switch duration analysis is novel, though ablations are conducted only on the validation set.
- Writing Quality: 6/10 — The algorithmic logic is clear, but the incremental nature of the contribution means that the length of the exposition does not fully align with the scale of the contribution.
- Value: 5/10 — The module has engineering value as a plug-and-play component, but quantitative gains are limited.
- Novelty: ⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐
- Value to Me: ⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)
- [\[CVPR 2026\] Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation](out_of_sight_out_of_track_adversarial_attacks_on_propagation-based_multi-object_.md)
- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](tcei_dual_level_adaptation_multi_object_tracking.md)
- [\[CVPR 2026\] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](tcei_test_time_calibration_experience_intuition_mot.md)
- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](storm_referring_multi_object_tracking.md)

</div>

<!-- RELATED:END -->
