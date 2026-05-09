---
title: >-
  [Paper Note] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking
description: >-
  [CVPR 2026][Video Understanding][Multi-object tracking] FC-Track is a lightweight post-association correction framework that explicitly corrects identity switch errors caused by target overlap in online MOT. It employs IoA (Intersection over Area)-based overlap-aware appearance feature filtering and a local mismatch reassignment strategy, reducing the long-term identity switch ratio to 29.55%.
tags:
  - CVPR 2026
  - Video Understanding
  - Multi-object tracking
  - identity switch correction
  - overlap-aware
  - post-association correction
  - online tracking
date: 2026-05-08
content_hash: 4e7825438b46ccc6
---

# FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking

**Conference**: CVPR 2026
**arXiv**: [2603.12758](https://arxiv.org/abs/2603.12758)
**Code**: To be confirmed
**Area**: Video Understanding
**Keywords**: Multi-object tracking, identity switch correction, overlap-aware, post-association correction, online tracking

## TL;DR
FC-Track is a lightweight post-association correction framework that explicitly corrects identity switch errors caused by target overlap in online MOT. It employs IoA (Intersection over Area)-based overlap-aware appearance feature filtering and a local mismatch reassignment strategy, reducing the long-term identity switch ratio to 29.55%.

## Background & Motivation
Multi-object tracking (MOT) is a core component in robotic perception, autonomous driving, and video analysis. The dominant tracking-by-detection paradigm first detects then associates, but frequent occlusions and target overlaps in crowded scenes make association errors inevitable.

**Root Cause**: Once an association error occurs (a detection being assigned to the wrong tracklet), the error propagates over time — subsequent frames use the incorrect tracklet's appearance features for matching, resulting in long-term ID switches that severely degrade tracking consistency.

**Limitations of Prior Work**:
1. Most methods treat association decisions as irreversible — no correction is made once matching is complete.
2. Methods that improve association accuracy (better appearance or motion models) can only reduce the probability of errors but cannot correct them after they occur.
3. Offline/global optimization methods can retroactively correct errors but fail to meet real-time requirements.
4. Existing online correction methods (e.g., OC-SORT) primarily address motion model errors without explicitly targeting association errors caused by overlap.

**Core Idea**: Insert a lightweight error-correction module after the association stage — detect overlapping tracklet pairs, freeze appearance feature updates during overlap, and use clean appearance features saved prior to overlap to determine whether an identity swap has occurred and correct it.

## Method

### Overall Architecture
FC-Track is embedded as a post-processing plugin into the standard online MOT pipeline. After each frame's standard detection → association step:
(1) Compute IoA between all tracklet pairs to identify overlapping pairs;
(2) Freeze appearance feature updates when IoA is high;
(3) Perform a secondary appearance similarity verification on matches within overlapping pairs, and reassign if an error is detected.

### Key Designs
1. **IoA Overlap-Aware Appearance Feature Filtering**:

    - **Function**: Suspend appearance feature updates during target overlap to prevent feature cross-contamination.
    - **Mechanism**: Compute the IoA (intersection area / reference box area) for all tracklet pairs. When IoA exceeds the update threshold $\tau_{update} = 0.3$, freeze the tracklet's appearance features, retaining the last clean features from a non-overlapping frame. When IoA exceeds the overlap threshold $\tau_{overlap} = 0.8$, form an overlapping tracklet pair $(t_{pri}, t_{aux})$.
    - **Design Motivation for IoA over IoU**: IoA is an asymmetric metric (using one box's area as the denominator), more accurately reflecting the degree to which a small target is occluded by a larger one. IoU is sensitive to size differences — when a small target is fully covered by a large one, IoU may remain low while IoA approaches 1.
    - **Prime/Auxiliary Role Assignment**: The tracklet used as the denominator in the IoA computation (i.e., the smaller, more heavily occluded one) is designated as *prime* and serves as the index key in subsequent reassignment.

2. **Local Mismatch Reassignment**:

    - **Function**: Perform secondary verification and correction of association results within overlapping tracklet pairs.
    - **Mechanism**: For an overlapping pair $(t_{pri}, t_{aux})$, take the detection $d_f$ matched to $t_{pri}$ and compute:
        - $S_{pri} = \text{Distance}(F_{det}[d_f], F_{trk}[t_{pri}])$ (appearance distance between detection and prime)
        - $S_{aux} = \text{Distance}(F_{det}[d_f], F_{trk}[t_{aux}])$ (appearance distance between detection and auxiliary)

      When $S_{pri} \geq \tau_{min}$ (the prime match distance is large enough to indicate unreliable matching) and $S_{pri} - S_{aux} \geq \tau_{dif}$ (auxiliary is clearly closer), reassignment is executed: detection $d_f$ is reassigned to $t_{aux}$.
    - **Design Motivation**: The strict dual-threshold condition ($\tau_{min} = 0.8$, $\tau_{dif} = 0.4$) ensures correction is triggered only under high-confidence conditions, avoiding erroneous corrections. Clean appearance features saved prior to overlap serve as the comparison reference rather than features contaminated during overlap.

3. **Integration into Two-Stage Matching**:

    - **Function**: Embed the correction module into each stage of two-stage association.
    - **Mechanism**: Mainstream MOT adopts two-stage matching (first high-confidence detections, then low-confidence detections); FC-Track applies one correction pass after each stage's association.
    - Ablation studies show that correction is more effective at stage one (high-confidence association) and has limited effect at stage two (low-confidence, ambiguous matching).

### Loss & Training
- FC-Track is a purely inference-time post-processing module — **no training is required**.
- All thresholds are fixed hyperparameters: $\tau_{update}=0.3$, $\tau_{overlap}=0.8$, $\tau_{min}=0.8$, $\tau_{dif}=0.4$.
- Implemented on top of TrackTrack (current state-of-the-art online tracker).

## Key Experimental Results

### Main Results

| Method | Dataset | HOTA↑ | MOTA↑ | IDF1↑ | AssA↑ | IDs↓ | FPS |
|--------|---------|-------|-------|-------|-------|------|-----|
| ByteTrack | MOT17 | 63.05 | 80.25 | 77.30 | 61.98 | 2196 | 29.6 |
| BoT-SORT | MOT17 | 65.05 | 80.55 | 80.23 | 65.49 | 1212 | 6.8 |
| TrackTrack | MOT17 | 66.94 | 81.71 | 82.78 | 66.80 | 837 | 5.9 |
| **FC-Track** | **MOT17** | **66.95** | **81.73** | **82.81** | **67.81** | **837** | **5.7** |
| TrackTrack | MOT20 | 65.61 | 77.52 | 80.82 | 67.35 | 719 | 0.7 |
| **FC-Track** | **MOT20** | **65.67** | **77.52** | **80.90** | **67.48** | **719** | **0.6** |

### Ablation Study — ID Switch Duration Analysis (MOT17 val)

| Tracker | Switch Count↓ | Avg. Duration (frames)↓ | Median Duration (frames)↓ | Long-term Switch Ratio↓ | IDTP↑ | IDFP↓ |
|---------|--------------|------------------------|--------------------------|------------------------|-------|-------|
| ByteTrack | 201 | 33.04 | 11 | 50.25% | 40434 | 13456 |
| BoT-SORT | 199 | 32.89 | 5 | 38.69% | 41757 | 12133 |
| TrackTrack | 236 | 22.88 | 5 | 36.86% | 42144 | 11746 |
| **FC-Track** | **308** | **18.33** | **3** | **29.55%** | **42305** | **11585** |

| Ablation Config | HOTA↑ | IDF1↑ | AssA↑ | Notes |
|----------------|-------|-------|-------|-------|
| Baseline (TrackTrack) | 69.40 | 81.86 | 73.57 | No correction |
| + Euclidean distance | 69.48 | 81.90 | 73.71 | Euclidean distance |
| + Cosine distance | **69.67** | **82.12** | **74.08** | Cosine distance (superior) |
| Stage 1 only | 69.67 | 82.12 | 74.08 | Stage-1 correction effective |
| Stage 2 only | 69.40 | 81.86 | 73.57 | Stage-2 correction ineffective |

### Key Findings
- Although FC-Track produces slightly more identity switches (308 vs. 236), the **average switch duration is substantially reduced** (18.33 vs. 22.88 frames), with a median of only 3 frames — indicating that errors are corrected rapidly.
- **The long-term identity switch ratio drops from 36.86% to 29.55%**, which is the core contribution: even when identity switches occur, the tracker recovers quickly.
- IDTP increases (42305 vs. 42144) while IDFP decreases (11585 vs. 11746), indicating that more frames are correctly associated.
- Cosine distance outperforms Euclidean distance, as the direction of appearance features is more discriminative than their magnitude.
- Correction is effective at stage one (high-confidence matching) but not at stage two (low-confidence matching), where correct associations are inherently harder to determine.

## Highlights & Insights
- The perspective is novel: rather than pursuing "error prevention," the paper focuses on "error recovery" — a more practical approach for real-world deployment.
- The method is extremely lightweight: no training, no additional networks, only 4 hyperparameters, functioning as a plug-and-play module with negligible computational overhead.
- The introduction of ID switch duration analysis provides richer diagnostic information than simple ID switch counts.
- The use of IoA over IoU is well-motivated — IoA is more sensitive to asymmetric occlusion where a small target is fully covered by a larger one.

## Limitations & Future Work
- Improvements on standard metrics (HOTA, IDF1) are marginal (e.g., +0.01 HOTA on MOT17); the main gains are captured by the ID switch duration analysis.
- Validation is limited to TrackTrack; generalizability should be assessed on additional baselines such as ByteTrack and BoT-SORT.
- The overlap threshold $\tau_{overlap}=0.8$ is quite strict; moderate occlusions (IoA 0.3–0.8) may be missed.
- Only pairwise overlaps are considered; scenarios involving three or more simultaneously overlapping targets are not handled.
- Appearance feature freezing may become stale after prolonged occlusion if the target's appearance changes; a more principled feature update strategy is needed.
- FPS decreases slightly (5.9→5.7), which may require optimization under stricter real-time constraints.

## Related Work & Insights
- OC-SORT constructs virtual trajectories during occlusion to correct motion model errors; this paper addresses errors at the appearance/association level — the two approaches are complementary.
- UnfcTrack models appearance change sequences using unfalsified control, a more complex but more comprehensive approach.
- Broader implication for the MOT community: post-association correction should be treated as a standard component of the MOT pipeline rather than an optional module.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The "correction over prevention" paradigm is novel, and the IoA-based correction design is well-reasoned, though technical depth is moderate.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Standard evaluation on MOT17 and MOT20 with ID switch duration analysis and complete ablations, but multi-baseline validation is lacking.
- **Writing Quality**: ⭐⭐⭐ Structure is clear but writing quality is average; some descriptions are unnecessarily verbose.
- **Value**: ⭐⭐⭐⭐ The plug-and-play lightweight module has practical value for industrial deployment; the ID switch duration analysis contributes methodologically.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)
- [\[CVPR 2026\] Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation](out_of_sight_out_of_track_adversarial_attacks_on_propagation-based_multi-object_.md)
- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](tcei_dual_level_adaptation_multi_object_tracking.md)
- [\[CVPR 2026\] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](tcei_test_time_calibration_experience_intuition_mot.md)
- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](storm_referring_multi_object_tracking.md)

<!-- RELATED:END -->
