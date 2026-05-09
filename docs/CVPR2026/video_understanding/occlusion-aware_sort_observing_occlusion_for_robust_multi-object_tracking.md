---
title: >-
  [Paper Note] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking
description: >-
  [CVPR 2026][Video Understanding][Multi-Object Tracking] This paper proposes OA-SORT, an occlusion-aware tracking framework that explicitly models target occlusion states to mitigate positional cost ambiguity and Kalman Filter estimation instability. The method achieves state-of-the-art improvements on DanceTrack, SportsMOT, and MOT17, with all components being plug-and-play compatible with multiple tracker architectures.
tags:
  - CVPR 2026
  - Video Understanding
  - Multi-Object Tracking
  - Occlusion Awareness
  - Kalman Filter
  - Data Association
  - Plug-and-Play
date: 2026-05-08
content_hash: 86a74d90c3381e32
---

# Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking

**Conference**: CVPR 2026  
**arXiv**: [2603.06034](https://arxiv.org/abs/2603.06034)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Multi-Object Tracking, Occlusion Awareness, Kalman Filter, Data Association, Plug-and-Play

## TL;DR

This paper proposes OA-SORT, an occlusion-aware tracking framework that explicitly models target occlusion states to mitigate positional cost ambiguity and Kalman Filter estimation instability. The method achieves state-of-the-art improvements on DanceTrack, SportsMOT, and MOT17, with all components being plug-and-play compatible with multiple tracker architectures.

## Background & Motivation

**Positional cost ambiguity caused by occlusion in 2D MOT**: When objects of the same category partially occlude one another, detectors struggle to distinguish foreground from background, producing inaccurate detections and thereby introducing ambiguity into the IoU cost matrix, which leads to identity switches (ID switches).

**Kalman Filter sensitivity to inaccurate detections**: The discrete linear KF accumulates errors when repeatedly updated with inaccurate detections caused by occlusion, resulting in unstable estimates—a problem that is exacerbated under non-linear motion patterns.

**Vulnerability of auxiliary cues to occlusion**: Appearance features become unreliable under occlusion due to contamination from occluding targets; motion direction partially reduces matching failures but does not resolve cost ambiguity; detection confidence is also sensitive to occlusion.

**Absence of explicit occlusion state modeling**: Existing methods either model occlusion indirectly (e.g., via active/inactive states) or employ motion compensation to recover lost targets, but none directly estimate occlusion severity and leverage it to correct association costs.

**Underutilization of depth information**: Although methods such as PD-SORT and SparseTrack exploit pseudo-depth to design association strategies, they remain susceptible to cost ambiguity induced by occlusion.

**Need for a general, training-free occlusion-aware framework**: The objective is to design plug-and-play, training-free components that can be readily integrated into diverse tracker architectures to improve robustness.

## Method

### Overall Architecture

OA-SORT builds upon Hybrid-SORT as the baseline, employing a three-stage association strategy (high-score detection association → low-score detection association → lost tracklet re-linking), augmented with three occlusion-aware components:

- **OAM (Occlusion-Aware Module)**: Analyzes target occlusion states and produces occlusion coefficients.
- **OAO (Occlusion-Aware Offset)**: Incorporates occlusion coefficients into positional costs to alleviate cost ambiguity.
- **BAM (Bias-Aware Momentum)**: Leverages occlusion coefficients to refine the KF update step and suppress estimation drift.

The pipeline proceeds as follows: (1) After KF prediction, OAM computes occlusion coefficients from the estimates; (2) During association, OAO integrates occlusion coefficients into the spatial consistency measure; (3) Tracklets associated with low-score detections have their KF motion parameters refined via BAM; (4) Before the current frame ends, OAM recomputes occlusion coefficients from the latest observations for subsequent BAM usage.

### Key Designs

**1. Depth Ordering**: Under a bird's-eye-view camera geometry, the vertical position of a bounding box's bottom edge is used to infer depth ordering—a lower bottom-edge $y$-coordinate indicates a target closer to the camera. A 5-pixel threshold is applied to reduce sensitivity.

**2. Occlusion Coefficient**: Based on depth ordering, the occlusion coefficient for a target is computed as the ratio of the occluded area to its own bounding box area: $Oc_i = A(\mathcal{O}_i) / A(\mathcal{D}_i)$, with union-based handling for simultaneous occlusion by multiple targets.

**3. Gaussian Map (GM) Refinement**: The raw occlusion coefficient may overestimate occlusion severity because bounding box borders enclose substantial background pixels. A 2D Gaussian weight map is introduced to assign higher weights to pixels near the target center, yielding a refined coefficient $\hat{Oc}_i = \sum_{(x,y) \in \mathcal{O}_i} GM_{x,y} / A(\mathcal{D}_i)$.

**4. OAO Spatial Consistency Score**: The occlusion coefficient is applied to the KF estimate (rather than the detection, which is unstable under occlusion) and combined with IoU in a weighted sum: $S = \tau \cdot (1 - \hat{Oc}^X) + (1-\tau) \cdot C_{IoU}(\mathcal{D}, X)$. This is activated only during the first-stage high-score detection association.

**5. BAM Adaptive Momentum**: For low-score detections, a momentum term is derived from the occlusion coefficient of the tracklet's most recent observation and an IoU measure: $BAM = C_{IoU}(X_{t|t-1}, Z_t) \cdot (1 - \hat{Oc}^{Z_{t-1}})$. The current observation is then blended with the KF prediction: $\hat{Z}_t = BAM \cdot Z_t + (1-BAM) \cdot H_t X_{t|t-1}$. Severer occlusion or larger detection deviation causes the update to rely more heavily on the KF estimate.

### Loss & Training

OA-SORT is a **training-free** framework requiring no additional training or fine-tuning. All hyperparameters are set empirically:

- The GM parameters $\sigma^x, \sigma^y$ are adjusted according to dataset motion characteristics (DanceTrack: $w/3\sqrt{2}, h/3$; SportsMOT: $w/4, h/3$; MOT17: $w/2, h/2$).
- The OAO balancing coefficient $\tau$ is tuned in the range 0.1–0.2 (DanceTrack: 0.15, SportsMOT: 0.2, MOT17: 0.1).
- BAM uniformly uses HMIoU as the spatial consistency measure.

## Key Experimental Results

### Main Results

**DanceTrack test set (non-linear motion + frequent occlusion)**

| Method | HOTA | AssA | MOTA | IDF1 |
|--------|------|------|------|------|
| Hybrid-SORT | 62.2 | 47.4 | 91.6 | 63.0 |
| **OA-SORT** | **63.1** | **48.5** | **91.7** | **64.2** |
| OA-Byte (ByteTrack+OA) | 49.0 | 33.7 | 89.6 | 55.9 |
| OA-OC (OC-SORT+OA) | 56.5 | 39.6 | 91.2 | 57.6 |
| OA-Sparse (SparseTrack+OA) | 57.8 | 41.8 | 91.5 | 60.2 |
| OA-PD (PD-SORT+OA) | 60.4 | 44.9 | 91.4 | 60.8 |

**SportsMOT test set (variable-speed motion + camera motion)**

| Method | HOTA | AssA | MOTA | IDF1 |
|--------|------|------|------|------|
| Hybrid-SORT | 73.0 | 61.6 | 94.3 | 73.3 |
| **OA-SORT** | **73.4** | **62.3** | **94.4** | **74.1** |
| Hybrid-SORT* | 74.8 | 63.2 | 96.2 | 75.1 |
| **OA-SORT*** | **75.2** | **63.8** | **96.3** | **75.8** |

**MOT17 test set (linear motion + long-term occlusion)**

| Method | HOTA | AssA | MOTA | IDF1 |
|--------|------|------|------|------|
| Hybrid-SORT | 63.6 | 63.2 | 80.6 | 78.4 |
| **OA-SORT** | **64.2** | **64.0** | 79.6 | **79.1** |
| Hybrid-SORT-REID | 64.0 | 63.5 | 79.9 | 78.7 |

### Ablation Study

**Component ablation (DanceTrack-val)**

| OAO | BAM | GM | HOTA | AssA | IDF1 | Time Overhead |
|-----|-----|----|------|------|------|---------------|
| - | - | - | 59.4 | 44.9 | 60.7 | 9.57ms |
| ✓ | - | - | 59.9 | 45.6 | 60.9 | +1.96ms |
| - | ✓ | - | 60.5 | 46.5 | 62.2 | +3.81ms |
| ✓ | ✓ | - | 60.6 | 46.7 | 62.0 | +9.25ms |
| ✓ | ✓ | ✓ | **61.5** | **48.0** | **63.7** | +14.99ms |

**Cross-tracker generalization (DanceTrack-val, all with OAO+BAM+GM)**

| Tracker | HOTA: Orig→New | IDF1: Orig→New |
|---------|----------------|----------------|
| SORT | 48.4→50.4 (+2.0) | 49.6→53.3 (+3.7) |
| ByteTrack | 47.1→49.3 (+2.2) | 52.7→55.7 (+3.0) |
| OC-SORT | 52.3→53.8 (+1.5) | 52.0→53.9 (+1.9) |
| TrackTrack | 59.3→59.9 (+0.6) | 61.1→62.0 (+0.9) |

### Key Findings

- GM is the most critical component, contributing +2.1 HOTA when introduced alone, as it effectively suppresses the interference of background pixels at bounding box borders on occlusion estimation.
- BAM alone outperforms OAO alone (+1.1 vs. +0.5 HOTA), indicating that refining KF estimates is more beneficial than correcting association costs.
- Performance degrades when $\tau$ exceeds approximately 0.2; excessively large occlusion offsets distort the spatial consistency representation.
- OA-SORT without ReID even surpasses Hybrid-SORT-REID (+0.5 AssA, +0.3 IDF1), demonstrating that occlusion awareness can partially substitute for appearance features.
- Sequences with more severe occlusion benefit from larger improvements (see Fig. 1).

## Highlights & Insights

- **Direct modeling of occlusion severity**: Unlike indirect occlusion handling methods, OAM explicitly computes occlusion coefficients and injects them into both the association stage and the KF update step.
- **Gaussian Map refinement of occlusion estimates**: The use of a 2D Gaussian weight map to downweight background pixels near bounding box borders is a novel contribution and the largest source of performance gain in the paper.
- **Fully plug-and-play and training-free**: All three modules can be used independently and have been validated across seven trackers: SORT, ByteTrack, OC-SORT, SparseTrack, PD-SORT, TrackTrack, and BOT-SORT.
- **Rigorous problem analysis**: Section 3 provides a systematic analysis of how occlusion induces cost ambiguity through detection error and estimation error accumulation, establishing a solid theoretical foundation for the proposed designs.

## Limitations & Future Work

- **Restrictive assumption of bottom-edge depth ordering**: When the lower half of a target is occluded or when targets are airborne (e.g., jumping), the bottom-edge position fails to accurately reflect depth relationships, degrading framework performance.
- **Lack of long-term occlusion modeling**: Occlusion states and their temporal evolution constitute a continuous process; the current method relies only on instantaneous occlusion information without modeling the long-term temporal dynamics of occlusion.
- **Computational overhead of GM**: Introducing GM increases average per-frame tracking time by approximately 15ms, which, while still satisfying real-time requirements, may become a bottleneck in very large-scale scenarios.
- **Dataset-specific hyperparameter tuning**: The parameters $\sigma^x$, $\sigma^y$, and $\tau$ require empirical tuning for different scenarios, necessitating additional calibration when generalizing to entirely new domains.

## Related Work & Insights

- **Position-association methods**: SORT → OC-SORT → Hybrid-SORT → PD-SORT/SparseTrack (leveraging pseudo-depth). OA-SORT extends this line by further introducing explicit occlusion modeling.
- **Feature-association methods**: StrongSORT++ and BOT-SORT-REID enhance association with appearance features, but feature reliability also degrades under occlusion. OA-SORT demonstrates that occlusion awareness can partially replace ReID.
- **Occlusion handling methods**: Stadler (indirect modeling via active/inactive states), Hibo (motion compensation to recover lost targets), and DiffMOT (diffusion-model-based prediction)—none of these directly estimate occlusion severity.

## Rating

- Novelty: ⭐⭐⭐⭐ — Explicit occlusion coefficients, Gaussian Map refinement, and dual-path injection constitute a novel and well-motivated design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets, seven tracker integrations, detailed ablations, and hyperparameter sensitivity analysis.
- Writing Quality: ⭐⭐⭐⭐ — Problem analysis is thorough, mathematical derivations are clear, and figures and tables are well-organized.
- Value: ⭐⭐⭐⭐ — The plug-and-play design offers strong practical utility, with consistent and reasonable performance gains.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[AAAI 2026\] PlugTrack: Multi-Perceptive Motion Analysis for Adaptive Fusion in Multi-Object Tracking](../../AAAI2026/video_understanding/plugtrack_multi-perceptive_motion_analysis_for_adaptive_fusion_in_multi-object_t.md)
- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](tcei_dual_level_adaptation_multi_object_tracking.md)
- [\[CVPR 2026\] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](dual-level_adaptation_for_multiobject_tracking_building_testtime_calibration_from.md)
- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](storm_referring_multi_object_tracking.md)

<!-- RELATED:END -->
