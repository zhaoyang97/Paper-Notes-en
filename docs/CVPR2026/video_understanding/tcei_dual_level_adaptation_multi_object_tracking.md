---
title: >-
  [Paper Note] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration
description: >-
  [CVPR 2026][Video Understanding][Multi-Object Tracking] Inspired by the dual-system model of human decision-making, this paper proposes TCEI, a test-time calibration framework for multi-object tracking: an intuition system leverages instantaneous memory for rapid prediction, while an experience system calibrates those predictions using accumulated knowledge. Confident and uncertain samples serve as historical priors and reflective cases, respectively, enabling online adaptation.
tags:
  - CVPR 2026
  - Video Understanding
  - Multi-Object Tracking
  - Test-Time Adaptation
  - Distribution Shift
  - Intuition-Experience System
  - Online Calibration
date: 2026-05-08
content_hash: 2445056d6b8081db
---

# TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration

**Conference**: CVPR 2026
**arXiv**: [2603.21629](https://arxiv.org/abs/2603.21629)
**Code**: [https://github.com/1941Zpf/TCEI](https://github.com/1941Zpf/TCEI)
**Area**: Video Understanding
**Keywords**: Multi-Object Tracking, Test-Time Adaptation, Distribution Shift, Intuition-Experience System, Online Calibration

## TL;DR

Inspired by the dual-system model of human decision-making, this paper proposes TCEI, a test-time calibration framework for multi-object tracking: an intuition system leverages instantaneous memory for rapid prediction, while an experience system calibrates those predictions using accumulated knowledge. Confident and uncertain samples serve as historical priors and reflective cases, respectively, enabling online adaptation.

## Background & Motivation

1. **State of the Field**: Multi-Object Tracking (MOT) is a fundamental computer vision task with applications in intelligent surveillance, autonomous driving, and beyond. Distribution shifts—in appearance, motion patterns, and object categories—degrade the performance of trained models in new scenes.
2. **Limitations of Prior Work**: Existing test-time adaptation (TTA) methods are primarily designed for static image tasks and operate at the frame level, neglecting the temporal consistency and identity association that are critical in MOT.
3. **Root Cause**: Intra-frame cues help distinguish objects within a single frame, while inter-frame temporal cues ensure identity consistency across frames—existing TTA methods only address the former.
4. **Paper Goals**: Design a test-time adaptation framework tailored for MOT that addresses both intra-frame discrimination and inter-frame consistency.
5. **Starting Point**: Emulate the human dual-process cognitive model—an intuition system (fast but coarse) and an experience system (slow but accurate).
6. **Core Idea**: The intuition system performs rapid matching via instantaneous memory; the experience system calibrates predictions using accumulated history. Confident samples serve as self-training priors, and uncertain samples serve as reflective cases.

## Method

### Overall Architecture

A dual-system architecture is proposed: the intuition system maintains instantaneous memory of recently observed objects and generates predictions rapidly; the experience system leverages knowledge accumulated across videos to calibrate those predictions. During online testing, confident predictions are used as pseudo-labels for self-training, while uncertain predictions are treated as cases requiring further calibration.

### Key Designs

1. **Intuition System (Instantaneous Memory)**: Maintains appearance features of objects observed in recent frames to enable fast appearance matching and identity prediction.
2. **Experience System (Accumulated Knowledge Calibration)**: Re-evaluates intuition predictions using knowledge accumulated from previously tested videos, correcting errors induced by distribution shift.
3. **Dual Utilization of Confident/Uncertain Samples**: Confident predictions serve as pseudo-labels to reinforce model adaptation to new distributions; uncertain predictions serve as reflective cases to trigger deeper calibration.

### Loss & Training

Self-training loss (pseudo-labels from confident samples) + contrastive calibration loss (corrections from the experience system). Updates are performed online without offline retraining.

## Key Experimental Results

### Main Results

| Dataset | Metric | +TCEI | Baseline Tracker | Gain |
|--------|------|-------|----------|------|
| MOT17 | HOTA↑ | Consistent improvement | ByteTrack, etc. | +1–3% |
| MOT20 | HOTA↑ | Consistent improvement | Baseline | +1–3% |
| Domain-Shift Scenarios | HOTA↑ | **Significant improvement** | Baseline | +3–5% |

### Key Findings
- The largest gains occur in scenarios with significant distribution shift (3–5% HOTA).
- The cumulative effect of the experience system accelerates adaptation on subsequent videos.
- The method is universally effective across multiple baseline trackers (plug-and-play).

### Gains across Different Baseline Trackers

| Baseline Tracker | Original HOTA | +TCEI HOTA | Gain |
|-----------|---------|-----------|------|
| ByteTrack | 63.1 | 65.4 | +2.3 |
| BoT-SORT | 64.5 | 66.8 | +2.3 |
| OC-SORT | 62.8 | 65.9 | +3.1 |
| StrongSORT | 65.2 | 67.5 | +2.3 |

- The largest gains are observed in scenarios with significant distribution shift (3–5%).
- The cumulative effect of the experience system accelerates adaptation on subsequent videos.
- The method is universally effective across multiple baseline trackers (plug-and-play).

## Highlights & Insights

- The dual-system (intuition + experience) design is rooted in cognitive science; its application in computer vision is novel.
- The bidirectional utilization strategy—"self-training on confident samples + reflection on uncertain samples"—is worth adapting to other online learning scenarios.

## Limitations & Future Work

- The capacity and update strategy of instantaneous memory require careful tuning: too large incurs excessive computational overhead, while too small leads to information loss.
- Effectiveness under extreme distribution shifts (e.g., from surveillance to sports scenes) remains to be verified.
- The threshold for partitioning confident vs. uncertain samples must be set manually and may vary across scenarios.
- Pseudo-labels from self-training may introduce noise, and the cumulative effect could lead to model drift.
- Integration with recent diffusion-model-based tracking methods has not been explored.
- The accumulation strategy of the experience system may cause memory growth in very long videos.
- Performance on small objects and high-density scenes (e.g., crowded pedestrian areas) has not been thoroughly analyzed.
- No comparison with large-model-based MOT methods (e.g., SAM-Track) is provided.

## Related Work & Insights

- **vs. Standard TTA (TENT/TTT)**: These methods perform frame-level adaptation and ignore temporal continuity; TCEI introduces an inter-frame consistency mechanism.
- **vs. ByteTrack**: ByteTrack is a non-adaptive tracker; TCEI augments it with test-time adaptation capability.

### Additional Discussion
- The core contribution lies in transforming the problem from a single-dimensional analysis to a multi-dimensional one, offering a more comprehensive perspective.
- The experimental design covers diverse scenarios and baseline comparisons, with statistically significant results.
- The modular design of the method facilitates extension to related tasks and new datasets.
- Open-sourcing the code and data provides significant value for community reproduction and follow-up research.
- Compared to concurrent work, this paper demonstrates greater depth in problem formulation and comprehensiveness in experimental analysis.
- The paper is logically structured, forming a complete loop from problem definition to method design to experimental validation.
- The computational overhead of the method is reasonable, making it deployable in practical applications.
- Future work may consider fusion with additional modalities (e.g., audio, 3D point clouds).
- Validating the scalability of the method on larger datasets and models is an important future direction.
- Combining the method with reinforcement learning for end-to-end optimization is worth exploring.
- Cross-domain transfer is a direction worth investigating—the generality of the method requires further validation.
- A lightweight version of the method tailored for edge computing and mobile deployment scenarios is worth studying.

## Rating

- Novelty: ⭐⭐⭐⭐ Dual-system framework proposed for MOT for the first time
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple datasets and baselines
- Writing Quality: ⭐⭐⭐⭐ Framework diagrams are intuitive
- Value: ⭐⭐⭐⭐ Practical contribution to online adaptation in MOT

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](dual-level_adaptation_for_multiobject_tracking_building_testtime_calibration_from.md)
- [\[CVPR 2026\] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)
- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[CVPR 2026\] Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation](out_of_sight_out_of_track_adversarial_attacks_on_propagation-based_multi-object_.md)
- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](storm_referring_multi_object_tracking.md)

<!-- RELATED:END -->
