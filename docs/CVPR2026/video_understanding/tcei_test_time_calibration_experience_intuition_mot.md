---
title: >-
  [Paper Note] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition
description: >-
  [CVPR 2026][Video Understanding][Multi-Object Tracking] Inspired by Kahneman's dual-process theory, the TCEI framework proposes a test-time adaptation method that combines an intuitive system (rapid inference via transient memory of recently observed objects) with an experiential system (calibration of intuitive predictions using knowledge accumulated from historical videos), achieving significant improvements in multi-object tracking under distribution shift without requiring backpropagation.
tags:
  - CVPR 2026
  - Video Understanding
  - Multi-Object Tracking
  - Test-Time Adaptation
  - Dual-System Theory
  - Distribution Shift
  - Identity Association
date: 2026-05-08
content_hash: d5c8d65595c5a09e
---

# Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition

**Conference**: CVPR 2026
**arXiv**: [2603.21629](https://arxiv.org/abs/2603.21629)
**Code**: [https://github.com/1941Zpf/TCEI](https://github.com/1941Zpf/TCEI)
**Area**: Video Understanding
**Keywords**: Multi-Object Tracking, Test-Time Adaptation, Dual-System Theory, Distribution Shift, Identity Association

## TL;DR

Inspired by Kahneman's dual-process theory, the TCEI framework proposes a test-time adaptation method that combines an intuitive system (rapid inference via transient memory of recently observed objects) with an experiential system (calibration of intuitive predictions using knowledge accumulated from historical videos), achieving significant improvements in multi-object tracking under distribution shift without requiring backpropagation.

## Background & Motivation

1. **Background**: Multi-object tracking (MOT) frequently encounters distribution shifts in appearance, motion patterns, and object categories between training and test data, degrading online inference performance. Test-time adaptation (TTA) is a promising paradigm for alleviating this issue.
2. **Limitations of Prior Work**: Existing TTA methods primarily target static image tasks (classification, segmentation) and adapt using only intra-frame information, neglecting the inter-frame temporal consistency and identity association requirements of MOT. Backpropagation-based TTA methods also suffer from low computational efficiency and catastrophic forgetting.
3. **Key Challenge**: In MOT, intra-frame cues are needed to distinguish objects, while inter-frame temporal cues ensure ID consistency — both are equally important, yet existing TTA methods address only the former.
4. **Goal**: Design a forward-pass TTA method for MOT that leverages historically observed objects to provide temporal guidance for current ID association.
5. **Key Insight**: Drawing an analogy to human dual-process decision-making — fast intuitive judgment (System 1) combined with slow, deliberative calibration (System 2).
6. **Core Idea**: The intuitive system provides rapid predictions via transient memory of recent objects, while the experiential system calibrates inconsistencies in intuitive predictions using knowledge accumulated across all processed videos.

## Method

### Overall Architecture

TCEI is a forward-pass TTA framework built on top of existing MOT trackers. The pipeline proceeds as follows: (1) the baseline tracker generates initial ID predictions → (2) the intuitive system uses high-confidence objects in transient memory as temporal priors and low-confidence objects as reflective cases to augment and scrutinize predictions → (3) the experiential system checks the consistency of intuitive predictions against historical experience and actively calibrates when inconsistencies arise.

### Key Designs

1. **Intuitive System**:
    - **Function**: Rapid prediction augmentation using transient memory of recently observed objects.
    - **Mechanism**: A transient memory buffer stores recently processed objects. High-confidence predictions serve as "temporal priors" — their similarity to current detections is used to enhance the accuracy of current ID predictions. Low-confidence/uncertain objects serve as "reflective cases" — signaling the model to avoid similarly unreliable predictions. Both signals integrate training-time knowledge with test-time observations.
    - **Design Motivation**: Analogous to human intuitive decision-making — quickly recalling recent experience and making preliminary judgments by referencing high- and low-confidence cases.

2. **Experiential System**:
    - **Function**: Calibrating intuitive predictions using long-range historical experience.
    - **Mechanism**: The experiential system maintains knowledge accumulated from all processed test videos. Experience embeddings evolve alongside query embeddings to capture object-specific features. When intuitive predictions are consistent with historical experience, they are preserved (stability); when inconsistencies arise, the system actively intervenes to calibrate (bias correction).
    - **Design Motivation**: The intuitive system relies only on recent objects and cannot provide long-range temporal information; the experiential system compensates for this limitation.

3. **Cache-based TTA Mechanism**:
    - **Function**: Enabling test-time optimization without backpropagation.
    - **Mechanism**: A key-value cache model stores historical samples. High-confidence objects populate a positive cache (providing priors), while low-confidence objects populate a negative cache (providing reflective signals). The cache is dynamically updated to always reflect the latest test environment. Experience embeddings evolve with query embeddings to capture object-specific rather than category-level features.
    - **Design Motivation**: Avoids the computational overhead of backpropagation and the unstable parameter updates caused by noisy samples, offering greater stability compared to backpropagation-based TTA methods such as TENT.

### Loss & Training

TCEI is a purely forward-pass method and involves no training or backpropagation. Both intuitive prediction and experiential calibration are performed at inference time through cache queries and similarity computation.

## Key Experimental Results

### Main Results

| Dataset | Metric | TCEI | Baseline Tracker | Gain |
|---------|--------|------|-----------------|------|
| MOT17 | HOTA/IDF1 | SOTA | Baseline | Significant |
| MOT20 | HOTA/IDF1 | SOTA | Baseline | Significant |
| DanceTrack | HOTA/IDF1 | SOTA | Baseline | Significant |
| All datasets | Consistency | Uniformly improved | — | Strong generalization |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Baseline tracker only | Baseline | No TTA adaptation |
| + Intuitive system (positive cache) | Improved | Temporal prior is effective |
| + Intuitive system (positive + negative cache) | Further improved | Reflective mechanism is effective |
| + Experiential system | SOTA | Long-range calibration provides additional gains |

### Key Findings

- TCEI consistently outperforms the unadapted baseline across three mainstream benchmarks, validating the value of test-time adaptation for MOT.
- The forward-pass approach is more stable than backpropagation-based TTA methods (e.g., TENT) and is less prone to catastrophic forgetting.
- Jointly exploiting high-confidence and low-confidence objects outperforms using only high-confidence objects.
- The long-range memory of the experiential system is particularly important for scenarios with large appearance variation (e.g., DanceTrack).
- The intuitive system constructs transient memory of recent objects, using high-confidence predictions as temporal priors to enhance current ID predictions.
- Low-confidence/uncertain objects serve as reflective cases, guiding the model to avoid similarly unreliable predictions.

## Highlights & Insights

- **The mapping from dual-process theory to MOT-TTA** is natural: recent memory → fast intuitive judgment → deliberative experiential calibration. This cognitive framework provides clear design principles for the method.
- **The negative cache/reflective mechanism** is an interesting design: leveraging failure and uncertain cases as a "pitfall guide."
- **Forward-pass TTA** is critical for latency-sensitive MOT scenarios, avoiding the overhead and instability associated with backpropagation.

## Limitations & Future Work

- Cache size and update strategy require careful tuning.
- Knowledge accumulation in the experiential system may introduce interference from outdated information in extremely long video sequences.
- Modeling of inter-object interaction relationships is not considered.

## Related Work & Insights

- **vs. TENT/FSTTA**: Backpropagation-based TTA methods incur high computational cost and exhibit instability; TCEI requires only forward passes.
- **vs. TDA/Tip-Adapter**: Cache-based TTA methods previously applied only to static images; TCEI extends this paradigm to video temporal modeling.
- **vs. ByteTrack/OC-SORT**: Conventional trackers lack test-time adaptation capability; TCEI can be incorporated as a plug-in module to enhance any tracker.
- The mapping from dual-process theory to MOT-TTA is natural: recent memory → fast intuitive judgment → deliberative experiential calibration.
- The negative cache/reflective mechanism, which leverages failure and uncertain cases as a "pitfall guide," represents an interesting design innovation.
- Experience embeddings evolve alongside query embeddings rather than serving as fixed templates, capturing object-specific rather than category-level features.
- MOTIP's ID decoder reformulates association as direct ID prediction; TCEI can serve as an upper-level adaptive module on top of it.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The interdisciplinary combination of dual-process cognitive theory and MOT test-time adaptation is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset validation on MOT17/MOT20/DanceTrack with complete ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear; the analogical framework drawn from human cognition is visually intuitive.
- **Value**: ⭐⭐⭐⭐ Represents the first systematic study of test-time adaptation for MOT; the forward-pass approach offers strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](tcei_test_time_calibration_experience_intuition_mot.md)
- [\[CVPR 2026\] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)
- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[CVPR 2026\] Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation](out_of_sight_out_of_track_adversarial_attacks_on_propagation-based_multi-object_.md)
- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](storm_referring_multi_object_tracking.md)

</div>

<!-- RELATED:END -->
