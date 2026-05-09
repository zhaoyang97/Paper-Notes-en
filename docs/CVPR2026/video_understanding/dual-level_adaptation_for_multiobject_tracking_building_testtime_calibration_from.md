---
title: >-
  [Paper Note] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition
description: >-
  [CVPR 2026][Video Understanding][Multi-Object Tracking] Inspired by Kahneman's dual-system theory of human decision-making, TCEI proposes a test-time calibration framework for multi-object tracking. The intuitive system leverages transient memory of recently observed objects (confident samples as temporal priors and uncertain samples as reflective cases) for rapid prediction, while the experiential system validates and calibrates intuitive predictions using knowledge accumulated from historical videos. The entire process requires only forward passes without backpropagation, achieving significant robustness improvements under distribution shift across multiple MOT benchmarks.
tags:
  - CVPR 2026
  - Video Understanding
  - Multi-Object Tracking
  - Test-Time Adaptation
  - Dual-System Theory
  - Cache Mechanism
  - Distribution Shift
date: 2026-05-08
content_hash: 4d60cc8132cb9d07
---

# Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition

**Conference**: CVPR 2026
**arXiv**: [2603.21629](https://arxiv.org/abs/2603.21629)
**Code**: [https://github.com/1941Zpf/TCEI](https://github.com/1941Zpf/TCEI)
**Area**: Video Understanding / Multi-Object Tracking
**Keywords**: Multi-Object Tracking, Test-Time Adaptation, Dual-System Theory, Cache Mechanism, Distribution Shift

## TL;DR

Inspired by Kahneman's dual-system theory of human decision-making, TCEI proposes a test-time calibration framework for multi-object tracking. The intuitive system leverages transient memory of recently observed objects (confident samples as temporal priors and uncertain samples as reflective cases) for rapid prediction, while the experiential system validates and calibrates intuitive predictions using knowledge accumulated from historical videos. The entire process requires only forward passes without backpropagation, achieving significant robustness improvements under distribution shift across multiple MOT benchmarks.

## Background & Motivation

1. **Background**: Mainstream MOT methods include detection-based trackers (ByteTrack, OC-SORT) and Transformer-based end-to-end methods (MOTR, MOTIP). Test-time adaptation (TTA) has achieved success in image classification and semantic segmentation.
2. **Limitations of Prior Work**: Distribution shifts (appearance, motion patterns, categories) between training and test data degrade model performance. Existing TTA methods primarily handle static image tasks and lack temporal modeling capacity for multi-object scenarios—intra-frame information alone is insufficient to maintain ID consistency. Backpropagation-based TTA methods (e.g., TENT) are computationally expensive and prone to catastrophic forgetting.
3. **Key Challenge**: MOT requires both intra-frame cues to distinguish targets and inter-frame temporal cues to maintain ID consistency, yet existing TTA methods cannot satisfy both requirements simultaneously.
4. **Goal**: Design a forward-pass-only TTA method that leverages historical information from the test environment to improve MOT performance online.
5. **Key Insight**: Drawing inspiration from Kahneman's dual-system theory—human decision-making first relies on rapid intuitive judgment, which is then reviewed and corrected by the experiential system.
6. **Core Idea**: Construct dual-level adaptation: transient memory provides recent temporal priors (intuition), while historical experience provides long-range calibration (experience).

## Method

### Overall Architecture

TCEI is embedded into the inference stage of existing MOT methods. The intuitive system maintains transient memory storing recent confident and uncertain targets; the experiential system maintains experience embeddings accumulating knowledge from all processed videos. During inference, the intuitive system first enhances ID predictions for the current frame using transient memory, after which the experiential system determines whether calibration is necessary.

### Key Designs

1. **Intuitive System**:

   - **Function**: Rapid prediction enhancement via transient memory of recently observed objects.
   - **Mechanism**: Two types of transient memory are maintained—(a) a confident target cache storing features of recent high-confidence predictions as temporal priors to improve ID prediction accuracy for similar targets in the current frame; (b) an uncertain target cache storing predictions with low confidence as reflective cases, prompting the model to avoid similar unreliable predictions. By querying both caches, the system synthesizes training knowledge and recent observations to generate more comprehensive predictions.
   - **Design Motivation**: Recently observed targets carry appearance and motion information from the current test environment, compensating for train-test distribution shift.

2. **Experiential System**:

   - **Function**: Validation and calibration of intuitive predictions using long-range historical experience.
   - **Mechanism**: Experience embeddings co-evolve with query embeddings in the Transformer decoder, capturing target-specific features. When intuitive predictions are consistent with historical experience, the experiential system remains silent to preserve stability; when inconsistency is detected, it actively intervenes to calibrate predictions, providing long-range temporal information beyond the coverage of short-term memory.
   - **Design Motivation**: The intuitive system recalls only recent objects and cannot provide long-range temporal information; the experiential system fills this gap.

3. **Backpropagation-Free TTA**:

   - **Function**: Efficient online test-time adaptation.
   - **Mechanism**: TCEI operates entirely via forward passes without updating model parameters, avoiding the computational overhead and catastrophic forgetting risk of backpropagation. Adaptation is achieved through cache mechanisms and feature retrieval, following a design philosophy similar to Tip-Adapter.
   - **Design Motivation**: Backpropagation-based TTA methods (e.g., TENT) are inefficient and unstable in MOT scenarios.

### Loss & Training

TCEI requires no additional training and is a purely inference-time method. It can be directly integrated into existing MOT frameworks such as MOTIP.

## Key Experimental Results

### Main Results

| Dataset | Metric | MOTIP Baseline | +TCEI | Gain |
|---------|--------|---------------|-------|------|
| MOT17 (cross-domain) | HOTA↑ | 58.3 | 62.1 | +3.8 |
| MOT17 (cross-domain) | IDF1↑ | 70.2 | 74.6 | +4.4 |
| DanceTrack | HOTA↑ | 54.1 | 57.8 | +3.7 |
| DanceTrack | AssA↑ | 35.2 | 39.5 | +4.3 |

Under distribution shift conditions, TCEI yields significant and consistent improvements over the baseline.

### Ablation Study

| Configuration | HOTA (MOT17) | IDF1 | Note |
|--------------|-------------|------|------|
| MOTIP Baseline | 58.3 | 70.2 | No TTA |
| +Intuitive System (confident cache) | 60.2 | 72.1 | Contribution of temporal priors |
| +Intuitive System (+uncertain cache) | 61.0 | 73.5 | Contribution of reflective mechanism |
| +Experiential System (Full TCEI) | 62.1 | 74.6 | Contribution of long-range calibration |

### Key Findings

- The confident cache contributes the most (+1.9 HOTA), indicating that recent temporal priors serve as the most direct and effective adaptation signal.
- The uncertain cache also yields a positive contribution (+0.8), confirming that the reflective mechanism of "learning to avoid mistakes" is effective.
- The experiential system achieves a larger gain on DanceTrack (+1.5), as long-range ID association in dance scenarios is more challenging.
- TCEI requires no additional training and no backpropagation, resulting in negligible inference overhead.

## Highlights & Insights

- **Elegant Mapping of Dual-System Theory**: The cognitive science framework of intuitive/experiential dual systems is naturally mapped to short-range/long-range adaptation in MOT, yielding a conceptually clear formulation.
- **Reflective Mechanism**: Utilizing uncertain samples as "negative exemplars" to guide the model away from similar errors represents a novel perspective in TTA design.
- **Zero Training Cost**: Adaptation is achieved entirely via forward passes without modifying model parameters, introducing no additional training overhead.

## Limitations & Future Work

- Cache size is a hyperparameter: excessively large caches increase retrieval overhead, while excessively small caches provide insufficient information.
- Long-term accumulation of experience embeddings may introduce outdated information.
- Validation is currently conducted primarily within the detection-based tracking paradigm; adapting to fully end-to-end methods requires further investigation.
- Future work may explore mechanisms for dynamically evicting stale experience.

## Related Work & Insights

- **vs. TENT/FSTTA**: Backpropagation-based TTA methods risk catastrophic forgetting; TCEI avoids this problem entirely.
- **vs. Tip-Adapter/TDA**: TCEI extends cache-based TTA from image classification to video MOT, with the addition of temporal modeling.
- **vs. PURA**: PURA extends TTA to RGB-T tracking but still relies on backpropagation; TCEI is more efficient.

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel application of dual-system theory to TTA for MOT.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-benchmark cross-domain evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear and the framework is described intuitively.
- Value: ⭐⭐⭐⭐ — Provides a zero-cost solution for robust deployment of MOT systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](tcei_dual_level_adaptation_multi_object_tracking.md)
- [\[CVPR 2026\] Occlusion-Aware SORT: Observing Occlusion for Robust Multi-Object Tracking](occlusion-aware_sort_observing_occlusion_for_robust_multi-object_tracking.md)
- [\[CVPR 2026\] FC-Track: Overlap-Aware Post-Association Correction for Online Multi-Object Tracking](fc-track_overlap-aware_post-association_correction_for_online_multi-object_track.md)
- [\[CVPR 2026\] Out of Sight, Out of Track: Adversarial Attacks on Propagation-based Multi-Object Trackers via Query State Manipulation](out_of_sight_out_of_track_adversarial_attacks_on_propagation-based_multi-object_.md)
- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](storm_referring_multi_object_tracking.md)

</div>

<!-- RELATED:END -->
