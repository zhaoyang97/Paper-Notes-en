---
title: >-
  [Paper Note] Tracking and Understanding Object Transformations
description: >-
  [NeurIPS 2025][Video Understanding][object tracking] This paper introduces the Track Any State task and the TubeletGraph zero-shot framework, which tracks objects undergoing drastic appearance changes in video (e.g.…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "object tracking"
  - "state change"
  - "zero-shot"
  - "spatiotemporal segmentation"
date: 2026-05-08
content_hash: 777ea96e61a6ebd7
---

# Tracking and Understanding Object Transformations

**Conference**: NeurIPS 2025
**arXiv**: [2511.04678](https://arxiv.org/abs/2511.04678)  
**Code**: [Available](https://tubelet-graph.github.io)  
**Area**: Video Understanding
**Keywords**: object tracking, state change, video understanding, zero-shot, spatiotemporal segmentation

## TL;DR

This paper introduces the Track Any State task and the TubeletGraph zero-shot framework, which tracks objects undergoing drastic appearance changes in video (e.g., an apple being cut, a butterfly emerging from a chrysalis) while simultaneously detecting and describing these transformations.

## Background & Motivation

In the real world, objects frequently undergo state transitions — apples are sliced into pieces, butterflies emerge from chrysalises. Tracking such changes is essential for understanding objects and dynamics, yet existing trackers typically lose their targets once an object undergoes a transformation.

The root cause is that all mainstream object trackers (template matching, optical flow, SAM2, etc.) rely on the assumption of continuous object appearance. When an object undergoes a state change, its appearance may change dramatically (a red apple → white flesh fragments; a chrysalis → an empty shell + butterfly), causing trackers to produce abundant false negatives — the model concludes that the original object has "disappeared."

A key observation is that tracking errors caused by state changes tend to be unidirectional: when an object's appearance changes, the model is biased toward predicting the object as "missing" (false negatives) rather than erroneously tracking other objects (false positives). This asymmetry provides an opportunity to recover lost targets.

The paper poses two core questions:
1. How can a transformed, missing object be located within the exponentially large search space of a video?
2. How can the underlying transformation be modeled and object ambiguity after a state change be resolved?

## Method

### Overall Architecture

TubeletGraph is a zero-shot system consisting of four steps: (1) partitioning the video into a set of spatiotemporal tubelets; (2) reasoning about candidate entities via spatial proximity and semantic consistency constraints; (3) prompting a multimodal LLM to describe the transformation; and (4) constructing a state graph.

### Key Designs

1. **Spatiotemporal Partition**: The first frame is segmented into entities using CropFormer, $\mathcal{E}_1 = \text{CF}(I_1) \cup \{\mathcal{M}_1\}$, and each entity is then propagated forward with SAM2 to form an initial set of tubelets. As time progresses, new tracking sequences are initiated at intermediate frames whenever regions emerge that are not covered by any existing tubelet. This reformulates the continuous problem of "searching every pixel in every frame for a missing object" into the discrete problem of "identifying which tubelet is the truly missing object," substantially reducing the search space.

2. **Spatial Proximity Constraint**: The multiple candidate masks predicted by SAM2 are used to estimate the spatial region where the transformed object is likely to appear. The score is defined as $S_{\text{prox}}(C,P) = \max_{j} |c_s \cap m_s^j| / |c_s|$, where $\{m_s^j\}$ are SAM2's three candidate masks at the candidate's appearance frame, with threshold $\tau_{\text{prox}}=0.3$. The motivation is that a transformed object's location does not change drastically over short time intervals.

3. **Semantic Consistency Constraint**: Masked-pooled CLIP features are used to compute semantic similarity: $S_{\text{sem}}(C,P) = \max_{i,j} f(p_i, I_i) \cdot f(c_j, I_j)^T$, with threshold $\tau_{\text{sem}}=0.7$. The motivation is that an object's identity and semantics are not fundamentally altered by a transformation (a chrysalis becomes a butterfly, not a bird), which also filters out false positives such as hands and tools.

4. **State Graph Construction**: For each new candidate tubelet satisfying both constraints, its appearance is treated as a marker of a state transformation. The system renders contours on the tubelet's initial frame and the first frame, then prompts GPT-4.1 to describe the transformation and object identity, from which a state graph is constructed.

### Loss & Training

TubeletGraph is a fully zero-shot system requiring no training. All components (SAM2.1-L, CropFormer-Hornet-3X, FC-CLIP-COCO, GPT-4.1) use default hyperparameters. Only the thresholds $\tau_{\text{prox}}=0.3$ and $\tau_{\text{sem}}=0.7$ are determined via grid search on the VOST training set.

## Key Experimental Results

### Main Results

| Method | Detect+Describe Changes | VOST $\mathcal{J}$ | VOST $\mathcal{J}_{tr}$ | VSCOS $\mathcal{J}$ | M3-VOS $\mathcal{J}$ | DAVIS17 $\mathcal{J}$ |
|---|---|---|---|---|---|---|
| SAM2.1 | ✗ | 48.4 | 32.4 | 72.0 | 71.3 | 85.7 |
| SAM2.1 (ft) | ✗ | 54.4 | 36.4 | - | - | - |
| DAM4SAM | ✗ | 48.8 | 33.6 | 71.3 | 72.2 | 86.2 |
| **TubeletGraph** | **✓** | **50.9** | **36.7** | **75.9** | **74.1** | 85.6 |

### Ablation Study

| Configuration | VOST $\mathcal{J}$ | Precision $\mathcal{P}$ | Recall $\mathcal{R}$ |
|---|---|---|---|
| SAM2.1 baseline | 48.4 | 71.3 | 54.5 |
| +Spatiotemporal partition (all added) | 25.7 | 18.6 | 71.5 |
| +Semantic | 49.2 | 63.7 | 64.8 |
| +Proximity | 50.7 | 67.7 | 63.8 |
| +Proximity+Semantic | **50.9** | 68.1 | 63.7 |

State graph evaluation (VOST-TAS): temporal localization precision 43.1, recall 20.4; action verb accuracy 81.8, object description accuracy 72.3.

### Key Findings

- SAM2's precision on transformed objects (71.3%) far exceeds its recall (54.5%), validating the "false-negative dominance" observation.
- Spatiotemporal partition alone raises recall to 71.5 (surpassing fine-tuned SAM2 at 65.5), but at the cost of a significant precision drop.
- The two constraints substantially recover precision (+49.5) while minimizing recall loss (−7.8).
- The system is highly robust to the thresholds $\tau_{\text{prox}}$ and $\tau_{\text{sem}}$: sweeping across multiple datasets yields only small variation in $\mathcal{J}$.
- Replacing GPT-4.1 with Qwen-2.5VL causes a dramatic drop in semantic accuracy (action: 81.8→31.8), highlighting the importance of high-quality VLMs for semantic description.

## Highlights & Insights

- **Valuable problem formulation**: Track Any State unifies tracking and state-change understanding into a single task, producing outputs that include both tracking masks and a state graph.
- **Elegant search space reduction**: Spatiotemporal partition transforms a continuous search into a discrete selection, offering an elegant solution to the problem of locating objects after transformation.
- **New benchmark VOST-TAS**: Comprising 57 video instances, 108 transformations, and 293 annotated result objects, this benchmark fills an evaluation gap in the field.

## Limitations & Future Work

- Computational efficiency is a bottleneck: constructing the spatiotemporal partition takes approximately 7 seconds per frame on an A6000 GPU, precluding real-time applications.
- Transformation detection is passive — it is triggered only upon false-negative recovery and cannot detect transformations that do not alter object appearance.
- Temporal localization recall is low (20.4%), leaving substantial room for improvement.
- The modular design may make systematic error attribution difficult.

## Related Work & Insights

The paper reveals an interesting insight: the failure mode of existing trackers under object transformation is structured (false-negative dominant) rather than random, and this structured failure pattern can be systematically exploited. The spatiotemporal partition idea in TubeletGraph can also be applied to multi-object tracking with minimal additional computational cost. The framework has direct applicability in robotic manipulation, particularly for modeling pre- and post-conditions of tasks such as cutting and folding.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (new task definition + new method + new benchmark)
- Experimental Thoroughness: ⭐⭐⭐⭐ (4 tracking datasets + state graph evaluation, with clear ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (fluent exposition with clear problem-driven narrative)
- Value: ⭐⭐⭐⭐ (opens a new research direction, though computational cost limits practical deployment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](../../ICCV2025/video_understanding/general_compression_framework_for_efficient_transformer_object_tracking.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[CVPR 2026\] UETrack: A Unified and Efficient Framework for Single Object Tracking](../../CVPR2026/video_understanding/uetrack_a_unified_and_efficient_framework_for_single_object_tracking.md)
- [\[CVPR 2026\] Event6D: Event-based Novel Object 6D Pose Tracking](../../CVPR2026/video_understanding/event6d_event-based_novel_object_6d_pose_tracking.md)
- [\[CVPR 2026\] STORM: End-to-End Referring Multi-Object Tracking in Videos](../../CVPR2026/video_understanding/storm_referring_multi_object_tracking.md)

</div>

<!-- RELATED:END -->
