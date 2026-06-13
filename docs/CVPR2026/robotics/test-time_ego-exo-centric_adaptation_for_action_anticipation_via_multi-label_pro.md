---
title: >-
  [Paper Note] Test-time Ego-Exo-centric Adaptation for Action Anticipation via Multi-Label Prototype Growing and Dual-Clue Consistency
description: >-
  [CVPR2026][Robotics][test-time adaptation] This paper introduces the Test-time Ego-Exo Adaptation for Action Anticipation (TE2A3) task and proposes the DCPGN network…
tags:
  - "CVPR2026"
  - "Robotics"
  - "test-time adaptation"
  - "Ego-Exo"
  - "Action Anticipation"
  - "Multi-Label"
  - "Prototype Learning"
  - "CLIP"
  - "dual-clue consistency"
date: 2026-05-08
content_hash: 7228d488c0552858
---

# Test-time Ego-Exo-centric Adaptation for Action Anticipation via Multi-Label Prototype Growing and Dual-Clue Consistency

**Conference**: CVPR2026  
**arXiv**: [2603.09798](https://arxiv.org/abs/2603.09798)  
**Code**: [ZhaofengSHI/DCPGN](https://github.com/ZhaofengSHI/DCPGN)  
**Area**: Robotics / Ego-Exo View Adaptation  
**Keywords**: test-time adaptation, Ego-Exo, Action Anticipation, Multi-Label, Prototype Learning, CLIP, dual-clue consistency

## TL;DR

This paper introduces the Test-time Ego-Exo Adaptation for Action Anticipation (TE2A3) task and proposes the DCPGN network, which leverages multi-label prototype growing and dual-clue (visual + textual) consistency to online-adapt a source-view trained model to the target view at test time for action anticipation, substantially outperforming existing TTA methods.

## Background & Motivation

1. **Human cross-view capability**: Humans can seamlessly switch between egocentric (Ego) and exocentric (Exo) perspectives via mirror neurons to anticipate future actions—a capability critical for human-robot collaboration and embodied AI.
2. **Existing methods require target-view data**: Most Ego-Exo adaptation approaches (pretrain-finetune / UDA) access target-view data during training, incurring additional computational and data collection costs.
3. **Single-view model degradation across views**: Models trained on one viewpoint suffer significant performance drops when directly applied to the other, due to large differences in camera angle and visual style.
4. **Opportunity for test-time adaptation (TTA)**: TTA methods can online-adjust models without labeled target-view data, yet existing TTA methods are designed for single-label tasks and struggle with multi-action candidate scenarios.
5. **Multi-action candidate challenge**: Real-world events often involve multiple concurrent atomic actions, whereas entropy-based TTA methods tend to focus on the single highest-confidence class, leading to suboptimal performance.
6. **Ego-Exo spatiotemporal gap**: The two viewpoints differ substantially in spatial layout (inconsistent scenes and distractors) and temporal dynamics (asynchronous action progression), making naive domain adaptation insufficient.

## Method

### Overall Architecture

DCPGN (Dual-Clue enhanced Prototype Growing Network) comprises two core modules:

- **ML-PGM** (Multi-Label Prototype Growing Module): Progressively accumulates multi-label knowledge to learn unbiased prototypes.
- **DCCM** (Dual-Clue Consistency Module): Integrates visual and textual clues to enforce dual-clue consistency.

During training, a BCE loss is used on labeled source-view data. At test time, the visual encoder (CLIP ViT-L/14) is frozen, while learnable prompts and prototypes are updated online.

### ML-PGM

1. **Multi-label pseudo-label assignment**: Top-K predictions from the logits of each test sample are selected as pseudo-labels (K=3 on EgoMe-anti; K=5 on EgoExoLearn), avoiding overconfidence induced by single-label strategies.
2. **Entropy-priority queue strategy**: A memory bank of capacity N=500 is maintained per class. When the bank is full, only the N samples with the lowest entropy (least uncertainty) are retained, ensuring reliability increases over time.
3. **Confidence-weighted prototype computation**: Class prototypes are computed as confidence-weighted sums over stored representations: $p_i^T = \sum_{k=1}^{N'} \eta(l_{i,k}^T) \cdot \bar{f}_{v,k}^T$, suppressing noise from negative-class samples.
4. **Prototype-based classification**: Similarities between test sample representations and all class prototypes yield prototype logits $L_p$.

### DCCM

1. **Visual clue**: The last frame of the observed video is extracted as a visual clue, carrying object-level spatial information about the scene.
2. **Textual clue**: A lightweight narrator (GRU + attention) generates descriptive text from frame features, serving as a temporal indicator of action progression. The narrator is trained on open-source video-text pairs and frozen at test time.
3. **CLIP inference**: Frozen CLIP visual and text encoders extract features from visual and textual clues respectively; similarities with learnable-prompt action class features yield visual logits $L_v$ and textual logits $L_t$.
4. **Dual-clue consistency loss**: A symmetric KL divergence is applied between the softmax distributions of $L_v$ and $L_t$: $L_C = KL(P_v \| P_t) + KL(P_t \| P_v)$, constraining cross-modal clue consistency and explicitly bridging the Ego-Exo spatiotemporal gap.

### Final Prediction and Loss

$$L_{final} = L_p + \alpha \cdot (L_v + L_t), \quad \alpha = 0.5$$

At test time, learnable prompts are optimized online via SGD without any data augmentation.

## Key Experimental Results

### Main Results (class-mean Top-5 recall)

| Method | EgoMe-anti Exo2Ego Noun/Verb | EgoMe-anti Ego2Exo Noun/Verb | EgoExoLearn Exo2Ego Noun/Verb | EgoExoLearn Ego2Exo Noun/Verb |
|---|---|---|---|---|
| No Adaptation | 71.94 / 32.46 | 64.24 / 30.07 | 31.91 / 34.36 | 35.28 / 33.03 |
| ML-TTA | 77.11 / 36.92 | 69.46 / 34.39 | 36.35 / 37.67 | 42.96 / 40.43 |
| **DCPGN (Ours)** | **79.03 / 43.84** | **72.01 / 40.10** | **46.26 / 42.98** | **48.48 / 46.51** |

On EgoExoLearn Exo2Ego, DCPGN outperforms ML-TTA by **9.91%** on Noun and **5.31%** on Verb.

### Ablation Study

| Configuration | EgoMe-anti E2E Noun | EgoExoLearn E2E Noun |
|---|---|---|
| Full DCPGN | 79.03 | 46.26 |
| w/o consistency loss $L_C$ | 78.67 (−0.36) | 44.80 (−1.46) |
| w/o visual clue | 76.92 (−2.11) | 41.32 (−4.94) |
| w/o textual clue | 77.56 (−1.47) | 41.94 (−4.32) |
| w/o entire DCCM | 76.11 (−2.92) | 38.43 (−7.83) |
| w/o confidence weighting | 74.63 (−4.40) | 37.76 (−8.50) |
| single-label assignment only | 72.74 (−6.29) | 34.70 (−11.56) |

Key findings: visual clues contribute more to Noun prediction, while textual clues are more critical for Verb prediction; multi-label assignment yields substantial gains over single-label assignment.

### Model Complexity

| Component | FLOPs (G) | Params (MB) |
|---|---|---|
| Baseline | 367.55 | 251.18 |
| ML-PGM | +0.00 | +8.54 |
| Narrator | +0.03 | +2.38 |
| Textual clue encoding | +4.06 | +54.04 |

ML-PGM introduces virtually zero additional computation, and the narrator is extremely lightweight.

## Highlights & Insights

1. **First TE2A3 task**: This work is the first to introduce TTA into cross-view Ego-Exo action anticipation, requiring no target-view training data.
2. **Multi-label prototype growing mechanism**: Top-K pseudo-label assignment combined with entropy-priority queues and confidence weighting effectively addresses the class-balance problem in multi-action candidate scenarios.
3. **Complementary dual-clue design**: Visual clues capture spatial object information while textual clues capture temporal action progression; KL divergence consistency explicitly bridges the Ego-Exo spatiotemporal gap.
4. **New benchmark EgoMe-anti**: A new benchmark suitable for this task is constructed based on the EgoMe dataset.
5. **Significant performance gains**: DCPGN surpasses the second-best method by 9.91% on Noun in EgoExoLearn, with thorough experiments and in-depth analysis.

## Limitations & Future Work

1. **Narrator requires additional training data**: The narrator must be pretrained on open-source video-text pairs, introducing a prerequisite dependency.
2. **Manual tuning of K**: The optimal K differs across datasets (3 vs. 5), lacking an adaptive selection mechanism.
3. **Fixed memory bank capacity**: N=500 is manually set and may be suboptimal when class-wise data distributions vary widely.
4. **Only Noun/Verb classification evaluated**: Finer-grained temporal localization or full event prediction is not addressed.
5. **Lack of real-time analysis**: Despite claiming online adaptation, inference latency and practical deployment feasibility are not reported.

## Related Work & Insights

- **vs. Tent/TPT/TDA and other traditional TTA**: These methods are designed for single-label tasks and over-focus on high-confidence classes in multi-action candidate scenarios; DCPGN's multi-label mechanism addresses this fundamental limitation.
- **vs. ML-TTA**: Although ML-TTA targets multi-label settings, it is designed for image-level classification and lacks video-level spatiotemporal modeling and Ego-Exo viewpoint gap handling.
- **vs. UDA methods (Sync, GCEAN)**: UDA methods require access to unlabeled target-view data during training, whereas DCPGN adapts entirely online at test time.
- **vs. pretrain-finetune methods (AE2, Exo2EgoDVC)**: These methods require labeled target-view data for finetuning, which DCPGN does not.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to introduce the TE2A3 task; the combination of multi-label prototype growing and dual-clue consistency is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Two benchmarks, four settings, comprehensive ablations, and visualization analyses are all well covered.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, method description is rigorous, and figures and tables are well presented.
- Value: ⭐⭐⭐⭐ — Provides a practical paradigm for cross-view online adaptation in human-robot collaboration and embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments](../../ICLR2026/robotics/test-time_mixture_of_world_models_for_embodied_agents_in_dynamic_environments.md)
- [\[ICCV 2025\] PacGDC: Label-Efficient Generalizable Depth Completion with Projection Ambiguity and Consistency](../../ICCV2025/robotics/pacgdc_label-efficient_generalizable_depth_completion_with_projection_ambiguity_.md)
- [\[AAAI 2026\] Object-Centric Latent Action Learning](../../AAAI2026/robotics/object-centric_latent_action_learning.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[ICLR 2026\] All-day Multi-scenes Lifelong Vision-and-Language Navigation with Tucker Adaptation](../../ICLR2026/robotics/all-day_multi-scenes_lifelong_vision-and-language_navigation_with_tucker_adaptat.md)

</div>

<!-- RELATED:END -->
