---
title: >-
  [Paper Note] CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection
description: >-
  [CVPR 2026][Object Detection][Test-time adaptation] This paper proposes the CD-Buffer framework, which drives complementary collaboration between a subtractive buffer (channel suppression) and an additive buffer (lightweight adapter compensation) via a unified domain discrepancy measure, enabling robust test-time object detection adaptation across adverse weather conditions of varying severity.
tags:
  - CVPR 2026
  - Object Detection
  - Test-time adaptation
  - adverse weather
  - channel adaptation
  - additive-subtractive complementarity
date: 2026-05-08
content_hash: d0d295d670873126
---

# CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection

**Conference**: CVPR 2026
**arXiv**: [2603.26092](https://arxiv.org/abs/2603.26092)
**Code**: [Website](https://wkfsksdl99.github.io/cd_buffer/)
**Area**: Object Detection
**Keywords**: Test-time adaptation, adverse weather, object detection, channel adaptation, additive-subtractive complementarity

## TL;DR

This paper proposes the CD-Buffer framework, which drives complementary collaboration between a subtractive buffer (channel suppression) and an additive buffer (lightweight adapter compensation) via a unified domain discrepancy measure, enabling robust test-time object detection adaptation across adverse weather conditions of varying severity.

## Background & Motivation

**State of the Field**: Test-time adaptation (TTA) addresses domain shift by updating source-pretrained models online, without offline retraining or target labels. Existing TTA methods fall into two paradigms: additive methods (introducing lightweight modules to learn target-specific adjustments) and subtractive methods (removing domain-sensitive channels).

**Limitations of Prior Work**: Additive methods (e.g., BufferTTA) perform well under moderate domain shift but struggle to recover under severe degradation; subtractive methods (e.g., PruningTTA) excel under severe shift but over-prune recoverable useful information under moderate shift. Each paradigm is effective only within a limited range of conditions.

**Root Cause**: In real-world scenarios, different feature channels within the same image may experience varying degrees of domain shift — some channels are severely degraded and require suppression, while others need only minor adjustment. Existing methods apply uniform treatment to all channels, failing to accommodate this heterogeneity.

**Paper Goals**: Design an adaptive mechanism that automatically balances "removal" and "compensation" strategies according to the degree of domain shift in each channel.

**Starting Point**: Measure channel-level domain discrepancy and use a unified metric to simultaneously drive two complementary operations.

**Core Idea**: Discrepancy-driven dual-buffer coupling — severely shifted channels are suppressed and receive strong compensation, moderately shifted channels are finely adjusted, and stable channels are largely left unchanged.

## Method

### Overall Architecture

Built upon Faster R-CNN with a ResNet-50 backbone. A subtractive buffer (learnable mask scores) is placed at each BN layer, and an additive buffer (lightweight $1\times1$ + $3\times3$ convolutional adapter) is placed on the residual path. Both buffers are coupled through a unified channel discrepancy measure $D_c$.

### Key Designs

1. **Feature-Level Domain Discrepancy**: Combines image-level and instance-level feature discrepancies:
    $D^I = \frac{\sum_{n=1}^{N}\|X_t^c - \bar{X}_s^c\|_1}{NHW}, \quad D^O = \frac{\sum_{m=1}^{M}\|x_t^c - \bar{x}_s^c\|_1}{Mhw}$
    $D = D^I + D^O$
   where $X_s$ denotes source-domain precomputed mean features and $x_s$ denotes instance-level (via RoI Align) mean features. **Design Motivation**: Object detection requires simultaneous consideration of global scene and local instance domain shift. High $D$ indicates severe channel deviation requiring intervention; low $D$ indicates that only fine-tuning is needed.

2. **Subtractive Buffer**: Introduces learnable mask scores $s \in \mathbb{R}^C$ (initialized from BN weights as $s_c = |\gamma_c|$), driving channel suppression via discrepancy-weighted regularization:
    $\mathcal{L}_{mask} = \frac{1}{C}\sum_c \|D_c \cdot s_c\|_1$
   Channels with high discrepancy have large $D_c$, producing stronger gradients that push $s_c$ down and cause those channels to be suppressed by a thresholded mask. A dynamic percentile threshold $\tau = \text{Percentile}(\{|s_c^{(l)}|\}, \rho_{target})$ controls the overall suppression rate (5%), with a straight-through estimator to ensure gradient flow. Stochastic reactivation prevents the permanent removal of useful channels.

3. **Additive Buffer**: A lightweight adapter $F_{add} = \frac{\text{Conv}_{1\times1}(F) + \text{Conv}_{3\times3}(F)}{2} \odot \boldsymbol{\alpha}$, with a learnable channel scaling factor $\boldsymbol{\alpha}$ (initialized to $10^{-2}$). The key innovation is the **inverse soft mask**:
    $\hat{m}_{soft}^{-1} = k \cdot \text{Norm}(\mathbf{1} - \hat{m}_{soft})$
   Channels strongly suppressed by the subtractive buffer (where $\hat{m}_{soft} \approx 0$) receive the strongest additive compensation (maximum $\hat{m}_{soft}^{-1}$), achieving automatic balance of "compensate as much as is removed." **Design Motivation**: While the subtractive buffer removes severely degraded features, it inevitably discards information; the inverse modulation of the additive buffer automatically provides the strongest compensation for precisely those channels.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{align} + \lambda_{reg} \cdot \mathcal{L}_{mask}$$
- $\mathcal{L}_{align}$: L1 alignment of source and target feature mean and variance
- Layer-wise gradient scaling: additive buffer gradients are amplified according to layer discrepancy $D^l$
- Joint optimization: additive buffer parameters, BN affine parameters, and mask scores

## Key Experimental Results

### Main Results

| Method | KITTI Fog 50m | Fog 75m | Fog 150m | Rain 200mm | Rain 100mm |
|--------|--------------|---------|----------|-----------|------------|
| Direct Test | 21.27 | 30.84 | 50.45 | 47.11 | 65.32 |
| BufferTTA (Additive) | 23.21 | 33.12 | 52.50 | 50.96 | 69.28 |
| PruningTTA (Subtractive) | 33.97 | 42.83 | 58.58 | 50.94 | 65.42 |
| ActMAD | 39.65 | 49.95 | 60.18 | 56.37 | 62.94 |
| **CD-Buffer** | **44.80** | **56.06** | **68.42** | **63.22** | 71.40 |

| Method | ACDC Fog | ACDC Snow | ACDC Rain | ACDC Night |
|--------|---------|-----------|-----------|------------|
| Direct Test | 16.50 | 11.04 | 7.82 | 4.83 |
| BufferTTA | 24.16 | 17.18 | 11.70 | 6.98 |
| **CD-Buffer** | **24.45** | 15.41 | **13.71** | **8.92** |

### Ablation Study

| Configuration | KITTI Fog 50m | Note |
|---------------|--------------|------|
| Additive buffer only | ~23.2 | Limited effectiveness under severe domain shift |
| Subtractive buffer only | ~34.0 | Removes degraded features but loses information |
| Dual buffer without coupling | Significantly below full method | Independent operations lack coordination |
| Full CD-Buffer | 44.80 | Discrepancy-driven coupling is optimal |

### Key Findings

1. **Complementarity Validation**: BufferTTA performs well under moderate shift (Rain 75mm) but poorly under severe shift (Fog 50m); PruningTTA exhibits the opposite pattern. CD-Buffer consistently outperforms across all severity levels.
2. **Continual TTA Stability**: In continual adaptation experiments on KITTI Fog 50m→75m→150m, CD-Buffer adapts fastest and remains the most stable. ActMAD achieves rapid initial improvement but exhibits unstable convergence.
3. The inverse soft mask coupling mechanism is the critical factor behind the performance gains — unifying two paradigms from independent strategies into a coordinated system.

## Highlights & Insights

- This work is the first to systematically reveal the complementary nature of additive and subtractive TTA paradigms, providing an intuitive explanatory framework.
- The discrepancy-driven coupling design is elegant: a single measure $D_c$ simultaneously drives two opposing operations, automatically achieving channel-level differentiated processing.
- The inverse soft mask is a highlight design: compensation intensity is derived directly from the subtractive buffer's mask scores, requiring no additional network.
- Batch-size independence: unlike BN statistics-based methods that are sensitive to small batches, CD-Buffer achieves adaptation through structural modifications.

## Limitations & Future Work

- Experiments are conducted only on Faster R-CNN + ResNet-50; applicability to end-to-end detectors such as DETR-based architectures has not been verified.
- The channel suppression rate is fixed at 5%; adaptive determination could be explored.
- Source-domain feature statistics must be precomputed and stored, increasing deployment complexity.
- Integration with self-supervised objectives (e.g., contrastive learning) has not been explored.

## Related Work & Insights

- Unlike ActMAD's multi-layer feature alignment, CD-Buffer provides finer-grained adaptation through channel-level differentiated processing.
- The inverse mask concept is generalizable to other scenarios requiring a balance between preservation and modification (e.g., knowledge distillation in model compression).
- This work contributes an "additive vs. subtractive" taxonomic framework to the TTA field, which may facilitate understanding of future work.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The additive/subtractive complementarity insight is novel; the discrepancy-driven coupling mechanism is elegantly designed
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-severity evaluation is comprehensive; continual TTA experiments are meaningful
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and method description is complete
- Value: ⭐⭐⭐⭐ Offers a new paradigm integration perspective for TTA

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Test-Time Adaptive Object Detection with Foundation Model](../../NeurIPS2025/object_detection/test-time_adaptive_object_detection_with_foundation_model.md)
- [\[CVPR 2026\] Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection](beyond_prompt_degradation_prototype-guided_dual-pool_prompting_for_incremental_o.md)
- [\[CVPR 2026\] CompAgent: An Agentic Framework for Visual Compliance Verification](compagent_an_agentic_framework_for_visual_compliance_verification.md)
- [\[AAAI 2026\] YOLO-IOD: Towards Real Time Incremental Object Detection](../../AAAI2026/object_detection/yolo-iod_towards_real_time_incremental_object_detection.md)
- [\[ICLR 2026\] CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection](../../ICLR2026/object_detection/cgsa_class-guided_slot-aware_adaptation_for_source-free_object_detection.md)

<!-- RELATED:END -->
