---
title: >-
  [Paper Note] Continual Multimodal Contrastive Learning
description: >-
  [NeurIPS 2025][Multimodal VLM][Continual Learning] This paper is the first to formally define the Continual Multimodal Contrastive Learning (CMCL) problem and proposes Dual-side Null-space gradient projection (DNS), which projects gradients from new data into subspaces that do not interfere with previously acquired knowledge. DNS achieves the best stability–plasticity trade-off across 7 datasets.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Continual Learning
  - Multimodal Contrastive Learning
  - Gradient Projection
  - Catastrophic Forgetting
  - Modality Binding
date: 2026-05-08
content_hash: 76d18f33eb11619a
---

# Continual Multimodal Contrastive Learning

**Conference**: NeurIPS 2025
**arXiv**: [2503.14963](https://arxiv.org/abs/2503.14963)
**Code**: [https://github.com/Xiaohao-Liu/CMCL](https://github.com/Xiaohao-Liu/CMCL)
**Area**: Multimodal VLM
**Keywords**: Continual Learning, Multimodal Contrastive Learning, Gradient Projection, Catastrophic Forgetting, Modality Binding

## TL;DR

This paper is the first to formally define the Continual Multimodal Contrastive Learning (CMCL) problem and proposes Dual-side Null-space gradient projection (DNS), which projects gradients from new data into subspaces that do not interfere with previously acquired knowledge. DNS achieves the best stability–plasticity trade-off across 7 datasets.

## Background & Motivation

**State of the Field**: Multimodal contrastive learning (MCL) aligns different modalities (vision, audio, text, etc.) into a unified representation space via contrastive objectives. Representative methods such as CLIP, ImageBind, and LanguageBind have demonstrated strong cross-modal representation capabilities.

**Limitations of Prior Work**: Existing MCL methods typically assume all modality data can be collected at once and trained jointly. In practice, multimodal data often arrives in batches—new modality-pair data continually emerges—making retraining from scratch prohibitively expensive. Continual fine-tuning on existing models, however, leads to catastrophic forgetting that disrupts previously learned cross-modal alignment.

**Root Cause**: Conventional continual learning methods (e.g., EWC, GEM, DER++) are designed for class-incremental or task-incremental settings and cannot handle the unique cross-modal complexity of MCL, where the contrastive objective remains consistent while the involved modality pairs continuously change. These methods suffer from a severe stability–plasticity trade-off in CMCL scenarios.

**Paper Goals**: (1) Formally define continual multimodal contrastive learning with rigorous mathematical definitions of stability and plasticity; (2) Design a method that preserves previously learned cross-modal alignment while effectively acquiring new modality pairs.

**Starting Point**: From a gradient update perspective, model parameter updates are essentially modifications to a global parameter matrix. If gradients can be projected into subspaces that do not affect old data representations, both objectives can be simultaneously satisfied. This draws inspiration from null-space projection in single-modal continual learning, but the multimodal setting requires simultaneous projection from two modality sides.

**Core Idea**: Project gradients from both modality sides simultaneously into the null space of old data features, ensuring that parameter updates do not interfere with previously learned cross-modal alignment.

## Method

### Overall Architecture

Given a pre-trained modality-binding model (e.g., ImageBind), trainable linear projection layers are appended on top. When new modality-pair data arrives, contrastive learning optimizes the projection layer parameters. The core of DNS is: at each training step, gradients are projected from both "sides" into specific subspaces, such that the projected gradient updates do not affect previously learned cross-modal alignment scores.

### Key Designs

1. **CMCL Problem Formalization and Dual-Objective Definition**:

    - Function: Provides a rigorous mathematical framework for continual multimodal contrastive learning.
    - Mechanism: Stability is defined as the alignment score $\mathbf{A}_{t-1;t}^{m_1,m_2}$ computed on old data with the current model remaining consistent with that of the old model $\mathbf{A}_{t-1;t-1}^{m_1,m_2}$; plasticity requires the model to learn effectively from new modality pairs. The authors prove that if the global parameter update is projected as $\bar{\mathbf{W}} = \tilde{\mathbf{W}} - \mathbf{P}'\tilde{\mathbf{W}}\mathbf{P}$, where $\mathbf{P}'$ and $\mathbf{P}$ are spatial projectors of old data features, the stability condition is automatically satisfied.
    - Design Motivation: The CMCL objective (contrastive loss) remains fixed while the modality pairs change—this necessitates dedicated stability/plasticity definitions distinct from those in classification-based continual learning.

2. **Dual-side Null-space Gradient Projection (DNS)**:

    - Function: Decomposes the global stability condition into locally actionable gradient projections.
    - Mechanism: The global parameter update is expanded into three terms associated with the gradients of the two modalities. Theorem 4 proves that the gradient of each modality can be projected independently—the gradient for modality $m_1$ is projected as $\Delta\mathbf{W}_t^{m_1} = \nabla\mathbf{W}_t^{m_1} - \tilde{\mathbf{P}}\nabla\mathbf{W}_t^{m_1}\mathbf{P}'$, where $\tilde{\mathbf{P}}$ and $\mathbf{P}'$ are constructed from the features of the corresponding old modalities. Replacing original gradients with projected ones ensures parameter updates do not disturb previously learned cross-modal alignment.
    - Design Motivation: Direct projection on global parameters is intractable; decomposing the projection to per-modality gradients substantially reduces implementation complexity.

3. **Multi-step and Multi-modality-pair Extension**:

    - Function: Generalizes the two-step two-modality theory to arbitrary steps and modality pairs.
    - Mechanism: An incremental centered feature covariance matrix $\bar{\mathbf{Z}}_{<t}^m$ accumulates feature information from all historical steps; the projector is updated via SVD at the end of each step. For different modality pairs, gradients of modalities not involved in training are simply set to zero.
    - Design Motivation: Real-world training involves multiple steps and modality pairs, requiring an efficient mechanism to maintain historical information without storing old data; the recursive covariance update enables replay-free continual learning.

### Loss & Training

Standard CLIP-style InfoNCE contrastive loss is used with the AdamW optimizer (lr=0.0001, weight decay=0.001) and batch size=64. Null-space projection is approximated via truncated SVD, with a minimum eigenvalue threshold $\lambda_{\min}$ of 0.01 (ImageBind/UniBind) or 0.0001 (LanguageBind).

## Key Experimental Results

### Main Results

Evaluated on 7 datasets (UCF101, ESC50, NYUDv2, VGGSound-S, Clotho, TVL, LLVIP) across 11 training steps, 7 modalities, and 3 backbone models.

| Backbone | Method | Cls. Acc | BWT_A | R@10 | BWT_R10 |
|----------|--------|---------|-------|------|---------|
| ImageBind | Vanilla | 47.32 | -5.72 | 38.56 | -3.34 |
| ImageBind | Co2L | 50.13 | -3.74 | 38.66 | -1.77 |
| ImageBind | **DNS** | **52.52** | **-0.02** | **40.89** | **-1.07** |
| LanguageBind | Vanilla | 51.86 | -15.71 | 36.02 | -10.19 |
| LanguageBind | CILA | 59.30 | -2.63 | 40.48 | -1.09 |
| LanguageBind | **DNS** | **64.07** | **-0.09** | **42.44** | -3.00 |
| UniBind | C-FLAT | 51.25 | -4.36 | 40.51 | -1.48 |
| UniBind | **DNS** | **52.86** | **+0.31** | **41.44** | **-1.19** |

### Ablation Study

| Analysis Dimension | Result | Notes |
|-------------------|--------|-------|
| Stability deviation | Per-step alignment score deviation near 0 | Empirically validates the theoretical bound of Theorem 5 |
| Plasticity higher-order term | $o(\eta)/\eta < 0$ | Satisfies the plasticity condition of Theorem 6 |
| Training loss | Converges normally at each step | Plasticity is preserved under projection constraints |
| Training time overhead | DNS adds <1s | More efficient than replay-based methods |

### Key Findings

- DNS achieves classification BWT near zero or even positive (+0.31 on UniBind), indicating that learning new knowledge can even benefit old tasks—an extremely rare outcome in continual learning.
- Forgetting is most severe in the LanguageBind setting (Vanilla BWT = -15.71); DNS reduces this to -0.09, an improvement of approximately 170×.
- DNS is a replay-free method requiring no storage of old data, with negligible additional training time of under one second.

## Highlights & Insights

- **Elegance of Dual-side Projection**: Unlike single-modal continual learning, which requires only one-sided projection, CMCL must simultaneously account for the feature spaces of two modalities. The authors cleverly decompose the global stability condition into local per-modality gradient projections, yielding a theoretically principled and practically simple solution.
- **Efficiency of the Replay-free Design**: Compared to methods requiring a replay buffer, DNS maintains only a recursively updated feature covariance matrix, with minimal storage and computational overhead.
- **Flexible Extension to Diverse Modality Pairs**: When a new step involves a different modality pair, gradients of non-participating modalities are simply set to zero without additional handling—highly practical in real-world multimodal scenarios.

## Limitations & Future Work

- Experiments evaluate linear layers appended to pre-trained models; directly fine-tuning entire encoders is not explored, which may limit applicability in large-scale training.
- The SVD truncation threshold requires separate tuning for different backbones (0.01 vs. 0.0001), lacking an adaptive strategy.
- Only pairwise modality alignment is considered; simultaneous training on three or more modalities is not addressed.
- Theoretical analysis assumes linear projection layers; generalization to nonlinear encoders requires further investigation.

## Related Work & Insights

- **vs. Adam-NSCL**: Also employs null-space projection but only for single-modal continual learning and cannot handle the cross-modal complexity of multimodal alignment. DNS's dual-side projection is a natural extension of null-space methods to multimodal settings.
- **vs. Co2L**: Uses contrastive distillation to prevent forgetting but requires old data replay; BWT remains -5.79 in the LanguageBind setting. DNS requires no replay and achieves BWT near zero.
- **vs. EWC**: Prevents forgetting via parameter importance weights, implicitly assuming parameter independence; performance is limited under cross-modal interactions in multimodal settings.

## Rating

- Novelty: ⭐⭐⭐⭐ First to formalize the CMCL problem and propose a dedicated method; dual-side projection offers theoretical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Seven datasets, three backbones, multiple baselines, and detailed stability/plasticity analysis.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear with a complete notation system, though readability is somewhat impacted by the density of mathematical symbols.
- Value: ⭐⭐⭐⭐ Fills an important gap at the intersection of multimodal contrastive learning and continual learning.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](bioclip_2_emergent_properties_from_scaling_hierarchical_contrastive_learning.md)
- [\[ICLR 2026\] KeepLoRA: Continual Learning with Residual Gradient Adaptation](../../ICLR2026/multimodal_vlm/keeplora_continual_learning_with_residual_gradient_adaptation.md)
- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](../../ICLR2026/multimodal_vlm/enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[NeurIPS 2025\] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning](on_the_value_of_cross-modal_misalignment_in_multimodal_representation_learning.md)
- [\[NeurIPS 2025\] Multimodal Negative Learning](multimodal_negative_learning.md)

<!-- RELATED:END -->
