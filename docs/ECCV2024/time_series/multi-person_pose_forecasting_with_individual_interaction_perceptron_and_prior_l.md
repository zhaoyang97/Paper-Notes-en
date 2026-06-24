---
title: >-
  [Paper Note] Multi-person Pose Forecasting with Individual Interaction Perceptron and Prior Learning
description: >-
  [ECCV 2024][Time Series][Multi-person pose forecasting] This paper proposes IAFormer (Interaction-Aware Pose Forecasting Transformer). By designing the Interaction Perceptron Module (IPM) to evaluate the level of individual interaction with events, and introducing the Interaction Prior Learning Module (IPLM) to accumulate prior knowledge of high-frequency interaction patterns, it achieves semantic-level multi-person pose forecasting, significantly outperforming existing metho…
tags:
  - "ECCV 2024"
  - "Time Series"
  - "Multi-person pose forecasting"
  - "interaction-aware"
  - "Transformer"
  - "prior learning"
  - "event role modeling"
date: 2026-05-08
content_hash: f9a8871323f397a7
---

# Multi-person Pose Forecasting with Individual Interaction Perceptron and Prior Learning

**Conference**: ECCV 2024  
**Authors**: Peng Xiao, Yi Xie, Xuemiao Xu, Weihong Chen, Huaidong Zhang  
**Code**: [https://github.com/ArcticPole/IAFormer](https://github.com/ArcticPole/IAFormer)  
**Area**: Multi-person Pose Forecasting / Time-Series Forecasting  
**Keywords**: Multi-person pose forecasting, interaction-aware, Transformer, prior learning, event role modeling

## TL;DR

This paper proposes IAFormer (Interaction-Aware Pose Forecasting Transformer). By designing the Interaction Perceptron Module (IPM) to evaluate the level of individual interaction with events, and introducing the Interaction Prior Learning Module (IPLM) to accumulate prior knowledge of high-frequency interaction patterns, it achieves semantic-level multi-person pose forecasting, significantly outperforming existing methods on multiple multi-person scene datasets.

## Background & Motivation

**Background**: Human pose forecasting aims to predict future poses based on historical pose sequences, which is a crucial problem for understanding human intent. Current methods mainly learn single-person or multi-person temporal motion patterns through deep learning models for prediction.

**Limitations of Prior Work**: Existing multi-person pose forecasting methods often ignore the differences in individual roles in events. In multi-person scenes, people participate in events to different degrees—some are core participants, while others are mere onlookers. Existing methods model interaction relationships by treating everyone equally, which limits performance in complex scenes where multiple interactions occur simultaneously. Furthermore, many methods tend to learn trajectory-level motion patterns (trajectory pose forecasting) rather than semantic-level action prediction (semantic pose forecasting).

**Key Challenge**: Interactions in multi-person scenarios are selective—events usually involve only some of the people in the scene. Without distinguishing the interaction strength between individuals and events, the model can be disturbed by the motion patterns of irrelevant individuals, failing to focus on truly meaningful interaction relationships.

**Goal**: (1) How to quantify each individual's level of interaction with the ongoing event? (2) How to leverage interaction level information to guide feature extraction and pose forecasting? (3) How to equip the model with prior knowledge of high-frequency interaction patterns to make more semantic predictions?

**Key Insight**: The authors put forward a key observation: "events usually involve only some of the people in the scene." Guided by this, they design an interaction perceptron module to assign a "participation score" to each person. Meanwhile, they use a prior learning module to accumulate common interaction patterns (such as handshaking, talking, passing a ball), enabling the prediction with semantic understanding.

**Core Idea**: By evaluating each individual's level of event participation through an interaction perceptron module, and combining it with the accumulated knowledge of high-frequency interaction patterns from prior learning, the model achieves semantic-aware multi-person pose forecasting.

## Method

### Overall Architecture

IAFormer takes the historical pose sequences of multiple people as input and outputs the future pose sequence of each individual. The overall workflow consists of: (1) First, evaluating each person's interaction level with the event through the Interaction Perceptron Module (IPM); (2) Based on the interaction evaluation results, extracting interaction-aware human features through attention mechanisms (distinguishing between high-interaction and low-interaction individuals); (3) Injecting prior knowledge of high-frequency interactions using the Interaction Prior Learning Module (IPLM); (4) Forecasting future poses of each person based on the fused features.

### Key Designs

1. **Interaction Perceptron Module (IPM)**:

    - **Function**: Computes an interaction level score for each individual in the scene to quantify their relevance to the current event.
    - **Mechanism**: IPM receives the pose features of all individuals and infers who the core participants in the event are by learning the relative motion patterns and spatial relationships between people. Specifically, it compares the similarity between each individual's motion features and the overall event features to produce a scalar score. Individuals with higher scores are considered core participants, and their features receive greater weights in subsequent attention calculations.
    - **Design Motivation**: In real-world multi-person scenarios, some people just pass by or stand in the background, contributing minimally to forecasting the pose changes of the core characters in the event. IPM allows the model to automatically filter out interference from irrelevant individuals and focus on meaningful interactions.

2. **Interaction-Aware Attention**:

    - **Function**: Adaptively extracts interaction-aware features for each individual based on the score results from IPM.
    - **Mechanism**: On top of the standard Transformer's attention mechanism, the interaction scores produced by IPM are used as modulation factors for attention weights. For high-scoring interaction pairs, the attention weights are amplified; for low-scoring pairs, the weights are suppressed. This enables the feature extraction process to distinguish between "interaction-relevant" and "interaction-irrelevant" information.
    - **Design Motivation**: Standard attention does not differentiate interaction intensity and treats relationships between all people equally, which introduces significant noise in complex multi-person scenes. Interaction-aware attention lets the model "know who to look at."

3. **Interaction Prior Learning Module (IPLM)**:

    - **Function**: Learns and accumulates a prior knowledge base of high-frequency interaction patterns, enabling predictions with semantic understanding.
    - **Mechanism**: IPLM maintains a learnable prior memory that stores prototype features of common interaction patterns (such as motion patterns like "two people talking face-to-face" or "one person passing an item to another"). During prediction, current scene interaction features are matched with prototypes in the prior memory, and matched priors are fused into the prediction features. This allows the model not only to perform trajectory-level extrapolation but also to understand "what they are doing" and make semantically reasonable predictions based on that.
    - **Design Motivation**: Purely data-driven pose forecasting easily falls into simple motion extrapolation, failing to handle sudden changes in motion patterns (such as suddenly turning or starting a new action). Prior learning provides the model with "common sense", allowing it to produce semantically reasonable predictions when trajectory cues are unclear.

### Loss & Training

Training mainly utilizes the L2 pose reconstruction loss as the primary loss to measure the distance between the predicted poses and the ground-truth future poses. Meanwhile, auxiliary supervision is applied to the interaction scores of IPM (e.g., based on annotations of whether there is physical contact or co-participation in the same action) to ensure the semantic correctness of the interaction evaluation. The prior memory of IPLM automatically extracts high-frequency interaction patterns from raw data through end-to-end learning.

## Key Experimental Results

### Main Results

| Dataset | Metrics (MPJPE mm) | Ours | Prev. SOTA | Gain |
|--------|-----------------|----------|---------|------|
| CMU-Mocap (2 persons) | MPJPE@1000ms | Significantly leads | - | - |
| UMPM (multi-person) | MPJPE@1000ms | Significantly leads | - | - |
| CHI3D (2-person interaction) | MPJPE@1000ms | Significantly leads | - | - |
| Human3.6M (single-person) | MPJPE@1000ms | Competitive | - | - |
| Synthetic crowd data | MPJPE@1000ms | Significantly leads | - | Clear multi-person advantage |

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| Full IAFormer | Optimal | Full model |
| w/o IPM | Obvious performance drop | Unable to distinguish interaction levels, disturbed by irrelevant individuals |
| w/o IPLM | Moderate drop | Lacks prior knowledge, reduced semantic prediction capability |
| w/o Interaction-Aware Attention | Large drop | Degenerates to standard attention, multi-person advantage disappears |
| IPM + Standard Attention | Some improvement but insufficient | Limited effect with only scoring but no fusion mechanism |

### Key Findings
- The IPM module contributes the most, especially in scenarios with more people (more than 5), indicating that interaction level evaluation is core.
- IPLM excels most on datasets with rich interaction patterns (CHI3D), showing that prior knowledge holds significant value in complex interactions.
- In single-person scenarios (Human3.6M), IAFormer remains competitive but its advantage is no longer pronounced, which is expected—the interaction modeling module has no utility in single-person settings.
- As the number of people in the scene increases, the advantage of IAFormer over the baseline is progressively amplified.

## Highlights & Insights
- **The observation that "events usually involve only some of the people"** is highly insightful. This simple yet overlooked observation directly guides the design of IPM, and the idea can be transferred to tasks like multi-person activity recognition and group behavior analysis.
- The design idea of the **Prior Learning Module**—maintaining a learnable database of interaction pattern prototypes—is an excellent way to introduce domain knowledge. This can be transferred to other motion prediction tasks that require "common sense reasoning" (e.g., vehicle interaction prediction, social robot navigation).
- Modeling interaction levels as continuous values rather than binary classification allows the model to handle progressive interaction participation, which is more flexible than hard grouping.

## Limitations & Future Work
- The capacity of the prior learning module is fixed, which might suffer from insufficient coverage of rare interaction patterns; dynamic expansion of the prior memory could be considered.
- The paper mainly validates on laboratory datasets; the effectiveness in large-scale scenarios like real city streetscapes remains to be verified.
- The interaction evaluation of IPM is primarily based on motion features and does not incorporate scene context (such as object detection, scene semantic segmentation), limiting the understanding of scene causal relationships.
- Long-term forecasting (>2 seconds) scenarios are not considered; in long-term forecasting, interaction roles might change dynamically.

## Related Work & Insights
- **vs MRT (Motion Representation Transformer)**: MRT models global relationships between all people without distinguishing interaction intensity. IAFormer introduces selective attention through IPM, which performs better in multi-person scenarios.
- **vs SoMoFormer**: SoMoFormer uses social force models to model interpersonal interactions but lacks semantic-level interaction understanding. IAFormer's IPLM supplements semantic priors.
- **vs TBIFormer**: TBIFormer considers body-part-level interactions, which is complementary to the person-level interaction evaluation of IAFormer. Combining the two might further improve performance.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of interaction level quantification and prior learning is novel in the field of pose forecasting.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple datasets and scenarios with different numbers of people, with a systematic ablation study.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, smooth method presentation, and well-conveyed key insights.
- Value: ⭐⭐⭐⭐ Proposes an effective interaction modeling framework for multi-person pose forecasting, and the open-source code increases practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Structure Learning from Time-Series Data with Lag-Agnostic Structural Prior](../../ICLR2026/time_series/structure_learning_from_time-series_data_with_lag-agnostic_structural_prior.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[ICML 2026\] Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting](../../ICML2026/time_series/parametric_prior_mapping_framework_for_non-stationary_probabilistic_time_series_.md)
- [\[ICLR 2026\] CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting](../../ICLR2026/time_series/cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time_se.md)
- [\[AAAI 2026\] M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting](../../AAAI2026/time_series/m2fmoe_multi-resolution_multi-view_frequency_mixture-of-experts_for_extreme-adap.md)

</div>

<!-- RELATED:END -->
