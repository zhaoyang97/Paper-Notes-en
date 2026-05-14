---
title: >-
  [Paper Note] Predicting Video Slot Attention Queries from Random Slot-Feature Pairs
description: >-
  [AAAI 2026][Video Understanding][Object-centric learning] This paper proposes RandSF.Q, which significantly improves query prediction quality in video object-centric learning (OCL) by leveraging next-frame features for i…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "Object-centric learning"
  - "video object discovery"
  - "Slot Attention"
  - "temporal modeling"
  - "self-supervised learning"
date: 2026-05-08
content_hash: 7d482f51a3bcefd3
---

# Predicting Video Slot Attention Queries from Random Slot-Feature Pairs

**Conference**: AAAI 2026
**arXiv**: [2508.01345](https://arxiv.org/abs/2508.01345)
**Code**: [https://github.com/Genera1Z/RandSF.Q](https://github.com/Genera1Z/RandSF.Q)
**Area**: Video Understanding
**Keywords**: Object-centric learning, video object discovery, Slot Attention, temporal modeling, self-supervised learning

## TL;DR

This paper proposes RandSF.Q, which significantly improves query prediction quality in video object-centric learning (OCL) by leveraging next-frame features for informative query prediction and learning transition dynamics from randomly sampled slot-feature pairs. The method surpasses state-of-the-art approaches by up to 10 points on object discovery benchmarks.

## Background & Motivation

### State of the Field
Video OCL aims to discover objects from video in a self-supervised manner, representing each object as a feature vector (slot) and tracking these objects across frames. Dominant approaches adopt a recurrent architecture: an aggregator (Slot Attention) aggregates the current frame into slots → a transitioner converts current slots into queries for the next frame → the aggregator processes the next frame using these queries.

### Core Problem — Two Overlooked Issues

**Issue (i1): Next-frame features are not utilized.**
All existing transitioners predict next-frame queries based solely on current (or historical) slots, despite the fact that next-frame features are already available and considerably more informative. This is analogous to forecasting tomorrow's weather from today's records alone while ignoring tomorrow's satellite imagery — a clearly superior information source that is nonetheless left unused.

**Issue (i2): Transition dynamics are not learned.**
Existing transitioners lack appropriate inductive biases for learning genuine transition dynamics. The authors conduct a striking diagnostic experiment: **removing the transitioner entirely and using current slots directly as next-frame queries actually yields better performance**. This demonstrates that existing transitioners are not merely ineffective but actively harmful.

### Starting Point
The paper proposes a new transitioner architecture (a Transformer decoder rather than an encoder) that conditions query prediction on both current slots and next-frame features, and introduces a training strategy based on randomly sampled slot-feature pairs to compel the transitioner to learn true transition dynamics.

## Method

### Overall Architecture

Built upon SlotContrast (current SOTA), the framework comprises four components:

1. **Encoder $\phi_e$** (frozen DINO2 ViT): encodes video frames into features $F_t \in \mathbb{R}^{h \times w \times c}$
2. **Aggregator $\phi_a$** (Slot Attention): aggregates features into slots $S_t$ and segmentation masks $M_t$
3. **Transitioner $\phi_r$** (**newly designed** Transformer decoder block): transforms $S_t$ and $F_{t+1}$ into next-frame queries $Q_{t+1}$
4. **Decoder $\phi_d$** (autoregressive Transformer decoder with random ordering): reconstructs features $F_t'$ from $S_t$

Objective: $\arg\min_{\phi_a, \phi_r, \phi_d} \text{MSE}(\{F_t'\}_{t=1}^T, \text{sg}(\{F_t\}_{t=1}^T))$

### Key Designs

#### 1. **Informative Query Prediction — Addressing (i1)**

**Core modification**: The transitioner is redesigned from a Transformer encoder to a **Transformer decoder**, enabling cross-attention over next-frame features.

**At inference**: The transitioner takes current slots $S_t$ as a starting point and conditions on next-frame features $F_{t+1}$ as supplementary information to predict queries:

$$\phi_r: S_t + E[1], F_{t+1} + E[0] \rightarrow Q_{t+1}$$

where $E \in \mathbb{R}^{\Delta \times c}$ is a learnable relative temporal embedding table. $E[0]$ denotes "at the target timestep" and $E[1]$ denotes "one step before the target."

**Comparison with prior methods**:
- Prior method $\phi_r^1$ (STEVE, SAVi, etc.): predicts from current $S_t$ only, using a Transformer encoder
- Prior method $\phi_r^2$ (STATM, SlotPi): predicts from all historical $\{S_i\}_{i=1}^t$, using multi-layer Transformer encoders
- **Ours**: predicts from $S_t + F_{t+1}$, using a single-layer Transformer decoder (substantially lighter than $\phi_r^2$)

**Why are next-frame features more informative?** By the aggregation equation $\phi_a: Q_t, F_t \rightarrow S_t$, the next-frame features $F_{t+1}$ contain all the most up-to-date information about the next-frame slots — and hence the next-frame queries.

#### 2. **Effective Query Prediction Learning — Addressing (i2)**

**Mechanism**: Rather than always predicting queries from the most recent slot-feature pair, the model is trained with slot-feature pairs **randomly sampled** from the available recurrent history.

**At training**: Timesteps for slots and features are randomly sampled from a temporal window $\Delta$:

$$\phi_r: S_{t_1} + E[t+1-t_1], F_{t_2} + E[t+1-t_2] \rightarrow Q_{t+1}$$

where $t_1 \sim \mathcal{U}\{t-\Delta+1, ..., t\}$ and $t_2 \sim \mathcal{U}\{t-\Delta+2, ..., t+1\}$.

**Design Motivation**: If the transitioner only needs to handle single-step temporal differences, it may converge to a trivial identity mapping. Random sampling of inputs from varying timesteps forces the transitioner to **genuinely understand transition dynamics**: how to infer target queries from arbitrary historical states and features.

**Temporal embeddings**: $E[t+1-t_i]$ is injected via addition (empirically superior to concatenation), informing the transitioner of the relative temporal offset between each input and the target timestep.

**Training vs. inference discrepancy**: At inference, the most recent slot $S_t$ and the latest feature $F_{t+1}$ are always used (i.e., $E[1]$ and $E[0]$) to maximize prediction accuracy.

#### 3. **End-to-End Training Without Additional Losses**

Like existing transitioners, the proposed transitioner is trained end-to-end via the overall MSE reconstruction loss, without requiring any auxiliary transition loss. The random slot-feature pair strategy improves transition dynamics learning without adding training complexity.

### Loss & Training

- Primary loss: MSE reconstruction loss (slots reconstruct DINO2 features as targets)
- Auxiliary loss: slot-slot contrastive loss (ssc) from SlotContrast, or temporal similarity loss (tsim) from VideoSAUR
- Window size $\Delta = 5$ or $6$ (consistent with training video clip length)
- Input resolution 256×256 (224×224); encoder is DINO2 ViT-S/14

## Key Experimental Results

### Main Results (Video Object Discovery)

| Method | MOVi-C ARIfg | MOVi-D ARIfg | YTVIS ARIfg | YTVIS mIoU |
|--------|-------------|-------------|------------|-----------|
| STEVE | - | 66.5 | - | - |
| VideoSAUR | 53.3 | 40.0 | 49.2 | 29.7 |
| SlotContrast | 59.9 | 63.9 | 49.4 | 32.8 |
| **RandSF.Q (tsim)** | **66.3** | **72.0** | **60.4** | **38.5** |
| **RandSF.Q (ssc)** | 67.4 | **77.5** | 58.0 | 37.2 |

**Key result**: On YTVIS, RandSF.Q surpasses SlotContrast by more than **10 percentage points** (ARIfg: 49.4→60.4) and by nearly **6 points** in mIoU (32.8→38.5).

### Ablation Study

| Next-frame features | Random slot-feature pairs | Relative temporal injection | ARI+ARIfg |
|:---:|:---:|:---:|-----------|
| ✓ | ✓ | ✓ | **108.0** |
| ✓ | - | ✓ | 99.7 |
| - | ✓ | ✓ | 81.6 |
| - | - | - | 64.6 |

| Sampling window $\Delta$ | 2 | 3 | 4 | 5 |
|:---:|:---:|:---:|:---:|:---:|
| ARI+ARIfg | 90.0 | 102.0 | 107.8 | **108.0** |

| Temporal injection method | Concatenation | Addition |
|:---:|:---:|:---:|
| ARI+ARIfg | 98.8 | **108.0** |

### Downstream Tasks

| Method | Object recognition top1↑ | Object recognition top3↑ | VQA per-question↑ |
|--------|--------------------------|--------------------------|-------------------|
| SlotContrast | 19.9 | 49.1 | 95.6 |
| **RandSF.Q** | **26.1** | **60.9** | **96.3** |

### Key Findings

1. **Leveraging next-frame features is the most critical factor**: removing it causes performance to drop sharply from 108.0 to 81.6 (−24.4%).
2. **Random sampling contributes the second-largest gain**: removing it reduces performance from 108.0 to 99.7 (−7.7%), though the result still far exceeds the baseline.
3. **Optimal window size aligns with training clip length**: performance peaks when $\Delta$ matches the temporal span of training video clips.
4. **Additive temporal embedding outperforms concatenation**: 108.0 vs. 98.8, as addition injects temporal information along every feature dimension.
5. **Validation of transition dynamics learning**: when non-latest slot-feature pairs are used at inference, performance degrades but remains substantially above SlotContrast, confirming that the transitioner has internalized genuine dynamics knowledge.

## Highlights & Insights

1. **The "removing the transitioner improves performance" finding is highly impactful**: it directly challenges the Transformer encoder transitioner that has been a de facto standard in video OCL for years.
2. **A simple insight about available information**: next-frame features are already accessible at inference time, yet all prior work overlooks this fact.
3. **Elegance of the random sampling training strategy**: no additional parameters or losses are introduced; simply changing the input pairing scheme during training compels the model to learn true dynamics.
4. **Transformer encoder → decoder transition**: this architectural choice is both principled (cross-attention over features is required) and lightweight (only a single decoder block is used).

## Limitations & Future Work

1. **Fixed slot count**: the method still requires a predefined number of slots and cannot adapt to the actual number of objects in a scene. Integrating adaptive slot-count techniques would necessitate redesigning the random sampling strategy.
2. **Window size constrained by training clip length**: longer videos may require larger windows, increasing computational cost.
3. **Evaluated only in self-supervised settings**: effectiveness in supervised or semi-supervised video OCL remains unexplored.
4. **Limited gains on CLEVRER**: the baseline already achieves high performance (95.6%) on simple synthetic videos, leaving little room for improvement.

## Related Work & Insights

- **Slot Attention (Locatello 2020)**: the foundational module for object-centric learning, upon which the proposed transitioner is built.
- **SlotContrast (Manasyan 2025)**: current SOTA and the baseline for this work.
- **VideoSAUR (Zadaianchuk 2024)**: the first method to leverage visual foundation models for video OCL.
- **DINO2 (Oquab 2023)**: serves as the frozen encoder providing features.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The two core insights (next-frame feature utilization + random sampling) are simple yet highly effective, and had been entirely overlooked by prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers synthetic and real datasets, downstream tasks, detailed ablations, and a dynamics validation matrix.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear; motivation experiments are convincing.
- Value: ⭐⭐⭐⭐⭐ — Achieves substantial progress in video OCL; the core ideas are generalizable to other recurrent architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning](../../CVPR2026/video_understanding/reconstruction-guided_slot_curriculum_addressing_object_over-fragmentation_in_vi.md)
- [\[AAAI 2026\] Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers](distillation_dynamics_towards_understanding_feature-based_di.md)
- [\[ICLR 2026\] VideoNSA: Native Sparse Attention Scales Video Understanding](../../ICLR2026/video_understanding/videonsa_native_sparse_attention_scales_video_understanding.md)
- [\[ICLR 2026\] Online Time Series Prediction Using Feature Adjustment](../../ICLR2026/video_understanding/online_time_series_prediction_using_feature_adjustment.md)
- [\[CVPR 2026\] Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning](../../CVPR2026/video_understanding/stay_in_your_lane_role_specific_queries_with_overlap_suppression_loss_for_dense_.md)

</div>

<!-- RELATED:END -->
