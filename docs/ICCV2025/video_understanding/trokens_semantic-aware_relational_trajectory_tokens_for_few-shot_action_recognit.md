---
title: >-
  [Paper Note] Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition
description: >-
  [ICCV 2025][Video Understanding][Few-Shot Action Recognition] This paper proposes the Trokens framework, which converts point trajectories into semantically-aware relational tokens via **semantic-aware trajectory point sampling** and **relational motion modeling** (comprising intra-trajectory HoD and inter-trajectory relative displacement descriptors). By fusing these tokens with appearance features, Trokens achieves state-of-the-art performance on six few-shot action recogni…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Few-Shot Action Recognition"
  - "Point Trajectory Tracking"
  - "Semantic Sampling"
  - "Motion Modeling"
  - "HoD"
date: 2026-05-08
content_hash: a6a977f9989ac11f
---

# Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition

**Conference**: ICCV 2025
**arXiv**: [2508.03695](https://arxiv.org/abs/2508.03695)  
**Code**: [Project Page](https://pulkitkumar95.github.io/trokens)  
**Area**: Video Understanding / Few-Shot Action Recognition
**Keywords**: Few-Shot Action Recognition, Point Trajectory Tracking, Semantic Sampling, Motion Modeling, HoD

## TL;DR

This paper proposes the Trokens framework, which converts point trajectories into semantically-aware relational tokens via **semantic-aware trajectory point sampling** and **relational motion modeling** (comprising intra-trajectory HoD and inter-trajectory relative displacement descriptors). By fusing these tokens with appearance features, Trokens achieves state-of-the-art performance on six few-shot action recognition benchmarks.

## Background & Motivation

The core of video understanding lies in the joint modeling of **motion** and **appearance** information. In few-shot action recognition, where training data is extremely scarce, this joint modeling becomes especially critical. Existing methods face two fundamental challenges:

### Challenge 1: How to Select Informative Tracking Points?

- **Dense sampling**: Comprehensive coverage but computationally expensive.
- **Uniform grid sampling** (e.g., TATs): Simple and efficient, but **cannot adapt to object scale**—small yet critical objects (e.g., knives, buttons) are easily missed, while large background regions are redundantly sampled.
- Example: In the action "spreading butter with a knife," uniform sampling may entirely miss the motion of the knife.

### Challenge 2: How to Effectively Model Trajectory Motion Patterns?

- Existing Transformer-based methods (e.g., TATs) treat trajectories **only as feature sampling anchors**, implicitly relying on self-attention to learn motion.
- However, positional embeddings primarily encode static positions and do not directly capture **temporal displacements** or **cross-trajectory relations**.
- Optical flow methods are limited to adjacent-frame analysis and degrade under occlusion.

The motivation behind Trokens is to leverage **semantic priors** to guide point sampling for adaptive coverage, while **explicitly modeling** both intra- and inter-trajectory motion dynamics.

## Method

### Overall Architecture

Trokens consists of four sequentially connected components:

1. **Appearance Feature Extraction**: DINOv2-base extracts video appearance tokens $\mathcal{F}^{\text{RGB}} \in \mathbb{R}^{H \times W \times T \times C}$.
2. **Semantic-Aware Point Sampling**: Adaptive sampling guided by clustering of DINO features.
3. **Relational Motion Modeling**: Intra-trajectory HoD + inter-trajectory relative displacements.
4. **Decoupled Spatiotemporal Transformer**: Fuses motion and appearance tokens for classification.

### Semantic-Aware Point Sampling

**Core Idea**: Exploiting the **natural semantic clustering property** of DINO patch tokens—tokens belonging to the same object naturally cluster together in feature space.

Procedure:
1. Extract patch token features from DINOv2.
2. Cluster features into $L$ semantic groups.
3. Uniformly sample $q = M/L$ points per group, where $M=256$ is the total number of trajectories.
4. Begin sampling from the frame in which a new semantic group first appears.
5. Track these points using the pretrained CoTracker to obtain semantically-aware trajectories $\mathcal{P} \in \mathbb{R}^{M \times T \times 2}$.

**Key Advantage**: Small objects (e.g., a knife) form their own semantic group and receive the same sampling density as large objects, ensuring that critical motion information is not overlooked.

### Relational Motion Modeling

#### Intra-Trajectory Motion Module

Inspired by HoG (Histogram of Oriented Gradients), this module adopts **HoD (Histogram of Oriented Displacements)** to encode the motion direction and magnitude within each trajectory.

For trajectory $\mathcal{P}^m$ at time $t$, the displacement is computed as:

$$\Delta x_t = x_t - x_{t-\delta}, \quad \Delta y_t = y_t - y_{t-\delta}$$

Displacement magnitude: $\Delta d_t = \sqrt{\Delta x_t^2 + \Delta y_t^2}$

Displacement orientation: $\theta_t = \arctan2(\Delta y_t, \Delta x_t)$

$\theta_t$ is quantized into $B=32$ orientation bins, with the displacement magnitude distributed to the two nearest bins weighted by distance, yielding a per-timestep HoD descriptor $\mathbf{H}_{\text{HoD}} \in \mathbb{R}^{T \times B}$, which is projected to a $C$-dimensional feature space via a fully connected layer:

$$\mathcal{F}_{\text{intra}}^{\text{motion}} = \text{FC}(f_{\text{HoD}}(\mathcal{P})) \in \mathbb{R}^{M \times T \times C}$$

Differences from the original HoD: (1) per-timestep computation preserves temporal order; (2) learnable projection enhances representational capacity; (3) generalization from human skeletal keypoints to arbitrary trajectories.

#### Inter-Trajectory Motion Module

This module captures **coordinated motion** across different trajectories (e.g., the relative motion between a knife and bread distinguishes "spreading butter" from "cutting").

For trajectory $\mathcal{P}^m$ at time $t$, the relative displacement with respect to all other trajectories is computed as:

$$\mathbf{d}_t^m = [(x_t^m - x_t^{m'}, y_t^m - y_t^{m'})]_{m'=1}^{M} \in \mathbb{R}^{2M}$$

The full descriptor $\mathbf{d} \in \mathbb{R}^{M \times T \times 2M}$ is projected via a fully connected layer:

$$\mathcal{F}_{\text{inter}}^{\text{motion}} = \text{FC}(\mathbf{d}) \in \mathbb{R}^{M \times T \times C}$$

### Motion-Aware Spatiotemporal Transformer

1. **Trajectory Alignment**: Appearance tokens aligned to trajectories are sampled from appearance features according to trajectory coordinates, yielding $\mathcal{F}_{\text{traj}}^{\text{RGB}}$.
2. **Feature Fusion**: Three feature streams are fused via element-wise addition:

$$\mathcal{F}^{\text{fuse}} = \mathcal{F}_{\text{traj}}^{\text{RGB}} + \mathcal{F}_{\text{intra}}^{\text{motion}} + \mathcal{F}_{\text{inter}}^{\text{motion}}$$

3. **Decoupled Attention**: Self-attention is applied separately along the intra-trajectory (temporal) and inter-trajectory (spatial) dimensions, and the results are summed.
4. **Classification Output**: A learnable CLS token aggregates the final features via cross-attention.

### Loss & Training

Standard few-shot dual loss:

$$\mathcal{L} = \mathcal{L}_{\text{CE}}(p_{\text{cls}}^Q, y) + \mathcal{L}_{\text{Contrastive}}(\mathcal{F}_{\text{final}}^Q, \mathcal{F}_{\text{final}}^S)$$

## Key Experimental Results

### Main Results: SSV2 Full (5-way K-shot)

| Method | 1-shot | 2-shot | 3-shot | 5-shot |
|--------|--------|--------|--------|--------|
| MoLo (CVPR'23) | 56.6 | 62.3 | 67.0 | 70.6 |
| TATs (ECCV'24) | 57.7 | 67.1 | 70.0 | 74.6 |
| **Trokens** | **61.5** | **69.9** | **73.8** | **76.7** |

On SSV2 Full, Trokens achieves a **+3.8%** gain at 1-shot and **+2.1%** at 5-shot over TATs. SSV2 is a motion-intensive dataset, validating the importance of explicit motion modeling.

### Cross-Dataset Generalization

| Dataset | Method | 1-shot | 3-shot | 5-shot |
|---------|--------|--------|--------|--------|
| SSV2 Small | TATs | 47.9 | 60.0 | 64.4 |
| SSV2 Small | **Trokens** | **53.4** | **65.3** | **68.9** |
| HMDB-51 | TATs | 60.0 | 71.8 | 77.0 |
| HMDB-51 | **Trokens** | **69.8** | **80.0** | **82.3** |
| UCF-101 | TATs | 92.0 | 96.8 | 95.5 |
| UCF-101 | **Trokens** | **94.0** | **97.3** | **97.9** |

Trokens achieves a **+9.8%** gain on HMDB-51 at 1-shot and **+5.5%** on SSV2 Small at 1-shot, representing substantial improvements.

### Ablation Study

Through class-level performance analysis (Figure 3), the paper identifies the sources of improvement:
- **Semantic Sampling**: Action categories involving small objects (e.g., "Unfolding something," "Twisting something") show notable gains.
- **Motion Modeling**: Categories requiring fine-grained temporal dynamics (e.g., "Pulling something from left to right") improve significantly.
- **Exposed Limitations**: Point tracking degrades under fast motion causing blur (e.g., "Rolling something on flat surface") and large camera movements (e.g., "Picking something up").

## Highlights & Insights

1. **Semantic-Aware Sampling as an Overlooked Key**: The idea of using DINO clustering to guide sampling is simple yet highly effective, especially for actions involving small objects.
2. **Modern Revival of Classical Methods**: HoD was originally a handcrafted feature from a decade ago; Trokens reformulates it as a learnable, per-timestep descriptor that thrives within a deep learning framework.
3. **Explicit Motion Modeling > Implicit Learning**: Although self-attention in Transformers can theoretically capture motion, explicit priors prove more effective in the low-data few-shot regime.
4. Element-wise addition for motion–appearance fusion, despite its simplicity, performs remarkably well.

## Limitations & Future Work

1. Reliance on external point tracking models such as CoTracker introduces additional computational overhead and dependencies.
2. The $\mathbb{R}^{2M}$ descriptor in the inter-trajectory module grows quadratically with the number of trajectories, limiting scalability.
3. Gains on appearance-biased datasets such as Kinetics are limited (only +1.0% at 1-shot), as the value of motion modeling is dataset-dependent.
4. Evaluation is conducted in a vision-only setting, without comparison to multimodal (vision–language) methods.

## Related Work & Insights

- **Few-Shot AR**: OTAM, TRX, STRM, MoLo, HYRSM, TATs
- **Point Tracking**: CoTracker, PIPs, TAPIR, TATs
- **Motion Features**: HoG, HoD, optical flow methods

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of semantic sampling, modernized HoD, and explicit relational motion modeling is novel and effective.
- Practicality: ⭐⭐⭐⭐ — End-to-end trainable; achieves comprehensive SOTA across six benchmarks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Six datasets, multiple shot/way settings, and class-level analysis.
- Writing Quality: ⭐⭐⭐⭐ — Motivation figures are clear; method derivation is complete.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[CVPR 2025\] Temporal Alignment-Free Video Matching for Few-Shot Action Recognition](../../CVPR2025/video_understanding/temporal_alignment-free_video_matching_for_few-shot_action_recognition.md)
- [\[CVPR 2025\] TAMT: Temporal-Aware Model Tuning for Cross-Domain Few-Shot Action Recognition](../../CVPR2025/video_understanding/tamt_temporal-aware_model_tuning_for_cross-domain_few-shot_action_recognition.md)
- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)

</div>

<!-- RELATED:END -->
