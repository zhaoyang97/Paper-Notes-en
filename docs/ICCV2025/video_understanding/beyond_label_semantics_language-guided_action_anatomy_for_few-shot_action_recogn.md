---
title: >-
  [Paper Note] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition
description: >-
  [ICCV 2025][Video Understanding][few-shot action recognition] This paper proposes the Language-Guided Action Anatomy (LGA) framework, which leverages large language models to decompose action labels into atomic-level action descriptions encoded as subject–motion–object triplets. On the video side, a clustering-based segmentation strategy partitions frame sequences into corresponding atomic action stages. Multimodal fusion and matching are then performed at the atomic level, yielding substantial improvements in few-shot action recognition performance.
tags:
  - ICCV 2025
  - Video Understanding
  - few-shot action recognition
  - LLM
  - atomic action
  - multimodal fusion
  - metric learning
date: 2026-05-08
content_hash: d8f408731caa1487
---

# Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition

**Conference**: ICCV 2025
**arXiv**: [2507.16287](https://arxiv.org/abs/2507.16287)
**Code**: N/A
**Area**: Video Understanding / Few-Shot Action Recognition
**Keywords**: few-shot action recognition, LLM, atomic action, multimodal fusion, metric learning

## TL;DR

This paper proposes the Language-Guided Action Anatomy (LGA) framework, which leverages large language models to decompose action labels into atomic-level action descriptions encoded as subject–motion–object triplets. On the video side, a clustering-based segmentation strategy partitions frame sequences into corresponding atomic action stages. Multimodal fusion and matching are then performed at the atomic level, yielding substantial improvements in few-shot action recognition performance.

## Background & Motivation

Few-shot action recognition (FSAR) aims to classify unseen-category videos given only a handful of annotated samples. Recent multimodal approaches—particularly those incorporating textual information—have achieved notable progress, yet existing methods typically exploit only the coarse-grained semantics of action labels. Human actions, however, contain rich fine-grained information: postural changes, motion dynamics, and object interactions manifest differently across temporal stages, and such critical knowledge cannot be adequately captured from action labels alone.

The authors observe that the key factors of an action comprise: (1) three core elements—subject, motion dynamics, and object interaction; and (2) the onset, progression, and completion stages along the temporal dimension, all of which are critical for category discrimination. This motivates fine-grained alignment between query and support videos so that every key aspect of an action is accounted for.

## Method

### Overall Architecture

The LGA framework consists of three core modules: **Action Anatomy**, **Fine-grained Multimodal Fusion**, and **Multimodal Matching**. After a visual backbone extracts features from input videos, the textual branch employs an LLM to decompose each label into atomic action descriptions, while the visual branch segments the frame sequence into corresponding atomic action stages. Feature fusion and matching are subsequently performed at the atomic level.

### Key Designs

1. **Textual Anatomy**:

    - An LLM decomposes each action label into an ordered sequence of atomic action descriptions.
    - Each description explicitly encodes the subject, motion, and object elements.
    - For example, "Jump into pool" → onset: "A person standing at the edge…"; progression: "The person leaping off…"; completion: "The person entering the water…"
    - Atomic descriptions are fed into a text backbone to extract features $\{t_i\}_{i=1}^{L}$.

2. **Visual Anatomy and CLUSTER-Segment**:

    - The frame feature sequence $\{f_i\}_{i=1}^{T}$ is partitioned into $L$ atomic action stages.
    - The CLUSTER-Segment strategy initializes each frame as an individual cluster, computes cosine similarity between adjacent clusters, and iteratively merges the most similar adjacent pair until $L$ clusters remain.
    - Overlapping frames are added between adjacent clusters to enhance robustness.
    - **Design Motivation**: Unlike uniform segmentation, this strategy adaptively captures sub-actions of varying durations.

3. **Fine-grained Multimodal Fusion Module**:

    - Multi-head cross-attention integrates atomic-level visual and textual features.
    - The query is formed by summing the atomic visual feature $f_{S_i}$ and the corresponding textual feature $t_i$: $\mathbf{Q_i} = t_i + f_{S_i}$.
    - Keys and values are formed by concatenating all atomic visual features: $\mathbf{K} = \mathbf{V} = \text{concat}(\{f_{S_i}\}_{i=1}^{L})$.
    - Each atomic action feature thus learns local semantic details while remaining aware of the global temporal structure.
    - The final action prototype is obtained by concatenating all stage features: $\tilde{f} = \text{concat}(\tilde{f}_{S_1}, \tilde{f}_{S_2}, \tilde{f}_{S_3})$.

4. **Multimodal Matching Module**:

    - **Video–video matching**: The Aligned Bidirectional Mean Hausdorff Metric (AB-MHM) is proposed to align temporal sequences at the atomic action level and compute the distance between query and support videos.
    - **Video–text matching**: The average-pooled feature of each stage in the query video is compared against the textual features of each class.
    - The two matching scores are combined via a weighted geometric mean: $p_{(y=i|q)} = (p^{v-v})^{\alpha} \times (p^{v-t})^{(1-\alpha)}$.

### Loss & Training

- Episode-based meta-learning is adopted during training.
- A joint objective combining cross-entropy loss and contrastive loss is used.
- During training, only video–video matching is used for classification to ensure stability; the full multimodal matching is employed at inference.
- The visual backbone is initialized with CLIP ViT-B/16, and 8 frames are sampled uniformly per video.

## Key Experimental Results

### Main Results

| Dataset | Setting | LGA | CLIP-FSAR | EMP-Net | Gain (vs. CLIP-FSAR) |
|--------|------|-----|-----------|---------|---------------------|
| HMDB51 | 1-shot | **86.8** | 77.1 | 76.8 | +9.7 |
| HMDB51 | 5-shot | **89.3** | 87.7 | 85.8 | +1.6 |
| Kinetics | 1-shot | **95.2** | 94.8 | 89.1 | +0.4 |
| UCF101 | 1-shot | **98.2** | 97.0 | 94.3 | +1.2 |
| SSv2-Small | 1-shot | **58.9** | 54.6 | 57.1 | +4.3 |
| SSv2-Small | 5-shot | **69.3** | 61.8 | 65.7 | +7.5 |
| SSv2-Full | 1-shot | **63.8** | 62.1 | 63.1 | +1.7 |
| SSv2-Full | 5-shot | **74.4** | 72.1 | 73.0 | +2.3 |

### Ablation Study

| Visual-An | Textual-An | V-V Match | V-T Match | HMDB51 1-shot | HMDB51 5-shot |
|-----------|------------|---------|---------|---------------|---------------|
| ✗ | ✗ | ✓ | ✗ | 75.8 | 87.7 |
| ✓ | ✗ | ✓ | ✗ | 79.9 | 86.0 |
| ✗ | ✓ | ✓ | ✗ | 79.6 | 87.2 |
| ✓ | ✓ | ✓ | ✗ | 80.8 | 88.2 |
| ✓ | ✓ | ✗ | ✓ | 83.1 | 86.2 |
| ✓ | ✓ | ✓ | ✓ | **86.8** | **89.3** |

### Key Findings

- Visual anatomy and textual anatomy individually yield 1-shot gains of 4.1% and 3.8%, respectively.
- Multimodal matching (V-V + V-T) is especially effective in the 1-shot setting (+6.0%), indicating that textual cues are more critical when visual information is scarce.
- Three atomic action stages (onset / progression / completion) is optimal; more stages introduce LLM hallucinations and temporal overlap.
- CLUSTER-Segment outperforms uniform segmentation (HARD) and TW-FINCH by adaptively accommodating sub-actions of varying durations.

## Highlights & Insights

- **LLM as an action knowledge engine**: The framework ingeniously leverages the world knowledge embedded in LLMs to anatomize actions rather than relying on raw label text, thereby making explicit the rich prior knowledge implicit in action labels.
- **Atomic-level alignment**: Unlike global-level visual–textual alignment, fine-grained alignment at the sub-action level more closely mirrors how humans understand actions.
- **AB-MHM metric**: Introducing atomic-level temporal alignment into the Hausdorff distance yields a non-parametric design with strong transferability and computational efficiency.

## Limitations & Future Work

- The number of atomic action stages is fixed at three, which may be insufficiently flexible for complex actions; adaptive determination of the segment count warrants investigation.
- The quality of LLM-generated descriptions depends on the capability of the underlying model and is subject to hallucination risk.
- Using visual or textual anatomy in isolation may degrade performance in the 5-shot setting due to cross-modal misalignment.
- Validation on larger-scale datasets and backbone architectures has not been conducted.

## Related Work & Insights

- Compared with methods that employ expanded descriptions (e.g., SAFSAR), LGA's atomic-level decomposition more effectively captures the temporal structure of actions.
- The CLUSTER-Segment strategy is generalizable to other tasks requiring temporal video segmentation.
- The complementarity of multimodal matching suggests dataset-specific sensitivity: HMDB51 benefits more from text-based matching, whereas SSv2 benefits more from visual matching.

## Rating

- Novelty: ⭐⭐⭐⭐ — Innovative combination of action anatomy and atomic-level fusion/matching, leveraging LLMs to extract action prior knowledge.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five benchmarks, detailed ablations, multi-dimensional analysis, and visualization.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear; the method is presented in a systematic and coherent manner.
- Value: ⭐⭐⭐⭐ — Establishes a new paradigm for exploiting LLMs in few-shot action recognition.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition](trokens_semantic-aware_relational_trajectory_tokens_for_few-shot_action_recognit.md)
- [\[AAAI 2026\] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition](../../AAAI2026/video_understanding/task-specific_distance_correlation_matching_for_few-shot_action_recognition.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[NeurIPS 2025\] Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](../../NeurIPS2025/video_understanding/seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)

<!-- RELATED:END -->
