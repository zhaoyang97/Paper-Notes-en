---
title: >-
  [Paper Note] Temporal Alignment-Free Video Matching for Few-Shot Action Recognition
description: >-
  [CVPR 2025][Video Understanding][Few-Shot Action Recognition] This paper proposes TEAM (TEmporal Alignment-free Matching), which aggregates video features with cross-attention using a fixed number of learnable pattern tokens. By doing so, it eliminates the dependence on predefined temporal units and brute-force alignment, achieving more flexible and efficient video matching for the FSAR task and reaching SOTA performance on multiple benchmarks.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Few-Shot Action Recognition"
  - "Temporal Alignment-free Matching"
  - "Pattern Tokens"
  - "Meta-learning"
  - "Feature Aggregation"
date: 2026-05-08
content_hash: 6f71bcc057cdc6c6
---

# Temporal Alignment-Free Video Matching for Few-Shot Action Recognition

**Conference**: CVPR 2025  
**arXiv**: [2504.05956](https://arxiv.org/abs/2504.05956)  
**Code**: [https://github.com/leesb7426/TEAM](https://github.com/leesb7426/TEAM)  
**Area**: Video Understanding  
**Keywords**: Few-Shot Action Recognition, Temporal Alignment-free Matching, Pattern Tokens, Meta-learning, Feature Aggregation

## TL;DR

This paper proposes TEAM (TEmporal Alignment-free Matching), which aggregates video features with cross-attention using a fixed number of learnable pattern tokens. By doing so, it eliminates the dependence on predefined temporal units and brute-force alignment, achieving more flexible and efficient video matching for the FSAR task and reaching SOTA performance on multiple benchmarks.

## Background & Motivation

**Background**: Few-shot action recognition (FSAR) aims to learn novel action classes with only a few annotated videos. Leading methods are based on metric learning, typically employing frame-level or tuple-level temporal alignment to measure the distance between support and query videos. Frame-level methods like OTAM search for optimal matching across frames, while tuple-level methods like TRX align sub-sequence units.

**Limitations of Prior Work**: (1) **Inflexibility**: Frame-level and tuple-level methods rely on predefined alignment units (frames or fixed-length tuples), which struggle to adapt when action durations and speeds vary significantly. (2) **Efficiency issues**: The alignment cost grows quadratically with the number of frames because pairwise similarities must be computed between all support and query units. (3) Existing FSAR benchmarks mostly consist of trimmed, short-term videos, which masks this efficiency issue.

**Key Challenge**: There is a fundamental conflict between alignment methods based on predefined units and the diverse, real-world durations of actions—fixed windows cannot accommodate infinitely varying action patterns.

**Goal**: (1) Eliminate the need for predefined temporal units in action representation; (2) Eliminate the brute-force alignment process in video matching.

**Key Insight**: Rather than partitioning videos into frames or tuples and aligning them individually, a set of fixed, globally learnable pattern tokens can be used to "absorb" the discriminative features of each video. Matching can then be performed by directly comparing the corresponding pattern tokens, without any alignment steps.

**Core Idea**: Use cross-attention to aggregate video frame features into fixed pattern tokens, where each token learns a global discriminative pattern. Matching is performed token-wise, thereby eliminating the quadratic complexity of temporal alignment.

## Method

### Overall Architecture

Given support and query videos, $T=8$ frames are uniformly sampled, and frame features are independently extracted using an image feature extractor (ResNet-50 or ViT-B). Subsequently, $M$ learnable pattern tokens $P = [P_1, P_2, ..., P_M]$ are defined. Through a dual-complementary aggregation scheme, instance tokens $P^+$ and exclusive tokens $P^-$ are generated. After adapting the tokens of the support set, the classification probability is calculated using token-wise similarity.

### Key Designs

1. **Instance Pattern Tokens ($P^+$)**:

    - **Function**: Aggregates positive features relevant to class discrimination within the video, capturing shared patterns across videos of the same class.
    - **Mechanism**: Each pattern token $P_m$ acts as a query, and the video frame features $F$ act as keys and values. Information is aggregated via cross-attention: $\bar{P}_m^+ = P_m + \text{CA}(P_m, F, F)$, which is then passed through an MLP to obtain $P_m^+$. During matching, the corresponding tokens are measured using cosine distance: $\text{PD}(P_n^S, P^Q) = \sum_{m=1}^{M} -d(P_{n,m}^{S+}, P_m^{Q+})$.
    - **Design Motivation**: Free from the constraint of frame numbers or speed, pattern tokens capture global discriminative patterns rather than relying on specific frame matches determined by alignment, offering both flexibility and efficiency.

2. **Exclusive Pattern Tokens ($P^-$)**:

    - **Function**: Encodes "heterogeneity"—that is, feature patterns belonging to other classes that do not exist in the current class video.
    - **Mechanism**: Contrary to the residual addition of instance tokens, subtraction is utilized: $\bar{P}_m^- = P_m - \text{CA}(P_m, F, F)$, which directly excludes information related to the current class, making exclusive tokens close to instance tokens of other classes. During classification, complementary evidence from both instance and exclusive tokens is considered, computing the exclusive probability using the nearest-class distance $\text{ND}$.
    - **Design Motivation**: Some videos might lack the typical discriminative features of a class, which might cause classification to fail when relying solely on instance tokens. Exclusive tokens assist the decision by identifying "what it is not", providing a complementary perspective for classification.

3. **Adaptation of Support Pattern Tokens**:

    - **Function**: Adjusts the support tokens based on the composition of novel classes in each episode to establish clearer boundaries between classes.
    - **Mechanism**: The cosine similarity of instance tokens between classes is calculated as $E_{n,o,m}^+ = P_{n,m}^{S+} \cdot P_{o,m}^{S+}$, serving as an indicator of semantic entanglement. The adapted instance tokens are obtained by enhancing self-information and suppressing shared information: $\tilde{P}_{n,o,m}^{S+} = P_m + (1 + E_{n,o,m}^+)\text{CA}(P_m, F_n^S, F_n^S) - E_{n,o,m}^+\text{CA}(P_m, F_o^S, F_o^S)$, which is then averaged across all other classes. Exclusive tokens are processed similarly but in the opposite direction.
    - **Design Motivation**: Pattern tokens trained globally possess strong discriminative power for base classes but may not be sufficiently refined for novel classes. The adaptation process dynamically optimizes the decision boundary in each episode by explicitly removing shared information between classes.

### Loss & Training

The final loss is the sum of cross-entropy from both the instance and exclusive components: $\mathcal{L} = \mathcal{L}^+ + \mathcal{L}^-$. During inference, the classification probability fuses the two distances: $p(y^Q = n) = \text{softmax}(\text{PD}(\hat{P}^S, P^Q) + \text{ND}(\hat{P}^S, P^Q); n)$.

Training is performed using SGD for 10,000 iterations. The number of pattern tokens $M$ is set within 50-80 depending on the dataset/setting (e.g., 60 for Kinetics 1-shot, 80 for 5-shot). A prototyping concept is adopted to handle many-shot scenarios.

## Key Experimental Results

### Main Results (5-way, ResNet-50 backbone)

| Method | HMDB51 1-shot | Kinetics 1-shot | UCF101 1-shot |
|------|-------------|----------------|--------------|
| OTAM | 54.5 | 72.2 | 79.9 |
| MoLo | 60.8 | 74.0 | 86.0 |
| GgHM | 61.2 | 74.9 | 85.2 |
| **Ours** | **62.8** | **75.1** | **87.2** |

Using a ViT-B backbone yields further improvements across all datasets (HMDB51: 70.9, Kinetics: 83.3, UCF101: 94.5).

### Ablation Study

| $P^+$ | $P^-$ | $\hat{P}$ | E | HMDB51 | Kinetics | UCF101 |
|-------|-------|----------|---|--------|----------|--------|
| ✓ | | | | 61.8 | 74.6 | 86.7 |
| ✓ | ✓ | | | 62.5 | 75.0 | 86.8 |
| ✓ | ✓ | ✓ | | 62.2 | 74.8 | 87.1 |
| ✓ | ✓ | ✓ | ✓ | **62.8** | **75.1** | **87.2** |

### Key Findings

- Using only instance tokens (row a) already outperforms most alignment methods, validating the core advantage of alignment-free matching.
- Exclusive tokens provide a stable, complementary boost (+0.4-0.7%).
- Adaptation without entanglement control leads to performance degradation (row c vs b), highlighting the importance of controlling the scale of adaptation.
- In cross-domain evaluations (Kinetics $\rightarrow$ HMDB51/UCF101), TEAM consistently outperforms all methods, demonstrating strong generalization capability.
- The performance is robust and insensitive to the number of pattern tokens within the range of 40-80.
- The efficiency advantage is significant: compared to the $O(T^2)$ complexity of frame-level alignment, TEAM requires only $O(M)$ (time is significantly reduced, as shown in Fig. 7).

## Highlights & Insights

- **Fundamental Simplification**: Transforms the temporal alignment problem into a feature aggregation problem, replacing variable frame matching with a fixed number of tokens, which is conceptually elegant and highly effective.
- **Complementary Dual-Token Design**: Instance tokens capture "what it is" and exclusive tokens encode "what it is not", establishing more robust class boundaries.
- **Adaptive Entanglement Control**: Dynamically adjusts the adaptation strength based on inter-class similarities to avoid over-adaptation.

## Limitations & Future Work

- On datasets with extremely high temporal complexity like SSv2-Small, TEAM is slightly inferior to the TATs method that utilizes an additional Point Tracker.
- The number of pattern tokens needs to be manually tuned per dataset (although it is not highly sensitive).
- The current method only uses frame-level global features, without modeling finer-grained spatio-temporal attention.
- The integration with large-scale pre-trained vision models (e.g., CLIP, VideoMAE) has not been explored.

## Related Work & Insights

- Token-based aggregation methods such as DETR's object query and BLIP-2's Q-Former provide similar design paradigms.
- The innovation of TEAM lies in the dual-complementary aggregation scheme and episode-level adaptation; these concepts can be transferred to other few-shot learning scenarios.
- The alignment-free matching approach could inspire other domains requiring sequence matching (such as text matching or time-series classification).

## Rating

- **Novelty**: 8/10 — The idea of transforming frame/tuple alignment into token aggregation is simple and effective.
- **Experimental Thoroughness**: 8/10 — Covers 4 datasets, multiple backbones, and various configurations, with detailed ablation studies.
- **Writing Quality**: 8/10 — Intuitive diagrams and clear methodological descriptions.
- **Value**: 7/10 — Practical improvements in the FSAR domain, though the broader impact might be relatively specific.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TAMT: Temporal-Aware Model Tuning for Cross-Domain Few-Shot Action Recognition](tamt_temporal-aware_model_tuning_for_cross-domain_few-shot_action_recognition.md)
- [\[AAAI 2026\] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition](../../AAAI2026/video_understanding/task-specific_distance_correlation_matching_for_few-shot_action_recognition.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](../../ICCV2025/video_understanding/beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[ICCV 2025\] Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition](../../ICCV2025/video_understanding/trokens_semantic-aware_relational_trajectory_tokens_for_few-shot_action_recognit.md)
- [\[CVPR 2025\] Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models](video-panda_parameter-efficient_alignment_for_encoder-free_video-language_models.md)

</div>

<!-- RELATED:END -->
