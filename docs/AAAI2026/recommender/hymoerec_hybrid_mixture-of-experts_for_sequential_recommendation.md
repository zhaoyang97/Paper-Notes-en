---
title: >-
  [Paper Note] HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation
description: >-
  [AAAI 2026 (Student Abstract)][Recommender Systems][Sequential Recommendation] This paper proposes HyMoERec, a hybrid mixture-of-experts architecture combining shared and specialized expert branches. By replacing the conventional feed-forward network in sequential recommendation models with an adaptive expert fusion mechanism, the model captures heterogeneous user behavior patterns and diverse item complexities, consistently outperforming state-of-the-art methods on the MovieLens-1M and Beauty datasets.
tags:
  - AAAI 2026 (Student Abstract)
  - Recommender Systems
  - Sequential Recommendation
  - Mixture-of-Experts
  - User Behavior Modeling
  - Feed-Forward Network
  - Adaptive Fusion
date: 2026-05-08
content_hash: 65863cc3701839d2
---

# HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation

**Conference**: AAAI 2026 (Student Abstract)  
**arXiv**: [2511.06388](https://arxiv.org/abs/2511.06388)  
**Code**: None  
**Area**: Recommender Systems  
**Keywords**: Sequential Recommendation, Mixture-of-Experts, User Behavior Modeling, Feed-Forward Network, Adaptive Fusion

## TL;DR

This paper proposes HyMoERec, a hybrid mixture-of-experts architecture combining shared and specialized expert branches. By replacing the conventional feed-forward network in sequential recommendation models with an adaptive expert fusion mechanism, the model captures heterogeneous user behavior patterns and diverse item complexities, consistently outperforming state-of-the-art methods on the MovieLens-1M and Beauty datasets.

## Background & Motivation

**Background**: Sequential recommendation aims to predict the next item of interest based on a user's historical interaction sequence. Transformer-based methods (e.g., SASRec, BERT4Rec) have achieved notable progress on this task, with self-attention layers and position-wise feed-forward networks (FFN) as core components.

**Limitations of Prior Work**: The FFN layers in existing models apply identical parameters to all user interactions and all items, essentially treating every input uniformly. In practice, however: (1) user behavior patterns are highly heterogeneous—some users exhibit stable and focused preferences, while others are exploratory and variable; (2) item complexity is diverse—popular items and long-tail items have different representation requirements. A uniform FFN cannot simultaneously accommodate this diversity.

**Key Challenge**: Limited model capacity versus rich user/item diversity—a single FFN lacks sufficient expressive power to capture all types of behavioral patterns and item characteristics.

**Goal**: Design a sequential recommendation architecture capable of adaptively handling different types of user behaviors and items.

**Key Insight**: Replace the uniform FFN with a Mixture-of-Experts (MoE) module, where different experts specialize in different behavioral patterns while shared experts retain the ability to capture universal patterns.

**Core Idea**: A hybrid MoE architecture combining shared and specialized expert branches, employing a gating network to adaptively select and fuse expert outputs, thereby providing customized feature transformations for different user behaviors and items.

## Method

### Overall Architecture

HyMoERec builds on a standard Transformer-based sequential recommendation backbone, replacing the position-wise FFN with a hybrid MoE module. The input is a user's item interaction sequence; after embedding, self-attention, and MoE layers, the model outputs a prediction distribution over the next item.

### Key Designs

1. **Hybrid MoE Architecture**:

    - **Function**: Simultaneously capture universal and specialized behavioral patterns.
    - **Mechanism**: The MoE layer consists of two components: (a) shared expert branch—all inputs pass through these experts, which learn general feature transformations across users and items; (b) specialized expert branch—a gating network dynamically selects the top-$k$ specialized experts per input, each learning to handle specific types of behavioral patterns. The final output is a weighted fusion of the shared branch and the selected specialized branch outputs.
    - **Design Motivation**: A purely shared FFN lacks adaptability, while a purely MoE design may suffer from training instability (expert collapse or load imbalance). The hybrid design leverages the shared branch to ensure baseline performance and training stability, while the specialized branch provides additional adaptive capacity.

2. **Adaptive Expert Fusion Mechanism**:

    - **Function**: Dynamically determine which experts to use and how to fuse their outputs for each input.
    - **Mechanism**: The gating network maps input embeddings to an expert weight distribution, applies softmax normalization, and selects the top-$k$ experts. Fusion weights are determined by the gating network output, routing similar behavioral patterns to the same experts and directing distinct patterns to different ones.
    - **Design Motivation**: Fixed routing cannot adapt to input diversity. Adaptive routing enables on-demand allocation of computational resources—simple inputs may rely solely on shared experts, while complex inputs benefit from additional processing by specialized experts.

3. **Training Stabilization Strategy**:

    - **Function**: Prevent expert collapse and load imbalance during MoE training.
    - **Mechanism**: A load-balancing auxiliary loss is introduced to penalize variance in expert utilization frequency, ensuring all experts are used uniformly. The shared expert branch also provides stable gradient flow, preventing overall training collapse.
    - **Design Motivation**: The classical MoE problem of "winner-takes-all"—where a small number of experts are overused while others degenerate—makes load balancing essential for practical MoE systems.

### Loss & Training

Standard sequential recommendation training objective (e.g., cross-entropy or BPR loss) combined with a load-balancing auxiliary loss.

## Key Experimental Results

### Main Results

| Dataset | Metric | HyMoERec | SOTA Baseline | Gain |
|---------|--------|-----------|---------------|------|
| MovieLens-1M | HR@10 / NDCG@10 | Best | SASRec, etc. | Consistent improvement |
| Beauty | HR@10 / NDCG@10 | Best | SASRec, etc. | Consistent improvement |

### Ablation Study

| Configuration | Performance | Note |
|---------------|-------------|------|
| Hybrid MoE (Full) | Best | Shared + specialized experts |
| Shared experts only | Near standard FFN | No specialization |
| Specialized experts only | Unstable | Lacks shared foundation |
| Standard FFN | Baseline | No expert branches |

### Key Findings

- Hybrid MoE consistently improves recommendation performance over standard FFN, validating the value of handling heterogeneous user behavior.
- Both the shared and specialized branches are indispensable—removing either leads to performance degradation or training instability.
- The model is effective for both long-tail users (few interactions) and active users (many interactions), demonstrating that MoE indeed adapts to diverse behavioral patterns.

## Highlights & Insights

- **Introducing MoE into the FFN layer of sequential recommendation** is a straightforward yet effective idea, representing a natural enhancement of existing architectures.
- **The balance of the hybrid design (shared + specialized)** is a key empirical insight for practical MoE systems—pure MoE tends to be less stable than the hybrid variant.

## Limitations & Future Work

- As a Student Abstract, the experimental scale and depth of analysis are limited.
- The paper does not analyze what behavioral patterns the individual experts actually learn.
- User profile information could be incorporated to assist expert routing.

## Related Work & Insights

- **vs. SASRec/BERT4Rec**: These methods use a uniform FFN; HyMoERec replaces it with a hybrid MoE to enhance adaptability.
- **vs. Switch Transformer MoE**: Switch Transformer validates the effectiveness of MoE in NLP; HyMoERec transfers this idea to the recommendation domain.

## Rating

- Novelty: ⭐⭐⭐ Hybrid MoE for recommendation is relatively novel, though the technical components are established
- Experimental Thoroughness: ⭐⭐⭐ Limited by Student Abstract format
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated
- Value: ⭐⭐⭐ Meaningful improvement for sequential recommendation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation](wavelet_enhanced_adaptive_frequency_filter_for_sequential_re.md)
- [\[AAAI 2026\] Bid Farewell to Seesaw: Towards Accurate Long-tail Session-based Recommendation via Dual Constraints of Hybrid Intents](bid_farewell_to_seesaw_towards_accurate_long-tail_session-based_recommendation_v.md)
- [\[ICLR 2026\] CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](../../ICLR2026/recommender/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)
- [\[NeurIPS 2025\] TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation](../../NeurIPS2025/recommender/tv-rec_time-variant_convolutional_filter_for_sequential_recommendation.md)
- [\[AAAI 2026\] FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)

</div>

<!-- RELATED:END -->
