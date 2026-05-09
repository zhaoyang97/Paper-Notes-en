---
title: >-
  [Paper Note] TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation
description: >-
  [NeurIPS 2025][Recommender Systems][Sequential recommendation] This paper proposes TV-Rec, a time-variant convolutional filter grounded in graph signal processing that replaces conventional fixed convolutions and self-attention mechanisms, achieving higher expressiveness for sequential recommendation with an average improvement of 7.49% across 6 benchmark datasets.
tags:
  - NeurIPS 2025
  - Recommender Systems
  - Sequential recommendation
  - time-variant convolutional filter
  - graph signal processing
  - attention replacement
  - user behavior modeling
date: 2026-05-08
content_hash: d8579138623e5346
---

# TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation

**Conference**: NeurIPS 2025
**arXiv**: [2510.25259](https://arxiv.org/abs/2510.25259)
**Code**: N/A
**Area**: Recommender Systems
**Keywords**: Sequential recommendation, time-variant convolutional filter, graph signal processing, attention replacement, user behavior modeling

## TL;DR

This paper proposes TV-Rec, a time-variant convolutional filter grounded in graph signal processing that replaces conventional fixed convolutions and self-attention mechanisms, achieving higher expressiveness for sequential recommendation with an average improvement of 7.49% across 6 benchmark datasets.

## Background & Motivation

Sequential recommendation aims to predict the next item of interest based on a user's historical interaction sequence. Convolutional filters have been widely adopted for their ability to capture local sequential patterns, yet they suffer from fundamental limitations:

**Insufficient expressiveness of fixed convolutions**: Traditional convolutional filters are position-invariant, making it difficult to capture global interactions.

**Reliance on self-attention as a supplement**: Most convolution-based models require additional self-attention layers to model global dependencies.

**Computational efficiency**: The $O(n^2)$ complexity of self-attention limits applicability to long sequences.

The core insight of this paper: inspired by graph signal processing (GSP), **time-variant graph filters** can simultaneously capture position-dependent temporal variations and global interaction patterns, thereby fully replacing self-attention.

## Method

### Overall Architecture

TV-Rec formulates user interaction sequences as a signal processing problem on a temporal graph:
1. Items in the sequence are treated as graph nodes.
2. Temporal relationships form graph edges.
3. User preferences are treated as signals on the graph.
4. Time-variant filters learn position-dependent signal transformations.

### Key Designs

1. **Time-Variant Graph Filter**:

    - Unlike fixed convolutional kernels, filter coefficients vary with sequence position.
    - Each position has independent filter parameters that capture interaction patterns at that specific moment.
    - Mathematical formulation: $\mathbf{h}_t = \sum_{k=0}^{K} \alpha_k(t) \mathbf{S}^k \mathbf{x}_t$
    - where $\alpha_k(t)$ are position-dependent polynomial coefficients and $\mathbf{S}$ is the graph shift operator.

2. **Multi-Order Interaction Modeling**:

    - Low-order terms (small $k$): capture local/recent interactions.
    - High-order terms (large $k$): capture global/long-range interactions.
    - Time-variant coefficients adaptively balance the importance of different orders.

3. **Efficient Implementation**:

    - Avoids explicit construction of attention matrices.
    - Exploits the sparsity of the graph shift operator to accelerate computation.
    - Complexity is $O(Kn)$, significantly lower than the $O(n^2)$ of self-attention.

### Loss & Training

Standard cross-entropy loss is employed:
$$\mathcal{L} = -\sum_{t} \log \frac{\exp(s_{y_{t+1}})}{\sum_{j} \exp(s_j)}$$

where $s_j$ denotes the predicted score for item $j$.

## Key Experimental Results

### Main Results (6 Datasets)

| Method | Beauty HR@10 | Beauty NDCG@10 | Sports HR@10 | ML-1M HR@10 | Yelp HR@10 | Amazon HR@10 |
|--------|-------------|---------------|-------------|------------|-----------|-------------|
| SASRec | 5.83 | 3.21 | 3.94 | 18.52 | 3.12 | 4.85 |
| BERT4Rec | 5.45 | 2.98 | 3.67 | 17.89 | 2.95 | 4.52 |
| FMLP-Rec | 6.12 | 3.45 | 4.21 | 19.23 | 3.35 | 5.12 |
| FEARec | 6.28 | 3.52 | 4.35 | 19.45 | 3.42 | 5.28 |
| BSARec | 6.35 | 3.58 | 4.42 | 19.67 | 3.48 | 5.35 |
| **TV-Rec** | **6.82** | **3.85** | **4.78** | **21.12** | **3.75** | **5.72** |
| Gain | +7.4% | +7.5% | +8.1% | +7.4% | +7.8% | +6.9% |

### Efficiency Comparison

| Method | Params (M) | Training Time (s/epoch) | Inference Latency (ms) | ML-1M NDCG@10 |
|--------|----------|------------------------|----------------------|--------------|
| SASRec | 1.2 | 42 | 8.5 | 10.85 |
| BERT4Rec | 2.4 | 78 | 12.3 | 10.42 |
| FEARec | 1.8 | 55 | 10.2 | 11.32 |
| **TV-Rec** | **0.9** | **28** | **5.2** | **12.35** |

### Key Findings

1. TV-Rec achieves state-of-the-art results on all 6 datasets, with an average improvement of 7.49%.
2. Completely removing self-attention yields even better performance, demonstrating that the time-variant filter is sufficiently expressive.
3. Parameter count and inference latency are significantly lower than attention-based methods.
4. The advantage is more pronounced on longer sequences, where time-variant filters better model long-range dependencies.

## Highlights & Insights

- **Theory-driven design**: The model is derived from graph signal processing theory rather than purely empirical approaches.
- **A viable replacement for attention**: Demonstrates that time-variant convolution can fully replace self-attention with greater efficiency.
- **Simplicity and efficiency**: The model features fewer parameters, faster inference, and strong performance, making it engineering-friendly.

## Limitations & Future Work

1. The design of the graph shift operator is currently relatively simple; richer graph structures warrant exploration.
2. The filter order $K$ requires hyperparameter tuning.
3. Integration of side information (e.g., item attributes, user profiles) has not been considered.
4. Performance in cold-start scenarios is not sufficiently discussed.

## Related Work & Insights

- **SASRec (Kang & McAuley, 2018)**: Pioneering work on self-attention-based sequential recommendation.
- **FMLP-Rec**: A fully MLP-based sequential recommendation model operating in the frequency domain.
- **FEARec**: A frequency-enhanced recommendation model.
- **BSARec**: A recommendation model based on bidirectional self-attention.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Theoretical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 4 |
| Overall Recommendation | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation](../../AAAI2026/recommender/wavelet_enhanced_adaptive_frequency_filter_for_sequential_re.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)
- [\[AAAI 2026\] HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation](../../AAAI2026/recommender/hymoerec_hybrid_mixture-of-experts_for_sequential_recommendation.md)
- [\[NeurIPS 2025\] MMPB: It's Time for Multi-Modal Personalization](mmpb_its_time_for_multi-modal_personalization.md)
- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](inference-time_reward_hacking_in_large_language_models.md)

</div>

<!-- RELATED:END -->
