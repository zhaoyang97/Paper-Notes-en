---
title: >-
  [Paper Note] CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation
description: >-
  [ICLR 2026][Recommender Systems][KV cache compression] By observing significant cross-user similarity (collaborative signals) in KV caches across different users in sequential recommendation, this paper proposes CollectiveKV, which decomposes KV into a low-dimensional user-specific component and a high-dimensional shared component retrieved from a global KV pool, achieving a compression ratio of 0.8% with no performance degradation.
tags:
  - ICLR 2026
  - Recommender Systems
  - KV cache compression
  - cross-user sharing
  - collaborative signals
  - sequential recommendation
  - SVD analysis
date: 2026-05-08
content_hash: 0884a393f09cbc3c
---

# CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation

**Conference**: ICLR 2026
**arXiv**: [2601.19178](https://arxiv.org/abs/2601.19178)
**Code**: To be confirmed
**Area**: Recommender Systems / Model Compression
**Keywords**: KV cache compression, cross-user sharing, collaborative signals, sequential recommendation, SVD analysis

## TL;DR
By observing significant cross-user similarity (collaborative signals) in KV caches across different users in sequential recommendation, this paper proposes CollectiveKV, which decomposes KV into a low-dimensional user-specific component and a high-dimensional shared component retrieved from a global KV pool, achieving a compression ratio of 0.8% with no performance degradation.

## Background & Motivation

**Background**: Sequential recommendation models (SIM, HSTU, etc.) adopt Transformer attention mechanisms to improve performance and introduce KV cache technology to precompute and cache K/V for reduced inference latency.

**Limitations of Prior Work**: Recommendation systems serve enormous user bases (hundreds of millions), each potentially with lengthy interaction histories, causing total KV cache volume to rapidly exceed GPU memory capacity and necessitating offloading to CPU/secondary storage, which introduces substantial transfer latency.

**Key Challenge**: KV compression methods from the LLM literature (e.g., token pruning, MLA dimensionality reduction) compress only single-user sequences and ignore the cross-user collaborative signals unique to recommendation scenarios.

**Goal**: Exploit cross-user KV similarity to achieve extreme compression—storing the majority of information in a globally shared pool while retaining only very low-dimensional personalized KV per user.

**Key Insight**: SVD decomposition of K/V reveals that principal components (>90% of information) exhibit strong cross-user correlation, while residuals (<10% of information) are user-specific—providing a quantitative basis for determining what can be shared.

**Core Idea**: A learnable global KV pool stores cross-user shared information; each user caches only low-dimensional personalized KV plus global indices, achieving an extreme compression ratio of 0.8%.

## Method

### Overall Architecture
The framework consists of two stages—prefill and decode. During prefill, the user sequence is linearly projected into low-dimensional user-specific KV (dimension $d_u$), while a router network computes and caches global KV indices. During decode, the cached indices are retrieved, high-dimensional shared KV (dimension $d_g$) is fetched from the GPU-resident global KV pool, and the concatenated result is used for attention computation.

### Key Designs

1. **KV Decomposition: User-Specific + Collective Shared**

   - **Function**: Decomposes KV into low-dimensional $\mathbf{K}_u \in \mathbb{R}^{n \times d_u}$ and high-dimensional $\mathbf{K}_c \in \mathbb{R}^{n \times d_g}$.
   - **Mechanism**: $\mathbf{K}_u = \mathbf{S} W_k + b_k$ (linear projection for dimensionality reduction); $\mathbf{K}_c[i] = P_k[\mathbf{I}_k[i]]$ (index-based retrieval from the global pool); final concatenation $\mathbf{K} = \text{concat}(\mathbf{K}_u, \mathbf{K}_c)$.
   - **Design Motivation**: SVD analysis demonstrates that principal components are shareable across users while residuals are personalized; the shared pool thus carries high-dimensional primary information while the low-dimensional projection retains user-specific characteristics.

2. **CollectiveKV Router**

   - **Function**: Maps sequence embeddings to global KV pool indices for each item.
   - **Mechanism**: $\mathbf{M} = \mathbf{S} W_r + b_r$; $\mathbf{I}_k[i] = \arg\max_j \mathbf{M}_{ij}$. During training, sigmoid gating ensures gradient propagation: $\mathbf{K}_c[i] = \sigma(\mathbf{M}[i, \mathbf{I}_k[i]]) \cdot P_k[\mathbf{I}_k[i]]$.
   - **Design Motivation**: Since argmax is non-differentiable, sigmoid gating combined with a peak loss ensures consistency between training and inference.

3. **Global KV Pool**

   - **Function**: $P_k, P_v \in \mathbb{R}^{m \times d_g}$ reside permanently in GPU memory and are shared across all users.
   - **Design Motivation**: Pool size $m$ is far smaller than the product of user count and sequence length, dramatically reducing storage; the high dimensionality $d_g$ preserves information capacity.

### Loss & Training
- Original recommendation loss + peak loss $\mathcal{L}_{\text{peak}} = -\frac{1}{n}\sum_i \log\sigma(\mathbf{M}[i, \mathbf{I}_k[i]])$ (ensuring sigmoid outputs approach 1).
- Load balance loss (KL divergence to encourage uniform selection of pool keys).
- End-to-end training with joint optimization of pool, router, and projection layers.

## Key Experimental Results

### Main Results (5 Models × 3 Datasets)

| Model | Dataset | GAUC (Base→+Ours) | AUC (Base→+Ours) | CR |
|-------|---------|-------------------|------------------|----|
| SIM | MicroVideo | 0.6954→**0.6973** | 0.6933→**0.7057** | **1.6%** |
| SDIM | MicroVideo | 0.6857→**0.6883** | 0.6749→**0.6871** | **1.2%** |
| SIM | KuaiVideo | 0.6577→**0.6604** | 0.6798→**0.6900** | **1.2%** |
| HSTU | MicroVideo | — | — | **0.8%** |

### Ablation Study

| Configuration | AUC | Notes |
|---------------|-----|-------|
| Full CollectiveKV | 0.7057 | Best |
| User-specific KV only | ~0.69 | Missing shared information |
| Collective KV only | ~0.69 | Missing personalization |
| Without peak loss | ~0.70 | Train-inference inconsistency |
| Without balance loss | ~0.70 | Low pool utilization |

### Key Findings
- **0.8% compression with no performance drop**: Results match or improve across 5 models × 3 datasets, indicating that shared KV provides a regularization/information-augmentation effect.
- SVD analysis offers an interpretable theoretical basis for compression—principal components exhibit strong cross-user correlation while residuals are user-specific.
- Inference latency is substantially reduced: secondary storage transfer volume shrinks by 50–100×, while in-GPU index lookup overhead is negligible.

## Highlights & Insights
- **Cross-user KV sharing is a compression dimension unique to recommendation systems**: LLM KV compression lacks this dimension (each inference serves a single sequence), whereas recommendation systems inherently possess collaborative signals—a neglected yet highly promising direction.
- **SVD decomposition provides a theoretical analysis tool for determining what can be shared**: The contrast in cross-user similarity between principal components and residuals is intuitive and compelling.
- **Sigmoid gating + peak loss in the router design** elegantly resolves the non-differentiability of discrete index selection.

## Limitations & Future Work
- The global KV pool resides permanently in GPU memory, constraining the feasible pool size $m$—how should $m$ be chosen in large-scale deployments?
- Validation is limited to CTR prediction tasks; applicability to ranking and generative recommendation remains unexplored.
- The router employs a simple linear layer; more sophisticated routing strategies may yield further improvements.

## Related Work & Insights
- **vs. MLA (DeepSeek)**: MLA reduces dimensionality within a single user's KV; CollectiveKV exploits cross-user sharing to achieve more extreme compression.
- **vs. Token pruning (Loki/Quest)**: Token pruning discards information; CollectiveKV does not discard but instead transfers information to the shared pool.
- **vs. HSTU**: HSTU introduces KV caching to recommendation without compression; CollectiveKV builds on this foundation to achieve 0.8% compression.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Cross-user KV sharing is an entirely new perspective, supported by theoretical SVD analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Broad coverage across 5 models × 3 datasets, though further ablation details are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ SVD analysis visualizations are clear and the overall logic is coherent.
- **Value**: ⭐⭐⭐⭐⭐ A 0.8% compression ratio carries substantial industrial deployment value.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](../../AAAI2026/recommender/exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)
- [\[AAAI 2026\] HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation](../../AAAI2026/recommender/hymoerec_hybrid_mixture-of-experts_for_sequential_recommendation.md)
- [\[AAAI 2026\] Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation](../../AAAI2026/recommender/wavelet_enhanced_adaptive_frequency_filter_for_sequential_re.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)
- [\[NeurIPS 2025\] TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation](../../NeurIPS2025/recommender/tv-rec_time-variant_convolutional_filter_for_sequential_recommendation.md)

<!-- RELATED:END -->
