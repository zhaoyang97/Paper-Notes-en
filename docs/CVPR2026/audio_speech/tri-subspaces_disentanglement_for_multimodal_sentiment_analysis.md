---
title: >-
  [Paper Note] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis
description: >-
  [CVPR 2026][Audio & Speech][Multimodal Sentiment Analysis] This paper proposes the TSD framework, which explicitly decomposes multimodal features into three complementary subspaces—globally shared, pairwise shared, and modality-private—and adaptively integrates these three levels of information via a subspace-aware cross-attention (SACA) fusion module, achieving state-of-the-art performance on CMU-MOSI and CMU-MOSEI.
tags:
  - CVPR 2026
  - "Audio & Speech"
  - Multimodal Sentiment Analysis
  - Tri-Subspace Disentanglement
  - Cross-Attention Fusion
  - Pairwise Sharing
  - HSIC
date: 2026-05-08
content_hash: a9e449f196c17879
---

# Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis

**Conference**: CVPR 2026
**arXiv**: [2602.19585](https://arxiv.org/abs/2602.19585)
**Code**: Unavailable
**Area**: Speech/Audio
**Keywords**: Multimodal Sentiment Analysis, Tri-Subspace Disentanglement, Cross-Attention Fusion, Pairwise Sharing, HSIC

## TL;DR

This paper proposes the TSD framework, which explicitly decomposes multimodal features into three complementary subspaces—globally shared, pairwise shared, and modality-private—and adaptively integrates these three levels of information via a subspace-aware cross-attention (SACA) fusion module, achieving state-of-the-art performance on CMU-MOSI and CMU-MOSEI.

## Background & Motivation

Multimodal sentiment analysis integrates linguistic, visual, and acoustic modalities. Most existing methods adopt a binary shared-private decomposition (e.g., MISA), partitioning features into globally shared and modality-specific components. However, a substantial portion of affective cues in human communication are shared only between certain modality pairs—for instance, in sarcastic scenarios, tone and facial expression jointly convey negative sentiment while the verbal content appears positive. Such pairwise-shared signals are either ignored or misclassified under the binary decomposition paradigm.

## Method

### Overall Architecture

Feature encoding → Tri-subspace disentanglement (shared / pairwise-shared / private) → SACA fusion → Sentiment prediction.

### Key Designs

#### 1. Tri-Subspace Encoders

- **Shared encoder** $I(\cdot; \theta_c)$: Shared parameters across modalities; extracts modality-agnostic globally consistent features $\mathbf{C}_m$.
- **Pairwise-shared encoder** $S_{mn}(\cdot; \theta_{mn})$: Parameters shared within each modality pair; extracts pairwise interactive features $\mathbf{S}_{mn}^{(m)}$.
- **Private encoder** $P_m(\cdot; \theta_m)$: Independent parameters per modality; preserves modality-specific information $\mathbf{P}_m$.

Three modalities yield 9 subspace representations (3 shared + 3 pairwise-shared + 3 private).

#### 2. Disentanglement Supervisor

A three-branch discriminator predicts the true source subspace (shared / pairwise-shared / private) of each embedding, preventing information leakage:
$$\mathcal{L}_{sup} = -\frac{1}{M}\sum_m [\log D_{com}(\mathbf{c}_m) + \sum_{n \neq m}\log D_{sub}(\mathbf{s}_{mn}^{(m)}) + \log D_{pri}(\mathbf{p}_m)]$$

#### 3. Subspace Constraint Losses

- **Shared consistency loss**: $\mathcal{L}_{com} = \sum \|\mathbf{c}_m - \mathbf{c}_n\|_2^2$
- **Pairwise collaboration loss**: $\mathcal{L}_{pair} = \sum \|\mathbf{s}_{mn}^{(m)} - \mathbf{s}_{mn}^{(n)}\|_2^2$
- **Private independence loss** (HSIC): $\mathcal{L}_{pri} = \sum \text{HSIC}(\mathbf{p}_{m_1}, \mathbf{p}_{m_2})$
- **Orthogonality loss**: $\mathcal{L}_{ort} = \sum \|\mathbf{C}_m^\top \mathbf{P}_m\|_F^2 + ...$

#### 4. Subspace-Aware Cross-Attention Fusion (SACA)

For each subspace, a context set is constructed from the remaining subspaces, and multi-head cross-attention is applied for enhancement. A gating network then computes adaptive weights $\psi_k$ to produce a weighted sum: $\mathbf{Y}_{final} = \sum_k \psi_k \cdot F_{\mathcal{S}}^{(k)}$.

### Loss & Training

Overall loss: $\mathcal{L}_{all} = \mathcal{L}_{task} + \mathcal{L}_{TS}$

## Key Experimental Results

### Main Results

| Dataset | Metric | TSD | Prev. SOTA (EMOE) | Gain |
|--------|------|-----|--------------|------|
| CMU-MOSI | MAE ↓ | **0.691** | 0.697 | −0.9% |
| CMU-MOSI | ACC7 ↑ | **49.0%** | 47.8% | +2.5% |
| CMU-MOSI | ACC2 ↑ | **86.5%** | 85.4% | +1.3% |
| CMU-MOSEI | ACC7 ↑ | **54.9%** | 54.1% | +1.5% |
| CMU-MOSEI | ACC2 ↑ | **86.2%** | 85.5% | +0.8% |

### Ablation Study

| Configuration | MOSI MAE | Note |
|------|----------|------|
| w/o pairwise-shared | +0.015 | Pairwise-shared signals are important for sentiment inference |
| w/o disentanglement supervisor | +0.012 | The supervisor effectively prevents information leakage |
| w/o SACA | +0.018 | SACA fusion substantially outperforms simple concatenation |
| HSIC replaced by L2 | +0.008 | HSIC better enforces statistical independence |

### Key Findings

- The tri-subspace decomposition outperforms the binary shared-private baseline across all metrics, with small variance over 5 random seeds.
- The framework also demonstrates strong transferability on multimodal intent recognition tasks.

## Highlights & Insights

1. Explicit modeling of the pairwise-shared subspace fills a theoretical gap left by the shared-private binary decomposition.
2. The hierarchical fusion design of SACA allows each subspace to attend to information from other subspaces before determining fusion weights.
3. The disentanglement supervisor is a natural application of adversarial training, offering finer granularity than the modality discriminator used in MISA.

## Limitations & Future Work

1. With three modalities, the number of pairwise subspaces is 3; extending to more modalities leads to combinatorial explosion ($C_n^2$).
2. All loss weights $\lambda_{1\text{-}4}$ require careful tuning.
3. Temporal dynamics are not modeled, as sentiment analysis is performed at the utterance level only.

## Related Work & Insights

- Compared to MISA: extends the decomposition from 2 subspaces to 3 by introducing the pairwise-shared dimension.
- The HSIC independence constraint is generalizable to other multi-view learning tasks that require feature orthogonalization.

## Rating

- Novelty: ⭐⭐⭐⭐ — The tri-subspace decomposition has clear theoretical motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated on two datasets with intent recognition transfer and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical notation is rigorous and well-organized.
- Value: ⭐⭐⭐ — Performance gains are modest, but the research direction is sound.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](../../AAAI2026/audio_speech/psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)
- [\[AAAI 2026\] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis](../../AAAI2026/audio_speech/pase_prototype-aligned_calibration_and_shapley-based_equilibrium_for_multimodal_.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](../../AAAI2026/audio_speech/improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[AAAI 2026\] TEXT: Text-Routed Sparse Mixture of Experts for Multimodal Sentiment Analysis with Explanation Enhancement and Temporal Alignment](../../AAAI2026/audio_speech/text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)
- [\[CVPR 2026\] UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark](unim_a_unified_any-to-any_interleaved_multimodal_benchmark.md)

<!-- RELATED:END -->
