---
title: >-
  [Paper Note] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection
description: >-
  [AAAI 2026][Audio & Speech][multimodal sentiment analysis] This paper proposes the MODS framework, which eliminates redundancy in non-linguistic modalities via Graph-based Dynamic Compression (GDC)…
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "multimodal sentiment analysis"
  - "dynamic modality selection"
  - "graph convolutional network"
  - "capsule network"
  - "cross-modal attention"
  - "sequence compression"
date: 2026-05-08
content_hash: 2652f235d63a08bc
---

# Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection

**Conference**: AAAI 2026
**arXiv**: [2511.06328](https://arxiv.org/abs/2511.06328)  
**Code**: To be confirmed  
**Area**: Audio & Speech
**Keywords**: multimodal sentiment analysis, dynamic modality selection, graph convolutional network, capsule network, cross-modal attention, sequence compression

## TL;DR

This paper proposes the MODS framework, which eliminates redundancy in non-linguistic modalities via Graph-based Dynamic Compression (GDC), and introduces a sample-level Dynamic Primary Modality Selector (MSelector) together with Primary-modality-Centric Cross-Attention (PCCA) to enable adaptive dominant modality selection on a per-sample basis for multimodal sentiment analysis (MSA).

## Background & Motivation

- In MSA, different modalities contribute unevenly to sentiment prediction; language typically carries the highest information density and serves as the default primary modality.
- Existing methods fix language as the primary modality (e.g., TCSP, ALMT), failing to accommodate samples where non-linguistic modalities are more discriminative.
- Although HCT-DMG proposes dynamic selection, it only supports batch-level selection (due to asynchronous sequence constraints) and neglects **sequence redundancy** in non-linguistic modalities.
- Audio and visual sequences have far lower information density than text; using them directly as the primary modality introduces noise.

## Core Problem

How can the strongest modality be dynamically selected as the primary modality at the sample level, while simultaneously addressing feature quality degradation caused by sequence redundancy in non-linguistic modalities?

## Method

### Overall Architecture

MODS = **GDC** (graph-based compression module) + **MSelector** (primary modality selector) + **PCCA** (primary-modality-centric cross-attention).

### Key Design 1: Graph-based Dynamic Compression (GDC)

A Capsule Network compresses long audio/visual sequences into graph nodes of the same length as the text sequence:

$$\text{Caps}_m^{i,j} = W_m^{ij} H_m^i$$

Dynamic routing iteratively updates routing coefficients $r_m^{i,j}$; noisy or redundant capsules automatically receive low weights, yielding high-quality nodes $N_m^j = \sum_i \text{Caps}_m^{i,j} \times r_m^{i,j}$.

Self-attention is then applied to construct edge weights, followed by a GCN to learn graph representations:

$$H_m^l = \text{ReLU}(D_m^{-1/2} E_m D_m^{-1/2} H_m^{l-1} W_m^l + b_m^l)$$

After compression, $H_a, H_v \in \mathbb{R}^{T_l \times d}$, aligned in length with the language sequence.

### Key Design 2: Primary Modality Selector (MSelector)

Attention-based aggregation is applied to each modality to obtain a vector $h_m$; the concatenated representation is passed through an MLP followed by softmax to produce three weights:

$$w = \text{softmax}(\text{MLP}(\text{concat}(h_a, h_l, h_v))), \quad p = \arg\max(w_a, w_t, w_v)$$

The modality with the highest weight is selected as the primary modality $p$; each modality's features are scaled by its corresponding weight before being forwarded to subsequent modules. This achieves **sample-level** dynamic selection.

### Key Design 3: Primary-modality-Centric Cross-Attention (PCCA)

Multi-layer iterative enhancement, where each layer consists of:
1. Two cross-attention operations $CA_{a \to p}$: auxiliary modality information flows toward the primary modality.
2. One self-attention $SA_p$: self-enhancement of the primary modality.
3. Fusion: $H_p^{[i+1]} = H_{p_{update}}^{[i]} + \sum_{a} H_{a \to p}^{[i]}$
4. Reverse cross-attention $CA_{p \to a}$: enhanced primary modality information is propagated back to the auxiliary modalities.

In the final layer, only $CA_{a \to p}$ is retained; the output $H_p$ is used for sentiment regression.

### Loss & Training

$$\mathcal{L}_{task} = \mathcal{L}_{reg} + \alpha \mathcal{L}_{NCE}$$

An InfoNCE loss reconstructs individual unimodal features from the fused representation, stabilizing primary modality selection.

## Key Experimental Results

| Method | MOSI MAE↓ | MOSI Acc-7↑ | MOSI Acc-2↑ | MOSEI Acc-2↑ | SIMS Acc-5↑ |
|--------|-----------|------------|------------|-------------|------------|
| Self-MM | 0.708 | 46.67 | 83.44/85.46 | 83.76/85.15 | 41.53 |
| MMIM | 0.718 | 46.64 | 83.38/85.82 | 82.08/85.14 | - |
| DTN | 0.716 | 47.5 | -/85.1 | -/85.5 | 44.26 |
| **MODS** | **0.688** | **49.27** | **83.53/85.83** | **84.52/85.88** | **45.51** |

- Achieves state-of-the-art performance across four benchmarks (MOSI, MOSEI, SIMS, SIMSv2).
- SIMS Acc-5: 45.51% (vs. DTN 44.26%); SIMSv2 Acc-5: 55.51% (vs. DTN 53.71%).
- Ablation: removing GDC drops MOSI Acc-7 from 49.27 to 45.34 (−3.93); fixing any single modality as the primary modality results in a 3–4 point decline.
- Case studies demonstrate that language is selected when it conveys positive sentiment while audio/visual signals are negative, and non-linguistic modalities are selected when language is neutral but audio/visual signals are positive.

## Highlights & Insights

- The first MSA method to achieve **sample-level** dynamic primary modality selection (as opposed to batch-level).
- The GDC design of constructing graph nodes via a capsule network is elegant: dynamic routing automatically filters redundancy and noise.
- PCCA uses the primary modality as a bridge for information flow, preventing noise propagation arising from direct interaction between auxiliary modalities.
- Significant improvements over fixed-primary-modality methods on modality-balanced datasets such as SIMS/SIMSv2 validate the value of dynamic selection.

## Limitations & Future Work

- The argmax operation in MSelector is non-differentiable; training relies on softmax weights as an approximation, which may yield insufficiently sharp selections.
- Validation is limited to three-modality scenarios; extending MSelector to more modalities requires redesign.
- GDC compresses audio/visual sequences to match the text length, which is a rigid choice and may not represent the optimal compression ratio for all samples.
- Pre-trained multimodal backbones (e.g., CLIP, Whisper) are not explored; only conventional feature extractors are employed.

## Related Work & Insights

| Dimension | MODS | HCT-DMG | PaSE | ALMT |
|-----------|------|---------|------|------|
| Primary Modality Selection | Sample-level dynamic | Batch-level dynamic | None (uniform) | Fixed (language) |
| Sequence Compression | GDC (Capsule + GCN) | None | None | None |
| Fusion Strategy | PCCA (primary-modality-centric) | Hierarchical | Prototype gating | Text-centric attention |
| Core Problem | Modality selection + redundancy | Modality selection | Modality competition | Modality interaction |

- Dynamic routing in capsule networks applied to sequence compression is a paradigm worth attention; it preserves salient information more effectively than pooling.
- The "primary-modality-centric" fusion paradigm avoids noisy cross-propagation between weak modalities and is particularly effective when input quality varies across modalities.
- Dynamic primary modality selection is transferable to multimodal LLM settings for handling input modalities of heterogeneous quality.

## Rating

⭐⭐⭐⭐ — The combination of sample-level dynamic selection and graph-based compression is well-motivated and effective, though the differentiability of the core module and its scalability warrant further investigation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis](pase_prototype-aligned_calibration_and_shapley-based_equilibrium_for_multimodal_.md)
- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](../../CVPR2026/audio_speech/tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[AAAI 2026\] TEXT: Text-Routed Sparse Mixture of Experts for Multimodal Sentiment Analysis with Explanation Enhancement and Temporal Alignment](text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)
- [\[ICML 2026\] Towards Understanding Modality Interaction in Multimodal Language Models via Partial Information Decomposition](../../ICML2026/audio_speech/towards_understanding_modality_interaction_in_multimodal_language_models_via_par.md)

</div>

<!-- RELATED:END -->
