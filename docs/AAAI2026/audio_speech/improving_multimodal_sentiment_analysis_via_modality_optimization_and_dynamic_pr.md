---
title: >-
  [Paper Note] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection
description: >-
  [AAAI 2026][Audio & Speech][multimodal sentiment analysis] Proposes the MODS framework, which eliminates redundant non-verbal modalities via Graph-based Dynamic Compression (GDC), and designs a sample-level dynamic primary modality selector (MSelector) along with a primary-modality-centric cross-attention (PCCA) mechanism to achieve adaptive dominant modality selection per sample in MSA.
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
content_hash: 0695607b1cb57ad6
---

# Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection

**Conference**: AAAI 2026  
**arXiv**: [2511.06328](https://arxiv.org/abs/2511.06328)  
**Code**: To be confirmed  
**Area**: Audio and Speech  
**Keywords**: multimodal sentiment analysis, dynamic modality selection, graph convolutional network, capsule network, cross-modal attention, sequence compression  

## TL;DR

Proposes the MODS framework, which eliminates redundant non-verbal modalities via Graph-based Dynamic Compression (GDC), and designs a sample-level dynamic primary modality selector (MSelector) along with a primary-modality-centric cross-attention (PCCA) mechanism to achieve adaptive dominant modality selection per sample in MSA.

## Background & Motivation

- In MSA, different modalities contribute unequally to sentiment prediction, with language typically having the highest information density and acting as the default primary modality.
- Existing methods fix language as the primary modality (e.g., TCSP, ALMT), failing to adapt to individual samples where non-verbal modalities have stronger emotional discriminative power.
- Although HCT-DMG proposes dynamic selection, it only supports batch-level selection (due to asynchronous sequence constraints) and ignores the **sequence redundancy** of non-verbal modalities.
- Audio/visual sequences have a much lower information density than text, and directly using them as the primary modality would introduce noise interference.

## Core Problem

How to dynamically select the strongest modality as the primary modality at the sample level, while simultaneously addressing the feature quality issues caused by sequence redundancy in non-verbal modalities?

## Method

### Overall Architecture

MODS = **GDC** (Graph-based Dynamic Compression) + **MSelector** (Primary Modality Selector) + **PCCA** (Primary-modality-Centric Cross-Attention).

### Key Designs

#### Key Design 1: Graph-based Dynamic Compression (GDC)

A Capsule Network is used to compress the long audio/visual sequences into graph nodes of the same length as the text:

$$\text{Caps}_m^{i,j} = W_m^{ij} H_m^i$$

The routing coefficients $r_m^{i,j}$ are updated iteratively through dynamic routing, whereby noisy/redundant capsules automatically receive lower weights, generating high-quality nodes $N_m^j = \sum_i \text{Caps}_m^{i,j} \times r_m^{i,j}$.

Subsequently, self-attention is used to construct edge weights, and graph representations are learned via a GCN:

$$H_m^l = \text{ReLU}(D_m^{-1/2} E_m D_m^{-1/2} H_m^{l-1} W_m^l + b_m^l)$$

After compression, $H_a, H_v \in \mathbb{R}^{T_l \times d}$, aligned with the language sequence length.

#### Key Design 2: Primary Modality Selector (MSelector)

Attention-based aggregation is performed on each modality to obtain the vector $h_m$, which are then concatenated and passed through an MLP + softmax to output three weights:

$$w = \text{softmax}(\text{MLP}(\text{concat}(h_a, h_l, h_v))), \quad p = \arg\max(w_a, w_t, w_v)$$

The modality with the highest weight is selected as the primary modality $p$. The features of each modality are multiplied by their corresponding weights before being sent to subsequent modules. This achieves **sample-level** dynamic selection.

#### Key Design 3: Primary-modality-Centric Cross-Attention (PCCA)

Multi-layer iterative enhancement, where each layer contains:
1. Two cross-attentions $CA_{a \to p}$: auxiliary modality information flows to the primary modality.
2. One self-attention $SA_p$: self-enhancement of the primary modality.
3. Fusion: $H_p^{[i+1]} = H_{p_{update}}^{[i]} + \sum_{a} H_{a \to p}^{[i]}$
4. Reverse cross-attention $CA_{p \to a}$: the enhanced primary modality information is passed back to the auxiliary modalities.

The final layer retains only $CA_{a \to p}$, outputting $H_p$ for sentiment regression.

### Loss & Training

$$\mathcal{L}_{task} = \mathcal{L}_{reg} + \alpha \mathcal{L}_{NCE}$$

The InfoNCE loss backward predicts each single modality's features from the fused features to stabilize primary modality selection.

## Key Experimental Results

| Method | MOSI MAE↓ | MOSI Acc-7↑ | MOSI Acc-2↑ | MOSEI Acc-2↑ | SIMS Acc-5↑ |
|------|----------|------------|------------|-------------|------------|
| Self-MM | 0.708 | 46.67 | 83.44/85.46 | 83.76/85.15 | 41.53 |
| MMIM | 0.718 | 46.64 | 83.38/85.82 | 82.08/85.14 | - |
| DTN | 0.716 | 47.5 | -/85.1 | -/85.5 | 44.26 |
| **MODS** | **0.688** | **49.27** | **83.53/85.83** | **84.52/85.88** | **45.51** |

- Achieves comprehensive SOTA across 4 datasets (MOSI, MOSEI, SIMS, SIMSv2).
- SIMS Acc-5 reaches 45.51% (vs 44.26% for DTN), and SIMSv2 Acc-5 reaches 55.51% (vs 53.71% for DTN).
- Ablation: Removing GDC drops MOSI Acc-7 from 49.27 to 45.34 (-3.93); fixing any single modality as the primary modality drops performance by 3-4 points.
- Case study shows that language is selected when language is positive/audio-visual is negative, and non-verbal modalities are selected when language is neutral/audio-visual is positive.

## Highlights & Insights

- The first MSA method to achieve **sample-level** dynamic primary modality selection (rather than batch-level).
- The design of using a capsule network to construct graph nodes in GDC is clever: dynamic routing automatically filters out redundancy/noise.
- PCCA uses the primary modality as a bridge for information flow, preventing interference from direct interaction between auxiliary modalities.
- Significantly outperforms fixed primary modality methods even on modality-balanced datasets like SIMS/SIMSv2, validating the effectiveness of dynamic selection.

## Limitations & Future Work

- The argmax operation in MSelector is non-differentiable, relying on softmax weights as approximations during training, which may lead to insufficiently sharp selections.
- Validated only in 3-modality scenarios; the design of MSelector needs to be reconsidered when scaling to more modalities.
- GDC compresses audio/visual sequences to the same length as text; this rigid choice of length may not be the optimal compression ratio for all samples.
- Pre-trained multimodal backbones (such as CLIP, Whisper) were not explored, using only traditional feature extractors.

## Related Work & Insights

| Dimension | MODS | HCT-DMG | PaSE | ALMT |
|------|------|---------|------|------|
| Primary Modality Selection | Sample-level dynamic | Batch-level dynamic | None (Equal) | Fixed language |
| Sequence Compression | GDC (Capsule+GCN) | None | None | None |
| Fusion Method | PCCA (Primary-centric) | Hierarchical | Prototype gating | Text-centric attention |
| Core Problem | Modality selection + redundancy | Modality selection | Modality competition | Modality interaction |

## Insights

- Utilizing the dynamic routing of capsule networks for sequence compression is a noteworthy paradigm that preserves key information better than pooling.
- The "primary-modality-centric" fusion paradigm prevents noisy cross-propagation between weak modalities, which is particularly effective in scenarios with unequal information quality.
- Dynamic primary modality selection can be generalized to scenarios in multimodal LLMs where input modalities of varying quality need to be processed.

## Rating

⭐⭐⭐⭐ — The combined design of sample-level dynamic selection and graph compression is reasonable and effective, though the differentiability and scalability of core modules could be improved.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis](pase_prototype-aligned_calibration_and_shapley-based_equilibrium_for_multimodal_.md)
- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](../../CVPR2026/audio_speech/tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[AAAI 2026\] A Text-Routed Sparse Mixture-of-Experts Model with Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis](text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)
- [\[ACL 2025\] Improving Language and Modality Transfer in Translation by Character-level Modeling](../../ACL2025/audio_speech/improving_language_and_modality_transfer_in.md)

</div>

<!-- RELATED:END -->
