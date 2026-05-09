---
title: >-
  [Paper Note] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis
description: >-
  [AAAI 2026][Audio & Speech][multimodal sentiment analysis] PaSE is a framework that explicitly addresses modality competition in multimodal sentiment analysis through a two-stage optimization strategy combining prototype-guided calibration alignment (via Entropic Optimal Transport) and Shapley-value-based gradient modulation.
tags:
  - AAAI 2026
  - "Audio & Speech"
  - multimodal sentiment analysis
  - modality competition
  - prototype alignment
  - Shapley value
  - optimal transport
  - gradient modulation
date: 2026-05-08
content_hash: c16537e57b9a61d7
---

# PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis

**Conference**: AAAI 2026  
**arXiv**: [2511.17585](https://arxiv.org/abs/2511.17585)  
**Code**: To be confirmed  
**Area**: Audio & Speech  
**Keywords**: multimodal sentiment analysis, modality competition, prototype alignment, Shapley value, optimal transport, gradient modulation

## TL;DR

PaSE is a framework that explicitly addresses modality competition in multimodal sentiment analysis through a two-stage optimization strategy combining prototype-guided calibration alignment (via Entropic Optimal Transport) and Shapley-value-based gradient modulation.

## Background & Motivation

- Multimodal Sentiment Analysis (MSA) fuses text, audio, and visual modalities, yet **modality competition** frequently occurs in practice: dominant modalities suppress weaker ones, causing fusion performance to fall short of expectations.
- For instance, on CMU-MOSI, incorporating audio/visual on top of a text-only baseline yields limited or even negative gains.
- Existing methods largely assume natural complementarity among modalities, lacking explicit modeling of modality competition dynamics.
- Current gradient modulation approaches (e.g., OGM-GE) rely on indirect signals such as gradient norms and lack principled quantification of individual modality contributions.

## Core Problem

How to explicitly quantify and balance the contributions of each modality in MSA, thereby alleviating the suppression of weaker modalities by dominant ones (typically text)?

## Method

### Overall Architecture

PaSE consists of three modules: **PCL** (intra-modal prototype calibration) → **CAL** (cross-modal alignment) → **two-stage optimization** (prototype-gated fusion + Shapley gradient modulation).

### Key Design 1: Prototype-guided Calibration Learning (PCL)

A prototype vector is maintained for each class within each modality via momentum updates ($\gamma=0.98$):

$$c_k^m \leftarrow \gamma c_k^m + (1-\gamma) \frac{1}{|B_k|}\sum_{i \in B_k} h_i^m$$

A contrastive learning loss pulls same-class samples closer and pushes apart different-class samples:

$$\mathcal{L}_{\text{intra}}^m = -\frac{1}{N}\sum_{i=1}^N \log \frac{e^{\phi(h_i^m, c_{y_i}^m)/\tau}}{\sum_{k=1}^K e^{\phi(h_i^m, c_k^m)/\tau}}$$

### Key Design 2: Cross-modal Alignment via Entropic Optimal Transport (CAL)

Class-level prototypes of each modality are treated as discrete distributions, and Entropic OT is applied to solve the cross-modal transport plan, minimizing the Wasserstein distance. A bidirectional symmetric matching loss and a consistency regularization term are introduced:

$$\mathcal{L}_{\text{match}} = \frac{1}{2}\left(\langle \mathbf{Q}^{(m \to n)}, \mathbf{C} \rangle_F + \langle \mathbf{Q}^{(n \to m)}, \mathbf{C}^\top \rangle_F\right)$$

$$\mathcal{L}_{\text{reg}} = \|\mathbf{Q}^{(m \to n)} - (\mathbf{Q}^{(n \to m)})^\top\|_F^2$$

### Key Design 3: Shapley-based Gradient Modulation (SGM)

Shapley values are employed to quantify the marginal contribution of each modality:

$$\psi_m(u) = \sum_{S \subseteq \mathcal{M} \setminus \{m\}} \frac{|S|!(k-|S|-1)!}{k!}[u(S \cup \{m\}) - u(S)]$$

After normalization, a modulation factor $\varphi_m = \exp(\tilde{\psi}_{\min}/\tilde{\psi}_m - 1)$ is computed, granting weaker modalities larger effective learning rates while suppressing dominant ones.

### Two-stage Training

- **Stage 1** (warm-up): Standard training with entropy-guided weights and Prototype-Gated Fusion, allowing the dominant modality to guide representation learning.
- **Stage 2**: Once validation-set entropy stabilizes, SGM is automatically activated to balance modality contributions via Shapley gradient modulation.

## Key Experimental Results

| Method | MOSI Acc-2↑ | MOSI Acc-7↑ | MOSEI Acc-2↑ | MOSEI Corr↑ |
|------|------------|------------|-------------|------------|
| MSAmba (AAAI'25) | 85.99/87.43 | 49.67 | 85.78/86.86 | 0.796 |
| Semi-IIN (AAAI'25) | 85.28/87.04 | 46.50 | 84.98/87.70 | 0.804 |
| **PaSE** | **86.40/88.32** | **50.92** | **86.07/88.10** | **0.831** |

- IEMOCAP four-class emotion F1: Happy 91.5, Sad 88.6, Angry 89.4, Neutral 73.2 — state-of-the-art across all categories.
- Ablation: removing SGM causes a 2.85% drop in MOSI Acc-2, the largest single-component impact; removing CAL causes a 1.52% drop.
- Full modality vs. best bimodal combination: average improvement of 4.02%, effectively resolving the "performance degradation upon adding modalities" problem.
- Comparison with GPT-4o-mini: PaSE (BERT-base) achieves 88.32 vs. 86.54 on MOSI, demonstrating that the lightweight model remains competitive.

## Highlights & Insights

- PaSE is the first to introduce **Shapley values** for gradient modulation in MSA, providing a theoretically principled quantification of modality contributions.
- The bidirectional symmetric alignment via Entropic OT combined with structural consistency regularization is more rigorous than simple contrastive loss.
- The two-stage training strategy is well-motivated: allowing the dominant modality to first establish representational structure before applying SGM avoids instability from premature modulation.
- t-SNE visualizations demonstrate that PaSE's fused representations achieve substantially better class separation than SelfMM/EUAR.

## Limitations & Future Work

- Computing Shapley values requires enumerating all modality subsets ($2^3$ combinations for three modalities); computational cost grows exponentially as the number of modalities increases.
- Validation is limited to MOSI/MOSEI/IEMOCAP; more challenging scenarios (e.g., sarcasm detection, multilingual settings) remain unexplored.
- Feature extractors are relatively dated (Facet, COVAREP); the framework has not been validated with stronger visual/audio backbones.
- The prototype update strategy is relatively simple (momentum EMA); more sophisticated prototype maintenance mechanisms have not been explored.

## Related Work & Insights

| Dimension | PaSE | OGM-GE | PMR | ConFEDE |
|------|------|--------|-----|---------|
| Modality contribution quantification | Shapley value | Gradient norm | Progressive reinforcement | Contrastive decomposition |
| Alignment method | Entropic OT | None | None | Contrastive learning |
| Fusion strategy | Prototype-Gated | Simple fusion | Tri-directional attention | Shared-private |
| Theoretical grounding | Game theory | Heuristic | None | None |

- Although Shapley value computation is costly, it is entirely feasible when the number of modalities is limited (3–5) and can be extended to other multimodal tasks.
- The two-stage paradigm of "letting the model learn freely first, then introducing modulation for balance" is more stable than applying modulation throughout training.
- The idea of using Entropic OT for prototype alignment is applicable to embedding space alignment across modalities in Vision-Language Models.

## Rating

⭐⭐⭐⭐ — Theoretical motivation is clear; the Shapley-based gradient modulation is novel and effective, though the experimental validation is limited in scope and relies on conservative backbone choices.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[AAAI 2026\] TEXT: Text-Routed Sparse Mixture of Experts for Multimodal Sentiment Analysis with Explanation Enhancement and Temporal Alignment](text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](../../CVPR2026/audio_speech/tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)

<!-- RELATED:END -->
