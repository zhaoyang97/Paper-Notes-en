---
title: >-
  [Paper Note] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis
description: >-
  [AAAI 2026][Audio & Speech][multimodal sentiment analysis] Proposes the PaSE framework to explicitly address modality competition in multimodal sentiment analysis through a two-stage optimization strategy involving prototype-guided calibration alignment (Entropic Optimal Transport) and Shapley-value gradient modulation.
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "multimodal sentiment analysis"
  - "modality competition"
  - "prototype alignment"
  - "Shapley value"
  - "optimal transport"
  - "gradient modulation"
date: 2026-05-08
content_hash: ccf655a34787a221
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis

**Conference**: AAAI 2026  
**arXiv**: [2511.17585](https://arxiv.org/abs/2511.17585)  
**Code**: To be confirmed  
**Area**: Audio & Speech  
**Keywords**: multimodal sentiment analysis, modality competition, prototype alignment, Shapley value, optimal transport, gradient modulation

## TL;DR

Proposes the PaSE framework to explicitly address modality competition in multimodal sentiment analysis through a two-stage optimization strategy involving prototype-guided calibration alignment (Entropic Optimal Transport) and Shapley-value gradient modulation.

## Background & Motivation

- Multimodal sentiment analysis (MSA) integrates text, audio, and visual modalities, but often suffers from **modality competition** in practice, where dominant modalities suppress weaker ones, resulting in sub-optimal fusion performance.
- For example, on the CMU-MOSI dataset, adding audio/visual modalities to a text-only baseline yields limited gains or even negative returns.
- Existing methods mostly assume natural complementarity between modalities, lacking explicit modeling of modality competition dynamics.
- Existing gradient modulation methods (e.g., OGM-GE) rely on indirect signals such as gradient norms, lacking a principled quantification of each modality's contribution.

## Core Problem

How to explicitly quantify and balance the contributions of different modalities in multimodal sentiment analysis to alleviate the suppression of weaker modalities by the dominant one (typically text)?

## Method

### Overall Architecture

PaSE consists of three modules: **PCL** (Intra-modal Prototype Calibration Learning) $\to$ **CAL** (Cross-modal Alignment) $\to$ **Two-stage Optimization** (Prototype-Gated Fusion + Shapley-based Gradient Modulation).

### Key Design 1: Prototype-guided Calibration Learning (PCL)

Maintains a prototype vector for each category within each modality (momentum-updated, with $\gamma=0.98$):

$$c_k^m \leftarrow \gamma c_k^m + (1-\gamma) \frac{1}{|B_k|}\sum_{i \in B_k} h_i^m$$

Pulls intra-class samples closer while pushing inter-class samples away using a contrastive learning loss:

$$\mathcal{L}_{\text{intra}}^m = -\frac{1}{N}\sum_{i=1}^N \log \frac{e^{\phi(h_i^m, c_{y_i}^m)/\tau}}{\sum_{k=1}^K e^{\phi(h_i^m, c_k^m)/\tau}}$$

### Key Design 2: Cross-modal Alignment via Entropic Optimal Transport (CAL)

Treats the category prototypes of each modality as discrete distributions and solves for the cross-modal transport plan using Entropic Optimal Transport (OT) to minimize the Wasserstein distance. Introduces bidirectional symmetric matching loss and consistency regularization:

$$\mathcal{L}_{\text{match}} = \frac{1}{2}\left(\langle \mathbf{Q}^{(m \to n)}, \mathbf{C} \rangle_F + \langle \mathbf{Q}^{(n \to m)}, \mathbf{C}^\top \rangle_F\right)$$

$$\mathcal{L}_{\text{reg}} = \|\mathbf{Q}^{(m \to n)} - (\mathbf{Q}^{(n \to m)})^\top\|_F^2$$

### Key Design 3: Shapley-based Gradient Modulation (SGM)

Quantifies the marginal contribution of each modality using Shapley values:

$$\psi_m(u) = \sum_{S \subseteq \mathcal{M} \setminus \{m\}} \frac{|S|!(k-|S|-1)!}{k!}[u(S \cup \{m\}) - u(S)]$$

After normalization, the modulation factor $\varphi_m = \exp(\tilde{\psi}_{\min}/\tilde{\psi}_m - 1)$ is calculated, granting a larger learning rate to weaker modalities while suppressing dominant ones.

### Two-stage Training

- **Stage 1** (Warm-up): Employs entropy-guided weights and Prototype-Gated Fusion for normal training, allowing dominant modalities to guide the learning process.
- **Stage 2**: Automatically switches to SGM once the validation set entropy stabilizes, using Shapley-based gradient modulation to balance modality contributions.

## Key Experimental Results

| Method | MOSI Acc-2↑ | MOSI Acc-7↑ | MOSEI Acc-2↑ | MOSEI Corr↑ |
|------|------------|------------|-------------|------------|
| MSAmba (AAAI'25) | 85.99/87.43 | 49.67 | 85.78/86.86 | 0.796 |
| Semi-IIN (AAAI'25) | 85.28/87.04 | 46.50 | 84.98/87.70 | 0.804 |
| **PaSE** | **86.40/88.32** | **50.92** | **86.07/88.10** | **0.831** |

- F1 scores for four emotion categories on IEMOCAP: Happy 91.5, Sad 88.6, Angry 89.4, Neutral 73.2, achieving overall state-of-the-art results.
- Ablation: Removing SGM causes a 2.85% drop in MOSI Acc-2, showing the largest impact; removing CAL leads to a 1.52% decrease.
- All modalities vs. Best bi-modal combination: An average gain of 4.02%, effectively resolving the "modality addition degradation" issue.
- Comparison with GPT-4o-mini: PaSE (BERT-base) achieves 88.32 vs. 86.54 on MOSI, showing that lightweight models still maintain an advantage.

## Highlights & Insights

- For the first time, **Shapley value** is introduced into gradient modulation for multimodal sentiment analysis, providing a theoretically principled quantification of modality contributions.
- The combined bidirectional symmetric alignment and structure-preserving regularization of Entropic OT is more rigorous than a simple contrastive loss.
- The two-stage training strategy is well-designed: it allows dominant modalities to construct the representation structure first, and then balances with SGM, preventing instability caused by premature modulation.
- t-SNE visualizations show that the class separation of PaSE's fused representation is far superior to that of SelfMM/EUAR.

## Limitations & Future Work

- Computing Shapley values requires traversing all subsets of modalities ($2^3$ configurations for 3 modalities), which leads to exponential growth in computational cost as the number of modalities increases.
- Validations are limited to MOSI, MOSEI, and IEMOCAP, lacking evaluation on more challenging scenarios (e.g., sarcasm detection, multilingual settings).
- The feature extractors used are relatively dated (Facet, COVAREP), and have not been validated with stronger visual/audio backbones.
- The prototype update strategy is simplistic (momentum EMA), leaving more sophisticated prototype maintenance mechanisms unexplored.

## Related Work & Insights

| Dimension | PaSE | OGM-GE | PMR | ConFEDE |
|------|------|--------|-----|---------|
| Modality Contribution Quantification | Shapley value | Gradient Norm | Progressive Reinforcement | Contrastive Decomposition |
| Alignment Method | Entropic OT | None | None | Contrastive Learning |
| Fusion Strategy | Prototype-Gated | Simple Fusion | Three-way Attention | Shared-Private |
| Theoretical Guarantee | Game Theory | Heuristic | None | None |

## Insights

- Although expensive to compute, Shapley values are entirely feasible when the number of modalities is limited (3-5), making this approach generalizable to other multimodal tasks.
- The two-stage paradigm of "allowing the model to learn freely first, then introducing modulation-based balancing" is more stable than applying modulation throughout training.
- The approach of utilizing Entropic OT for prototype alignment can be applied to align embedding spaces of different modalities in VLMs.

## Rating

⭐⭐⭐⭐ — Clear theoretical motivation; the Shapley gradient modulation approach is novel and effective, but the experimental validation scenarios and backbones are somewhat conservative.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](../../CVPR2026/audio_speech/tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[AAAI 2026\] A Text-Routed Sparse Mixture-of-Experts Model with Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis](text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)

</div>

<!-- RELATED:END -->
