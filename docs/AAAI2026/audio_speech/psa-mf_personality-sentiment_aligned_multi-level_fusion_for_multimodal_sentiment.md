---
title: >-
  [Paper Note] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis
description: >-
  [AAAI2026][Audio & Speech][Multimodal Sentiment Analysis] This paper is the first to introduce a pre-trained personality model into Multimodal Sentiment Analysis (MSA) for extracting personalized sentiment features. Through personality-sentiment contrastive alignment and a progressive multi-level fusion architecture (pre-fusion → cross-modal interaction → enhanced fusion), the proposed PSA-MF achieves state-of-the-art performance on CMU-MOSI and CMU-MOSEI.
tags:
  - AAAI2026
  - "Audio & Speech"
  - Multimodal Sentiment Analysis
  - Personality-Sentiment Alignment
  - Multi-Level Fusion
  - Contrastive Learning
  - BERT
  - Personalized Sentiment
date: 2026-05-08
content_hash: 5e50d811123ca393
---

# PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis

**Conference**: AAAI2026  
**arXiv**: [2512.01442](https://arxiv.org/abs/2512.01442)  
**Authors**: Heng Xie, Kang Zhu, Zhengqi Wen, Jianhua Tao, Xuefei Liu, Ruibo Fu, Changsheng Li  
**Code**: Not released  
**Area**: Audio & Speech  
**Keywords**: Multimodal Sentiment Analysis, Personality-Sentiment Alignment, Multi-Level Fusion, Contrastive Learning, BERT, Personalized Sentiment  

## TL;DR

This paper is the first to introduce a pre-trained personality model into Multimodal Sentiment Analysis (MSA) for extracting personalized sentiment features. Through personality-sentiment contrastive alignment and a progressive multi-level fusion architecture (pre-fusion → cross-modal interaction → enhanced fusion), the proposed PSA-MF achieves state-of-the-art performance on CMU-MOSI and CMU-MOSEI.

## Background & Motivation

### Challenges in Multimodal Sentiment Analysis

With the explosive growth of social media video content, Multimodal Sentiment Analysis (MSA)—fusing textual, visual, and audio modalities for sentiment recognition—has become a critical technology in human-computer interaction, risk prediction, and related fields. Compared to unimodal analysis, MSA leverages cross-modal complementarity to improve accuracy and robustness, but faces two core challenges: how to extract effective sentiment representations from each modality, and how to effectively fuse sentiment information across modalities.

### Limitations of Prior Work

Existing MSA methods extract only shallow information during feature extraction, overlooking the significant differences in emotional expression attributable to individual personality traits. Research in psychology and affective computing has demonstrated a close relationship between Big Five personality traits and emotional expression—individuals with different personality profiles exhibit markedly different emotional responses and expression styles to the same situation. Nevertheless, existing methods do not incorporate personalized information during sentiment feature encoding. In the fusion stage, prior methods directly merge features from each modality without accounting for differences in feature granularity: text features extracted by pre-trained models carry far higher information density than pre-extracted visual and audio features, and naive same-level fusion fails to reconcile the differences in information density and semantic depth across modalities.

### Theoretical Basis for Personality–Emotion Interaction

Eladhari et al. proposed a personality-emotion mapping model; Zhang et al. demonstrated the mutual correlation between emotion and personality traits through multi-task learning; Mohammadi et al. found that personality plays an important role in emotion generation. These studies provide a theoretical basis for incorporating personality information into sentiment analysis. However, existing datasets typically contain only sentiment labels without personality annotations, and how to achieve personalized sentiment modeling under label-only conditions remains an open problem.

## Core Problem

How can multimodal sentiment analysis (1) leverage a pre-trained personality model to extract personalized sentiment features from text and align them with the sentiment space, and (2) design a progressive multi-level fusion architecture to gradually bridge the semantic gap across modalities, thereby improving sentiment recognition performance?

## Method

### Overall Architecture

PSA-MF consists of three major modules: (1) feature extraction and personality-sentiment alignment; (2) multimodal pre-fusion; and (3) cross-modal interaction and enhanced fusion.

### Unimodal Feature Extraction

- **Text**: The first $N$ layers of a fine-tuned BERT are used to extract sentiment embeddings $\text{CLS}_s$; a pre-trained Personality BERT extracts personality embeddings $\text{CLS}_p$.
- **Visual/Audio**: LSTM encodes pre-extracted facial action units (FACET, 35-dim) and acoustic features (COVAREP): $h_m = \text{LSTM}(X_m; \theta_{\text{LSTM}_m}), m \in \{v, a\}$

### Personality-Sentiment Alignment Module

Sentiment and personality features are linearly projected into a shared space and aligned via CLIP-style contrastive learning:

$$\mathcal{L}_{cl} = -\log \frac{\exp(\text{sim}(T_s^i, T_p^i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(T_s^i, T_p^j)/\tau)}$$

A composite contrastive loss incorporating cosine similarity weighting is defined as: $\mathcal{L}_{ccl} = \text{sim}(T_s^i, T_p^i) \cdot \mathcal{L}_{cl}$

**Personalized Sentiment Constraint Loss**—dynamically regulates alignment strength and constrains features to remain within the correct sentiment space:

$$\mathcal{L}_{ps} = (1 - \text{sim}(T_s^i, T_p^i)) \cdot \|W_y \cdot T_s^i - y_i\|_1$$

Total alignment loss: $\mathcal{L}_{\text{Align}} = \mathcal{L}_{ccl} + \mathcal{L}_{ps}$

### Multimodal Pre-Fusion

The remaining $(12-N)$ layers of BERT serve as a multimodal encoder; the text [CLS], visual, and audio features are concatenated and fed in for preliminary cross-modal alignment:

$$\text{CLS}_m = \text{BERT}_m([\text{CLS}_s, X_v, X_a]; \theta_{\text{BERT}_m})$$

A cross-modal contrastive loss $\mathcal{L}_{clm}$ is simultaneously applied to align the text representation with visual and audio representations.

### Cross-Modal Interaction and Enhanced Fusion

The pre-fused feature $M_s$ serves as the query and interacts with visual/audio features via multi-head attention: $V_t = \text{Att}_v(M_s, h_v, h_v)$, $A_t = \text{Att}_a(M_s, h_a, h_a)$

**Serial Fusion**: A linear layer merges the three cross-modal representations: $F_s = W_p \cdot [V_t', A_t', M_s']$

**Parallel Fusion**: A convolutional operation (kernel=3) compresses the stacked three-stream features: $F_p = \text{Conv}([V_t, A_t, M_s])$

Final prediction: $\hat{y} = \text{Sub}([F_s, F_p])$; total loss: $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Align}} + \mathcal{L}_{clm} + \|\hat{y}_i - y_i\|_1$

## Key Experimental Results

Datasets: CMU-MOSI (2,199 segments) and CMU-MOSEI (23,453 segments). Training performed on RTX 3090.

### Main Results (MOSI / MOSEI)

| Method | MOSI MAE↓ | MOSI Acc2↑ | MOSI F1↑ | MOSEI MAE↓ | MOSEI Acc2↑ | MOSEI F1↑ |
|--------|----------|-----------|---------|-----------|-----------|---------|
| TFN | 0.901 | 80.8 | 80.7 | 0.593 | 82.5 | 82.1 |
| MuLT | 0.871 | 83.0 | 82.8 | 0.580 | 82.5 | 82.3 |
| MISA | 0.783 | 83.4 | 83.6 | 0.555 | 85.5 | 85.3 |
| HyCon | 0.713 | 85.2 | 85.1 | 0.601 | 85.4 | 85.1 |
| FGTI | 0.702 | 85.8 | 85.8 | 0.536 | 86.0 | 86.0 |
| ULMD | 0.700 | 85.82 | 85.71 | 0.531 | 85.95 | 85.91 |
| **PSA-MF** | **0.686** | **86.43** | **86.19** | **0.521** | **86.30** | **86.28** |

PSA-MF improves Acc2 over FGTI by 0.63% and F1 by 0.39% on MOSI; on MOSEI, MAE is reduced by 1.9% relative to ULMD.

### Ablation Study (MOSI)

| Variant | MAE↓ | Corr↑ | Acc2↑ | F1↑ |
|---------|------|-------|-------|-----|
| w/o personality features | 0.711 | 0.795 | 84.60 | 84.47 |
| w/o BERT pre-fusion | 0.735 | 0.778 | 84.76 | 84.44 |
| w/o enhanced fusion | 0.806 | 0.788 | 85.21 | 84.96 |
| w/o $\mathcal{L}_{ps}$ | 0.724 | 0.784 | 83.99 | 83.92 |
| w/o $\mathcal{L}_{clm}$ | 0.754 | 0.784 | 85.06 | 84.92 |
| **PSA-MF** | **0.686** | **0.807** | **86.43** | **86.19** |

Removing the personalized sentiment constraint loss $\mathcal{L}_{ps}$ yields the largest performance drop (Acc2 −2.44%), underscoring its critical role in balancing personality alignment with sentiment classification.

## Highlights & Insights

- **First integration of a personality pre-trained model**: PSA-MF is the first MSA framework to incorporate Personality BERT for extracting personalized sentiment features, addressing the neglect of personality factors in conventional methods while remaining applicable to datasets with only sentiment labels.
- **Elegant personality-sentiment alignment design**: CLIP-style contrastive learning achieves semantic-level alignment, while the personalized sentiment constraint loss dynamically regulates alignment strength to prevent over-fitting to personality features at the expense of true sentiment.
- **Progressive multi-level fusion architecture**: The three-stage pipeline—pre-fusion (mitigating modality heterogeneity) → cross-modal interaction (personality-driven modality-specific reconstruction) → enhanced fusion (serial + parallel dual-stream)—progressively bridges the cross-modal semantic gap.
- **Insightful layer analysis**: A layer-by-layer alignment experiment across all 13 layers reveals that personality-sentiment alignment is optimal at the deeper pure-text layer (Layer 11) but degrades at the multimodal fusion layer, providing valuable design guidance.

## Limitations & Future Work

- **Outdated visual/audio features**: The method still relies on pre-extracted FACET (35-dim facial action units) and COVAREP features, without leveraging modern visual/audio pre-trained models such as VideoMAE or HuBERT, limiting representational capacity.
- **Limited dataset scale**: Evaluation is conducted only on CMU-MOSI and CMU-MOSEI, two classic but relatively small benchmarks; generalization to larger-scale or more recent MSA datasets remains unverified.
- **Generalizability of the personality model**: The Personality BERT used is trained on a YouTube-video Big Five dataset; whether its personality representations transfer to different cultural and linguistic contexts is unknown.
- **Architectural complexity**: The multi-level fusion design combined with multiple loss functions introduces numerous hyperparameters, potentially increasing training and tuning costs.

## Related Work & Insights

- **TFN/LMF** (tensor fusion): Perform only tensor-level modal fusion without considering personalized information or modality heterogeneity; PSA-MF outperforms TFN by a large margin on Acc2 (+5.6% on MOSI).
- **MuLT/MISA** (cross-attention): Leverage cross-modal attention for alignment but do not introduce personality information; PSA-MF surpasses MISA by 3.2% in Acc2 on MOSEI.
- **ULMD** (feature disentanglement): Decouples invariant/specific representations via modality separators but requires multiple encoders/decoders and complex constraints; PSA-MF adopts a simpler architecture and achieves 0.61% higher Acc2 on MOSI.
- **FGTI** (multi-granularity fusion): Enhances modality specificity through self-supervised learning but does not incorporate the personality dimension; PSA-MF achieves 0.63% higher Acc2 on MOSI.

### Broader Implications

- The personality-sentiment alignment paradigm can be generalized to other subjective evaluation tasks, such as user review analysis and mental health detection, by incorporating individual trait information into the feature space.
- The progressive multi-level fusion design pattern (pre-fusion → interaction → enhanced fusion) offers a general paradigm for handling multimodal heterogeneity.
- The personalized sentiment constraint loss—using $(1-\text{sim})$ as a dynamic weight—constitutes an elegant adaptive regularization technique.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to introduce a personality pre-trained model into MSA with a dedicated alignment mechanism; the contribution is clearly defined.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablation is comprehensive and the layer analysis is insightful; however, the datasets and visual/audio features are dated, and validation on more recent benchmarks is lacking.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with clear mathematical derivations and intuitive figures.
- Value: ⭐⭐⭐⭐ — Personalized sentiment analysis is an important research direction; incorporating the personality dimension into MSA opens a promising new avenue.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis](pase_prototype-aligned_calibration_and_shapley-based_equilibrium_for_multimodal_.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](../../CVPR2026/audio_speech/tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[AAAI 2026\] TEXT: Text-Routed Sparse Mixture of Experts for Multimodal Sentiment Analysis with Explanation Enhancement and Temporal Alignment](text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)

</div>

<!-- RELATED:END -->
