---
title: >-
  [Paper Note] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis
description: >-
  [AAAI2026][Audio & Speech][Multimodal Sentiment Analysis] This work introduces a pre-trained personality model into Multimodal Sentiment Analysis (MSA) for the first time to extract personalized sentiment features. Through personality-sentiment contrastive learning alignment and a progressive multi-level (pre-fusion $\rightarrow$ cross-modal interaction $\rightarrow$ enhanced fusion) fusion architecture, it achieves SOTA performance on CMU-MOSI and CMU-MOSEI.
tags:
  - "AAAI2026"
  - "Audio & Speech"
  - "Multimodal Sentiment Analysis"
  - "Personality-Sentiment Alignment"
  - "Multi-Level Fusion"
  - "Contrastive Learning"
  - "BERT"
  - "Personalized Sentiment"
date: 2026-05-08
content_hash: a17d90520a86dbc3
---

# PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis

**Conference**: AAAI2026  
**arXiv**: [2512.01442](https://arxiv.org/abs/2512.01442)  
**Authors**: Heng Xie, Kang Zhu, Zhengqi Wen, Jianhua Tao, Xuefei Liu, Ruibo Fu, Changsheng Li  
**Code**: Not released  
**Area**: Audio & Speech  
**Keywords**: Multimodal Sentiment Analysis, Personality-Sentiment Alignment, Multi-Level Fusion, Contrastive Learning, BERT, Personalized Sentiment  

## TL;DR

This work introduces a pre-trained personality model into Multimodal Sentiment Analysis (MSA) for the first time to extract personalized sentiment features. Through personality-sentiment contrastive learning alignment and a progressive multi-level (pre-fusion $\rightarrow$ cross-modal interaction $\rightarrow$ enhanced fusion) fusion architecture, it achieves SOTA performance on CMU-MOSI and CMU-MOSEI.

## Background & Motivation

### Challenges of Multimodal Sentiment Analysis

With the explosive growth of social media video content, Multimodal Sentiment Analysis (MSA)—which integrates text, visual, and acoustic modality information for sentiment recognition—has become a key technology in fields like human-computer interaction and risk prediction. Compared to unimodal analysis, MSA improves sentiment recognition accuracy and stability through modality complementarity, but it also faces two core challenges: how to extract effective sentiment representations from each modality, and how to effectively fuse sentiment information from multiple modalities.

### Limitations of Prior Work

Existing MSA methods only extract shallow information during the feature extraction stage, ignoring the significant differences in sentiment expression caused by different personality traits. Psychology and affective computing research have shown a close correlation between personality traits (the Big Five) and sentiment expression—individuals with different personalities respond to and express sentiments differently in identical situations. However, existing methods do not incorporate personalized information when encoding sentiment features. In the fusion stage, prior methods directly merge features from each modality without considering hierarchical feature differences: text features extracted by pre-trained models have a much higher information density than pre-extracted visual and acoustic features, making simple same-level fusion difficult to reconcile the differences in information density and semantic depth across modalities.

### Theoretical Foundation of Personality-Sentiment Interaction

Eladhari et al. proposed a personality-sentiment mapping model, Zhang et al. demonstrated the correlation between sentiment and personality traits through multi-task learning, and Mohammadi et al. discovered that personality plays an important role in sentiment generation. These studies provide a theoretical foundation for introducing personality information into sentiment analysis. However, existing datasets usually contain only sentiment labels without personality annotations, making personalized sentiment modeling under the condition of having only sentiment labels an unsolved problem.

## Core Problem

How to (1) utilize a pre-trained personality model to extract personalized sentiment features from text and align them with the sentiment space, and (2) design a progressive multi-level fusion architecture to gradually alleviate the semantic gap across modalities, thereby improving multimodal sentiment analysis performance?

## Method

### Overall Architecture

PSA-MF consists of three main modules: (1) Feature extraction and personality-sentiment alignment; (2) Multimodal pre-fusion; and (3) Cross-modal interaction and enhanced fusion.

### Unimodal Feature Extraction

- **Text**: Uses the first $N$ layers of a fine-tuned BERT to extract the sentiment embedding $\text{CLS}_s$, and uses a pre-trained Personality BERT to extract the personality embedding $\text{CLS}_p$.
- **Visual/Audio**: Uses LSTMs to encode pre-extracted facial action units (FACET, 35-dimensional) and acoustic features (COVAREP): $h_m = \text{LSTM}(X_m; \theta_{\text{LSTM}_m}), m \in \{v, a\}$.

### Personality-Sentiment Alignment Module

After linearly mapping the sentiment and personality features into a shared space, they are aligned via CLIP-style contrastive learning:

$$\mathcal{L}_{cl} = -\log \frac{\exp(\text{sim}(T_s^i, T_p^i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(T_s^i, T_p^j)/\tau)}$$

The composite contrastive loss is weighted by individual cosine similarity: $\mathcal{L}_{ccl} = \text{sim}(T_s^i, T_p^i) \cdot \mathcal{L}_{cl}$.

**Personalized Sentiment Constraint Loss**—dynamically adjusts the alignment strength and bounds the features within the correct sentiment space:

$$\mathcal{L}_{ps} = (1 - \text{sim}(T_s^i, T_p^i)) \cdot \|W_y \cdot T_s^i - y_i\|_1$$

Total alignment loss: $\mathcal{L}_{\text{Align}} = \mathcal{L}_{ccl} + \mathcal{L}_{ps}$.

### Multimodal Pre-Fusion

The remaining $(12-N)$ layers of BERT are used as a multimodal encoder, where the text [CLS], visual, and acoustic features are concatenated and input for preliminary cross-modal alignment:

$$\text{CLS}_m = \text{BERT}_m([\text{CLS}_s, X_v, X_a]; \theta_{\text{BERT}_m})$$

Meanwhile, a cross-modal contrastive learning loss $\mathcal{L}_{clm}$ is applied to align the text with the visual/audio representations.

### Cross-Modal Interaction & Enhanced Fusion

The pre-fused feature $M_s$ acts as the query, interacting with visual/acoustic features via multi-head attention: $V_t = \text{Att}_v(M_s, h_v, h_v)$, $A_t = \text{Att}_a(M_s, h_a, h_a)$.

**Serial Fusion**: A linear layer merges the three cross-modal representations: $F_s = W_p \cdot [V_t', A_t', M_s']$.

**Parallel Fusion**: A convolution operation (kernel size = 3) compresses the three stacked features: $F_p = \text{Conv}([V_t, A_t, M_s])$.

Final prediction: $\hat{y} = \text{Sub}([F_s, F_p])$, and the total loss: $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Align}} + \mathcal{L}_{clm} + \|\hat{y}_i - y_i\|_1$.

## Key Experimental Results

Datasets: CMU-MOSI (2,199 segments) and CMU-MOSEI (23,453 segments). Trained on RTX 3090.

### Main Results (MOSI / MOSEI)

| Method | MOSI MAE↓ | MOSI Acc2↑ | MOSI F1↑ | MOSEI MAE↓ | MOSEI Acc2↑ | MOSEI F1↑ |
|------|----------|-----------|---------|-----------|-----------|---------|
| TFN | 0.901 | 80.8 | 80.7 | 0.593 | 82.5 | 82.1 |
| MuLT | 0.871 | 83.0 | 82.8 | 0.580 | 82.5 | 82.3 |
| MISA | 0.783 | 83.4 | 83.6 | 0.555 | 85.5 | 85.3 |
| HyCon | 0.713 | 85.2 | 85.1 | 0.601 | 85.4 | 85.1 |
| FGTI | 0.702 | 85.8 | 85.8 | 0.536 | 86.0 | 86.0 |
| ULMD | 0.700 | 85.82 | 85.71 | 0.531 | 85.95 | 85.91 |
| **PSA-MF** | **0.686** | **86.43** | **86.19** | **0.521** | **86.30** | **86.28** |

On MOSI, PSA-MF improves Acc2 by 0.63% and F1 by 0.39% compared to FGTI; on MOSEI, it reduces MAE by 1.9% compared to ULMD.

### Ablation Study (MOSI)

| Variant | MAE↓ | Corr↑ | Acc2↑ | F1↑ |
|------|------|-------|-------|-----|
| w/o Personality Features | 0.711 | 0.795 | 84.60 | 84.47 |
| w/o BERT Pre-fusion | 0.735 | 0.778 | 84.76 | 84.44 |
| w/o Enhanced Fusion | 0.806 | 0.788 | 85.21 | 84.96 |
| w/o $\mathcal{L}_{ps}$ | 0.724 | 0.784 | 83.99 | 83.92 |
| w/o $\mathcal{L}_{clm}$ | 0.754 | 0.784 | 85.06 | 84.92 |
| **PSA-MF** | **0.686** | **0.807** | **86.43** | **86.19** |

Removing the personalized sentiment constraint loss $\mathcal{L}_{ps}$ has the greatest impact (Acc2 drops by 2.44%), indicating its critical role in balancing personality alignment and sentiment classification.

## Highlights & Insights

- **First to Introduce Pre-trained Personality Models**: This work integrates Personality BERT into MSA for the first time to extract personalized sentiment features, addressing the limitation of prior methods that ignore personality factors, whilst remaining applicable to datasets with only sentiment labels.
- **Exquisite Personality-Sentiment Alignment Design**: Utilizing CLIP-style contrastive learning achieves semantic-level alignment, while the personalized sentiment constraint loss dynamically adjusts the alignment intensity to prevent overfitting to personality features and deviating from the ground-truth sentiment.
- **Multi-level Progressive Fusion Architecture**: Moving from pre-fusion (mitigating modality heterogeneity) $\rightarrow$ cross-modal interaction (personality-driven reconstruction of modality-specific details) $\rightarrow$ enhanced fusion (serial + parallel dual-stream), the method gradually bridges the semantic gap across modalities.
- **In-depth Layer Analysis Experiments**: Layer-by-layer alignment analysis across 13 layers reveals that the personality-sentiment alignment performs best in the deep text layers (Layer 11) and declines in the multimodal fusion layers, providing valuable design insights.

## Limitations & Future Work

- **Outdated Visual/Audio Features**: The model still relies on FACET (35-dimensional facial action units) and COVAREP pre-extracted features, without exploiting modern visual/acoustic pre-trained models (e.g., VideoMAE, HuBERT), which limits the expressive capability of these features.
- **Limited Dataset Scale**: Evaluation is restricted to CMU-MOSI and CMU-MOSEI, two classic but relatively small datasets, and has not yet been validated on larger-scale or newer MSA datasets.
- **Generalizability of the Personality Model**: The Personality BERT utilized is trained on a YouTube Big Five dataset; whether its personality representations generalize to different cultural and linguistic contexts remains unknown.
- **Complex Fusion Architecture**: The design of multi-level fusion and multiple loss functions introduces a large number of hyperparameters, which may lead to high training and tuning costs.

## Related Work & Insights

- **TFN/LMF** (Tensor Fusion): Perform only tensor-level fusion without considering personalized information or modality heterogeneity. Ours significantly outperforms TFN in Acc2 (+5.6% on MOSI).
- **MuLT/MISA** (Cross-Attention): Align modalities via cross-modal attention but do not incorporate personality information. Ours achieves a 3.2% Gain in Acc2 on MOSEI compared to MISA.
- **ULMD** (Feature Decoupling): Decouples invariant and modality-specific representations via modality separators but requires designing multiple encoders/decoders and complex constraints. Our architecture is cleaner and achieves 0.61% higher Acc2 on MOSI.
- **FGTI** (Multi-Granularity Fusion): Enhances modality specificity via self-supervised learning but lacks the personality dimension. Ours yields a 0.63% Gain in Acc2 on MOSI.

## Inspirations & Connections

- The idea of personality-sentiment alignment can be generalized to other subjective assessment tasks, such as user review analysis and mental health detection, by introducing individual trait information into the feature space.
- The design pattern of multi-level progressive fusion (pre-fusion $\rightarrow$ interaction $\rightarrow$ enhancement) offers a general sentiment paradigm for addressing multimodal heterogeneity.
- The design of the personalized sentiment constraint loss—using $(1-\text{sim})$ for dynamic weighting—presents an elegant adaptive regularization technique.

## Rating

- Novelty: ⭐⭐⭐⭐ — Introduces a pre-trained personality model into MSA for the first time and designs an alignment mechanism with clear innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Sufficient ablation studies, insightful layer analysis, but the datasets and visual/audio features are somewhat outdated, lacking validation on the latest datasets.
- Writing Quality: ⭐⭐⭐⭐ — Complete structure, clear formula derivations, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Personalized sentiment analysis is an important direction, opening a new path by introducing the personality dimension to MSA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis](pase_prototype-aligned_calibration_and_shapley-based_equilibrium_for_multimodal_.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](../../CVPR2026/audio_speech/tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[AAAI 2026\] Improving Multimodal Sentiment Analysis via Modality Optimization and Dynamic Primary Modality Selection](improving_multimodal_sentiment_analysis_via_modality_optimization_and_dynamic_pr.md)
- [\[AAAI 2026\] A Text-Routed Sparse Mixture-of-Experts Model with Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis](text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)
- [\[AAAI 2026\] MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement](mf-speech_achieving_fine-grained_and_compositional_control_in_speech_generation_.md)

</div>

<!-- RELATED:END -->
