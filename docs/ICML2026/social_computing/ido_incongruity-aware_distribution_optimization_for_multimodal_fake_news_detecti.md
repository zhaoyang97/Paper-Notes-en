---
title: >-
  [Paper Note] IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection
description: >-
  [ICML 2026][Social Computing][Multimodal Fake News] IDO significantly improves F1 by 3-7% over SOTA on Weibo / Twitter / Fakeddit and enhances generalization to unseen fake news by **explicitly modeling inter-modal incon…
tags:
  - "ICML 2026"
  - "Social Computing"
  - "Multimodal Fake News"
  - "Inter-modal Incongruity"
  - "Distribution Optimization"
  - "Cross-modal Alignment"
date: 2026-05-08
content_hash: 5a332b09de85a458
---

# IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection

**Conference**: ICML 2026  
**arXiv**: [2605.29116](https://arxiv.org/abs/2605.29116)  
**Code**: To be confirmed  
**Area**: Social Computing / Multimodal Learning / Fake News Detection  
**Keywords**: Multimodal Fake News, Inter-modal Incongruity, Distribution Optimization, Cross-modal Alignment

## TL;DR
IDO significantly improves F1 by 3-7% over SOTA on Weibo / Twitter / Fakeddit and enhances generalization to unseen fake news by **explicitly modeling inter-modal incongruity** as a learnable distribution optimization objective—simultaneously pulling multimodal embeddings of real news closer and enlarging the incongruity of fake news.

## Background & Motivation

**Background**: Multimodal fake news detection utilizes joint signals from text and images to identify misinformation. Existing methods are mostly based on cross-modal fusion and binary classification—capturing modal information through contrastive learning or Graph Neural Networks.

**Limitations of Prior Work**: (1) Existing methods distinguish real and fake news as simple binary categories, lacking a precise characterization of "fake news features"; (2) Real and fake news exhibit different degrees of inter-modal incongruity (real news: high consistency; fake news: low consistency/incongruity), yet they are modeled identically; (3) Poor generalization to OOD fake news—novel types of fake news outside the training distribution are easily misclassified.

**Key Challenge**: The essential feature of fake news—**inter-modal semantic incongruity**—is not explicitly modeled, causing models to learn dataset-specific patterns rather than generalizable fake news features.

**Goal**: To treat inter-modal incongruity as an explicit optimization objective to improve the model's generalization capability toward unknown fake news.

**Key Insight**: It is observed that real news text-image pairs are highly consistent (matching descriptions), while fake news items are often inconsistent (images are irrelevant or contradict the text); a general discriminative signal can be obtained by strengthening this difference through distribution optimization.

**Core Idea**: Treat real news as a "high-consistency distribution" and fake news as a "low-consistency distribution"—simultaneously pulling real news consistency closer and pushing fake news incongruity further apart via **bidirectional distribution optimization**.

## Method

### Overall Architecture
(1) **Dual-stream encoding**: Text and images are processed via separate pre-trained encoders; (2) **Incongruity quantification**: Define cross-modal incongruity $d_{\text{incon}}(\mathbf{t}, \mathbf{v}) = 1 - \cos(\text{proj}_t(\mathbf{t}), \text{proj}_v(\mathbf{v}))$; (3) **Distribution optimization**: Real news $d \to 0$, fake news $d \to 1$; (4) **Joint training**: Combined classification loss and distribution optimization loss.

### Key Designs

1. **Learnable Quantification of Incongruity**:

    - Function: Defines a differentiable measure for cross-modal incongruity.
    - Mechanism: Heterogeneous modalities are mapped to an aligned space through shared semantic space projections $\text{proj}_t, \text{proj}_v$; the incongruity is defined as $d_{\text{incon}}(\mathbf{t}, \mathbf{v}) = 1 - \cos(\text{proj}_t(\mathbf{t}), \text{proj}_v(\mathbf{v}))$; to capture local incongruity, fine-grained patch alignment $d_{\text{local}} = \frac{1}{N} \sum_{i=1}^N \min_j d(\mathbf{t}_i, \mathbf{v}_j)$ is utilized, resulting in a final score $d = \alpha d_{\text{global}} + (1-\alpha) d_{\text{local}}$.
    - Design Motivation: A single global similarity score may overlook local incongruities (e.g., contradictions between image corners and text portions); fine-grained patch weighting captures incongruity more comprehensively.

2. **Bidirectional Distribution Optimization Loss**:

    - Function: Simultaneously optimizes the distributions of real news consistency and fake news incongruity.
    - Mechanism: For real news samples $(\mathbf{t}_r, \mathbf{v}_r)$, the loss is $\mathcal{L}_{\text{real}} = \mathbb{E}_{\text{real}}[d_{\text{incon}}(\mathbf{t}_r, \mathbf{v}_r)]$; for fake news samples $(\mathbf{t}_f, \mathbf{v}_f)$, the loss is $\mathcal{L}_{\text{fake}} = \max(0, m - \mathbb{E}_{\text{fake}}[d_{\text{incon}}(\mathbf{t}_f, \mathbf{v}_f)])$ with a margin $m = 0.7$; the total loss is $\mathcal{L}_{\text{IDO}} = \mathcal{L}_{\text{real}} + \lambda \mathcal{L}_{\text{fake}}$.
    - Design Motivation: Unidirectional losses (optimizing only one class) often lead to skewed classification boundaries; bidirectional distribution optimization maintains balance.

3. **Incongruity-Aware Classification Head**:

    - Function: Incorporates incongruity as an explicit feature input to the classifier to enhance discriminative signals.
    - Mechanism: The classifier input consists of $[\mathbf{t}; \mathbf{v}; d_{\text{global}}; d_{\text{local}}; d_{\text{global}} - d_{\text{local}}]$; an MLP outputs binary probabilities, and the model is trained jointly with cross-entropy loss.
    - Design Motivation: The classification head directly utilizes incongruity signals, while end-to-end joint optimization ensures that the distribution optimization objective aligns with the classification goal.

## Key Experimental Results

### Main Results

| Dataset | Method | Acc | F1 | AUC |
|--------|------|-----|-----|-----|
| Weibo | EANN | 78.2 | 76.5 | 84.3 |
| Weibo | MVAE | 81.7 | 80.4 | 87.6 |
| Weibo | MCAN | 84.5 | 83.7 | 90.2 |
| Weibo | **IDO** | **88.9** | **88.1** | **94.5** |
| Twitter | MCAN | 79.3 | 78.4 | 85.6 |
| Twitter | CAFE | 82.1 | 81.5 | 88.3 |
| Twitter | **IDO** | **87.6** | **86.8** | **92.7** |
| Fakeddit | MCAN | 76.5 | 75.2 | 83.4 |
| Fakeddit | CAFE | 79.7 | 78.9 | 86.5 |
| Fakeddit | **IDO** | **85.3** | **84.6** | **91.2** |

### OOD Generalization Test

| Train → Test | EANN F1 | MCAN F1 | **IDO F1** | Gain |
|------------|--------|--------|---------|------|
| Weibo → Twitter | 52.3 | 58.7 | **71.4** | +12.7 |
| Twitter → Fakeddit | 49.7 | 55.4 | **68.9** | +13.5 |
| Fakeddit → Weibo | 54.1 | 61.2 | **73.8** | +12.6 |

### Ablation Study

| Configuration | Weibo F1 | Twitter F1 |
|------|---------|-----------|
| Baseline (Classifier only) | 81.2 | 78.5 |
| + Global Incongruity | 85.7 | 83.4 |
| + Local Incongruity | 86.4 | 84.2 |
| + Bidirectional Optimization | 87.6 | 85.9 |
| **Full IDO** | **88.9** | **87.6** |

### Key Findings
- **Strong discriminative power of incongruity**: Visualizations show a clear separation in the distribution of incongruity between real and fake news.
- **Large improvements in OOD generalization**: Cross-dataset F1 gains of 12-14 percentage points verify that incongruity is a generalizable feature.
- **Local alignment complements global alignment**: Local incongruity successfully captures subtle contradictions between images and text.
- **Margin selection**: $m = 0.7$ is found to be optimal; smaller margins lead to insufficient separation, while larger margins cause overfitting.

## Highlights & Insights
- **Essential Feature Modeling**: Identifying inter-modal incongruity as an essential feature of fake news and explicitly optimizing for it.
- **Elegant Bidirectional Optimization**: Simultaneously pulling real consistency and pushing fake incongruity avoids the bias common in unidirectional losses.
- **Significant Cross-dataset Generalization**: Leading OOD performance validates that the model learns universal features rather than dataset-specific patterns.

## Limitations & Future Work
- Incongruity $\neq$ Fake News: High consistency does not guarantee veracity (e.g., sophisticated fake news with carefully matched manipulated content).
- Multimodal Expansion: Currently limited to text and image modalities.
- Incongruity Interpretability: There may be a semantic gap between the model-learned incongruity and human understanding.
- Future Work: Introduce additional modalities (audio, video); integrate with external knowledge bases for fact verification; develop explainable visualizations for incongruity.

## Related Work & Insights
- **vs EANN/MVAE**: These follow traditional fusion for classification without explicit modeling of incongruity.
- **vs MCAN**: While capturing alignment via cross-modal attention, it still follows binary classification; IDO explicitly optimizes the incongruity distribution.
- **vs CAFE**: Uses contrastive learning to pull real news and push fake news; IDO uses incongruity as a more precise discriminative signal.
- **Insight**: The bidirectional design of distribution optimization can be extended to other binary classification tasks such as sentiment analysis or fraud detection.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of incongruity modeling and bidirectional distribution optimization is novel, though some components are inspired by existing work.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive testing across 3 datasets, 4 baselines, OOD generalization, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation and precise description of methodologies.
- Value: ⭐⭐⭐⭐⭐ Fake news detection possesses significant social value; OOD generalization is a critical bottleneck for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection](../../ACL2026/social_computing/livefact_a_dynamic_time-aware_benchmark_for_llm-driven_fake_news_detection.md)
- [\[ICML 2026\] MIND: Multi-Rationale Integrated Discriminative Reasoning Framework for Multi-Modal Fake News](mind_multi-rationale_integrated_discriminative_reasoning_framework_for_multi-mod.md)
- [\[AAAI 2026\] FactGuard: Event-Centric and Commonsense-Guided Fake News Detection](../../AAAI2026/social_computing/factguard_event-centric_and_commonsense-guided_fake_news_detection.md)
- [\[ICML 2026\] ObjEmbed: Towards Universal Multimodal Object Embeddings](objembed_towards_universal_multimodal_object_embeddings.md)
- [\[CVPR 2026\] Bridging Pixels and Words: Mask-Aware Local Semantic Fusion for Multimodal Media Verification](../../CVPR2026/social_computing/bridging_pixels_and_words_mask-aware_local_semantic_fusion_for_multimodal_media_.md)

</div>

<!-- RELATED:END -->
