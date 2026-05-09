---
title: >-
  [Paper Note] Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision
description: >-
  [AAAI2026][Self-Supervised Learning][urban computing] This paper proposes UrbanLN, a framework that improves urban region representation learning from LLM-generated captions via a long-caption-aware positional encoding interpolation strategy and a dual-level (data and model) noise suppression mechanism.
tags:
  - AAAI2026
  - Self-Supervised Learning
  - urban computing
  - region representation
  - cross-modal pre-training
  - CLIP
  - noise suppression
  - self-distillation
date: 2026-05-08
content_hash: db7ef4538580710c
---

# Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision

**Conference**: AAAI2026
**arXiv**: [2511.07062](https://arxiv.org/abs/2511.07062)
**Code**: To be confirmed
**Area**: Self-Supervised Learning
**Keywords**: urban computing, region representation, cross-modal pre-training, CLIP, noise suppression, self-distillation

## TL;DR
This paper proposes UrbanLN, a framework that improves urban region representation learning from LLM-generated captions via a long-caption-aware positional encoding interpolation strategy and a dual-level (data and model) noise suppression mechanism.

## Background & Motivation
Urban region representation learning aims to extract meaningful features from unlabeled urban data for downstream tasks such as population estimation, GDP prediction, and carbon emission forecasting. Recent works (e.g., UrbanCLIP, UrbanVLP) have begun leveraging multimodal large language models (MLLMs) to generate textual descriptions of urban imagery, enhancing visual representations through image-text contrastive learning. However, existing methods face two fundamental bottlenecks:

1. **Semantic bottleneck in long-text processing**: MLLM-generated descriptions typically exceed 100 words, yet the CLIP text encoder has a token limit of 77. Direct truncation results in substantial loss of fine-grained semantic information.
2. **Noise-induced failure in knowledge integration**: MLLM-generated captions commonly contain hallucinations, information omissions, and over-generalizations. UrbanCLIP relies on manual curation (not scalable), while UrbanVLP uses fixed category templates (causing severe semantic loss).

## Core Problem
How can cross-modal pre-training on urban imagery (1) enable CLIP to effectively process long captions for capturing fine-grained urban semantics, and (2) simultaneously suppress noise in LLM-generated descriptions at both the data and model levels?

## Method

### Overall Architecture
UrbanLN consists of three core components: a multi-model collaborative high-quality caption generation pipeline, a cross-modal pre-training framework with long-text support and noise suppression, and a lightweight prediction head for downstream tasks.

### 1. Multi-Model Collaborative Caption Generation (Data-Level Noise Suppression)

#### Multi-MLLM Captioning
Multiple MLLMs (LLaMA-Adapter V2, ShareGPT4V-7B, Qwen2.5-VL-7B, DeepSeek-VL2-tiny, InternVL3-8B) independently generate long captions. The diversity across models mitigates semantic bias introduced by any single model and serves as a form of text data augmentation.

#### Divide-and-Conquer Refinement
- SAM is applied to segment images and extract crops of salient visual elements.
- Each salient region is described by an MLLM with a short local caption to supplement details potentially omitted in the long caption.
- A factual parser extracts visual element phrases from captions; OWLv2 then scores and filters out hallucinated phrases with scores below 0.01.
- The same MLLM regenerates a more complete caption based on both the original long caption and the local short captions.

#### Consensus-based Evaluation
Without ground-truth references, multi-model consensus is used as a proxy for caption quality. The CAPTURE metric measures similarity between any two candidate captions (via exact matching of objects, attributes, and relations, plus synonym and soft matching). The candidate with the highest average CAPTURE score against all others is selected as the final caption.

### 2. Information-Preserved Stretching Interpolation (IPSI)
To overcome CLIP's 77-token limitation, IPSI is proposed:
- The first 20 positional encodings are kept unchanged (these are well-trained and effectively capture absolute positional information).
- Only the remaining 57 positions are interpolated with ratio $\lambda=4$, extending the maximum input length from 77 to 248.
- Linear weighted interpolation ensures smooth transitions between positional encodings.

This strategy breaks the long-text processing bottleneck with negligible additional computational cost.

### 3. Momentum-based Self-Distillation (MSD, Model-Level Noise Suppression)
- A momentum version of the student model serves as the teacher (EMA update, momentum=0.995).
- Two dynamic queues (length 4096) store the most recent image and text representations encoded by the teacher.
- **Contrastive loss $\mathcal{L}_C$**: Standard image-text contrastive learning loss.
- **Distillation loss $\mathcal{L}_D$**: KL divergence between the student's similarity distribution and the teacher's pseudo-target distribution.
- Final loss: $\mathcal{L} = (1-\mu)\mathcal{L}_C + \mu\mathcal{L}_D$, where $\mu=0.5$.

The teacher's pseudo-targets provide supplementary supervision beyond the original image-text pairs, guiding the student toward noise-robust cross-modal representations.

## Key Experimental Results

### Datasets and Tasks
- Four cities: Beijing (BJ), Shanghai (SH), Shenzhen (SZ), New York (NY).
- Downstream tasks: population (Pop), GDP, nighttime light intensity (Night), restaurant review count (Com), carbon emissions (CO₂), POI count, crime rate (Crime).
- Evaluation metrics: R², RMSE, MAE.

### Main Results (BJ Dataset, R² Metric)

| Model | Pop | GDP | Night | Com | CO₂ |
|-------|-----|-----|-------|-----|-----|
| UrbanVLP | 0.619 | 0.372 | 0.454 | 0.555 | 0.487 |
| **UrbanLN+SV** | **0.705** | **0.440** | **0.514** | **0.591** | **0.677** |
| Gain | 13.9% | 18.3% | 13.2% | 6.5% | 39.0% |

Average improvements on BJ: R² +18.23%, RMSE −7.84%, MAE −8.32%.

### Highlights on NY Dataset
- UrbanLN+SV achieves the best results on population, crime, and POI prediction, with an average improvement of 30.97%.
- Crime prediction R² improves from 0.467 (UrbanVLP) to 0.723, a gain of 54.8%.

### Ablation Study
- Removing IPSI: average R² drops by 26.45%, making it the most critical component.
- Removing Refinement: average R² drops by 10.45%.
- Removing Consensus: random caption selection leads to performance degradation.
- Removing MSD: significant performance drop, confirming the importance of model-level noise suppression.

### Cross-City Transfer
In experiments where the model is pre-trained on a source city and evaluated on a target city, the model maintains high prediction accuracy, demonstrating that the learned representations encode generalizable urban semantic understanding.

## Highlights & Insights
1. **Elegant IPSI design**: By interpolating only the last 57 positional encodings while preserving the first 20, IPSI extends CLIP's input length by 3.2× at negligible cost—ablation results confirm it contributes the most.
2. **Dual-level noise suppression**: Data-level multi-model collaboration, divide-and-conquer refinement, and consensus evaluation are combined with model-level momentum self-distillation to systematically address LLM caption noise.
3. **Multi-model consensus as a substitute for manual annotation**: Caption quality can be assessed without ground-truth descriptions, making the approach practically scalable.
4. **Strong cross-city transfer**: The framework learns generalizable urban semantics rather than city-specific features.

## Limitations & Future Work
1. **Limited performance of satellite imagery on fine-grained tasks**: On the NY dataset, UrbanLN+SI underperforms street-view-based approaches on crime and POI prediction, an inherent limitation of satellite image resolution and viewpoint.
2. **High complexity of the caption generation pipeline**: The pipeline requires five MLLMs, SAM, a factual parser, and OWLv2; while treated as data preprocessing, deployment costs are non-trivial.
3. **Only ViT-B/16 backbone evaluated**: The effects of larger ViT variants or alternative visual encoders are not explored.
4. **Upper bound of positional encoding interpolation**: With $\lambda=4$, the maximum length is extended to 248 tokens, but handling captions exceeding this limit is not discussed.
5. **Relatively narrow set of downstream tasks**: Evaluation focuses primarily on regression tasks; more diverse evaluations such as classification, retrieval, and segmentation are lacking.

## Related Work & Insights

| Method | Text Source | Long-Text Handling | Noise Handling | Multimodal Fusion |
|--------|-------------|-------------------|----------------|-------------------|
| UrbanCLIP | Single MLLM | Truncated to 77 tokens | Manual curation | CLIP contrastive learning |
| UrbanVLP | Fixed template generation | Truncated to 77 tokens | Scene segmentation ratio guidance | Token-level contrastive learning |
| **UrbanLN** | **Multi-MLLM + divide-and-conquer refinement** | **IPSI extended to 248** | **Dual-level suppression** | **CLIP + momentum self-distillation** |

UrbanLN achieves substantive improvements over prior work in caption quality, long-text processing capacity, and noise robustness.

The following broader insights are worth noting:
1. **Generalizability of IPSI**: The "preserve first $N$, interpolate the rest" strategy can be extended to other scenarios requiring CLIP token limit expansion (e.g., vision-language alignment for medical reports or legal documents).
2. **Multi-model consensus as an unsupervised quality signal**: Cross-model consensus is a practical reference-free quality proxy in unannotated data evaluation settings.
3. **Momentum self-distillation for noise robustness**: Originating from ALBEF/MoCo, this mechanism is broadly applicable to other noisy supervision scenarios.

## Rating
- Novelty: ⭐⭐⭐ (IPSI and dual-level noise suppression are original contributions, though each component builds on prior work)
- Experimental Thoroughness: ⭐⭐⭐⭐ (four cities, seven tasks, complete ablation, cross-city transfer, latency analysis)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, well-motivated)
- Value: ⭐⭐⭐⭐ (strong practical utility in urban computing, though the application domain is relatively niche)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] BCE3S: Binary Cross-Entropy Based Tripartite Synergistic Learning for Long-tailed Recognition](bce3s_binary_cross-entropy_based_tripartite_synergistic_learning_for_long-tailed.md)
- [\[ICCV 2025\] Improving Large Vision and Language Models by Learning from a Panel of Peers](../../ICCV2025/self_supervised/improving_large_vision_and_language_models_by_learning_from_a_panel_of_peers.md)
- [\[NeurIPS 2025\] Long-Tailed Recognition via Information-Preservable Two-Stage Learning](../../NeurIPS2025/self_supervised/long-tailed_recognition_via_information-preservable_two-stage_learning.md)
- [\[AAAI 2026\] Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning](explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea.md)
- [\[CVPR 2026\] Representation Learning for Spatiotemporal Physical Systems](../../CVPR2026/self_supervised/representation_learning_for_spatiotemporal_physica.md)

</div>

<!-- RELATED:END -->
