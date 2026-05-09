---
title: >-
  [Paper Note] AP-OOD: Attention Pooling for Out-of-Distribution Detection
description: >-
  [ICLR 2026][LLM/NLP][out-of-distribution detection] This paper proposes AP-OOD, which replaces the mean pooling in Mahalanobis distance-based OOD detection with learnable attention pooling, addressing the information loss caused by mean aggregation of token-level anomaly signals. On text OOD detection, AP-OOD reduces FPR95 on XSUM summarization from 27.84% to 4.67%, while supporting a smooth transition from unsupervised to semi-supervised settings.
tags:
  - ICLR 2026
  - LLM/NLP
  - out-of-distribution detection
  - attention pooling
  - Mahalanobis distance
  - token-level information
  - language models
date: 2026-05-08
content_hash: c230b6974c054b12
---

# AP-OOD: Attention Pooling for Out-of-Distribution Detection

**Conference**: ICLR 2026
**arXiv**: [2602.06031](https://arxiv.org/abs/2602.06031)
**Code**: [https://github.com/ml-jku/ap-ood](https://github.com/ml-jku/ap-ood)
**Area**: Text Generation
**Keywords**: out-of-distribution detection, attention pooling, Mahalanobis distance, token-level information, language models

## TL;DR
This paper proposes AP-OOD, which replaces the mean pooling in Mahalanobis distance-based OOD detection with learnable attention pooling, addressing the information loss caused by mean aggregation of token-level anomaly signals. On text OOD detection, AP-OOD reduces FPR95 on XSUM summarization from 27.84% to 4.67%, while supporting a smooth transition from unsupervised to semi-supervised settings.

## Background & Motivation

**State of the Field**: Language models deployed in production may encounter OOD inputs (e.g., trained on BBC article summarization but receiving CNN articles), leading to unreliable outputs such as hallucinations. Mahalanobis distance over token embeddings is the predominant detection approach.

**Limitations of Prior Work**: Existing methods (e.g., Ren et al. 2023) apply **mean pooling** over token embeddings before computing Mahalanobis distance — however, mean aggregation suppresses anomaly signals. When in-distribution (ID) and OOD sequences have similar mean embeddings but different token-level distributions, detection fails entirely. Figure 1 illustrates this failure mode.

**Root Cause**: Variable-length representations (token sequences) must be compressed into a scalar OOD score, yet naive aggregation discards token-level patterns critical for distinguishing ID from OOD inputs.

**Paper Goals**: Design an aggregation strategy beyond mean pooling that preserves token-level information for OOD detection.

**Starting Point**: Decompose the Mahalanobis distance into directional components, then replace mean projection along each direction with attention-weighted projection, allowing the model to learn which tokens are most informative for OOD detection.

**Core Idea**: Replace mean pooling with learnable attention pooling in the Mahalanobis distance computation, enabling OOD detection to exploit token-level information.

## Method

### Overall Architecture
AP-OOD extracts token embeddings $Z \in \mathbb{R}^{D \times S}$ from a pretrained encoder-decoder model, applies attention pooling with $M$ learnable query vectors $w_j$ to produce sequence-level representations, and computes the attention-pooled distance to corpus prototypes as the OOD score.

### Key Designs

1. **Attention-Pooled Mahalanobis Distance**

    - **Function**: Replace mean aggregation with an attention mechanism for pooling token embeddings.
    - **Mechanism**: The standard Mahalanobis distance admits a directional decomposition $d^2 = \sum_j (w_j^T \bar{z} - w_j^T \mu)^2$. Mean pooling $\bar{z} = \frac{1}{S}\sum_s z_s$ is replaced by attention pooling $\bar{z} = Z \cdot \text{softmax}(\beta Z^T w)$, so that each direction $w_j$ simultaneously defines the measurement direction and the token attention weights.
    - **Design Motivation**: Figures 1–2 demonstrate the contrast — mean pooling renders ID and OOD indistinguishable, whereas attention pooling can "see" token-level anomaly patterns.

2. **Multi-Query Multi-Head Extension**

    - **Function**: Replace the vector $w_j$ with a matrix $W_j \in \mathbb{R}^{D \times T}$, giving each head $T$ queries.
    - **Mechanism**: $\bar{Z} = Z \cdot \text{softmax}(\beta Z^T W)$ (softmax normalized over the full $S \times T$ matrix), with distances computed via the Frobenius inner product $\text{Tr}(W_j^T \bar{Z})$.
    - **Design Motivation**: Multiple queries capture richer token-level patterns, improving detection capability.

3. **Semi-Supervised Extension**

    - **Function**: Smoothly incorporate a small number of available OOD samples into training.
    - **Mechanism**: A distance-maximization term for OOD samples is added to the loss function, with a coefficient controlling the transition from unsupervised to supervised.
    - **Design Motivation**: In practice, some OOD types may be partially known; the method should be able to exploit such information.

### Loss & Training
- Only the query matrices $W_j$ are trained (very few parameters); the encoder is frozen.
- Loss function: $\mathcal{L} = \frac{1}{N}\sum_i d^2(Z_i, \tilde{Z}) - \sum_j \log(\|W_j\|^2)$
- Mini-batch attention pooling reduces memory consumption.
- Setting $\beta=0$ recovers the standard Mahalanobis distance (theoretical guarantee).

## Key Experimental Results

### Main Results

| Task | Metric | Prev. SOTA | Ours |
|------|--------|-----------|------|
| XSUM Summarization | FPR95↓ | 27.84% | **4.67%** |
| WMT15 En→Fr | FPR95↓ | 77.08% | **70.37%** |

The improvement in FPR95 is substantial, with a reduction of over 23 percentage points on XSUM.

### Ablation Study
- Setting $\beta=0$ (degenerating to Mahalanobis distance) significantly degrades performance, confirming the substantive contribution of attention pooling.
- Increasing the number of heads $M$ and queries $T$ consistently yields further gains.
- In the semi-supervised setting, even a small number of OOD samples further improves performance.

### Key Findings
- Mean pooling is the primary bottleneck in both summarization and translation tasks — the mean embeddings of OOD and ID sequences overlap heavily.
- The learned queries $w$ tend to focus on "anomalous" tokens in the sequence — those carrying the strongest OOD signal.
- The transition from unsupervised to semi-supervised is smooth, allowing the method to flexibly adapt to varying amounts of available OOD data.

## Highlights & Insights
- The formalization of the **"mean hides anomalies"** problem (Figures 1–2) is exceptionally intuitive — a picture is worth a thousand words.
- The theoretical framework unifying attention pooling with Mahalanobis distance is elegant: $\beta=0$ recovers the classical method, while $\beta>0$ generalizes to token-level aggregation.
- Only the query vectors are learned (negligible parameter count and computational overhead), making this a true post-hoc method.

## Limitations & Future Work
- Validation is limited to two tasks (summarization and translation); broader NLP settings (QA, dialogue, etc.) remain to be explored.
- The method relies on pretrained encoder-decoder architectures; applicability to decoder-only LLMs requires investigation.
- Only input-side OOD is addressed; distributional shift on the generation side is not considered.
- The attention temperature $\beta$ may require task-specific tuning.

## Related Work & Insights
- **vs. Ren et al. (2023)**: The baseline using mean pooling with Mahalanobis distance; AP-OOD directly replaces mean pooling with attention pooling.
- **vs. classifier-based OOD methods (MSP/Energy, etc.)**: These assume the presence of a classification head; AP-OOD is applicable to generative models.
- **vs. Mahalanobis distance (Lee et al. 2018)**: A classic image OOD method; AP-OOD extends it to sequential data.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of attention pooling and Mahalanobis distance is natural yet previously unexplored.
- **Experimental Thoroughness**: ⭐⭐⭐ Results on XSUM are exceptionally strong, but the experimental scope is narrow (2 tasks).
- **Writing Quality**: ⭐⭐⭐⭐⭐ The illustrative examples in Figures 1–2 are excellent; theoretical derivations are clear.
- **Value**: ⭐⭐⭐⭐ Provides a simple and effective improvement to NLP OOD detection.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Weak-to-Strong Generalization under Distribution Shifts](../../NeurIPS2025/llm_nlp/weak-to-strong_generalization_under_distribution_shifts.md)
- [\[CVPR 2026\] CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection](../../CVPR2026/llm_nlp/cops_conditional_prompt_synthesis_for_zero-shot_anomaly_detection.md)
- [\[AAAI 2026\] From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection](../../AAAI2026/llm_nlp/from_classification_to_ranking_enhancing_llm_reasoning_capabilities_for_mbti_per.md)
- [\[ICLR 2026\] Statistical Advantage of Softmax Attention: Insights from Single-Location Regression](statistical_advantage_of_softmax_attention_insights_from_single-location_regress.md)
- [\[AAAI 2026\] Vision Transformers are Circulant Attention Learners](../../AAAI2026/llm_nlp/vision_transformers_are_circulant_attention_learners.md)

<!-- RELATED:END -->
