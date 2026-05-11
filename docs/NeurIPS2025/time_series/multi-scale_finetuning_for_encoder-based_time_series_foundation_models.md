---
title: >-
  [Paper Note] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models
description: >-
  [NeurIPS 2025][Time Series][Time series foundation models] This paper proposes MSFT (Multi-Scale FineTuning), which leverages causal analysis to reveal that naive fine-tuning suffers from scale confounding…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Time series foundation models"
  - "multi-scale modeling"
  - "fine-tuning"
  - "causal inference"
  - "parameter-efficient fine-tuning"
date: 2026-05-08
content_hash: f5aa7f188c11fd48
---

# Multi-Scale Finetuning for Encoder-based Time Series Foundation Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.14087](https://arxiv.org/abs/2506.14087)
**Code**: [https://github.com/zqiao11/MSFT](https://github.com/zqiao11/MSFT)
**Area**: Time Series / Foundation Model Fine-tuning
**Keywords**: Time series foundation models, multi-scale modeling, fine-tuning, causal inference, parameter-efficient fine-tuning

## TL;DR

This paper proposes MSFT (Multi-Scale FineTuning), which leverages causal analysis to reveal that naive fine-tuning suffers from scale confounding, and designs a multi-scale modeling framework for efficient fine-tuning of encoder-based time series foundation models, significantly outperforming both naive fine-tuning and from-scratch SOTA methods.

## Background & Motivation

Time series foundation models (TSFMs) demonstrate strong zero-shot forecasting performance, yet how to efficiently fine-tune them for downstream tasks remains an underexplored problem. Existing fine-tuning strategies (full fine-tuning, linear probing) suffer from two core issues:

**The multi-scale nature of time series is neglected**: Time series data exhibits different temporal patterns at different sampling scales. For example, hourly energy consumption reflects local usage patterns, while daily aggregation captures macro-level trends. Naive fine-tuning learns only at the original scale and is prone to overfitting to single-scale patterns.

**The multi-scale capability of TSFMs is underutilized**: TSFMs are pre-trained on multi-scale datasets and inherently possess multi-scale forecasting ability, yet naive fine-tuning restricts the model to learning at a single scale, wasting the multi-scale knowledge acquired during pre-training.

The authors analyze this from a causal inference perspective: scale $S$ acts as a confounder that simultaneously affects the input $X$ (temporal patterns at different scales) and the knowledge activated in the model $M$ (scale-specific pre-trained knowledge), introducing a backdoor path $X \leftarrow S \rightarrow M \rightarrow Y$ that induces spurious correlations. It is therefore necessary to eliminate the confounding effect via do-calculus intervention $P(Y|do(X))$.

## Method

### Overall Architecture

MSFT operates under a causal intervention framework, realizing backdoor adjustment as $P(Y|do(X)) = \sum_s P(Y|X,S=s,M=g(X,s))P(s)$. The pipeline proceeds as follows:
1. Downsample the original time series to generate multi-scale sequences (by factors of $2^k$)
2. Independently tokenize and encode each scale
3. Concatenate multi-scale token sequences and capture both within-scale and cross-scale dependencies via decoupled dependency modeling
4. Weighted aggregation of multi-scale predictions

### Key Designs

1. **Scale-specific knowledge activation**: Pre-trained parameters are frozen; independent linear adapters (input projection layers) and independent LoRA modules (attention layers) are introduced for each scale. This avoids interference caused by differing token resolutions across scales and realizes $M=g(X,s)$ in the formulation—activating scale-specific TSFM knowledge.

2. **Decoupled token dependency modeling**: Composed of two components:

   - **In-scale attention**: A mask $\mathbf{M}_{in}$ ensures that tokens attend only to tokens within the same scale, preventing spurious attention caused by temporal index misalignment across scales.
   - **Cross-scale aggregator**: Bidirectional (coarse-to-fine C2F and fine-to-coarse F2C) layer-wise fusion of adjacent-scale information. Tokens are mapped to a shared space via linear projection $\phi_{i,j}^l$ and then fused according to temporal alignment: C2F uses Repeat upsampling, and F2C uses AvgPool downsampling.

3. **Multi-scale mixing output**: Each scale produces an independent prediction $\hat{Y}_i$; the training objective is a weighted loss $\mathcal{L}_{pred} = \sum_i w_i \mathcal{L}_{pred,i}$, where weights $w_i$ are learned via softmax. At inference, predictions from each scale are upsampled to the original resolution and then combined by weighted summation, producing an ensemble effect that mitigates overfitting.

### Loss & Training

The original prediction loss of each TSFM (MSE or NLL) is used, weighted and summed according to the learned scale weights. Pre-trained parameters are frozen; only the adapters, LoRA modules, cross-scale aggregator, and scale weights are trained.

## Key Experimental Results

### Main Results (Long-term Forecasting, MSE, lower is better)

| Dataset | Metric | MSFT (Moirai-Base) | Full FT (Moirai-Base) | TimeMixer | SimpleTM | Gain |
|--------|------|------|----------|------|------|------|
| ETTm1 | MSE | **0.332** | 0.368 | 0.381 | 0.381 | −9.8% vs FT |
| ETTm2 | MSE | **0.247** | 0.258 | 0.275 | 0.275 | −4.3% vs FT |
| Weather | MSE | **0.213** | 0.232 | 0.240 | 0.243 | −8.2% vs FT |
| Electricity | MSE | **0.169** | 0.173 | 0.182 | 0.166 | −2.3% vs FT |

MSFT consistently outperforms naive fine-tuning, LoRA, AdaLoRA, and other parameter-efficient fine-tuning methods across three TSFM backbones (Moirai, Moment, UniTS).

### Ablation Study

| Configuration | ETTm1 MSE | Weather MSE | Note |
|------|---------|------|------|
| MSFT (full) | **0.332** | **0.213** | All components |
| w/o cross-scale aggregator | 0.340 | 0.220 | Cross-scale fusion is important |
| w/o scale-specific LoRA | 0.345 | 0.222 | Scale-specific knowledge activation is important |
| w/o multi-scale mixing | 0.338 | 0.218 | Weighted fusion is beneficial |
| Single scale (K=0) | 0.361 | 0.230 | Multi-scale modeling is the core |

### Key Findings
- Multi-scale modeling is the primary source of improvement; each component (scale-specific adapters, decoupled dependency modeling, weighted mixing) contributes independently.
- MSFT surpasses not only various fine-tuning baselines but also from-scratch SOTA deep learning methods (TimeMixer, SimpleTM, etc.).
- The framework is equally applicable to probabilistic forecasting tasks (leveraging Moirai's probabilistic forecasting capability).

## Highlights & Insights
- The causal analysis perspective provides a theoretically novel and compelling diagnosis of naive fine-tuning—scale acts as a confounder that introduces spurious correlations.
- The framework design is clean and general, compatible with different encoder-based TSFM architectures.
- Freezing pre-trained parameters and employing lightweight adapters yields high parameter efficiency while avoiding catastrophic forgetting.

## Limitations & Future Work
- The framework targets only encoder-based TSFMs; decoder-only (e.g., TimesFM) and encoder-decoder (e.g., Chronos) architectures are not explored.
- The number of scales $K$ requires manual selection; no adaptive mechanism is provided.
- The downsampling strategy is fixed as average pooling; alternative approaches (e.g., wavelet transforms) merit exploration.

## Related Work & Insights
- **vs TimeMixer**: TimeMixer trains a multi-scale model from scratch, whereas MSFT leverages the multi-scale capability of pre-trained TSFMs for fine-tuning—different approaches sharing the same objective.
- **vs LoRA**: Standard LoRA does not differentiate between scales; MSFT's scale-specific LoRA more effectively activates the corresponding pre-trained knowledge.
- **vs Scaleformer**: Scaleformer refines predictions iteratively from coarse to fine, while MSFT employs bidirectional fusion for greater flexibility.

## Rating
- Novelty: ⭐⭐⭐⭐ The causal perspective on fine-tuning is novel, though multi-scale modeling itself is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three backbones, multiple datasets, comparisons with diverse fine-tuning methods, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Logically clear, with causal analysis and method design organically integrated.
- Value: ⭐⭐⭐⭐ Provides a practical and theoretically grounded general framework for TSFM fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning](less_is_more_unlocking_specialization_of_time_series_foundation_models_via_struc.md)
- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)
- [\[NeurIPS 2025\] Synthetic Series-Symbol Data Generation for Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)

</div>

<!-- RELATED:END -->
