---
title: >-
  [Paper Note] VoT: Event-Driven Reasoning and Multi-Level Alignment Unlock the Value of Text for Time Series Forecasting
description: >-
  [ICLR 2026][Time Series][Multimodal time series forecasting] This paper proposes VoT, a multimodal time series forecasting method that fully exploits the value of textual information through event-driven reasoning (lever…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Multimodal time series forecasting"
  - "event-driven reasoning"
  - "text alignment"
  - "adaptive frequency fusion"
  - "LLM reasoning"
date: 2026-05-08
content_hash: 06990da92306ca90
---

# VoT: Event-Driven Reasoning and Multi-Level Alignment Unlock the Value of Text for Time Series Forecasting

**Conference**: ICLR 2026
**arXiv**: [2603.15452](https://arxiv.org/abs/2603.15452)  
**Area**: Time Series
**Keywords**: Multimodal time series forecasting, event-driven reasoning, text alignment, adaptive frequency fusion, LLM reasoning

## TL;DR

This paper proposes VoT, a multimodal time series forecasting method that fully exploits the value of textual information through event-driven reasoning (leveraging LLMs to perform structured reasoning over exogenous text for numerical prediction) and multi-level alignment (representation-level endogenous text alignment + prediction-level adaptive frequency fusion). VoT comprehensively outperforms existing methods on real-world datasets spanning 10 domains.

## Background & Motivation

Most existing time series forecasting methods rely solely on numerical data. However, abrupt real-world events — such as the 2008 financial crisis or the COVID-19-induced spike in unemployment in 2020 — are difficult to predict from historical numerical patterns alone. Textual information can provide event-driven forecasting guidance, yet existing multimodal methods face two key challenges:

### Challenge 1: Insufficient Utilization of Text

| Method Type | Issue |
|-------------|-------|
| Endogenous text methods (e.g., LLM-Mixer) | Use statistical summaries/data descriptions that largely overlap with time series information |
| Exogenous text methods (e.g., CMIN, DualTime) | Perform only representation-level fusion, failing to mine deep semantics |
| **VoT (Ours)** | **Simultaneously employs LLM reasoning + feature extraction, supporting both endogenous and exogenous text** |

### Challenge 2: Difficulty in Modality Alignment

Text describes event-driven abrupt changes, while time series captures subtle numerical fluctuations — a significant modality gap exists between the two, making naive fusion insufficient for achieving complementarity.

## Method

### Overall Architecture: Dual-Branch Design

VoT comprises two complementary branches:
- **Event-driven prediction branch**: Generates numerical predictions from exogenous text via LLM reasoning
- **Numerical prediction branch**: Aligns endogenous text with time series representations before generating predictions

The outputs of both branches are integrated through adaptive frequency fusion.

### Key Design 1: Event-Driven Reasoning

**Three-step generation pipeline**:

1. **Template generation**: An LLM generates a structured template $\mathcal{D}$ based on data descriptions and samples
2. **Summary generation**: The template is used to extract prediction-relevant summaries $\mathcal{S}_i$ from raw exogenous text
3. **Reasoning-based prediction**: A Reasoner (reasoning-capable LLM) generates numerical predictions based on the summary and historical time series:

$$\hat{\mathbf{Y}}^{\text{event}}_i, \mathcal{R}_i = \text{Reasoner}(\mathcal{P}_{\text{reason}}, \mathcal{S}_i, \mathbf{X}_i)$$

**Historical In-Context Learning (HIC)**:
- **Training phase**: The Reasoner first generates predictions, then produces corrective reasoning $\mathcal{C}_i$ against ground-truth values, building a knowledge base $\mathcal{K} = \{(\text{Embed}(\mathcal{S}_i), \mathcal{C}_i)\}_{i=1}^M$
- **Inference phase**: The corrective reasoning from the most similar historical sample is retrieved as an ICL exemplar:

$$\hat{\mathbf{Y}}^{\text{event}}_j = \text{Reasoner}(\mathcal{P}_{\text{ICL}}, \mathcal{C}_{\tilde{i}}, \mathcal{S}_j, \mathbf{X}_j)$$

### Key Design 2: Multi-Level Alignment

**Representation level — Endogenous Text Alignment (ETA)**:
- Two sets of learnable queries $\mathbf{Q}^{\text{tr}}, \mathbf{Q}^{\text{se}}$ extract trend and seasonal semantics from text representations
- Cross-attention is used to achieve time series–text alignment
- **Decomposed contrastive learning** aligns trend and seasonal components at the sample level

**Prediction level — Adaptive Frequency Fusion (AFF)**:

Predictions from both branches are decomposed into low/mid/high-frequency components via FFT:

$$\mathcal{F}^{\text{num}} = \text{FFT}(\hat{\mathbf{Y}}^{\text{num}}), \quad \mathcal{F}^{\text{event}} = \text{FFT}(\hat{\mathbf{Y}}^{\text{event}})$$

Learnable weights $w_*^b$ adaptively fuse components across frequency bands:

$$\mathcal{F}_{\text{fused}} = \sum_* \sum_b w_*^b \mathcal{F}_*^b, \quad \hat{\mathbf{Y}}_{\text{final}} = \text{iFFT}(\mathcal{F}_{\text{fused}})$$

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ts}} + \mathcal{L}_{\text{align}} + \mathcal{L}_{\text{final}}$$

## Key Experimental Results

### Main Results: Comparison with Time Series and Multimodal Methods (Average over 10 Domains)

| Method | Agriculture MSE | Climate MSE | Economy MSE | Health MSE | Weather MSE |
|--------|----------------|-------------|-------------|------------|-------------|
| iTransformer | 0.220 | 1.135 | 0.222 | 1.519 | 1.231 |
| PatchTST | 0.228 | 1.184 | 0.210 | 1.432 | 1.145 |
| GPT4TS | 0.220 | 1.184 | 0.217 | 1.341 | - |
| TaTS | 0.215 | 1.180 | 0.215 | 1.356 | - |
| CALF | 0.250 | 1.286 | 0.207 | 1.491 | - |
| **VoT** | **0.209** | **1.078** | **0.201** | **1.205** | **0.968** |

### Dominant Performance

- **VoT ranks first on all 20 metrics (MSE + MAE) across all 10 domains**
- Consistent improvements over pure time series methods (e.g., Health: 1.205 vs. 1.432 for PatchTST, −15.9%)
- Comprehensive gains over existing multimodal methods (e.g., Climate: 1.078 vs. 1.180 for TaTS, −8.6%)

### Key Findings

1. **Effectiveness of text-based reasoning**: The event-driven reasoning branch effectively captures the impact of external events on time series
2. **Value of HIC**: Historical corrective reasoning used as ICL exemplars substantially improves reasoning accuracy
3. **Necessity of frequency fusion**: Different datasets exhibit different dependencies on textual vs. numerical information across frequency bands
4. **Cross-domain consistency**: The method is effective across 10 highly diverse domains including finance, climate, health, and transportation

## Highlights & Insights

1. **First multimodal time series forecasting method combining LLM reasoning with feature extraction**: Rather than solely extracting representations, VoT leverages the reasoning capability of LLMs to generate numerical predictions
2. **Innovative HIC design**: Error–correction pairs are stored during training; similar corrections are retrieved at inference as ICL guidance, enhancing reasoning without fine-tuning
3. **Elegant frequency-domain fusion**: Text and numerical weights are learned separately for each frequency band, adapting to domain-specific characteristics
4. **Complementary dual-branch architecture**: The event-driven branch handles abrupt changes while the numerical branch handles regular fluctuations; AFF achieves their optimal combination

## Limitations & Future Work

1. LLM reasoning (Reasoner) and knowledge base construction introduce substantial computational and latency overhead
2. The method depends on the availability and quality of exogenous text; domains lacking text may degrade to pure time series models
3. HIC retrieval quality is constrained by the size and diversity of the knowledge base
4. Hyperparameter choices for the frequency decomposition strategy may vary across domains
5. Noise and temporal misalignment in text are not explicitly addressed

## Rating ⭐⭐⭐⭐⭐

The method is comprehensive and elegantly designed. Achieving top rankings across all 20 metrics in 10 domains is a rare accomplishment. VoT genuinely integrates the reasoning capability of LLMs into time series forecasting, and both HIC and AFF reflect strong novelty. The primary limitation is the considerable computational overhead, which requires careful consideration in practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering](../../ICML2026/time_series/patra_pattern-aware_alignment_and_balanced_reasoning_for_time_series_question_an.md)
- [\[ICML 2026\] TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models](../../ICML2026/time_series/tsrbench_a_comprehensive_multi-task_multi-modal_time_series_reasoning_benchmark_.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[NeurIPS 2025\] MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting](../../NeurIPS2025/time_series/masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting.md)
- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)

</div>

<!-- RELATED:END -->
