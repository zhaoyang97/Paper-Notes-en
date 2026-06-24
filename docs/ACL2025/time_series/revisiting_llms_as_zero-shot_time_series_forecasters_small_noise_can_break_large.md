---
title: >-
  [Paper Note] Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models
description: >-
  [ACL 2025 (Short Paper)][Time Series][Large Language Models] This paper systematically evaluates the effectiveness of LLMs as zero-shot time-series forecasters and discovers that LLMs are extremely sensitive to input noise—even a small amount of noise can lead to a drastic performance degradation, making them underperform even simple domain-specific models (such as DLinear). The authors suggest that future research should focus on fine-tuning LLMs to better handle numerical s…
tags:
  - "ACL 2025 (Short Paper)"
  - "Time Series"
  - "Large Language Models"
  - "Zero-Shot Forecasting"
  - "Noise Sensitivity"
  - "Robustness"
date: 2026-05-08
content_hash: 43f191fd89ff92f0
---

# Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models

**Conference**: ACL 2025 (Short Paper)  
**arXiv**: [2506.00457](https://arxiv.org/abs/2506.00457)  
**Code**: [GitHub](https://github.com/junwoopark92/revisiting-LLMs-zeroshot-forecaster)  
**Area**: Time Series  
**Keywords**: Large Language Models, Zero-Shot Forecasting, Time Series, Noise Sensitivity, Robustness

## TL;DR

This paper systematically evaluates the effectiveness of LLMs as zero-shot time-series forecasters and discovers that LLMs are extremely sensitive to input noise—even a small amount of noise can lead to a drastic performance degradation, making them underperform even simple domain-specific models (such as DLinear). The authors suggest that future research should focus on fine-tuning LLMs to better handle numerical sequences.

## Background & Motivation

**Background**: LLMs have demonstrated powerful zero-shot capabilities in various tasks such as natural language processing, sparking research interest in directly applying them to time-series forecasting. Work like LLMTime translates numerical sequences into text prompts for zero-shot forecasting, achieving remarkable results in certain scenarios.

**Limitations of Prior Work**: However, the community remains divided regarding the effectiveness of zero-shot time-series forecasting with LLMs—some studies claim exceptional performance, while others question whether LLMs truly "understand" numerical patterns. Existing positive results might be influenced by specific evaluation settings, such as testing only on clean synthetic data or particular datasets.

**Key Challenge**: Real-world time-series data almost always contains noise (sensor errors, measurement uncertainties, random fluctuations, etc.), but most existing evaluations are conducted under idealized conditions. If the forecasting capability of LLMs drastically degrades in the presence of real-world noise, the practical value of their zero-shot forecasting becomes highly questionable.

**Goal**: To systematically compare the performance of zero-shot LLM forecasting against domain-specific models under strictly controlled experimental conditions, with a particular focus on robustness differences under noisy conditions.

**Key Insight**: The authors approach this from the perspective of robustness—comparing not only the performance on clean data, but also systematically injecting different types and levels of noise to observe the degradation of each model.

**Core Idea**: To reveal the fundamental vulnerability of zero-shot LLM forecasting through noise sensitivity analysis, and to suggest that fine-tuning rather than prompting is the correct direction for applying LLMs to time series.

## Method

### Overall Architecture

This paper is an evaluation study rather than a methodological proposal. The experimental framework is as follows: (1) select multiple LLMs (GPT-3.5, GPT-4o, LLaMA) and multiple domain-specific baselines (DLinear, RLinear, N-BEATS, ARIMA, etc.); (2) perform forecasting on multiple real-world (Monash, Informer benchmark) and synthetic datasets; (3) re-evaluate after injecting various types of noise (Gaussian noise, constant noise, missing values, periodic noise) into the data; (4) compare the performance changes of each model under clean and noisy conditions.

### Key Designs

1. **Multi-Type Noise Injection Framework**:

    - Function: Systematically tests model performance under different noise conditions.
    - Mechanism: Defines four types of noise—(a) Gaussian noise: adding a random perturbation of $\mathcal{N}(0, \sigma^2)$ at each time step; (b) Constant noise: inserting a fixed outlier at random positions; (c) Missing value noise: randomly setting a portion of time steps to empty; (d) Periodic noise: superimposing a sine wave with a specific frequency. Each noise type is injected at different intensity levels (low/medium/high).
    - Design Motivation: Noise in real-world time series takes various forms; testing a single type of noise is insufficient for a comprehensive robustness evaluation. These four noise types cover the most common data quality issues.

2. **Zero-Shot LLM Forecasting Protocol**:

    - Function: Standardizes the evaluation protocol for LLMs as zero-shot forecasters.
    - Mechanism: Follows the paradigm of LLMTime—converting historical numerical sequences into comma-separated text strings as prompts for the LLM, and letting the LLM complete the predicted values. Proper scaling and formatting are applied to the numerical values. Multiple generations are sampled and the median is taken as the point prediction. Evaluation metrics are MAE and MSE.
    - Design Motivation: A unified evaluation protocol ensures a fair comparison and aligns with established literature practices to guarantee comparability.

3. **Cross-Dimensional Comparison**:

    - Function: Unveils the weaknesses of LLM forecasting from multiple perspectives.
    - Mechanism: Beyond comparing absolute performance, it also analyzes (a) the degradation ratio caused by noise, (b) the impact of different dataset characteristics on sensitivity, and (c) the relationship between model scale and robustness. Synthetic data (using mathematical functions with known generation processes) is further utilized to control variables and observe how the LLM's capacity to recognize simple patterns changes under noise.
    - Design Motivation: Comprehensive comparison helps locate the root cause—whether LLMs fail to understand numerical values themselves, or are particularly sensitive to numerical perturbations in textual representations.

### Loss & Training

This paper does not involve training. Domain-specific models use their respective standard training pipelines, while LLMs are accessed directly via API calls or local inference.

## Key Experimental Results

### Main Results

Comparison of average MAE on five Informer real-world datasets (Clean vs Gaussian Noise):

| Model | Clean Data MAE | Noisy Data MAE | Degradation Ratio |
|------|------------|------------|--------|
| GPT-4o (Zero-Shot) | 0.412 | 0.687 | -66.7% |
| GPT-3.5 (Zero-Shot) | 0.458 | 0.731 | -59.6% |
| LLaMA-3 (Zero-Shot) | 0.523 | 0.812 | -55.3% |
| DLinear | 0.289 | 0.324 | -12.1% |
| RLinear | 0.301 | 0.338 | -12.3% |
| N-BEATS | 0.275 | 0.312 | -13.5% |
| ARIMA | 0.342 | 0.378 | -10.5% |

### Ablation Study

Impact of different noise types on GPT-4o on the Monash dataset:

| Noise Type | Clean MAE | Noisy MAE | Degradation Ratio |
|---------|---------|-----------|--------|
| No Noise | 0.389 | - | - |
| Gaussian Noise (σ=0.1) | - | 0.542 | -39.3% |
| Constant Noise (5%) | - | 0.611 | -57.1% |
| Missing Values (10%) | - | 0.498 | -28.0% |
| Periodic Noise | - | 0.467 | -20.1% |

### Key Findings

- **LLMs perform worse than domain-specific models even on clean data**—even under the most favorable conditions, GPT-4o's MAE is about 40% higher than DLinear's.
- **The noise amplification effect is highly significant**: The performance of LLMs drops by 50-67% under Gaussian noise, whereas domain-specific models only drop by 10-14%, further widening the performance gap.
- Constant noise (outliers) has the most severe impact on LLMs, which may be related to the tokenization mechanism of LLMs—outlier numerical values generate unusual token sequences.
- Synthetic data experiments show that even for a simple sigmoid function, the shape of the LLM's forecasting becomes severely distorted after adding a small amount of noise.
- Increasing model scale (GPT-3.5 → GPT-4o) helps on clean data, but provides limited improvement under noise, indicating that scale is not the key to solving robustness.

## Highlights & Insights

- **The title and core finding of "small noise breaks large models" are highly cautionary**: This directly challenges the narrative of LLMs as "all-powerful," providing a valuable correction to the community's over-optimistic sentiment.
- **The noise classification and systematic evaluation framework are reusable**: The evaluation protocol with four noise types and multi-level intensities can serve as a standard component for future evaluations of the numerical capabilities of LLMs.
- **The directional recommendation is very pragmatic**: Instead of dismissing the potential of LLMs in time-series forecasting, the paper clearly points out that the path of "fine-tuning > zero-shot" is more promising, offering clear guidance for subsequent research.

## Limitations & Future Work

- As a short paper, the experimental scope is limited, not covering a wider range of LLMs (such as Claude, Gemini) or larger-scale time-series datasets.
- The study focuses solely on zero-shot settings, without comparing the performance of few-shot or in-context learning in noisy scenarios.
- It lacks a deep analysis of the root causes of noise sensitivity in LLMs—whether it is a tokenization issue, a numerical precision problem, or a deficiency in pattern recognition.
- The paper proposes the suggestion that "LLMs should be fine-tuned" but does not provide empirical validation of fine-tuning strategies.

## Related Work & Insights

- **vs LLMTime (Gruver et al., 2023)**: LLMTime demonstrated the possibility of zero-shot time-series forecasting with LLMs for the first time. This paper directly challenges and extends its conclusions, pointing out that its positive results on clean data cannot generalize to real-world noisy environments.
- **vs Time-LLM (Jin et al., 2024)**: Time-LLM adopts a fine-tuning route, which aligns with the recommended direction of this paper.
- **vs Traditional Statistical Models (ARIMA)**: Even the most classic statistical methods significantly outperform zero-shot LLM forecasting under noise, highlighting the importance of domain knowledge.

## Rating

- Novelty: ⭐⭐⭐ The analysis of noise sensitivity is a valuable perspective, though the core experimental design is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐ Covers multiple models, datasets, and noise types, but the depth is limited due to the short paper format.
- Writing Quality: ⭐⭐⭐⭐ The arguments are clear, the findings are presented well, and the conclusions are not over-claimed.
- Value: ⭐⭐⭐⭐ Makes a significant contribution to the community's discussion on LLMs for time-series forecasting, with practical guidance in its recommended directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters](../../ICML2025/time_series/visionts_visual_masked_autoencoders_are_free-lunch_zero-shot_time_series_forecas.md)
- [\[ICLR 2026\] Zero-shot Forecasting by Simulation Alone](../../ICLR2026/time_series/zero-shot_forecasting_by_simulation_alone.md)
- [\[NeurIPS 2025\] TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning](../../NeurIPS2025/time_series/tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md)
- [\[NeurIPS 2025\] Decomposition of Small Transformer Models](../../NeurIPS2025/time_series/decomposition_of_small_transformer_models.md)
- [\[ICLR 2026\] Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](../../ICLR2026/time_series/relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)

</div>

<!-- RELATED:END -->
