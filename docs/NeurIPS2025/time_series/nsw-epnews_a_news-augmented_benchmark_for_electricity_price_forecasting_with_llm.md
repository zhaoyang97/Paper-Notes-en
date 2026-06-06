---
title: >-
  [Paper Note] NSW-EPNews: A News-Augmented Benchmark for Electricity Price Forecasting with LLMs
description: >-
  [NeurIPS 2025][Time Series][Electricity price forecasting] This paper introduces NSW-EPNews, the first electricity price forecasting benchmark augmented with news text…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Electricity price forecasting"
  - "LLM forecasting"
  - "multimodal benchmark"
  - "hallucination detection"
  - "prompt engineering"
date: 2026-05-08
content_hash: 217990e95b6372ef
---

# NSW-EPNews: A News-Augmented Benchmark for Electricity Price Forecasting with LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2506.11050](https://arxiv.org/abs/2506.11050)  
**Code**: [Figshare Dataset](https://figshare.com/s/e25f3a98679d347f2a2e)  
**Area**: Time Series
**Keywords**: Electricity price forecasting, LLM forecasting, multimodal benchmark, hallucination detection, prompt engineering

## TL;DR

This paper introduces NSW-EPNews, the first electricity price forecasting benchmark augmented with news text, systematically evaluating both traditional models and LLMs on multimodal electricity price prediction. Key findings show that news features provide marginal gains for traditional models, while LLMs suffer from severe hallucination issues.

## Background & Motivation

Electricity price forecasting is a core task in energy management systems. Existing methods rely heavily on historical numerical data, overlooking concurrent textual signals (e.g., market news, policy announcements) and meteorological data that influence prices. Although LLMs show promise in integrating structured features with free-form text, the community lacks a systematic benchmark to evaluate whether LLMs can reliably translate news sentiment and weather cues into accurate numerical forecasts.

Specifically, the following key issues exist:

**Lack of multimodal benchmarks**: Existing electricity forecasting datasets contain only numerical time series, with no news text data.

**Unknown LLM reliability**: Whether LLMs can produce trustworthy predictions in high-stakes energy markets remains unvalidated.

**Hallucination problem**: LLMs may generate plausible-looking price sequences that are inconsistent with the input data.

## Method

### Overall Architecture

The NSW-EPNews benchmark comprises three core components: data construction, prompt template design, and an evaluation framework with hallucination detection. The data covers New South Wales, Australia from 2015 to 2024, including over 175,000 half-hourly spot prices, daily temperature records, and curated market news summaries from WattClarity. The forecasting task is defined as a 48-step ahead prediction (i.e., forecasting 48 half-hourly prices for the next day).

### Key Designs

1. **Data Preprocessing and News Classification**: Raw news text is processed via GPT-4o using a four-module classification pipeline. The four modules handle role assignment, classification criteria, key attributes, and summarization rules, compressing lengthy electricity market reports into machine-parsable structured signals. Each news item is labeled with an impact level (high/medium/low) to investigate whether models can differentially leverage news of varying importance. For price data, since NEM transitioned from 30-minute to 5-minute frequency after October 2021, median downsampling is applied to maintain consistency.

2. **Four Prompt Template Designs**: To comprehensively evaluate LLMs, four prompt formats are designed, varying along two dimensions:

    - Zero-shot vs. few-shot (with or without example Q&A pairs)
    - With or without chain-of-thought (CoT) reasoning

   Each prompt integrates 48 historical price points, news summaries, temperature, and date information. Ablation prompts containing only historical prices are also designed to analyze the marginal contribution of news and temperature data.

3. **Hallucination and Error Detection Framework**: Four types of LLM output anomalies are defined and detected:

    - **Echoing Failure**: ≥10 predicted values are directly copied from the historical input.
    - **Trivial Transformation**: Predicted values equal historical values plus a fixed offset (matching ≥20 points).
    - **Degenerate Copying**: A single value appears more than 5 times in the output.
    - **Format Violation**: The output cannot be parsed as a price list.

### Loss & Training

Traditional baseline models (ARIMA, LR, XGBoost) are trained using standard supervised learning: ARIMA relies solely on the autoregressive price structure, while LR and XGBoost additionally incorporate TF-IDF vectorized news as features. LLMs (GPT-4o, Gemini 1.5 Pro) are queried via API for zero-shot/few-shot inference without fine-tuning. Evaluation metrics include MSE, RMSE, MAE, and MAPE.

## Key Experimental Results

### Main Results

| Model | Prompt Type | MSE (50% data) | MAE (50% data) | MSE (10% data) | MAE (10% data) |
|-------|-------------|----------------|----------------|----------------|----------------|
| ARIMA | — | 124,926 | 54.58 | 393,097 | 89.09 |
| Linear Regression | — | 129,537 | 69.54 | 393,663 | 90.20 |
| XGBoost | — | 263,069 | 113.33 | 453,917 | 129.61 |
| GPT-4o | Zero-shot | 209,929 | 53.65 | 692,109 | 98.27 |
| GPT-4o | Few-shot+CoT | 187,494 | 53.40 | 717,159 | 102.96 |
| Gemini 1.5 Pro | Zero-shot+CoT | 143,822 | 47.03 | 164,605 | 60.75 |
| Gemini 1.5 Pro | Few-shot+CoT | 187,494 | 53.40 | 319,180 | 73.85 |

### Ablation Study (Hallucination Detection — GPT-4o)

| Prompt Type | Avg. Echo Rate | Avg. Trivial Transform Rate | Avg. Degenerate Copy Rate | Avg. Format Violation Rate |
|-------------|---------------|-----------------------------|---------------------------|---------------------------|
| Zero-shot | 87.4% | 0.39% | 9.4% | 0.02% |
| Zero-shot+CoT | 53.8% | 0.45% | 9.4% | 0.13% |
| Few-shot | 51.1% | 5.76% | 8.9% | 0.08% |
| Few-shot+CoT | 56.6% | 1.90% | 8.4% | 0.07% |
| Ablation (price only) | 95.0% | 0.46% | 7.6% | 0.30% |

### Key Findings

1. **Marginal gains for traditional models**: News and temperature features yield negligible improvements in forecasting accuracy for ARIMA, LR, and XGBoost.
2. **Large and unstable LLM errors**: GPT-4o and Gemini 1.5 Pro exhibit MAE exceeding 100 AUD/MWh on certain data splits.
3. **Severe hallucination**: GPT-4o's echoing failure rate reaches 87% under zero-shot settings, indicating that the model largely copies historical inputs rather than generating genuine forecasts.
4. **Gemini outperforms GPT-4o**: Gemini 1.5 Pro with CoT surpasses GPT-4o on most metrics, but still falls short of the stability achieved by traditional models.
5. **CoT reduces echoing but does not guarantee accuracy**: CoT reduces the echo rate from 87% to 54%, yet prediction errors do not necessarily decrease correspondingly.

## Highlights & Insights

- **First multimodal electricity price forecasting benchmark**: Systematically integrates price, news, and weather modalities, filling a gap in the field.
- **Exposing LLMs' "pseudo-capability"**: High echo rates reveal that LLMs are not performing genuine numerical reasoning but rather engaging in pattern matching and copying.
- **Practical hallucination detection methodology**: The four-category hallucination detection algorithm is concise and effective, and is generalizable to other numerical forecasting scenarios.

## Limitations & Future Work

1. Fine-tuning LLMs is not explored, which could substantially improve task adaptation.
2. News is sourced exclusively from WattClarity; multi-source news may provide richer signals.
3. More advanced LLM integration strategies such as RAG (Retrieval-Augmented Generation) are not evaluated.
4. TF-IDF vectorization of news provides a coarse text representation for traditional models.

## Related Work & Insights

This work connects two research directions: traditional time series forecasting (ARIMA, XGBoost) and LLM-based forecasting (TimeGPT, GPT4MTS). The hallucination detection framework can inspire reliability evaluation for future deployment of LLMs in high-stakes domains such as finance and meteorology.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First electricity price forecasting benchmark augmented with news text; problem definition is clear.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison across multiple models, prompt strategies, and data splits; fine-tuning experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with detailed data presentation.
- **Value**: ⭐⭐⭐⭐ Provides important reference for understanding the limitations of LLMs in numerical forecasting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DemandCast: Global hourly electricity demand forecasting](demandcast_global_hourly_electricity_demand_forecasting.md)
- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)
- [\[NeurIPS 2025\] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)
- [\[ICLR 2026\] scits scientific time series understanding and generation with llms](../../ICLR2026/time_series/scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICCV 2025\] VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models](../../ICCV2025/time_series/vlrmbench_a_comprehensive_and_challenging_benchmark_for_vision-language_reward_m.md)

</div>

<!-- RELATED:END -->
