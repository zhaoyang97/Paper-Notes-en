---
title: >-
  [Paper Note] Event-Aware Sentiment Factors from LLM-Augmented Financial Tweets: A Transparent Framework for Interpretable Quant Trading
description: >-
  [ICML 2025][Time Series][Financial Sentiment Analysis] This study leverages Large Language Models (LLMs) to perform multi-label event classification and annotation on financial tweets, transforming unstructured social media text into structured, interpretable, event-driven quantitative factors. It discovers that specific event categories (e.g., rumor/speculation) possess significant negative Alpha signals (with Sharpe ratios as low as -0.38).
tags:
  - "ICML 2025"
  - "Time Series"
  - "Financial Sentiment Analysis"
  - "Large Language Models"
  - "Event-Driven Factors"
  - "Alpha Signal Discovery"
  - "Quantitative Trading"
date: 2026-05-08
content_hash: aaa6f768d35202f8
---

# Event-Aware Sentiment Factors from LLM-Augmented Financial Tweets: A Transparent Framework for Interpretable Quant Trading

**Conference**: ICML 2025  
**arXiv**: [2508.07408](https://arxiv.org/abs/2508.07408)  
**Code**: Yes (The paper promises to open-source all code and methodology)  
**Area**: Time Series Analysis  
**Keywords**: Financial Sentiment Analysis, Large Language Models, Event-Driven Factors, Alpha Signal Discovery, Quantitative Trading

## TL;DR

This study leverages Large Language Models (LLMs) to perform multi-label event classification and annotation on financial tweets, transforming unstructured social media text into structured, interpretable, event-driven quantitative factors. It discovers that specific event categories (e.g., rumor/speculation) possess significant negative Alpha signals (with Sharpe ratios as low as -0.38).

## Background & Motivation

Traditional financial models rely on structured data like fundamental analysis and technical price patterns to explain market movements. However, research in behavioral finance indicates that investor sentiment—the collective emotions and psychological state of market participants—can serve as an independent driver of asset prices. Social media platforms like Twitter (now X) provide an unprecedented real-time window into public discourse, making them crucial data sources for capturing market sentiment.

**Limitations of Prior Work**:

**High Noise**: Signals from simple sentiment polarity (positive/negative) are highly noisy, and their predictive power decays easily due to arbitrage behavior.

**Lack of Interpretability**: Polarity scores only reveal "how the market feels" (positive or negative) but fail to explain "why"—namely, the underlying semantics or real-world events driving the sentiment.

**Single-Dimensional Processing**: Existing methods treat sentiment as a single signal, failing to distinguish between different types of market narratives—such as merger rumors or boycotts—which may have contrastive return characteristics.

**Core Idea**: The true value of social media data lies not only in sentiment intensity but more importantly in its rich semantic structure. Automatically assigning multi-label event categories to high-intensity tweets using LLMs enables the construction of more robust and interpretable predictive signals.

## Method

### Overall Architecture

The proposed methodology transforms unstructured social media text into structured, tradeable Alpha signals, consisting of four core phases:

1. **Data Acquisition and Preprocessing**: Cleaning the tweet corpus and aligning it with market data.
2. **LLM Sentiment and Event Annotation**: Leveraging LLMs for sentiment scoring and multi-label event classification.
3. **Cross-Sectional Factor Construction**: Constructing event-driven cross-sectional factors based on annotation results.
4. **Factor Evaluation and Backtesting**: Conducting rigorous factor performance evaluation and strategy backtesting.

### Key Designs

#### 1. Data Pipeline

- **Tweet Corpus**: Utilizing the dataset from Sowinska et al., which contains 862,231 English tweets linked to stock cash tags. After cleaning, a high signal-to-noise ratio subset of 85,176 tweets is retained.
- **Preprocessing Steps**: Standard NLP pipeline—lowercasing, token normalization, and masking of `$cashtag` and `@user`.
- **Market Data**: Aligned stock-level price and volume data used to calculate daily log returns $r_t = \log(P_t / P_{t-1})$, serving as the dependent variable for all predictive evaluations.

#### 2. LLM-Augmented Event Annotation System

This is the core innovation of the paper, assigning both sentiment intensity and multi-label semantic tags to each tweet:

**Sentiment Polarity (Net Tone)**:

- Assigning a continuous sentiment score (net tone) to each tweet, reflecting the directional emotional intensity of the text.
- Employing a stacked LDA topic model combined with logistic regression to predict forward returns, producing polarity scores aligned with market reactions.
- The framework supports replacing this with LLM-prompted polarity scoring.

**Multi-Label Event Tagging**:

- Utilizing commercial-grade LLMs for **zero-shot multi-label classification**.
- Designing a curated dictionary containing **70+ types of finance-related events**, including:
    - Rumor/Speculation
    - Retail Investor Buzz
    - Brand Boycott
    - Other event categories.
- Each tweet can be assigned one or more tags.
- The net tone of multi-labeled tweets is duplicated across each assigned tag for subsequent aggregation.

#### 3. Cross-Sectional Event Factor Construction

For each event tag $e$, stock $i$, and trading day $t$, the factor exposure $F_{i,t,e}$ is defined as the aggregated net tone score of relevant tweets in that event category. These factors are then used to:

- Align with 1-to-7-day forward returns to evaluate statistical validity and market tradability.
- Construct cross-sectional ranking strategies.
- Perform residual analysis to verify orthogonality to market Beta.

#### 4. Relationship with and Extension of the SESTM Framework

This paper builds upon the SESTM (Supervised Sentiment Topic Model) framework proposed by Ke et al. (2019):

- **Original SESTM Method**: Statistically models topic distributions from financial news through predictive vocabulary filtering and unsupervised topic inference.
- **Ours**: Adapting it to the more volatile and informal domain of social media, utilizing LLMs as semantic augmenters to replace manually tuned dictionaries and unsupervised topic inference, thereby achieving multi-label classification.

### Loss & Training

The training strategy of this paper adopts a **two-stage pipeline**:

1. **Stage 1 — Sentiment Scoring**: Employs supervised learning using a stacked LDA topic model + logistic regression, with forward returns serving as the supervisory signal, to train the sentiment polarity scorer.
2. **Stage 2 — Event Annotation**: Utilizes the zero-shot inference capability of LLMs without requiring additional training, directly performing multi-label classification through meticulously designed prompts and an event dictionary.

The advantages of this design include:

- The sentiment scoring aligns with market signals through supervised learning.
- The event annotation leverages the pre-trained knowledge of LLMs, eliminating the need for domain-specific labeled data.
- The overall framework features high modularity and extensibility.

## Key Experimental Results

### Main Results

The paper evaluates the predictive power of event-driven factors across multiple holding periods (1-7 days):

| Event Category | Metric | Ours | Baseline (Simple Sentiment Polarity) | Explanation |
|---------|------|---------|-------------------|------|
| Rumor/Speculation | Sharpe Ratio | **-0.38** | ~0 | Strongest negative Alpha signal |
| Rumor/Speculation | IC | **>0.05** | <0.03 | Significant at the 95% confidence level |
| Retail Investor Buzz | Sharpe Ratio | Negative | ~0 | Significant contrarian indicator |
| Brand Boycott | Sharpe Ratio | Negative | ~0 | Event-level signals are stronger |
| Overall Event Factors | Statistical Significance | 95% confidence | Unstable | Multi-period validation |

Key Finding: Specific event tags **consistently generate negative Alpha**, indicating that these event-driven sentiment signals can serve as **contrarian indicators**—when the market exhibits high emotional response to specific event types, subsequent returns tend to reverse.

### Ablation Study

| Configuration | Key Metrics | Explanation |
|------|---------|------|
| Sentiment Polarity Only (No Event Tags) | Low IC, Insignificant Sharpe | Simple polarity lacks discriminative power |
| LLM Event Tagging + Sentiment Polarity | IC > 0.05, Sharpe = -0.38 | Event tags add critical explanatory power |
| Market Beta Residual Analysis | Alpha orthogonal to Beta | Confirms factors represent true Alpha |
| Multi-Holding Periods (1-7 days) | All significant | Factors are robust across multiple horizons |
| High Sentiment Intensity Filtering | Stronger signals | Event annotation is more valuable for high-sentiment tweets |

### Key Findings

1. **Event-conditioned signals show distinct predictive profiles**: Different event categories display starkly distinct return curves, confirming the necessity of decomposing sentiment into multi-dimensional event factors.
2. **Consistency of negative Alpha**: Rumor/speculation events consistently generate negative Alpha across 1-7 day horizons, with Sharpe ratios as low as -0.38 and IC exceeding 0.05.
3. **Orthogonality to market risk**: Residual analysis demonstrates that the predictive power of event factors is orthogonal to market Beta, representing a valid source of Alpha.
4. **Interpretability of LLM tags**: Each factor is semantically bound to actual market events, providing clear economic explanations for the strategies.

## Highlights & Insights

1. **Paradigm shift from "what" to "why"**: This paper elevates sentiment analysis from single-polarity scoring to multi-dimensional event attribution, answering "why" market sentiment behaves the way it does.
2. **A new role for LLMs as semantic augmenters**: Instead of using LLMs for end-to-end prediction, this paper leverages their zero-shot classification capabilities to provide structured inputs for traditional quantitative pipelines, balancing both interpretability and performance.
3. **Practical value of negative Alpha**: Identifying event categories that consistently generate negative Alpha provides direct guidance for building contrarian trading strategies.
4. **Reproducible research paradigm**: All code and methodology are open-sourced, lowering the barrier to entry for quantitative trading research.
5. **Modular design**: Decoupled sentiment scoring and event annotation allow individual components to be replaced and upgraded independently.

## Limitations & Future Work

1. **Data timeliness**: The tweet dataset used in the study dates back to 2017. Since market structures and social media ecosystems have changed significantly, the generalizability of factor performance in current markets needs verification.
2. **LLM dependency and cost**: Large-scale tweet tagging relies on commercial-grade LLMs, where API call costs and latencies might constrain real-time trading applications.
3. **Completeness of the event dictionary**: Do the 70+ event types cover all market-relevant narratives? Does the dictionary design introduce selection bias?
4. **Sample size limitations**: Standard filtering left only 85,176 tweets, which may pose small-sample challenges for cross-sectional factor construction.
5. **Factor decay issues**: The paper does not thoroughly discuss changes in factor efficacy over longer time horizons and potential Alpha decay.
6. **Multi-market validation**: Validated only on US equities; applicability to other markets (e.g., A-share, HKEX) remains unknown.
7. **Real-time deployment challenges**: Whether the latency from tweet acquisition to factor generation affects the feasibility of strategy execution.

## Related Work & Insights

- **SESTM (Ke et al., 2019)**: A supervised sentiment topic model that extracts sentiment factors from financial news through predictive vocabulary filtering and topic modeling. It serves as the direct inspiration for the methodology in this paper.
- **Bollen et al. (2011)**: Discovered that the "calm" dimension of Twitter sentiment can predict daily movements of the DJIA, pioneering the application of social media sentiment analysis in finance.
- **FinBERT (Sowinska & Madhyastha, 2020)**: A pre-trained BERT model tailored for the financial domain, advancing NLP applications in understanding financial text.
- **The GameStop Event**: Illustrates how retail sentiment expressed on social media can exert monumental impacts on the market, further validating the significance of the direction pursued in this paper.

**Inspirational Directions**:

- Expanding this framework to Chinese financial social media (e.g., Xueqiu, Eastmoney comments).
- Integrating multi-modal information (e.g., chart screenshots, video analysis) to enhance event recognition.
- Introducing temporal sequence modeling (such as Transformers or State Space Models) to capture the dynamic evolution of event factors.

## Rating

| Dimension | Rating (1-5) | Explanation |
|------|-----------|------|
| Novelty | 3.5 | The combination of LLMs and event annotation is innovative, but represents an incremental innovation built upon the SESTM framework. |
| Technical depth | 3.0 | The methodological framework is clear, but technical complexity is moderate as it heavily relies on the zero-shot capabilities of LLMs. |
| Experimental Thoroughness | 3.5 | The backtesting evaluation is systematic, but the dataset is relatively old and limited in scale. |
| Reproducibility | 4.5 | All code and methodologies are open-source, making it highly reproducible. |
| Practical Value | 4.0 | Direct practical significance for quantitative trading. |
| **Overall** | **3.5** | Well-directed and highly practical work, although technical depth and data validation still have room for improvement. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Context-Aware Sentiment Forecasting via LLM-based Multi-Perspective Role-Playing Agents](../../ACL2025/time_series/context_aware_sentiment_forecasting_agents.md)
- [\[AAAI 2026\] Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths](../../AAAI2026/time_series/interpreting_fedspeak_with_confidence_a_llm-based_uncertainty-aware_framework_gu.md)
- [\[ACL 2026\] A Unified Framework for Modeling Heterogeneous Financial Data via Dual-Granularity Prompting](../../ACL2026/time_series/a_unified_framework_for_modeling_heterogeneous_financial_data_via_dual-granulari.md)
- [\[ICML 2025\] TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [\[ICLR 2026\] HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming](../../ICLR2026/time_series/hivid_llm-guided_video_saliency_for_content-aware_vod_and_live_streaming.md)

</div>

<!-- RELATED:END -->
