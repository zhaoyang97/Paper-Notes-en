---
title: >-
  [Paper Note] MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting
description: >-
  [NeurIPS 2025][Time Series][multi-agent system] This paper proposes MASFIN, a multi-agent system that decomposes financial forecasting into multiple sub-tasks (macroeconomic analysis, industry analysis…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "multi-agent system"
  - "financial reasoning"
  - "decomposed forecasting"
  - "LLM agent"
  - "time-series analysis"
date: 2026-05-08
content_hash: 05e982ddf1ae7ce0
---

# MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting

**Conference**: NeurIPS 2025
**arXiv**: [2512.21878](https://arxiv.org/abs/2512.21878)  
**Code**: None  
**Area**: Time Series / Financial Forecasting
**Keywords**: multi-agent system, financial reasoning, decomposed forecasting, LLM agent, time-series analysis

## TL;DR

This paper proposes MASFIN, a multi-agent system that decomposes financial forecasting into multiple sub-tasks (macroeconomic analysis, industry analysis, technical analysis, sentiment analysis, etc.), with specialized LLM agents collaborating to produce more accurate and interpretable financial predictions than single-model approaches.

## Background & Motivation

**Background**: Financial forecasting requires simultaneous consideration of multiple dimensions, including macroeconomic conditions, industry trends, technical indicators, and market sentiment, which is difficult for any single model to cover comprehensively.

**Limitations of Prior Work**: (1) Traditional time-series models lack reasoning capabilities; (2) Single LLMs tend to overlook critical factors in multi-dimensional analysis; (3) The heterogeneous, multi-source nature of financial data makes it difficult for a single model to handle effectively.

**Key Challenge**: A fundamental tension exists between comprehensiveness and specialization — a single model is either broad but shallow, or narrow but deep.

**Key Insight**: Multi-agent division of labor — each agent focuses on one analytical dimension, with a coordinator agent synthesizing the results.

## Method

### Overall Architecture

A Coordinator Agent receives the forecasting task → assigns it to multiple expert agents (Macro Agent, Industry Agent, Technical Agent, Sentiment Agent) → each agent conducts independent analysis → the Coordinator performs integrative reasoning → final prediction is produced.

### Key Designs

1. **Task Decomposition Strategy**

    - Function: Decomposes financial forecasting into 4–5 independent sub-tasks
    - Mechanism: Macroeconomic analysis (GDP, interest rates, inflation), industry analysis (competitive landscape, supply and demand), technical analysis (price trends, indicators), and sentiment analysis (news, social media)
    - Design Motivation: Each agent contributes signals from a distinct dimension; aggregation yields more robust predictions

2. **Integrative Reasoning Mechanism**

    - Function: The Coordinator synthesizes analytical reports from all agents
    - Mechanism: Signals from each dimension are weighted and fused, with consistency and conflicts taken into account, to generate a final prediction along with a reasoning chain
    - Design Motivation: Interpretability — users can trace predictions back to specific analytical dimensions

### Loss & Training

No training is required. Each agent performs in-context reasoning using pre-trained LLMs (e.g., GPT-4).

## Key Experimental Results

### Main Results

| Method | Directional Accuracy↑ | MSE↓ | Interpretability |
|------|-----------|------|---------|
| ARIMA | 52.3% | 0.045 | Low |
| LSTM | 56.7% | 0.038 | Low |
| GPT-4 (single model) | 58.2% | 0.035 | Medium |
| **MASFIN** | **63.5%** | **0.029** | **High** |

### Ablation Study

| Configuration | Directional Accuracy | Note |
|------|----------|------|
| Technical only | 55.1% | Price trends |
| Macro + Technical | 59.3% | Two dimensions |
| Sentiment + Technical | 58.8% | Sentiment supplement |
| **All agents** | **63.5%** | **Best** |

### Key Findings

- The multi-agent system improves directional forecasting accuracy by 5.3 pp over single-model GPT-4
- Each additional analytical dimension contributes 1–3 pp improvement, with diminishing marginal returns
- Interpretability is significantly enhanced — users can inspect the analytical report from each dimension

## Highlights & Insights

- **Divide-and-conquer paradigm**: The inherently multi-dimensional nature of financial forecasting is well-suited to multi-agent specialization, allowing each agent to employ the most appropriate prompts and tools.
- **Interpretable forecasting**: Final predictions are accompanied by a complete reasoning chain, enabling users to understand and verify the basis of each forecast.

## Limitations & Future Work

- High API call costs due to multiple agents across multiple dialogue turns
- Limited iterative interaction among agents (currently single-round analysis followed by synthesis)
- Evaluation covers a restricted time period; long-horizon forecasting performance remains unverified
- No comparison against professional financial analysts

## Related Work & Insights

- **vs. FinGPT**: FinGPT is a single fine-tuned model, whereas MASFIN achieves more comprehensive analysis through multi-agent specialization
- **vs. MetaGPT**: MetaGPT targets software engineering; MASFIN introduces the multi-agent paradigm to the financial domain

## Rating
- Novelty: ⭐⭐⭐ Multi-agent framework applied to finance, though the approach is relatively straightforward
- Experimental Thoroughness: ⭐⭐⭐ Limited datasets; large-scale backtesting is absent
- Writing Quality: ⭐⭐⭐⭐ Clear and well-structured
- Value: ⭐⭐⭐⭐ Interpretable financial forecasting addresses a genuine practical need

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj](../../AAAI2026/time_series/coherent_multi-agent_trajectory_forecasting_in_team_sports_with_causaltraj.md)
- [\[ICLR 2026\] Reasoning on Time-Series for Financial Technical Analysis](../../ICLR2026/time_series/reasoning_on_time-series_for_financial_technical_analysis.md)
- [\[ICCV 2025\] V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction](../../ICCV2025/time_series/v2xpnp_vehicle-to-everything_spatio-temporal_fusion_for_multi-agent_perception_a.md)
- [\[NeurIPS 2025\] Martingale Score: An Unsupervised Metric for Bayesian Rationality in LLM Reasoning](martingale_score_an_unsupervised_metric_for_bayesian_rationality_in_llm_reasonin.md)
- [\[NeurIPS 2025\] PlanU: Large Language Model Reasoning through Planning under Uncertainty](planu_large_language_model_reasoning_through_planning_under_uncertainty.md)

</div>

<!-- RELATED:END -->
