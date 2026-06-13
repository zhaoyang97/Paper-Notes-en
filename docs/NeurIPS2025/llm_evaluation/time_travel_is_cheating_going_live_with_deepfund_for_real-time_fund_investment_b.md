---
title: >-
  [Paper Note] Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking
description: >-
  [NeurIPS 2025][LLM Evaluation][LLM trading] This paper introduces DeepFund — the first live fund investment benchmark for LLMs — which employs a multi-agent architecture (Financial Planner + Analyst Team + Portfolio Mana…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "LLM trading"
  - "live benchmarking"
  - "multi-agent"
  - "fund investment"
  - "information leakage"
  - "financial evaluation"
date: 2026-05-08
content_hash: 61157255abfa4b32
---

# Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking

**Conference**: NeurIPS 2025
**arXiv**: [2505.11065](https://arxiv.org/abs/2505.11065)  
**Code**: [GitHub](https://github.com/HKUSTDial/DeepFund)  
**Area**: LLM Evaluation
**Keywords**: LLM trading, live benchmarking, multi-agent, fund investment, information leakage, financial evaluation

## TL;DR
This paper introduces DeepFund — the first live fund investment benchmark for LLMs — which employs a multi-agent architecture (Financial Planner + Analyst Team + Portfolio Manager) connected to real-time market data, eliminating the information leakage caused by LLM "time travel" in traditional backtesting. Over 24 trading days of live testing across 9 flagship LLMs, only Grok 3 achieves positive returns, revealing fundamental limitations of current LLMs in active fund management.

## Background & Motivation

**Background**: LLMs have demonstrated strong capabilities on financial tasks such as report summarization, earnings call analysis, and asset classification. There is growing interest in leveraging LLMs for trading strategy generation and fund investment management. Existing benchmarks (e.g., TAT-QA, FinanceBench, InvestorBench) evaluate LLM financial understanding and trading performance.

**Limitations of Prior Work**: Existing benchmarks rely on historical backtesting to evaluate LLM trading strategies; however, LLM pretraining data very likely includes historical market data from the backtesting period. This leads to severe **information leakage** — LLMs can effectively "time travel," exploiting future information to achieve inflated performance on historical data. The varying knowledge cutoff dates across LLMs (e.g., GPT-4o at 2023.10, DeepSeek-V3 at 2024.7) further exacerbates evaluation unfairness.

**Key Challenge**: Under a backtesting setup, it is impossible to distinguish whether an LLM genuinely "predicts" market movements or merely "recalls" historical data, rendering backtesting results fundamentally unreliable.

**Goal**: To construct an LLM fund investment evaluation framework grounded entirely in live market data, ensuring zero information leakage.

**Key Insight**: Forward testing using live data strictly after each model's pretraining cutoff date, rather than backtesting.

**Core Idea**: Transform LLM fund investment evaluation from "historical backtesting" to "live forward testing," completely eliminating time-travel cheating.

## Method

### Overall Architecture
The DeepFund system consists of three components: (1) a **Live Environment** that continuously ingests real-time market data, fund asset information, and trading history; (2) a **Multi-Agent Workflow** that simulates real fund management processes — Financial Planner assigns tasks → Analyst Team conducts parallel analysis → Portfolio Manager makes decisions; and (3) an **LLM Factory** that supports flexible switching between different LLM backends. All agents are driven by the same LLM to ensure consistency.

### Key Designs

1. **Live Environment**:

    - Function: Provides leak-free, real-time market conditions.
    - Mechanism: A modular API gateway connects to data sources such as Yahoo Finance and Alpha Vantage, continuously retrieving real-time stock prices, company news, macroeconomic indicators, insider trading information, and other multi-source data. All data is published after each model's pretraining cutoff date.
    - Design Motivation: To fundamentally eliminate information leakage by using "future" data that the model could not have encountered during pretraining.

2. **Multi-Agent Decision Framework**:

    - Function: Simulates the collaborative decision-making process of real fund management.
    - Mechanism:
        - **Financial Planner**: Determines analytical priorities and assigns tasks to appropriate analysts (supporting both deterministic and dynamic modes).
        - **Analyst Team**: Six specialized analysts (Technical, Fundamental, Insider, Company News, Macro Economic, Policy), each analyzing domain-specific data and outputting standardized signals (Bullish/Bearish/Neutral) with detailed rationales.
        - **Portfolio Manager**: Synthesizes multiple analytical signals to make trading decisions (Buy/Sell/Hold), manages risk (position sizing and cash allocation), and maintains a dual-memory architecture (historical trades + current portfolio state).
    - Design Motivation: A single agent cannot effectively handle multi-source heterogeneous financial data; division of labor better mirrors the working model of real fund management teams.

3. **Evaluation Framework**:

    - Function: Quantifies LLM trading performance across multiple dimensions.
    - Mechanism: Standard financial metrics are employed, including Cumulative Return (CR), Buy-and-Hold Cumulative Return ($\text{CR}_\text{bnh}$), Sharpe Ratio, Maximum Drawdown (MDD), Win Rate (WR), Beta, and Alpha.
    - Design Motivation: To align with professional fund evaluation standards rather than relying solely on return rates.

### Loss & Training
No model training is involved. DeepFund is an evaluation framework that performs inference via the standard APIs of each LLM.

## Key Experimental Results

### Overall Trading Performance (2025.3.17–4.17, 24 Trading Days)

| LLM | CR(%) | SR | MDD(%) | WR(%) | Beta |
|-----|-------|-----|--------|-------|------|
| Grok 3 mini Beta | **+1.1** | 0.51 | 5.5 | 61 | 0.42 |
| Gemini 2.5 Flash | -1.9 | -1.37 | 6.4 | 61 | 0.35 |
| Claude 3.7 Sonnet | -3.7 | -1.45 | 10.1 | 70 | 0.64 |
| Llama 4 Scout | -4.3 | -2.42 | 8.9 | 61 | 0.36 |
| DeepSeek-V3 | -5.7 | -1.39 | 14.5 | 57 | 0.94 |
| GPT-4.1 | -5.9 | -1.87 | 12.8 | 52 | 0.77 |
| Qwen2.5-Max | -6.7 | -3.12 | 10.7 | 65 | 0.48 |
| GLM-4-Air | -7.5 | -2.31 | 13.2 | 57 | 0.78 |
| Doubao-1.5-pro | -8.1 | -2.35 | 13.6 | 65 | 0.84 |
| S&P 500 | -6.91 | — | 13.7 | — | 1.00 |

### Signal and Decision Validity

| Metric | Total | Valid | Validity Rate |
|------|------|--------|-------|
| Analyst Signals | 4320 | 4144 | 96% |
| Trading Decisions | 1080 | 1059 | 98% |

### Grok vs. DeepSeek Comparative Analysis

| Dimension | Grok 3 | DeepSeek-V3 |
|------|--------|-----------|
| Initial Cash Allocation | Conservative (60% cash reserve) | Aggressive (90% deployed immediately) |
| Trading Frequency | Low-frequency, long-term holding | High-frequency, momentum-driven |
| Sector Diversification | Good (energy + consumer goods) | Poor (concentrated in energy + financials) |
| Maximum Drawdown | 5.5% | 14.5% |
| Buy Decision Validity | 7/11 (64%) | 1/3 (33%) |
| Response to Tariff Shock | High cash buffer absorbs decline; buys on dip | Low cash prevents loss-cutting |

### Key Findings
- **Only 1 of 9 flagship LLMs achieves profitability**: Grok 3 is the sole winner with a marginal gain of +1.1%; most models underperform a Buy-and-Hold strategy.
- **Chinese LLMs generally underperform US LLMs**: During the tariff shock period, Chinese LLMs (Qwen, GLM, Doubao) incur larger losses.
- **Risk management is the decisive differentiator**: Grok's success lies not in superior stock-picking (its signal quality is comparable to DeepSeek's) but in conservative cash management and sector diversification.
- **No model successfully anticipates the strong rebound on April 9** (AAPL: $172 → $198 USD), exposing a shared weakness among LLMs in predicting extreme events.

## Highlights & Insights
- **"Time travel is cheating"**: The paper title itself constitutes a sharp critique of prevailing LLM financial evaluation paradigms. The live benchmarking concept is generalizable to any LLM evaluation involving time-series data.
- **LLM "trading personality" analysis**: Grok resembles a cautious fund manager (low frequency, diversified, high cash); DeepSeek resembles a retail speculator (high frequency, concentrated, all-in). This anthropomorphic framing offers valuable insight into LLM decision-making styles.
- **A practical multi-agent paradigm**: The Financial Planner → Analyst Team → Portfolio Manager division of labor serves as a sound demonstration of LLM agents in financial settings.
- **Signal quality ≠ profitability**: A 96% signal validity rate and 70% win rate (Claude) do not guarantee positive returns, indicating that risk control and position sizing matter more than signal accuracy.

## Limitations & Future Work
- **Evaluation period is too short**: Only 24 trading days, coinciding with extreme volatility (FOMC + trade war), so conclusions may be specific to this particular market environment.
- **Coverage limited to 5 US stocks**: All are large-cap blue chips from Berkshire Hathaway's portfolio; applicability to small-cap, growth-oriented, or other styles remains unverified.
- **Transaction costs are excluded**: Commissions, slippage, and market impact — real-world frictions that would likely further worsen losses — are not accounted for.
- **Single LLM drives all agents**: Real-world funds might assign different LLMs to different roles (e.g., a quantitatively oriented LLM for technical analysis).
- **Future directions**: (1) Extend evaluation to 6–12 months covering bull, bear, and volatile market regimes; (2) expand to global markets and additional asset classes; (3) incorporate transaction costs and market microstructure constraints; (4) explore heterogeneous LLM teams.

## Related Work & Insights
- **vs. InvestorBench**: InvestorBench evaluates LLM trading via historical backtesting, which is susceptible to information leakage; DeepFund resolves this fundamentally through live forward testing.
- **vs. FinRL-Meta**: FinRL-Meta focuses on benchmarking reinforcement learning strategies and is not LLM-specific; DeepFund is designed specifically for LLM agents.
- **vs. LiveBench/LiveCodeBench**: The underlying philosophy is analogous — continuously updated data is used to eliminate data contamination; DeepFund extends this idea to the domain of financial investment.
- **Broader implication**: Current LLMs perform far worse than marketing narratives suggest in financial scenarios that require genuine future prediction. The gap between "appearing capable" and "actually capable" warrants serious attention.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First live LLM fund investment benchmark; the framing of the "time travel" problem is highly incisive.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 flagship LLMs + multi-dimensional financial metrics + in-depth Grok/DeepSeek case comparison.
- Writing Quality: ⭐⭐⭐⭐⭐ Vivid narrative structure (Q1–Q4 framework); the "trading personality" analysis is exceptionally readable.
- Value: ⭐⭐⭐⭐⭐ A striking contribution to the financial AI community — most LLMs lose money in live trading, puncturing the backtesting bubble.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](../../ICCV2025/llm_evaluation/batclip_bimodal_online_test-time_adaptation_for_clip.md)
- [\[AAAI 2026\] OptScale: Probabilistic Optimality for Inference-time Scaling](../../AAAI2026/llm_evaluation/optscale_probabilistic_optimality_for_inference-time_scaling.md)
- [\[AAAI 2026\] Test-time Diverse Reasoning by Riemannian Activation Steering](../../AAAI2026/llm_evaluation/test-time_diverse_reasoning_by_riemannian_activation_steering.md)
- [\[ICLR 2026\] GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time](../../ICLR2026/llm_evaluation/guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti.md)
- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](../../ACL2026/llm_evaluation/beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)

</div>

<!-- RELATED:END -->
