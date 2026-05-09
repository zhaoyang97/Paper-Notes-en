---
title: >-
  [Paper Note] Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading
description: >-
  [NeurIPS 2025][LLM Agent][financial agents] This paper proposes FinAgent, an orchestration framework that maps each component of a traditional algorithmic trading system to a dedicated AI agent (Planner, Orchestrator, Alpha/Risk/Portfolio/Backtest/Execution/Audit/Memory agents), employs the MCP protocol for control communication and the A2A protocol for inter-agent communication, and validates the framework's feasibility on stock and BTC trading tasks.
tags:
  - NeurIPS 2025
  - LLM Agent
  - financial agents
  - algorithmic trading
  - multi-agent orchestration
  - MCP protocol
  - quantitative trading
date: 2026-05-08
content_hash: 93d98d84023137b5
---

# Orchestration Framework for Financial Agents: From Algorithmic Trading to Agentic Trading

**Conference**: NeurIPS 2025
**arXiv**: [2512.02227](https://arxiv.org/abs/2512.02227)
**Code**: [GitHub](https://github.com/)
**Area**: LLM Agent
**Keywords**: financial agents, algorithmic trading, multi-agent orchestration, MCP protocol, quantitative trading

## TL;DR

This paper proposes FinAgent, an orchestration framework that maps each component of a traditional algorithmic trading system to a dedicated AI agent (Planner, Orchestrator, Alpha/Risk/Portfolio/Backtest/Execution/Audit/Memory agents), employs the MCP protocol for control communication and the A2A protocol for inter-agent communication, and validates the framework's feasibility on stock and BTC trading tasks.

## Background & Motivation

Developing a traditional Algorithmic Trading (AT) system requires specialized teams over several years, following the pipeline: data processing → signal extraction → portfolio management → execution → evaluation. The core motivations of this paper are:

**Maturity of AI agent technology**: Frameworks such as ReAct (reasoning-acting), tool use, generative agents, reflection and memory, and multi-agent coordination have demonstrated substantial potential.

**Unique challenges of financial markets**: Strong temporal dynamics and extremely low signal-to-noise ratios make financial markets a highly demanding testbed for AI agents.

**Democratizing financial intelligence**: The goal is to make professional-grade trading strategy capabilities accessible to ordinary users.

**Mechanism**: Each component of an AT system is mapped one-to-one to a corresponding AI agent, constructing an end-to-end orchestration framework.

## Method

### Overall Architecture

The FinAgent framework consists of multiple agent pools, each responsible for one trading stage:

| Agent Pool | Function |
|:---|:---|
| **Planner** | Formulates the overall trading plan |
| **Orchestrator** | Schedules and coordinates agent pools |
| **Data Agents** | Retrieves, cleans, and aligns data from multiple sources |
| **Alpha Agents** | Proposes factor structures based on literature (no access to evaluation-period data) |
| **Risk Agents** | Computes risk exposures and sets constraints (concentration, volatility, drawdown) |
| **Portfolio Agents** | Tests long-short rules, capital and turnover constraints |
| **Execution Agents** | Converts weights to orders, accounting for slippage and transaction costs |
| **Backtest Agents** | Walk-forward backtesting and attribution analysis |
| **Audit Agents** | Validates return curves, drawdowns, and contributions |
| **Memory Agent** | Records states, prompts, tool calls, and decisions |

### Communication Protocol Design

**MCP (Model Context Protocol) for control messages**:
- The Orchestrator sends task descriptions to each agent pool via MCP (node type, task ID, input schema, strategy flags, timeout, retry budget).
- Awaits responses (acknowledgment, status, logs, artifact ID).
- Heartbeat monitoring and completion tracking.

**A2A (Agent-to-Agent) protocol for inter-agent communication**:
- Each agent pool interacts with the Memory Agent via A2A (reading historical context, uploading logs and key results).
- Message types: ask, tell, propose, confirm.
- Role-tagged with context IDs.
- Progress is shared on a schedule; failures can be taken over by peers or reassigned by the Orchestrator.

### Key Designs: Data Isolation

**Core principle**: LLM agents never access data from the evaluation window. Specifically:
- Alpha Agents design factors solely based on published literature and do not access evaluation-period data.
- All numerical signal construction and return mapping are performed by tool modules (not LLMs).
- Returns and performance metrics are hidden from LLM agents.
- Signal diagnostics (e.g., rank-IC) are computed by tool modules.

### BTC Trading Strategy Details

**Feature engineering**: Over 100 input features covering price, volume, volatility, and trend.
- Base price features: 5-minute smoothed returns, 15-minute volatility, 60-minute EWM volatility, VWAP deviation.
- Technical indicators: RSI (14/30), MACD, Bollinger Bands.
- Multi-scale momentum: 1/3/5/10/15/30/60/240 minutes.
- Volatility features: GARCH-style terms, volatility clustering.

**Prediction model**: XGBoost regression predicting next-minute return $r_{t+1}$.
- 300 trees, maximum depth 6, learning rate 0.08.
- Rolling walk-forward training, retrained every 24 hours.
- Feature importance filtering: top 70% retained.

**Signal blending**: Weighted combination of model predictions and price-action rules.

$$w_{\text{model}}(q_t) = \begin{cases} 0.10, & q_t < 0.05 \\ 0.20, & 0.05 \leq q_t < 0.10 \\ 0.40, & q_t \geq 0.10 \end{cases}$$

**Market regime detection**: Strong trend, breakout, range-bound, and high-volatility regimes, each with distinct signal weights.

## Key Experimental Results

### Main Results: Trading Performance Comparison

| Metric | Ours (Stocks) | SPY | QQQ | Equal-Weight | Ours (BTC) | BTC Buy&Hold |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Total Return ↑ | 20.42% | 16.60% | 21.59% | 47.46% | 8.39% | 3.80% |
| Annualized Return ↑ | 31.08% | 25.07% | 32.94% | 76.07% | - | - |
| Volatility ↓ | **11.83%** | 13.49% | 18.38% | 22.54% | **24.23%** | 25.82% |
| Sharpe ↑ | 2.63 | 1.86 | 1.79 | 3.37 | 0.378 | 0.170 |
| Max Drawdown ↑ | **-3.59%** | -8.89% | -14.13% | -16.21% | **-2.80%** | -5.26% |

### BTC Strategy Detailed Performance

| Metric | Agent Strategy | Buy&Hold |
|:---|:---:|:---:|
| Cumulative Return | 8.39% | 3.80% |
| Excess Return | +4.59pp | - |
| Sharpe | 0.380 | 0.168 |
| Calmar | 166.06 | 23.30 |
| Max Drawdown | -2.80% | -5.26% |
| Win Rate | 64.7% | 58.8% |
| Trades | 17 (1.04/day) | - |
| Avg. Holding Period | 16.07 hours | - |

### Key Findings

1. **Strong risk control**: The proposed method achieves the lowest volatility and smallest drawdown on both stocks and BTC.
2. **The equal-weight baseline outperforms on stocks** (47.46% vs. 20.42%), indicating that a conservative strategy may underperform a fully-invested position in a strongly trending market.
3. **The BTC experiment is more convincing**: Over 17 days of high-frequency trading, the strategy achieves +4.6pp excess return with drawdown only half that of Buy&Hold.
4. **Moderate trading frequency**: The BTC strategy averages only 1.04 trades per day, making it a medium-frequency rather than ultra-high-frequency strategy.

## Highlights & Insights

1. **Clear mapping from AT to Agentic Trading**: Mapping each stage of the mature algorithmic trading pipeline to an agent inherits the structural advantages of AT while introducing agent flexibility.
2. **Dual-protocol design (MCP + A2A) is well-motivated**: MCP serves the Orchestrator's control plane while A2A serves the agents' data plane, with clean separation of concerns.
3. **Data isolation** is the standout design: preventing LLMs from accessing evaluation-period data eliminates look-ahead bias and ensures backtest integrity.
4. **The Memory Agent** provides system auditability and state persistence.
5. **High reusability of the end-to-end framework**: The same DAG topology applies to both stocks and BTC, differing only in data sources and scheduling.

## Limitations & Future Work

1. **Short backtesting windows**: Stocks cover only 8 months (04–12/2024) and BTC only 17 days, yielding insufficient statistical significance.
2. **Stock strategy underperforms the equal-weight baseline**: 20.42% vs. 47.46% during the strong 2024 tech rally; robustness across different market regimes is not analyzed.
3. **Small stock universe**: Only 7 mega-cap technology stocks, which is not representative.
4. **No direct comparison with other agentic trading systems** (e.g., TradingAgents, AI Hedge Fund).
5. **LLM hallucination risk**: Alpha Agents generate factor structures from literature and may produce unreasonable factors.
6. **No live trading validation**: All results are from simulated backtests; latency, liquidity, and other live-trading issues are not addressed.
7. **Cost analysis is absent**: The expense of extensive LLM API calls may be substantial.

## Related Work & Insights

- **TradingAgents** (24.8K stars), **AI Hedge Fund** (42.3K stars): highly popular open-source agentic trading projects in the community.
- **FinGPT**, **FinMem**: foundational work on LLMs and memory mechanisms in the financial domain.
- **MCP Protocol**: Anthropic's Model Context Protocol; this paper is among the early works applying it to the financial domain.

## Rating

- ⭐⭐⭐ (3/5)
- **Novelty** ⭐⭐⭐: The framework design is conceptually clear, but the core contribution is largely a combination of existing components.
- **Experimental Thoroughness** ⭐⭐: Short backtesting periods, small universe, and lack of baseline comparisons weaken the empirical case.
- **Writing Quality** ⭐⭐⭐⭐: System architecture is described in detail; the technical specifics of the BTC strategy are particularly thorough.
- **Value** ⭐⭐⭐⭐: The framework design offers meaningful reference value; the MCP/A2A protocol choices provide practical guidance.
- **Theoretical Depth** ⭐⭐: Oriented toward engineering practice with limited theoretical analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/llm_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)
- [\[NeurIPS 2025\] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)
- [\[NeurIPS 2025\] Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents](agentic_plan_caching_test-time_memory_for_fast_and_cost-efficient_llm_agents.md)
- [\[NeurIPS 2025\] Agentic NL2SQL to Reduce Computational Costs](agentic_nl2sql_to_reduce_computational_costs.md)

</div>

<!-- RELATED:END -->
