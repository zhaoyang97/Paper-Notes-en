---
title: >-
  [Paper Note] MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting
description: >-
  [NeurIPS 2025][Multi-Agent][multi-agent system] This paper proposes MASFIN, a multi-agent system that decomposes financial forecasting into multiple sub-tasks (macroeconomic analysis, industry analysis, technical analysis, sentiment analysis, etc.), with specialized LLM agents collaborating to produce more accurate and interpretable financial predictions than single-model approaches.
tags:
  - "NeurIPS 2025"
  - "Multi-Agent"
  - "multi-agent system"
  - "financial reasoning"
  - "decomposed forecasting"
  - "LLM agent"
  - "time-series analysis"
date: 2026-05-08
content_hash: 3c072d5b62a330dd
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

- [\[ICLR 2026\] PixelCraft: A Multi-Agent System for High-Fidelity Visual Reasoning on Structured Images](../../ICLR2026/multi_agent/pixelcraft_a_multi-agent_system_for_high-fidelity_visual_reasoning_on_structured.md)
- [\[ICLR 2026\] From What to Why: A Multi-Agent System for Evidence-based Chemical Reaction Condition Reasoning](../../ICLR2026/multi_agent/from_what_to_why_a_multi-agent_system_for_evidence-based_chemical_reaction_condi.md)
- [\[ACL 2025\] DocAgent: A Multi-Agent System for Automated Code Documentation Generation](../../ACL2025/multi_agent/docagent_a_multi-agent_system_for_automated_code_documentation_generation.md)
- [\[AAAI 2026\] AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation](../../AAAI2026/multi_agent/agentodrl_a_large_language_model-based_multi-agent_system_fo.md)
- [\[AAAI 2026\] Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases](../../AAAI2026/multi_agent/thucy_an_llm-based_multi-agent_system_for_claim_verification_across_relational_d.md)

</div>

<!-- RELATED:END -->
