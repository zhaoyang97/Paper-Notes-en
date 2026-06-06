---
title: >-
  [Paper Note] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination
description: >-
  [ACL 2026][Multi-Agent][LLM Trading Agent] Ours proposes the ATLAS multi-agent financial trading framework and the Adaptive-OPRO prompt optimization method. It prepares heterogeneous market information through specialize…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "LLM Trading Agent"
  - "Prompt Optimization"
  - "Multi-Agent Collaboration"
  - "Financial Decision Making"
  - "Adaptive Strategy"
date: 2026-05-08
content_hash: 9dbc53876e0f68d5
---

# ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination

**Conference**: ACL 2026  
**arXiv**: [2510.15949](https://arxiv.org/abs/2510.15949)  
**Code**: Pending release  
**Area**: LLM Agent / Finance  
**Keywords**: LLM Trading Agent, Prompt Optimization, Multi-Agent Collaboration, Financial Decision Making, Adaptive Strategy

## TL;DR

Ours proposes the ATLAS multi-agent financial trading framework and the Adaptive-OPRO prompt optimization method. It prepares heterogeneous market information through specialized analyst agents and dynamically optimizes the central trading agent's instruction prompts based on delayed noisy feedback, significantly surpassing baselines across various market volatility environments.

## Background & Motivation

**Background**: LLMs show potential in financial decision-making for processing multi-source data and reasoning complex scenarios, but translating this capability into reliable trading systems faces major challenges.

**Limitations of Prior Work**: (1) Lack of a unified framework for systematic integration of heterogeneous information sources (technical indicators, news, fundamentals); (2) Static decision strategies are insufficient for market dynamics under delayed and noisy reward signals; (3) Existing methods often use manual prompts and fail to adapt to different market environments.

**Key Challenge**: Financial trading is inherently a sequential decision-making problem with temporal coupling between decisions and delayed rewards—yet existing prompt optimization methods (e.g., OPRO) assume instantaneous feedback and independent instances.

**Goal**: Build a unified LLM trading agent framework to solve the two core problems of information integration and behavioral adaptation.

**Key Insight**: Extend prompt optimization from single-turn immediate feedback to delayed noisy feedback scenarios in sequential decision-making.

**Core Idea**: Adaptive-OPRO—adapting the meta-optimization concept of OPRO to trading scenarios, achieving stable prompt iterative optimization through rolling evaluation windows and template separation.

## Method

### Overall Architecture

ATLAS consists of three core components: (1) Market Intelligence Pipeline—three specialized analyst agents processing technical, news, and fundamental information; (2) Decision and Execution Layer—the Central Trading Agent (CTA) generates orders and executes them in the StockSim simulator; (3) Feedback Mechanism—Adaptive-OPRO dynamically optimizes the CTA's instruction prompts based on execution feedback.

### Key Designs

1.  **Market Intelligence Pipeline (Three-Expert Architecture)**:

    - **Function**: Structures heterogeneous information sources into consistent decision inputs.
    - **Mechanism**: Market Analyst generates multi-time-scale technical summaries (2y/6m/3m), News Analyst aggregates news into structured fields (sentiment, key events, market relevance), and Fundamental Analyst extracts material changes from financial reports and corporate events.
    - **Design Motivation**: Separation of information preparation and decision-making; each analyst focuses on a specific modality.

2.  **Adaptive-OPRO Prompt Optimization**:

    - **Function**: Dynamically updates trading instructions based on delayed noisy feedback.
    - **Mechanism**: Maintains an instruction prompt $P_t$ and optimization history $\mathcal{H} = \{(P_i, s_i)\}$, evaluating every $K=5$ trading days. An Optimizer LLM generates new instructions $P_{t+1} = U(M, \mathcal{H}, s_t, \text{summary})$; the score mapping is $s = \text{clip}_{[0,100]}(50 + 250 \cdot \text{ROI})$.
    - **Design Motivation**: Original OPRO assumes immediate feedback and cannot handle credit assignment and delayed rewards in trading.

3.  **Template Separation Stability Mechanism**:

    - **Function**: Prevents prompt updates from breaking runtime interfaces.
    - **Mechanism**: Divides prompts into (a) editable static instructions (strategy, priority, constraints) and (b) non-editable dynamic runtime content (state, observation, tool output), allowing only the static portion to be edited.
    - **Design Motivation**: Prompt updates in sequential systems may accidentally break placeholders or output formats; forced editing locality is required.

### Loss & Training

This involves online prompt optimization rather than traditional training. ROI is calculated after each evaluation window (5 trading days) and mapped to a [0,100] score. The Optimizer LLM diagnoses failure modes, proposes revisions, summarizes changes, and predicts behavioral impacts. Candidate prompts are accepted only when template integrity is maintained.

## Key Experimental Results

### Main Results

| Model | Method | ROI(%) ↑ | Sharpe ↑ | Max DD(%) ↓ | Win Rate(%) |
|------|------|---------|---------|------------|-------------|
| LLaMA-3.3-70B | Baseline | -9.19±1.54 | -0.091 | 16.90 | 30.28 |
| LLaMA-3.3-70B | Adaptive-OPRO | **-6.16±2.08** | **-0.066** | **14.05** | **54.36** |
| GPT-o4-mini | Baseline | -1.30±1.71 | -0.017 | 9.68 | 29.17 |
| GPT-o4-mini | Adaptive-OPRO | **9.06±0.73** | **0.094** | 11.48 | **65.28** |
| GPT-o3 | Baseline | -6.11 | -0.080 | 11.58 | 42.59 |
| Claude Sonnet 4 | Adaptive-OPRO | **0.35±1.78** | **0.008** | 14.76 | 43.45 |
| Buy & Hold | - | -8.59 | -0.071 | 20.45 | 0.00 |

### Ablation Study

| Comparison | Findings |
|------|------|
| Baseline vs Reflection | Reflection methods are unstable and perform worse on some models. |
| Baseline vs Adaptive-OPRO | Adaptive-OPRO consistently outperforms the Baseline across all models. |
| Different Modalities | Increasing information sources is not always beneficial, depending on the market environment. |
| High vs Low Volatility | Adaptive-OPRO's advantage is more pronounced in high-volatility markets. |

### Key Findings

- Adaptive-OPRO consistently outperforms Baseline and Reflection methods across all LLM families.
- GPT-o4-mini + Adaptive-OPRO is the only configuration achieving a positive ROI (9.06%).
- Additional information modalities (news, fundamentals) are not always beneficial—they may degrade performance in noisy markets.
- Reporting results from multiple runs (mean±std) is crucial for evaluating randomness.

## Highlights & Insights

- Adaptive-OPRO is the first systematic extension of OPRO to sequential decision-making scenarios.
- The template separation design elegantly solves the interface stability problem in prompt optimization.
- The finding that "more information is not necessarily better" provides important practical guidance.
- Order-level decisions (type, size, timing, price) are closer to real trading than simple directional scores.

## Limitations & Future Work

- Evaluated only in single-stock trading scenarios, without considering portfolio management.
- Sensitivity of evaluation window size $K=5$ is not fully analyzed.
- The StockSim simulator may not fully reflect real market microstructure.
- Future work can extend to multi-asset, multi-market, and longer investment horizons.

## Related Work & Insights

- OPRO (Yang et al., 2024): Original prompt optimization method assuming immediate feedback.
- CryptoTrade (Li et al., 2024): Reflective trading integrating on-chain/off-chain signals.
- TradingAgents (Xiao et al., 2025): Multi-agent trading with structured debates.
- FINCON (Yu et al., 2024): Multi-agent collaboration reinforced by conceptual language.
- Ours' Adaptive-OPRO can be generalized to other sequential decision scenarios with delayed feedback.

## Rating

- Novelty: ⭐⭐⭐⭐ Adaptive-OPRO extends prompt optimization to delayed feedback scenarios in sequential decision-making.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 7 LLM families, multiple market environments, and multiple repetitions.
- Writing Quality: ⭐⭐⭐⭐ Clear framework description and reasonable formalization.
- Value: ⭐⭐⭐⭐ Provides reference for both LLM financial applications and prompt optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](../../ICML2026/multi_agent/maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[ICML 2026\] MASPOB: Multi-Agent Prompt Optimization via GNN Surrogate + LinUCB + Coordinate Ascent](../../ICML2026/multi_agent/maspob_bandit-based_prompt_optimization_for_multi-agent_systems_with_graph_neura.md)
- [\[ACL 2026\] Explicit Trait Inference for Multi-Agent Coordination](explicit_trait_inference_for_multi-agent_coordination.md)

</div>

<!-- RELATED:END -->
