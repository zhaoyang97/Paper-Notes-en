---
title: >-
  [Paper Note] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination
description: >-
  [ACL 2026][LLM Agent][LLM trading agent] This paper proposes ATLAS, a multi-agent financial trading framework, and Adaptive-OPRO, a prompt optimization method. ATLAS employs specialized analyst agents to prepare heteroge…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "LLM trading agent"
  - "prompt optimization"
  - "multi-agent coordination"
  - "financial decision-making"
  - "adaptive strategy"
date: 2026-05-08
content_hash: fae76e67b897f782
---

# ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination

**Conference**: ACL 2026
**arXiv**: [2510.15949](https://arxiv.org/abs/2510.15949)  
**Code**: To be released  
**Area**: LLM Agent / Finance
**Keywords**: LLM trading agent, prompt optimization, multi-agent coordination, financial decision-making, adaptive strategy

## TL;DR

This paper proposes ATLAS, a multi-agent financial trading framework, and Adaptive-OPRO, a prompt optimization method. ATLAS employs specialized analyst agents to prepare heterogeneous market information and dynamically optimizes the instruction prompt of a central trading agent based on delayed and noisy feedback, achieving significant improvements over baselines across diverse market volatility conditions.

## Background & Motivation

**Background**: LLMs have demonstrated potential in financial decision-making by processing multi-source data and reasoning over complex scenarios, yet translating this capability into reliable trading systems remains a major challenge.

**Limitations of Prior Work**: (1) No unified framework exists for systematically integrating heterogeneous information sources (technical indicators, news, fundamentals); (2) static decision strategies are insufficient for adapting to dynamic market conditions under delayed and noisy reward signals; (3) existing methods typically rely on hand-crafted prompts that cannot adapt to varying market environments.

**Key Challenge**: Financial trading is inherently a sequential decision-making problem with temporal coupling between decisions and delayed reward signals—yet existing prompt optimization methods (e.g., OPRO) assume immediate feedback and independent instances.

**Goal**: To build a unified LLM trading agent framework that addresses two core challenges: information integration and behavioral adaptation.

**Key Insight**: Extending prompt optimization from single-turn immediate feedback to sequential decision-making with delayed and noisy feedback.

**Core Idea**: Adaptive-OPRO—adapting the meta-optimization paradigm of OPRO to the trading setting, achieving stable iterative prompt optimization via rolling evaluation windows and template separation.

## Method

### Overall Architecture

ATLAS consists of three core components: (1) a **Market Intelligence Pipeline**—three specialized analyst agents handling technical, news, and fundamental information respectively; (2) a **Decision & Execution Layer**—a Central Trading Agent (CTA) generating orders executed within the StockSim simulator; (3) a **Feedback Mechanism**—Adaptive-OPRO dynamically optimizing the CTA's instruction prompt based on execution feedback.

### Key Designs

1. **Market Intelligence Pipeline (Three-Expert Architecture)**:

    - **Function**: Structures heterogeneous information sources into consistent decision inputs.
    - **Mechanism**: The Market Analyst generates multi-timescale technical summaries (2-year/6-month/3-month); the News Analyst aggregates news into structured fields (sentiment, key events, market relevance); the Fundamental Analyst extracts substantive changes from earnings reports and corporate events.
    - **Design Motivation**: Separating information preparation from decision-making allows each analyst to specialize in a specific modality.

2. **Adaptive-OPRO Prompt Optimization**:

    - **Function**: Dynamically updates trading instructions based on delayed and noisy feedback.
    - **Mechanism**: Maintains instruction prompt $P_t$ and optimization history $\mathcal{H} = \{(P_i, s_i)\}$; evaluates every $K=5$ trading days; generates new instructions via an optimizer LLM as $P_{t+1} = U(M, \mathcal{H}, s_t, \text{summary})$; score mapping: $s = \text{clip}_{[0,100]}(50 + 250 \cdot \text{ROI})$.
    - **Design Motivation**: Original OPRO assumes immediate feedback and cannot handle credit assignment and delayed rewards inherent in trading.

3. **Template Separation Stability Mechanism**:

    - **Function**: Prevents prompt updates from disrupting the runtime interface.
    - **Mechanism**: Splits the prompt into (a) editable static instructions (strategy, priorities, constraints) and (b) non-editable dynamic runtime content (state, observations, tool outputs), permitting edits only to the static portion.
    - **Design Motivation**: In sequential systems, prompt updates may inadvertently corrupt placeholders or output formats, necessitating enforced edit locality.

### Loss & Training

Rather than conventional training, ATLAS employs online prompt optimization. After each evaluation window (5 trading days), ROI is computed and mapped to a $[0, 100]$ score. The optimizer LLM diagnoses failure patterns, proposes revisions, summarizes changes, and predicts behavioral impact. Candidate prompts are accepted only when template integrity is preserved.

## Key Experimental Results

### Main Results

| Model | Method | ROI(%) ↑ | Sharpe ↑ | Max DD(%) ↓ | Win Rate(%) |
|-------|--------|---------|---------|------------|-------------|
| LLaMA-3.3-70B | Baseline | -9.19±1.54 | -0.091 | 16.90 | 30.28 |
| LLaMA-3.3-70B | Adaptive-OPRO | **-6.16±2.08** | **-0.066** | **14.05** | **54.36** |
| GPT-o4-mini | Baseline | -1.30±1.71 | -0.017 | 9.68 | 29.17 |
| GPT-o4-mini | Adaptive-OPRO | **9.06±0.73** | **0.094** | 11.48 | **65.28** |
| GPT-o3 | Baseline | -6.11 | -0.080 | 11.58 | 42.59 |
| Claude Sonnet 4 | Adaptive-OPRO | **0.35±1.78** | **0.008** | 14.76 | 43.45 |
| Buy & Hold | - | -8.59 | -0.071 | 20.45 | 0.00 |

### Ablation Study

| Comparison | Finding |
|-----------|---------|
| Baseline vs. Reflection | The Reflection method is unstable and performs worse than Baseline on certain models |
| Baseline vs. Adaptive-OPRO | Adaptive-OPRO consistently outperforms Baseline across all models |
| Different information modalities | Adding more information sources is not always beneficial; effectiveness depends on market conditions |
| High vs. low volatility | Adaptive-OPRO shows a more pronounced advantage in high-volatility markets |

### Key Findings

- Adaptive-OPRO consistently outperforms both Baseline and Reflection methods across all LLM families.
- GPT-o4-mini + Adaptive-OPRO is the only configuration achieving positive ROI (9.06%).
- Additional information modalities (news, fundamentals) are not always beneficial—they may degrade performance in noisy markets.
- Reporting multiple runs with mean±std is essential for evaluating randomness in this setting.

## Highlights & Insights

- Adaptive-OPRO represents the first systematic extension of OPRO to sequential decision-making scenarios.
- The template separation design elegantly resolves the interface stability problem in prompt optimization.
- The finding that "more information is not always better" carries important practical implications.
- Order-level decisions (type, size, timing, price) more closely reflect real trading than simple directional scoring.

## Limitations & Future Work

- Evaluation is limited to single-stock trading; portfolio management is not considered.
- Sensitivity analysis of the evaluation window size $K=5$ is insufficient.
- The StockSim simulator may not fully capture real market microstructure.
- Future work could extend to multi-asset, multi-market, and longer investment horizon settings.

## Related Work & Insights

- **OPRO** (Yang et al., 2024): The original prompt optimization method, assuming immediate feedback.
- **CryptoTrade** (Li et al., 2024): Reflective trading integrating on-chain and off-chain signals.
- **TradingAgents** (Xiao et al., 2025): Multi-agent trading with structured debate.
- **FINCON** (Yu et al., 2024): Multi-agent collaboration with conceptualized language reinforcement.
- The proposed Adaptive-OPRO is generalizable to other sequential decision-making scenarios with delayed feedback.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Adaptive-OPRO extends prompt optimization to sequential decision-making with delayed feedback.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 7 LLM families, diverse market conditions, and multiple repetitions.
- **Writing Quality**: ⭐⭐⭐⭐ Framework description is clear with reasonable formalization.
- **Value**: ⭐⭐⭐⭐ Offers reference value for both LLM financial applications and prompt optimization research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](../../NeurIPS2025/llm_agent/mat-agent_adaptive_multi-agent_training_optimization.md)
- [\[ACL 2026\] HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation](hag_hierarchical_demographic_tree-based_agent_generation_for_topic-adaptive_simu.md)

</div>

<!-- RELATED:END -->
