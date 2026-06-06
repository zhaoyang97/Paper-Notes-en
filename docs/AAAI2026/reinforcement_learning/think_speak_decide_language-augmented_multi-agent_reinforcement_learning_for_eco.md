---
title: >-
  [Paper Note] Think, Speak, Decide: Language-Augmented Multi-Agent Reinforcement Learning for Economic Decision-Making
description: >-
  [AAAI 2026][Reinforcement Learning][Multi-Agent Reinforcement Learning] This paper proposes the LAMP framework, which integrates LLM-driven language reasoning with MARL policy optimization through a Think–Speak–Decide th…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Multi-Agent Reinforcement Learning"
  - "Large Language Models"
  - "Economic Decision-Making"
  - "Language-Augmented Policy"
  - "Communication"
date: 2026-05-08
content_hash: 57d7c701b7d632bf
---

# Think, Speak, Decide: Language-Augmented Multi-Agent Reinforcement Learning for Economic Decision-Making

**Conference**: AAAI 2026
**arXiv**: [2511.12876](https://arxiv.org/abs/2511.12876)  
**Code**: [https://github.com/hey0223/LAMP](https://github.com/hey0223/LAMP)  
**Area**: Reinforcement Learning
**Keywords**: Multi-Agent Reinforcement Learning, Large Language Models, Economic Decision-Making, Language-Augmented Policy, Communication

## TL;DR

This paper proposes the LAMP framework, which integrates LLM-driven language reasoning with MARL policy optimization through a Think–Speak–Decide three-stage pipeline. The framework enables economic decision-making agents to understand and leverage natural language information (e.g., news, dialogues), achieving cumulative returns exceeding pure MARL baselines by 63.5% and LLM-only baselines by 34.0% in economic simulation environments.

## Background & Motivation

Economic decision-making relies on two types of information: **structured numerical signals** (prices, tax rates, etc.) and **unstructured linguistic information** (peer dialogues, media narratives, etc.). However:

**Limitations of Prior Work — Traditional MARL**: Standard multi-agent reinforcement learning assumes clean, structured communication protocols and cannot handle the noisy, semantically rich, and potentially deceptive natural language information present in the real world.

**Limitations of Prior Work — LLMs**: Although LLMs excel at processing language signals, most existing work uses LLMs to directly generate actions or simulate behavior, lacking systematic policy optimization, which is insufficient for solving complex economic problems or producing robust, actionable strategies.

**Core Problem**: In complex multi-agent economic environments, how can agents interpret and leverage natural language information to support optimal decision-making?

This question carries significant practical implications — scenarios such as labor markets, corporate pricing, and government policy design all involve the interactive processing of large amounts of linguistic information.

## Method

### Overall Architecture

LAMP (Language-Augmented Multi-Agent Policy) is a language-augmented multi-agent policy learning framework that follows a unified **Think–Speak–Decide** three-stage pipeline:

- **Think**: Interprets numerical observations, extracts short-term shocks and long-term trends, and caches high-value reasoning trajectories.
- **Speak**: Crafts and exchanges strategic messages based on reasoning results, and updates beliefs by parsing peer communications.
- **Decide**: Fuses numerical data, reasoning results, and reflective states as inputs to the MARL policy network to generate actions.

### Problem Formulation

The paper models the economic decision-making problem as a **partially observable Markov game** $\mathcal{M} = \langle N, S, O, A, P, R, \delta \rangle$, comprising one government agent and $N_h$ household agents. Each household's observation is augmented with linguistic information: $m_t^i = \mathcal{E}(\mathcal{L}(a_t^i, e_t^i, O_t^g))$, where $\mathcal{L}$ denotes the LLM-generated text and $\mathcal{E}$ the embedding model.

The household optimization objective is to maximize lifetime utility of consumption and leisure:

$$\max \mathbb{E}_0 \sum_{t=0}^{T_N} \beta^t \left(\frac{c_t^{1-\eta}}{1-\eta} - \frac{h_t^{1+\gamma}}{1+\gamma}\right)$$

### Key Designs

#### 1. Think Module: Dual-Path Economic Interpretation

The Think module transforms global numerical signals into shared news, providing both short-term and long-term economic interpretations:

- **Long-term news**: Generated at fixed checkpoints $L_i$ to capture structural trends: $\mathcal{R}_{L_i}^{long} = \mathcal{L}_L(O_{L_{i-1}:L_i}^g)$
- **Short-term news**: Triggered when key indicators — wealth Gini coefficient $G_w$, social welfare $\mathcal{W}$, and per capita GDP $Y$ — change beyond a threshold $\sigma$
- **Experience pool**: A short-term buffer stores each agent's top-$k_1$ reasoning trajectories; long-term experience is managed via a FAISS index supporting $k_3$-nearest-neighbor retrieval

**Design Motivation**: Mimics the way real-world economic participants rely on news media for information rather than directly receiving raw numerical data. The experience pool retains reasoning trajectories from successful strategies, enabling agents to reuse effective policies in similar scenarios.

#### 2. Speak Module: Strategic Communication and Opponent Modeling

The Speak module enables strategic information exchange among agents:

- The LLM generates three candidate statements for each agent.
- A **self-attention selector** $\mathcal{S}$ scores the candidates and samples one to broadcast to all agents.
- A **reflection module** $\mathcal{L}_{reflect}$ parses received messages and produces an assessment of each peer, including:
    - Estimated wealth tier $w_t^{i \to j} \in \{\text{low, mid, high}\}$
    - Numerical belief confidence $\tau_t^{i \to j} \in [0, 10]$
    - Self-reflection summary $\alpha_t^i$

**Design Motivation**: Enables agents to achieve coordination by exchanging strategic information and inferring opponent states, reaching higher welfare without brute-force exploration.

#### 3. Decide Module: Language-Fused Decision Making

The Decide module integrates language embeddings with numerical observations into the RL policy:

- All text (reasoning and reflection) is encoded by a text encoder $\mathcal{E}_{text}$ and pooled into a fixed-length vector.
- A projection layer $P: \mathbb{R}^D \to \mathbb{R}^d$ reduces dimensionality and aligns features: $\tilde{m}_t^i = \frac{P(h_t^i)}{\|P(h_t^i)\|_2}$
- The encoder is frozen; only the projection layer is updated to ensure training stability.
- The MADDPG framework is adopted: a centralized critic minimizes the Bellman error, while decentralized actors are updated via deterministic policy gradients.

### Loss & Training

- Based on the CTDE (Centralized Training with Decentralized Execution) paradigm.
- The centralized critic uses global state $x_t = (O_t^g, m_t^{1:N_h})$ and joint actions $a_t$.
- Standard MADDPG Bellman error loss and deterministic policy gradients are applied.
- The LLM backbone uses Qwen2.5-72B-Instruct-INT4.

## Key Experimental Results

### Main Results

Experiments are conducted on the TaxAI economic simulator across three scenarios: S1 (economic stability), S2 (economic slowdown), and S3 (crisis shock).

**S1 Scenario Main Results (Economic Stability)**:

| Category | Algorithm | Avg. Reward ↑ | Social Welfare ↑ | Consumption | Labor |
|----------|-----------|--------------|-----------------|-------------|-------|
| **Ours** | **LAMP** | **8.52 ± 0.13** | **2.56e+03 ± 37.7** | **2.30e+05** | **3.13e+05** |
| Traditional | MADDPG | 5.21 ± 0.16 | 1.17e+03 ± 551 | 5.32e+05 | 7.82e+05 |
| Traditional | Rule-Based | 7.60 ± 0.33 | 2.28e+03 ± 99.9 | 3.19e+05 | 5.68e+05 |
| LLM | ReAct | 7.44 ± 0.26 | 2.23e+03 ± 79.2 | 6.21e+05 | 1.02e+06 |
| LLM | CoT | 6.75 ± 0.34 | 2.03e+03 ± 103 | 4.35e+05 | 1.03e+06 |

**Key Gains**:
- vs. MADDPG: Reward +63.5%, Social Welfare +118.8%
- vs. Rule-Based: Reward +12.1%, Social Welfare +12.3%
- vs. ReAct (strongest LLM baseline): Reward +14.5%, Social Welfare +14.8%

### Ablation Study

| Configuration | Avg. Reward ↑ | Social Welfare ↑ | Consumption | Labor | Survival Years ↑ |
|--------------|--------------|-----------------|-------------|-------|-----------------|
| LAMP (Full) | 8.52 | 2.56e+03 | 2.30e+05 | 3.13e+05 | 300 |
| w/o Speak | 8.42 (-1%) | 2.53e+03 (-1%) | 3.24e+05 (+41%) | 5.36e+05 (+71%) | 300 |
| w/o Experience Pool | 8.45 (-1%) | 1.25e+03 (-51%) | 5.12e+05 (+122%) | 4.50e+05 (+44%) | 150 (-50%) |
| w/o Long-term Reasoning | 5.31 (-38%) | 1.15e+03 (-55%) | 2.27e+05 (-2%) | 4.10e+05 (+31%) | 219 (-27%) |
| w/o Short-term Reasoning | 8.18 (-4%) | 1.67e+03 (-35%) | 3.51e+05 (+53%) | 5.25e+05 (+68%) | 208 (-30%) |
| w/o Temporal Scheduling | 8.52 (0%) | 1.19e+03 (-53%) | 3.48e+05 (+51%) | 5.70e+05 (+82%) | 141 (-53%) |

### Key Findings

1. **Long-term reasoning is most critical**: Removing it leads to a 38% drop in reward and a 55% drop in social welfare, indicating that agents become myopic.
2. **The experience pool improves stability**: Removing it halves survival years and causes consumption to surge by 122%, suggesting agents fall into oscillation.
3. **The Speak module improves efficiency**: Removing it yields only a 1% reward drop, but consumption and labor increase substantially (+41%/+71%), indicating that without communication, agents compensate through brute force.
4. **Temporal scheduling is crucial**: Randomly triggered reasoning reduces survival years from 300 to 141.
5. **Adaptive strategy from LLM reasoning**: Upon detecting rising inequality, the LLM revises a "work more" stance toward reducing working hours and increasing savings.

## Highlights & Insights

1. **Elegant Think–Speak–Decide pipeline design**: Structures the cognitive process of human economic decision-making — thinking, communicating, and deciding — into trainable modules.
2. **Language as "compressed information"**: LLM reasoning can distill key insights from numerous fluctuating economic variables, which is difficult to achieve with purely data-driven approaches.
3. **Experience pool balances performance and interpretability**: It enables reuse of successful strategies while retaining reasoning trajectories as an auditable knowledge base.
4. **Ablation experiments reveal the decoupling of efficiency and performance**: The primary contribution of the Speak module lies in improving efficiency rather than absolute performance.

## Limitations & Future Work

1. **High computational cost**: Every step requires LLM calls (Qwen2.5-72B), constraining practical deployment.
2. **Validation limited to TaxAI**: Generalizability to broader economic scenarios (e.g., financial markets, supply chains) remains unknown.
3. **Deceptive communication by LLMs**: The paper does not thoroughly investigate whether agents learn to send deceptive messages.
4. **Frozen text encoder**: End-to-end fine-tuning may yield further performance gains.
5. **Scalability**: Communication complexity and LLM call overhead as the number of agents increases.

## Related Work & Insights

- **TaxAI** (Mi et al., 2024): Provides an economic simulation environment for tax policy design.
- **EconAgent** and **EconGym**: Applications of LLMs in economic evaluation and benchmarking.
- **MADDPG** (Lowe et al., 2017): Multi-Agent Deep Deterministic Policy Gradient.
- **FAMA / MAPoRL**: Pioneering work on MARL and LLM integration.
- Insight: LLMs can serve as "information compressors" to assist RL rather than directly replacing RL-based decision-making.

## Rating

- Novelty: ⭐⭐⭐⭐ — The Think–Speak–Decide pipeline represents a meaningful exploration of the MARL+LLM integration paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three scenarios, multiple baselines, and detailed ablations, but limited to a single environment.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and consistent notation, though notation-heavy in places.
- Value: ⭐⭐⭐⭐ — Provides a promising direction for language-augmented economic decision-making.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](explaining_decentralized_multi-agent_reinforcement_learning_policies.md)
- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](../../NeurIPS2025/reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[ICML 2026\] Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](../../ICML2026/reinforcement_learning/multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)

</div>

<!-- RELATED:END -->
