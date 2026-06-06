---
title: >-
  [Paper Note] Weasel: Domain-Generalizable Web Agents through Importance-Diversity Data Selection
description: >-
  [ICML 2026][LLM Agent][Data Selection] By combining goal-relevance and diversity in a trajectory step selection method, Weasel achieves a 9.7-12.5x training speedup while reducing training data to 20% of the original and…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Data Selection"
  - "Web Agent"
  - "Out-of-Distribution Generalization"
  - "Trajectory Curation"
  - "Training Efficiency"
date: 2026-05-08
content_hash: 9ed8b6de8fbf9452
---

# Weasel: Domain-Generalizable Web Agents through Importance-Diversity Data Selection

**Conference**: ICML 2026  
**arXiv**: [2605.20291](https://arxiv.org/abs/2605.20291)  
**Code**: https://github.com/fatemehpesaran310/weasel  
**Area**: LLM Agent / Web Agent  
**Keywords**: Data Selection, Web Agent, Out-of-Distribution Generalization, Trajectory Curation, Training Efficiency

## TL;DR
By combining goal-relevance and diversity in a trajectory step selection method, Weasel achieves a 9.7-12.5x training speedup while reducing training data to 20% of the original and significantly improving the generalization of Web Agents on unseen domains.

## Background & Motivation

**Background**: LLM-driven Web Agents have progressed through large-scale instruction data and strong base models, but most research evaluates within benchmarks, failing to test true generalization capabilities.

**Limitations of Prior Work**: (1) Agent performance drops significantly on websites or interaction patterns outside the training distribution; (2) Offline web interaction data is often lengthy and noisy, with expert trajectories containing high redundancy; (3) In AgentTrek, a single trajectory can reach 45 state-action pairs, and the Accessible Tree (AXTree) of each web state can reach up to 180K tokens.

**Key Challenge**: Selecting a subset of data that is both relevant and diverse within a limited budget—this is an NP-hard problem.

**Goal**: Design a trajectory selection method that simultaneously (1) improves out-of-distribution generalization and (2) reduces computational costs.

**Key Insight**: Model trajectory curation as a constrained optimization problem combining goal-conditioned importance and pairwise diversity, solved efficiently using a greedy algorithm.

**Core Idea**: Balance individual importance scores with pairwise diversity distances to select an information-dense subset from long trajectories within a fixed budget, achieving less data, high efficiency, and strong generalization.

## Method

### Overall Architecture
Three components—(1) **Trajectory Selection**: Scores steps based on goal relevance and diversity, selecting a subset greedily; (2) **Goal-Centric Pruning**: Retains local context around the target action in the AXTree; (3) **Reasoning Style Synthesis**: For reasoning-native models, replaces heterogeneous reasoning traces with self-generated reasoning processes consistent with the model's style.

### Key Designs

1. **Importance-Diversity Trajectory Subset Selection**:

    - **Function**: Selects a high-value subset of steps from long and redundant trajectories.
    - **Mechanism**: Individual importance $\Phi(t) = \text{BERTScore}(g, s_t)$; pairwise diversity $D(i,j) = \max(\delta(s_i, s_j), \delta(y_i, y_j))$. The objective is $\max_J \sum_{j\in J} \Phi(j) + \lambda \sum_{i<j, i,j\in J} D(i,j)$ subject to $|J| = T_0 \ll T$. A greedy algorithm first selects the best pair, then iteratively adds elements with the maximum marginal gain $i_m = \arg\max_{k \notin J_{m-1}} \Phi(k) + \lambda \sum_{i \in J_{m-1}} D(k,i)$.
    - **Design Motivation**: Maximizing importance alone tends to select redundant, similar states; adding diversity ensures coverage of heterogeneous web pages/states/interaction patterns. The problem is an NP-hard max-sum diversification task; the greedy approach falls within the top 1% optimal solutions for 99.7% of trajectories, with an approximation ratio of $0.9999\pm0.0005$.

2. **Goal-Centric AXTree Pruning**:

    - **Function**: Preserves the local context related to the target action within the web state while removing peripheral redundancy.
    - **Mechanism**: Given a linearized AXTree node sequence $V_t$ and a target position $k_t^*$, a continuous window of size $2w+1$ is retained; fixed-length prefixes are used for non-node actions (e.g., goto).
    - **Design Motivation**: AgentTrek single states can reach 180K tokens; retaining content near the target significantly reduces computation (2x speedup). Experiments verify that success rates drop linearly as the offset distance increases.

3. **Reasoning Style Synthesis**:

    - **Function**: Generates reasoning processes for reasoning-native models that are consistent with their pre-training style.
    - **Mechanism**: For each selected step $t \in J^*$, the target model generates $\hat{r}_t$ based on $g, h_t, \tilde{s}_t, a_t$; the training objective is $\max_\theta \sum_{\tau \in \mathcal{D}} \sum_{t \in J^*(\tau)} \log \pi_\theta(a_t, \hat{r}_t | g, h_t, \tilde{s}_t)$.
    - **Design Motivation**: Reasoning-native models like Qwen3 learn specific reasoning styles during pre-training; training on heterogeneous model traces causes style mismatch, which harms generalization. Table 4 shows style synthesis alone improves SR from 17.0% to 21.2%.

## Key Experimental Results

### Main Results

| Dataset | Model | Training Config | WebArena-Lite | WebArena | MiniWob | Training Speedup |
|--------|------|----------|---------------|----------|---------|----------|
| AgentTrek | Qwen2.5-7B | Full (52K) | 10.9 | 8.7 | 44.6 | 1.0× |
| AgentTrek | Qwen2.5-7B | Weasel (10K) | 14.5 | 9.5 | 48.0 | 11.3× |
| AgentTrek | Gemma3-4B | Full (52K) | 9.1 | 4.3 | 28.6 | 1.0× |
| AgentTrek | Gemma3-4B | Weasel (10K) | 11.5 | 5.5 | 30.6 | 12.5× |
| AgentTrek | Qwen3-8B | Full (52K) | 17.7 | 18.2 | 59.4 | 1.0× |
| AgentTrek | Qwen3-8B | Weasel (10K) | 21.2 | 19.2 | 61.9 | 10.7× |

### Ablation Study

| Method | Data Selection | Reasoning Synthesis | WebArena-Lite |
|------|----------|----------|---------------|
| Base Qwen3-8B | ✗ | ✗ | 16.4 |
| SFT (Random) | ✗ | ✗ | 16.5 |
| SFT + Reasoning Synthesis | ✗ | ✓ | 18.2 |
| Weasel w/o Reasoning Synthesis | ✓ | ✗ | 17.0 |
| **Weasel (Full)** | **✓** | **✓** | **21.2** |

### Key Findings
- Consistently outperforms full-data SFT across three LLMs with 9.7-12.5x training speedup, reaching or exceeding full fine-tuning performance with only 20% of the data.
- Necessity of Diversity: State-only diversity yields 9.7%, action-only 13.9%, while combined yields 14.5%.
- Importance-Diversity Balance: Importance-only yields 10.9%, diversity-only 7.9%, while combined yields 14.5%.
- Cross-Domain Transfer: Ported to AITW Android GUI settings, a 3.1K subset outperforms random sampling (5.8% → 6.6%).

## Highlights & Insights
- **Elegant Problem Modeling**: Formulated as max-sum diversification, providing both theoretical grounding (greedy approximation guarantees) and practical efficiency.
- **Multidimensional Diversity Design**: Simultaneously considers diversity in both state and action spaces.
- **Key Insight on Reasoning Style Matching**: Discovery of reasoning-native models' sensitivity to the reasoning style of training data, with performance jumping from 17% to 21%.
- **End-to-End Solution**: Data selection + state pruning + reasoning adaptation forms a complete pipeline.

## Limitations & Future Work
- Greedy algorithm theoretical guarantees do not apply to pseudo-distances (non-metrics), serving only as a heuristic.
- Improvements in multimodal GUI experiments are marginal (5.8% → 6.6%); vision-dominant scenarios require further optimization.
- BERTScore evaluations may themselves be biased.
- Improvements: Exploring learned weights for importance/diversity; combining with in-context learning; researching more flexible style adaptation for reasoning models.

## Related Work & Insights
- **vs WebRL / WebAgent-R1**: Online interaction RL vs. our offline data curation, avoiding environment rollout costs.
- **vs General Data Selection**: General methods focus on model-independent sample representativeness; Weasel is designed specifically for web agents using goal-conditioned importance.
- **vs Other State Pruning**: Lee et al. introduce extra parameters via learned modules or retrieval; Weasel is a lightweight parameter-free design.
- **vs Instruction Tuning Optimization**: Preference optimization targets alignment; Weasel optimizes both generalization and efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce max-sum diversification to web agent data selection.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 LLMs + multiple benchmarks + cross-domain validation + full ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with rigorous formalization.
- Value: ⭐⭐⭐⭐⭐ Directly beneficial for practical web agent deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption Through Empirical and Theoretical Analysis](../../AAAI2026/llm_agent/promoting_sustainable_web_agents_benchmarking_and_estimating_energy_consumption_.md)
- [\[ICML 2026\] Scaling Small Agents Through Strategy Auctions](scaling_small_agents_through_strategy_auctions.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](../../AAAI2026/llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)

</div>

<!-- RELATED:END -->
