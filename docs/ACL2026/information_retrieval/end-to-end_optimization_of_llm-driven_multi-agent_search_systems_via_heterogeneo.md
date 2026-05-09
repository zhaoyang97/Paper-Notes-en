---
title: >-
  [Paper Note] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning
description: >-
  [ACL 2026][Multi-Agent Search] This paper proposes MHGPO (Multi-Agent Heterogeneous Group Policy Optimization), a critic-free multi-agent RL method that achieves end-to-end optimization in a three-agent search system (Rewriter→Reranker→Answerer) through heterogeneous-group relative advantage estimation and backward reward propagation. The method captures implicit cross-agent dependencies and cross-trajectory correlations, significantly outperforming MAPPO and GRPO baselines on multi-hop QA benchmarks such as HotpotQA.
tags:
  - ACL 2026
  - Multi-Agent Search
  - MARL
  - Group Optimization
  - End-to-End Optimization
  - RAG
date: 2026-05-08
content_hash: 9af32f22231684f6
---

# End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning

**Conference**: ACL 2026
**arXiv**: [2506.02718](https://arxiv.org/abs/2506.02718)
**Code**: None
**Area**: Information Retrieval / Multi-Agent RL
**Keywords**: Multi-Agent Search, MARL, Group Optimization, End-to-End Optimization, RAG

## TL;DR

This paper proposes MHGPO (Multi-Agent Heterogeneous Group Policy Optimization), a critic-free multi-agent RL method that achieves end-to-end optimization in a three-agent search system (Rewriter→Reranker→Answerer) through heterogeneous-group relative advantage estimation and backward reward propagation. The method captures implicit cross-agent dependencies and cross-trajectory correlations, significantly outperforming MAPPO and GRPO baselines on multi-hop QA benchmarks such as HotpotQA.

## Background & Motivation

**State of the Field**: Multi-Agent Search Systems (MASS) coordinate multiple specialized LLM agents equipped with search tools to decompose tasks and perform retrieval-augmented reasoning. A common architecture is Rewriter (decomposing questions into retrieval queries) → Reranker (selecting relevant passages from retrieved results) → Answerer (generating the final answer).

**Limitations of Prior Work**: (1) Prompt engineering and single-agent SFT optimization are labor-intensive and poorly adaptive; (2) MAPPO requires large critic networks to evaluate joint actions, leading to instability and high memory overhead; (3) Group optimization algorithms such as GRPO are effective in single-context settings but do not extend straightforwardly to multi-context MASS, where multi-agent rollouts span multiple agents with disjoint local contexts; (4) Upstream agent outputs influence downstream behavior without direct gradient paths (indirect dependencies), and rollouts from the same root query explore related but distinct intermediate decisions (implicit cross-trajectory relationships).

**Root Cause**: MASS requires system-level optimization rather than single-agent optimization—yet existing MARL methods either rely on expensive critics (MAPPO) or cannot handle cross-agent dependencies across multiple contexts (GRPO).

**Paper Goals**: Design an efficient critic-free multi-agent RL method capable of capturing indirect cross-agent dependencies and implicit cross-trajectory correlations, shifting the optimization focus from local agent performance to global system success.

**Starting Point**: Parameter sharing combined with group optimization—all agents share a single LLM backbone; relative advantage estimation over heterogeneous groups compares rollouts from different prompts, and backward reward propagation attributes terminal rewards to upstream agents.

**Core Idea**: Heterogeneous group advantage estimation—by comparing rollouts originating from the same root query but differing in intermediate decisions (forming heterogeneous groups), the optimization focus shifts from "selecting the optimal local action under fixed upstream outputs" to "rewarding system behaviors that lead to global success."

## Method

### Overall Architecture

Input question → Multi-agent rollout sampling (generating $G$ complete trajectories) → Terminal reward (Answerer F1 compared against gold answer) → Backward reward propagation (from Answerer back to Reranker and Rewriter) → Heterogeneous group advantage estimation → Update shared LLM backbone (PPO objective + KL regularization).

### Key Designs

1. **Backward Reward Propagation**:

    - Function: Attributes system-level terminal rewards to upstream agents.
    - Mechanism: The terminal reward originates from the Answerer's output and is propagated backward along the trajectory to each upstream agent. For the $i$-th output of agent $k$, its shared reward is the aggregation (default: average) of rewards from all immediate downstream agents that consumed that output. Agent-specific format penalties are added to yield the final reward.
    - Design Motivation: Even without a direct gradient path between an upstream agent (e.g., Rewriter) and the terminal output, backward-propagated rewards expose indirect dependencies—poor retrieval queries lead to poor final answers.

2. **Heterogeneous Group Advantage Estimation**:

    - Function: Learns globally optimal behavior from cross-trajectory correlations.
    - Mechanism: Standard GRPO computes relative advantages only among rollouts sharing the same input (homogeneous groups). MHGPO allows groups to contain rollouts from different prompts (heterogeneous groups)—for example, different Reranker inputs resulting from different Rewriter queries for the same question. Through cross-trajectory comparison, the advantage signal no longer merely selects the optimal local action under a fixed prefix, but instead rewards system behaviors that lead to global success.
    - Design Motivation: In MASS, downstream agents receive inputs determined by upstream rollouts—the same agent receives different inputs under different upstream decisions, naturally forming heterogeneous groups.

3. **Three Rollout Sampling Strategies**:

    - Function: Balance sampling efficiency and optimization quality.
    - Mechanism: IS (Independent Sampling—purely homogeneous groups, high redundancy); FoF (Fork-of-First—sampling $G$ times at the first agent with one-to-one downstream pairing, efficient but only the entry agent has homogeneous groups); RR (Round-Robin—randomized forking points, balancing global coordination and local stability).
    - Design Motivation: IS incurs severe redundancy ($n \times G$ samples); FoF is efficient but downstream agents lack homogeneous comparison baselines; RR probabilistically varies forking points to trade off efficiency and stability.

### Loss & Training

PPO objective with KL regularization. Parameter sharing reduces MARL to multi-task learning. Training runs for 1 epoch with $G=4$, using Llama3.1-8B-Instruct as the backbone, a Wikipedia dump as the retrieval corpus, and Contriever as the retrieval backend.

## Key Experimental Results

### Main Results

**Performance on HotpotQA / 2WikiMultihopQA / MuSiQue**

| Method | HotpotQA F1 | 2WikiMHQA F1 (OOD) | MuSiQue F1 (OOD) |
|---|---|---|---|
| Llama3.1-8B (no RL) | 22.78 | 20.82 | 2.81 |
| PPO | 24.52 | 9.20 | 8.02 |
| GRPO | 27.42 | 11.03 | 9.29 |
| Search-o1 | — | — | — |
| **MHGPO-FoF** | **Highest** | **Substantially higher** | **Substantially higher** |
| **MHGPO-RR** | **Top tier** | **Top tier** | **Top tier** |

### Ablation Study

**Comparison of Sampling Strategies**

| Strategy | Sampling Efficiency | Training Stability | Performance |
|---|---|---|---|
| IS | Low (high redundancy) | High | Moderate |
| FoF | High | Moderate | High |
| FoF (os) | Moderate | Moderate | High+ |
| RR | Moderate–High | High | **Best** |

### Key Findings

- MHGPO substantially outperforms PPO and GRPO—the critic-free design is more stable, and heterogeneous groups capture cross-agent dependencies.
- PPO training is unstable with a severe drop in OOD performance (2WikiMHQA F1 of only 9.20); MHGPO generalizes better out-of-distribution.
- The RR strategy achieves the best balance between efficiency and performance—probabilistic forking points provide homogeneous comparison opportunities for all agents.
- Parameter sharing combined with the critic-free design substantially reduces memory and computational overhead.

## Highlights & Insights

- This is the first systematic study of group optimization algorithms applied to multi-agent search systems.
- Heterogeneous group advantage estimation is a natural extension of GRPO that shifts the optimization focus from local to global.
- Backward reward propagation is a concise and effective solution for handling indirect cross-agent dependencies.

## Limitations & Future Work

- Validation is limited to a three-agent MASS architecture; effectiveness on more complex topologies remains unknown.
- Parameter sharing may constrain role differentiation among agents.
- Training runs for only 1 epoch; the effect of additional training rounds has not been explored.

## Related Work & Insights

- **vs. MAPPO**: MAPPO requires large critic networks; MHGPO replaces these with group relative advantages, yielding greater efficiency and stability.
- **vs. GRPO**: GRPO supports only homogeneous groups and single-context settings; MHGPO extends to heterogeneous groups and multi-context settings.
- **vs. Search-o1**: Search-o1 integrates retrieval within a single model; MHGPO optimizes a modular multi-agent system.

## Rating

- Novelty: ⭐⭐⭐⭐ Heterogeneous group advantage estimation and backward reward propagation represent meaningful extensions of GRPO/MARL.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets with OOD evaluation, though the agent architecture is relatively simple.
- Writing Quality: ⭐⭐⭐⭐ Theoretical formalization is rigorous, and the connection to GRPO is analyzed clearly.
- Value: ⭐⭐⭐⭐⭐ Provides a practical and efficient framework for end-to-end RL optimization of LLM-based multi-agent systems.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)
- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[AAAI 2026\] OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval](../../AAAI2026/information_retrieval/opera_a_reinforcement_learning--enhanced_orchestrated_planner-executor_architect.md)
- [\[ACL 2026\] ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs](chairo_contextual_hierarchical_analogical_induction_and_reasoning_optimization_f.md)
- [\[ACL 2026\] Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring](bayesian_active_learning_with_gaussian_processes_guided_by_llm_relevance_scoring.md)

<!-- RELATED:END -->
