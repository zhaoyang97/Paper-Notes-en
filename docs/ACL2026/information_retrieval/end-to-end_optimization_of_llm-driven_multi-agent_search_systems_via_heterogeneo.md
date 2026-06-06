---
title: >-
  [Paper Note] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning
description: >-
  [ACL 2026][Information Retrieval & RAG][Multi-Agent Search] This paper proposes MHGPO (Multi-Agent Heterogeneous Group Policy Optimization), a critic-free multi-agent RL method. By employing heterogeneous group relative…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Multi-Agent Search"
  - "MARL"
  - "Group Optimization"
  - "End-to-End Optimization"
  - "RAG"
date: 2026-05-08
content_hash: 3de3852df668b6b3
---

# End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning

**Conference**: ACL 2026  
**arXiv**: [2506.02718](https://arxiv.org/abs/2506.02718)  
**Code**: None  
**Area**: Information Retrieval / Multi-Agent RL  
**Keywords**: Multi-Agent Search, MARL, Group Optimization, End-to-End Optimization, RAG

## TL;DR

This paper proposes MHGPO (Multi-Agent Heterogeneous Group Policy Optimization), a critic-free multi-agent RL method. By employing heterogeneous group relative advantage estimation and backward reward propagation, it achieves end-to-end optimization in a three-agent search system (Rewriter→Reranker→Answerer). It captures implicit cross-agent dependencies and cross-trajectory correlations, significantly outperforming MAPPO and GRPO baselines on multi-hop QA benchmarks such as HotpotQA.

## Background & Motivation

**Background**: Multi-Agent Search Systems (MASS) decompose tasks and perform retrieval-augmented reasoning by coordinating multiple specialized LLM agents equipped with search tools. Common architectures consist of Rewriter (decomposing questions into search queries) → Reranker (selecting relevant snippets from retrieval results) → Answerer (generating the final answer).

**Limitations of Prior Work**: (1) Prompt engineering and single-agent SFT optimization methods are engineering-intensive and lack adaptability; (2) MAPPO requires large critic networks to evaluate joint actions, leading to instability and high memory overhead; (3) Group optimization algorithms like GRPO are effective in single-context settings but do not extend directly to multi-context MASS, where multi-agent rollouts span multiple agents with disjoint local contexts; (4) Upstream agent outputs affect downstream behavior without direct gradient paths (indirect dependence), and rollouts from the same root query explore related but distinct intermediate decisions (implicit cross-trajectory relationships).

**Key Challenge**: MASS requires system-level optimization rather than single-agent optimization—however, existing MARL methods either rely on expensive critics (MAPPO) or fail to handle multi-context cross-agent dependencies (GRPO).

**Goal**: Design an efficient, critic-free multi-agent RL method capable of capturing indirect cross-agent dependencies and implicit cross-trajectory correlations, shifting the optimization focus from local agent performance to global system success.

**Key Insight**: Parameter sharing + Group Optimization—all agents share a single LLM backbone. Optimization is performed through relative advantage estimation across heterogeneous groups to compare rollouts from different prompts, using backward reward propagation to attribute terminal rewards to upstream agents.

**Core Idea**: Heterogeneous Group Advantage Estimation—by comparing rollouts originating from the same root query but having different intermediate decisions (forming a heterogeneous group), the optimization focus shifts from "selecting the best local action given a fixed upstream output" to "rewarding system behaviors that lead to global success."

## Method

### Overall Architecture

Input Question → Multi-agent rollout sampling (generating $G$ complete trajectories) → Terminal reward (F1 score of Answerer compared with gold answer) → Backward reward propagation (from Answerer back to Reranker and Rewriter) → Heterogeneous group advantage estimation → Update shared LLM backbone (PPO objective + KL regularization).

### Key Designs

1. **Backward Reward Propagation**:

    - **Function**: Attributes system-level terminal rewards to upstream agents.
    - **Mechanism**: Terminal rewards start from the Answerer's output and propagate backward along the trajectory to each upstream agent. For the $i$-th output of agent $k$, its shared reward is the aggregation (defaulting to the mean) of rewards from all direct successor agents that consumed that output. Agent-specific formatting penalties are added to obtain the final reward.
    - **Design Motivation**: Even if there is no direct gradient path between an upstream agent (e.g., Rewriter) and the terminal output, backward-propagated rewards expose indirect dependencies—poor search queries lead to poor final answers.

2. **Heterogeneous Group Advantage Estimation**:

    - **Function**: Learns globally optimal behavior from cross-trajectory correlations.
    - **Mechanism**: Standard GRPO only calculates relative advantages among rollouts of the same input (homogeneous group). MHGPO allows groups to contain rollouts from different prompts (heterogeneous groups)—for instance, different Reranker inputs caused by different Rewriter queries for the same problem. Through cross-trajectory comparison, the advantage signal no longer just selects the optimal local action under a fixed prefix but instead rewards system behaviors leading to global success.
    - **Design Motivation**: In MASS, downstream agent inputs depend on upstream rollouts. The same agent receives different inputs under different upstream decisions, forming naturally heterogeneous groups.

3. **Three Rollout Sampling Strategies**:

    - **Function**: Balances sampling efficiency and optimization quality.
    - **Mechanism**: IS (Independent Sampling: pure homogeneous groups, high redundancy); FoF (Fork-on-First: samples $G$ times at the first agent transition, one-to-one downstream; efficient but only the entry agent has a homogeneous group); RR (Round-Robin: randomizes the branching point, balancing global coordination and local stability).
    - **Design Motivation**: IS has severe redundancy ($n \times G$ samples). FoF is efficient but lacks homogeneous comparison benchmarks for downstream agents. RR trades off efficiency and stability by probabilistic branching.

### Loss & Training

PPO objective function + KL regularization. Parameter sharing reduces MARL to multi-task learning. Training is conducted for 1 epoch with $G=4$ using Llama3.1-8B-Instruct as the backbone, Wikipedia dump as the retrieval corpus, and Contriever as the retrieval backend.

## Key Experimental Results

### Main Results

**Performance on HotpotQA / 2WikiMultihopQA / MuSiQue**

| Method | HotpotQA F1 | 2WikiMHQA F1(OOD) | MuSiQue F1(OOD) |
|------|------------|-------------------|-----------------|
| Llama3.1-8B (No RL) | 22.78 | 20.82 | 2.81 |
| PPO | 24.52 | 9.20 | 8.02 |
| GRPO | 27.42 | 11.03 | 9.29 |
| Search-o1 | - | - | - |
| **MHGPO-FoF** | **Highest** | **Significantly Higher** | **Significantly Higher** |
| **MHGPO-RR** | **State-of-the-art** | **State-of-the-art** | **State-of-the-art** |

### Ablation Study

**Comparison of Sampling Strategies**

| Strategy | Sampling Efficiency | Training Stability | Performance |
|------|---------|----------|------|
| IS | Low (High Redundancy) | High | Medium |
| FoF | High | Medium | High |
| FoF (os) | Medium | Medium | High+ |
| RR | Medium-High | High | **Highest** |

### Key Findings

- MHGPO significantly outperforms PPO and GRPO—the critic-free design is more stable, and heterogeneous groups capture cross-agent dependencies.
- PPO training is unstable, and OOD performance drops significantly (2WikiMHQA F1 is only 9.20), whereas MHGPO exhibits better OOD generalization.
- The RR strategy achieves the best balance between efficiency and performance—probabilistic branching points provide homogeneous comparison opportunities for all agents.
- Parameter sharing + critic-free design significantly reduce memory and computational overhead.

## Highlights & Insights

- This is the first systematic study of group optimization algorithms applied to multi-agent search systems.
- Heterogeneous group advantage estimation is a natural extension of GRPO, shifting the optimization focus from local to global.
- Backward reward propagation provides a simple and effective solution for handling indirect cross-agent dependencies.

## Limitations & Future Work

- Validation was limited to a three-agent MASS architecture; the effectiveness on more complex topologies remains unknown.
- Parameter sharing may restrict role differentiation between agents.
- Training was limited to 1 epoch; the effects of more training iterations have not been explored.

## Related Work & Insights

- **vs MAPPO**: MAPPO requires large critic networks; MHGPO replaces them with group relative advantages, making it more efficient and stable.
- **vs GRPO**: GRPO only supports homogeneous groups and single contexts; MHGPO extends this to heterogeneous groups and multiple contexts.
- **vs Search-o1**: Search-o1 integrates retrieval within a single model; MHGPO optimizes modular multi-agent systems.

## Rating

- Novelty: ⭐⭐⭐⭐ Heterogeneous group advantage estimation and backward reward propagation are meaningful extensions of GRPO/MARL.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation on multiple datasets including OOD assessment, though agent architectures are relatively simple.
- Writing Quality: ⭐⭐⭐⭐ Theoretical formalization is rigorous, and the analysis of the connection with GRPO is clear.
- Value: ⭐⭐⭐⭐⭐ Provides a practical and efficient solution for end-to-end RL optimization of LLM multi-agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)
- [\[ICML 2026\] Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning](../../ICML2026/information_retrieval/graph-r1_towards_agentic_graphrag_framework_via_end-to-end_reinforcement_learnin.md)
- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)

</div>

<!-- RELATED:END -->
