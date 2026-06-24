---
title: >-
  [Paper Note] Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search
description: >-
  [ICML 2025][LLM Agent][Atomic Facts] Proposes LWM-Planner, which extracts "atomic facts" from interaction trajectories to enhance LLM world model simulation and combines this with recursive lookahead search to improve agent planning purely in-context. It significantly outperforms ReAct and Reflexion on tasks like ALFWorld.
tags:
  - "ICML 2025"
  - "LLM Agent"
  - "Atomic Facts"
  - "Lookahead Search"
  - "in-context learning"
  - "world model"
date: 2026-05-08
content_hash: 7523fd7df37dcdb3
---

# Improving LLM Agent Planning with In-Context Learning via Atomic Fact Augmentation and Lookahead Search

**Conference**: ICML 2025  
**arXiv**: [2506.09171](https://arxiv.org/abs/2506.09171)  
**Code**: To be confirmed  
**Area**: LLM Agent  
**Keywords**: LLM Agent, Atomic Facts, Lookahead Search, in-context learning, world model

## TL;DR

Proposes LWM-Planner, which extracts "atomic facts" from interaction trajectories to enhance LLM world model simulation and combines this with recursive lookahead search to improve agent planning purely in-context. It significantly outperforms ReAct and Reflexion on tasks like ALFWorld.

## Background & Motivation

### Core Challenges in LLM Agent Planning
- Inability to efficiently utilize historical experience (stuffing entire trajectories into the context is highly inefficient).
- Lack of an explicit world model to simulate future states.

### Limitations of Prior Work
- ReAct: Lacks cross-episode learning capabilities.
- Reflexion: Only generates high-level reflections/suggestions, which are not structured enough.
- RAP(MCTS): Requires environment interaction to expand the search tree, which is highly costly.

### Key Insight
LLMs possess substantial prior knowledge about world dynamics. Extracting concise "atomic facts" (e.g., "object X is in container Y") from experiences can significantly enhance simulation and planning capabilities.

## Method

### Overall Architecture: LWM-Planner
Maintains short-term interaction history and a long-term set of atomic facts, selecting actions via recursive lookahead search.

### Key Design 1: Atomic Fact Extraction
At the end of each episode, minimal units of knowledge in the form of "obstacle_at(3,0)" are extracted from the trajectory and added to the long-term fact base.

### Key Design 2: LLM as a Latent World Model
The LLM simultaneously acts as three roles (all enhanced by atomic facts):
- **Action Proposer**: Generates candidate actions.
- **World Model**: Predicts the next state, reward, and termination condition.
- **Value Estimator**: Estimates the long-term value of leaf nodes.

### Key Design 3: Recursive Lookahead Search
- Depth limit $d=3$, branching factor $b=4$.
- $Q(o,a) = r' - \lambda_{step} + \gamma \cdot \hat{V}(o')$
- Result caching reduces redundant LLM calls, and a temperature of 0 ensures determinism.

### Theoretical Motivation
The framework is formalized as an MDP with fact-based state abstraction, where the performance loss is bounded by three errors: $\epsilon_{sim}$, $\delta_{model}$, and $\epsilon_{plan}$.

## Key Experimental Results

### Main Results (Normalized Cumulative Return)

| Method | TextFrozenLake | CrafterMini | ALFWorld-A | ALFWorld-B | ALFWorld-C |
|------|---------------|-------------|------------|------------|------------|
| LWM-Planner | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** |
| ReAct + FEC | 89.6 | 99.9 | 22.0 | 67.7 | 54.4 |
| ReAct | -165.7 | 86.7 | 59.1 | 55.9 | 64.1 |

### Step Efficiency (Steps Required per Successful Episode)

| Method | TextFrozenLake | CrafterMini | ALFWorld-A |
|------|---------------|-------------|------------|
| LWM-Planner | **6.0** | **46.5** | **8.4** |
| ReAct + FEC | — | 41.4 | 14.6 |
| ReAct | — | 50.7 | 24.7 |

### Key Findings
1. LWM-Planner achieves the highest returns across all environments with near-optimal step counts.
2. Ablation analysis of ReAct+FEC demonstrates the effectiveness of fact extraction, which is further enhanced by lookahead search.
3. Fully in-context learning, requiring no weight updates.
4. Atomic facts accumulate across episodes, enabling continuous self-improvement of the agent.

## Highlights & Insights

1. The concept of "atomic facts" is elegant and concise: it is more efficient than retrieving entire trajectories and more precise than high-level reflections.
2. A unified design where the LLM simultaneously acts as the world model, value function, and policy.
3. The combination of search and experience is more robust than either pure reflection (Reflexion) or pure search.
4. Pure in-context learning means that adaptation to new environments requires absolutely no training.

## Limitations & Future Work

1. High volume of LLM inference calls, making multiple LLM queries per step costly.
2. The quality of atomic facts depends entirely on the LLM's reflection capabilities.
3. The experimental environments are relatively simple; performance in more complex, real-world environments remains to be validated.
4. Managing the long-term growth of the fact database could become a bottleneck.

## Related Work & Insights

- **vs ReAct (Yao et al. 2023)**: ReAct alternates between reasoning and acting but lacks a world model; this work leverages atomic facts to enhance the LLM for multi-step simulation.
- **vs Reflexion (Shinn et al. 2023)**: Reflexion produces high-level reflective suggestions, whereas this work extracts structured atomic facts for the world model, offering a finer granularity of information.
- **vs RAP (Kagaya et al. 2024)**: RAP retrieves full history trajectories to perform MCTS, which leads to long and redundant contexts; this work achieves higher efficiency by performing lookahead search with refined facts.
- **vs Dyna Architecture (Sutton 1990)**: This work represents an LLM-based implementation of Dyna—replacing the traditional parameterized world model with a fact set and utilizing LLM simulation for planning.
- **Insights**: Atomic facts can be integrated with symbolic reasoning systems or scaled to shared world knowledge in multi-agent cooperative scenarios.
- **Potential Direction**: Combining atomic facts with RAG to perform hierarchical fact management in large-scale environments.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The trinity design of atomic facts, LLM world model, and lookahead search is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient verification on TextFrozenLake and ALFWorld, but more complex environments remain to be tested.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical motivation and algorithmic descriptions, with a natural analogy to Dyna.
- Value: ⭐⭐⭐⭐⭐ Provides a systematic solution for online pure in-context learning in LLM agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tree Search for LLM Agent Reinforcement Learning](../../ICLR2026/llm_agent/tree_search_for_llm_agent_reinforcement_learning.md)
- [\[ACL 2025\] FACT-AUDIT: An Adaptive Multi-Agent Framework for Dynamic Fact-Checking Evaluation of Large Language Models](../../ACL2025/llm_agent/fact_audit_factcheck.md)
- [\[ACL 2026\] TheraAgent: Self-Improving Therapeutic Agent for Precise and Comprehensive Treatment Planning](../../ACL2026/llm_agent/theraagent_self-improving_therapeutic_agent_for_precise_and_comprehensive_treatm.md)
- [\[NeurIPS 2025\] LLM Agents for Knowledge Discovery in Atomic Layer Processing](../../NeurIPS2025/llm_agent/llm_agents_for_knowledge_discovery_in_atomic_layer_processing.md)
- [\[CVPR 2026\] Learning to Adapt: Self-Improving Web Agent via Cognitive-Aware Exploration](../../CVPR2026/llm_agent/learning_to_adapt_self-improving_web_agent_via_cognitive-aware_exploration.md)

</div>

<!-- RELATED:END -->
