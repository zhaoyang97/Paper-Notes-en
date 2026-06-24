---
title: >-
  [Paper Note] KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search
description: >-
  [ICML2025][LLM Agent][Knowledge Base Question Answering] KBQA-o1 is proposed, which combines a ReAct Agent with Monte Carlo Tree Search (MCTS) to perform knowledge base question answering through heuristic search driven by policy and reward models. Under low-resource settings, it improves the GrailQA F1 from 48.5% (GPT-3.5-turbo SOTA) to 78.5% using Llama-3.1-8B.
tags:
  - "ICML2025"
  - "LLM Agent"
  - "Knowledge Base Question Answering"
  - "KBQA"
  - "Monte Carlo Tree Search"
  - "MCTS"
  - "ReAct Agent"
  - "Low-resource"
  - "Logical Form Generation"
date: 2026-05-08
content_hash: b4e56566423a41fd
---

# KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search

**Conference**: ICML2025  
**arXiv**: [2501.18922](https://arxiv.org/abs/2501.18922)  
**Code**: [LHRLAB/KBQA-o1](https://github.com/LHRLAB/KBQA-o1)  
**Area**: LLM Agent  
**Keywords**: Knowledge Base Question Answering, KBQA, Monte Carlo Tree Search, MCTS, ReAct Agent, Low-resource, Logical Form Generation

## TL;DR
KBQA-o1 is proposed, which combines a ReAct Agent with Monte Carlo Tree Search (MCTS) to perform knowledge base question answering through heuristic search driven by policy and reward models. Under low-resource settings, it improves the GrailQA F1 from 48.5% (GPT-3.5-turbo SOTA) to 78.5% using Llama-3.1-8B.

## Background & Motivation
Knowledge Base Question Answering (KBQA) aims to answer natural language queries using large-scale structured knowledge bases like Freebase and Wikidata. Existing LLM-based methods face three major challenges:

**Insufficient KB Environment Perception**: End-to-end methods directly generate logical forms, making it difficult to perceive entities and relations in the KB, which leads to the generation of invalid queries.

**Local Optima vs. Search Space Explosion**: CoT methods are prone to getting stuck in local optima, while ToT methods expand the search space but face exponential growth.

**High Reliance on Labeled Data**: Training open-source LLMs requires a large volume of high-quality human annotations, which is extremely costly for large-scale KBs.

The core idea of KBQA-o1 is to model KBQA as an agent exploration process within the KB environment, utilizing MCTS to balance exploration efficiency with search quality, and reducing reliance on manual annotation through incremental fine-tuning via automatic labeling.

## Method

### Overall Architecture
KBQA-o1 comprises three core components: Agent initialization, MCTS heuristic exploration, and incremental fine-tuning.

### 1. ReAct-based Agent Design
- Eight **atomic query tools** are designed (Extract_entity, Find_relation, Merge, Order, Compare, Time_constraint, Count, Finish).
- Agent state space $\mathcal{H}$: Updated at each step by a Thought-Action-Observation triplet.
- The exploration space dynamically depends on the current state and the KB environment, with tool parameters selected from KB candidate sets.

### 2. MCTS Heuristic Search
- **Policy Model $\pi_{\text{policy}}$**: Takes the current state as input to predict the complete exploration sequence to the terminal state, trained using SFT.
- **Reward Model $\pi_{\text{reward}}$**: Takes the question as input to evaluate the quality of the generated logical form, trained using SFT.
- **Scoring Function**: $R_\pi(y|x) = \beta + \alpha \log \pi(y|x)$, where $\beta=100$ represents a perfect score.

Four stages of MCTS:
- **Selection**: The UCT algorithm balances exploration and exploitation to select child nodes.
- **Expansion**: The policy model generates candidates via beam search, matches them with KB executable options using SimCSE, and scores them using the policy model to select the top-$d$.
- **Simulation**: The best child node is simulated forward until the terminal state.
- **Back-propagation**: The reward model evaluates the complete trajectory, and Q-values are propagated back layer by layer.

### 3. Incremental Fine-tuning
- The policy and reward models are initialized with a small amount of labeled data.
- MCTS explores unlabeled questions to generate automatically annotated data.
- The reward model filters low-quality annotations, and incremental fine-tuning improves both models.

### Loss & Training
- Policy model: $\mathcal{L}_{\text{SFT}}(\pi_{\text{policy}}, \mathcal{D}_a) = -\mathbb{E}[\sum_{t=1}^l \log \pi_{\text{policy}}(\sum_{i=t}^l \mathbf{e}_i | \mathbf{h}_{t-1})]$
- Reward model: $\mathcal{L}_{\text{SFT}}(\pi_{\text{reward}}, \mathcal{D}_a) = -\mathbb{E}[\log \pi_{\text{reward}}(F_{\mathbf{h}_l} | \mathcal{Q})]$

## Key Experimental Results

### GrailQA (40-shot Low-resource Setting)

| Method | LLM | I.I.D F1 | Comp. F1 | Zero-shot F1 | Overall F1 |
|------|-----|----------|----------|--------------|------------|
| KB-BINDER | GPT-3.5-turbo | 43.3 | 36.6 | 44.0 | 42.2 |
| ARG-KBQA | GPT-3.5-turbo | 51.5 | 41.8 | 52.1 | 48.5 |
| **KBQA-o1** | Llama-3.1-8B | **85.5** | **77.6** | **76.1** | **78.5** |
| KBQA-o1 | Qwen2.5-72B | 87.4 | 83.0 | 81.9 | 82.1 |
| Fully Supervised TIARA | - | 91.2 | 74.8 | 80.7 | 81.9 |

### WebQSP (100-shot) & GraphQ (100-shot)

| Dataset | Method | F1 |
|--------|------|-----|
| WebQSP | ARG-KBQA (GPT-3.5) | 58.8 |
| WebQSP | KBQA-o1 (Llama-3.3-70B) | **67.0** |
| GraphQ | KBQA-o1 (Llama-3.3-70B) | **35.1** |

### Key Findings
- The 8B model KBQA-o1 **significantly outperforms** the SOTA method of GPT-3.5-turbo (78.5% vs. 48.5%), achieving a 30-point improvement.
- The improvement is particularly prominent in difficult scenarios such as Compositional and Zero-shot.
- It supports various open-source models such as Llama-3, Qwen2.5, and Gemma-2, exhibiting plug-and-play characteristics.
- Both MCTS agent exploration and incremental fine-tuning make significant design contributions in the ablation study.

## Highlights & Insights
1. **Modeling KBQA as an Agent+MCTS problem** is the core innovation of this work, transferring the concept of AlphaGo to knowledge base question answering.
2. The **atomic query tool design** is ingenious, with 8 tools covering all logical form construction requirements of KBQA.
3. **SimCSE matching** cleverly aligns model generation with KB environment options, resolving the issue of insufficient KB perception.
4. **Automatic annotation + incremental fine-tuning** significantly reduces annotation dependence, allowing 40-shot performance to approach or even surpass fully supervised methods.
5. The multi-model general plug-and-play design enhances the practicality of the method.

## Limitations & Future Work
1. The inference overhead brought by MCTS search is relatively large (N rollouts), resulting in high online deployment latency.
2. It is evaluated only on the Freebase dataset, and the generalization to other KBs such as Wikidata is not fully discussed.
3. The policy and reward models need to be fine-tuned separately, making the training pipeline complex.
4. The quality of automatically annotated data depends on the generalization capability of the reward model, which may lead to error accumulation.

## Related Work & Insights
- **RAP (Hao et al., 2023)**: First to combine MCTS with LLMs for reasoning; KBQA-o1 extends it to KB environments.
- **ReAct (Yao et al., 2023b)**: The foundation of the Agent framework.
- **KB-BINDER / KB-Coder**: Prior low-resource KBQA methods based on in-context learning (ICL) with GPT-3.5.
- Insight: MCTS-driven Agent exploration can be extended to other structured knowledge query tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ (First application of Agent+MCTS to KBQA, integrated framework design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three datasets, multiple models, comprehensive ablation experiments)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, theory and experiments complement one another)
- Value: ⭐⭐⭐⭐⭐ (Breakthrough work in low-resource KBQA, open-source and ready to use)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](../../ICLR2026/llm_agent/tooltree_efficient_llm_tool_planning_via_dual-feedback_monte_carlo_tree_search_a.md)
- [\[ICML 2026\] Agentic Monte Carlo: Simulating Reinforcement Learning for Black-Box Agents](../../ICML2026/llm_agent/agentic_monte_carlo_simulating_reinforcement_learning_for_black-box_agents.md)
- [\[ACL 2025\] A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems](../../ACL2025/llm_agent/multi_agent_dialect_bias_privacy_qa.md)
- [\[ACL 2026\] LiTS: A Modular Framework for LLM Tree Search](../../ACL2026/llm_agent/lits_a_modular_framework_for_llm_tree_search.md)
- [\[ICML 2025\] GuardAgent: Safeguard LLM Agents via Knowledge-Enabled Reasoning](guardagent_safeguard_llm_agents_by_a_guard_agent_via_knowledge-enabled_reasoning.md)

</div>

<!-- RELATED:END -->
