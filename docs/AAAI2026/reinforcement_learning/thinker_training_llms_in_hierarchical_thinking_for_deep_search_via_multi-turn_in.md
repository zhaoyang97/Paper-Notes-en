---
title: >-
  [Paper Note] Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction
description: >-
  [AAAI2026][Reinforcement Learning][deep search] This paper proposes the Thinker framework, which achieves structured deep search reasoning through hierarchical thinking (breadth decomposition + depth solving) and dual representation (natural language + logical functions). Combined with knowledge boundary determination to reduce unnecessary retrieval, the model is trained via SFT and significantly outperforms RL-based deep search methods across multiple QA benchmarks.
tags:
  - "AAAI2026"
  - "Reinforcement Learning"
  - "deep search"
  - "hierarchical thinking"
  - "RAG"
  - "multi-turn interaction"
  - "knowledge boundary"
date: 2026-05-08
content_hash: 224edf4353a38b14
---

# Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction

**Conference**: AAAI2026
**arXiv**: [2511.07943](https://arxiv.org/abs/2511.07943)  
**Code**: [OpenSPG/KAG-Thinker](https://github.com/OpenSPG/KAG-Thinker)  
**Area**: Reinforcement Learning
**Keywords**: deep search, hierarchical thinking, RAG, multi-turn interaction, knowledge boundary

## TL;DR
This paper proposes the Thinker framework, which achieves structured deep search reasoning through hierarchical thinking (breadth decomposition + depth solving) and dual representation (natural language + logical functions). Combined with knowledge boundary determination to reduce unnecessary retrieval, the model is trained via SFT and significantly outperforms RL-based deep search methods across multiple QA benchmarks.

## Background & Motivation
LLMs face knowledge insufficiency and hallucination problems in complex multi-hop reasoning tasks. Existing deep search methods are primarily trained via end-to-end reinforcement learning and exhibit four typical failure modes:

- **Interleaved Solving**: sub-problem solving processes are entangled and disorganized
- **Unclear Hierarchy**: lack of clear hierarchical structure in problem decomposition
- **Inconsistent Granularity**: inconsistent granularity in sub-problem decomposition
- **Inefficient Search**: retrieval is triggered for every sub-problem, even when the LLM already knows the answer

Core insight: human experts solve problems through structured thinking — first decomposing them into independent sub-problems, then solving each in turn. RL-based methods struggle to constrain the logical rigor of the reasoning process.

## Method

### Breadth Decomposition
Complex questions are decomposed at once into $n$ atomic-granularity sub-problems, each solvable independently:

- Four logical form functions are defined: **Retrieval**, **Math**, **Deduce**, and **Output**
- Each sub-problem adopts a **dual representation**:
    - **Step** (natural language): for general-purpose retrievers (e.g., E5, BGE-M3)
    - **Action** (logical function): for structured knowledge graph retrieval
- Inter-sub-problem dependencies are conveyed via variable passing (`#n`, `o_n`, `s_n`), ensuring logical coherence

Example: "Who died first, the director of *Hit Parade of 1947* or the director of *Khiladi 420*?" is decomposed into 5 sub-problems with dependencies explicitly passed through variables.

### Depth Solving
Iterative depth solving is applied to Retrieval-type sub-problems:

1. Execute retrieval → focused analysis → reasoning
2. If the current retrieved content is insufficient to answer, generate a new logical form and continue retrieval
3. Iterate until: (a) an answer is obtained (via `<<answer>>` tag), or (b) the maximum search depth $D$ is reached

### Knowledge Boundary Determination
A **"generate first, then evaluate"** strategy is adopted to avoid unnecessary external retrieval:

1. The LLM first attempts to answer the sub-problem using internal knowledge
2. Dual confidence assessment:
    - **Prompt-based**: self-reflective verification, outputting True/False
    - **Likelihood-based**: the minimum generation probability of the answer token sequence $C = \min_t p(y_t|\boldsymbol{x}, y_{<t})$ is compared against a threshold $\tau$
3. Retrieval is skipped only when both assessments return True

### Training Strategy
Multi-turn interactive SFT is adopted (rather than RL):
- The full interaction sequence $[S, U_1, A_1, \ldots, U_n, A_n]$ is concatenated
- Cross-entropy loss is computed only on Assistant response tokens
- Advantages: controllable reasoning style, suitable for vertical domain customization; compatible with parallel processing

## Key Experimental Results

### Main Results (Table 1, EM metric)

| Method | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg |
|------|-----|----------|-------|----------|-------|---------|-----------|-----|
| Search-R1 (7B) | 0.393 | 0.610 | 0.397 | 0.370 | 0.414 | 0.146 | 0.368 | 0.385 |
| ReSearch (7B) | 0.407 | 0.611 | 0.423 | 0.419 | 0.412 | 0.205 | 0.400 | 0.411 |
| **Thinker (7B)** | **0.450** | **0.642** | **0.484** | **0.421** | **0.469** | **0.221** | **0.480** | **0.452** |

- Average improvement over ReSearch: **+4.1%** (single-hop +4.5%, multi-hop +3.9%)
- More pronounced gains at 3B scale: average **+7.9%** over ReSearch

### Logical Rigor Evaluation (Table 2, GPT-4 judged)

| Method | Hierarchy | Interleaved | Granularity | Efficiency | Overall |
|------|-----------|-------------|-------------|------------|---------|
| Search-R1 | 0.813 | 0.955 | 0.852 | 0.903 | 0.638 |
| ReSearch | 0.872 | 0.967 | 0.877 | 0.922 | 0.705 |
| **Thinker** | **0.975** | **0.989** | **0.955** | **0.958** | **0.904** |

### Sample Efficiency (Table 4)
- Using only **1% of data (a few hundred samples)** achieves near-SOTA performance (Avg 0.406 vs. ReSearch 0.411)
- Full data: Avg 0.452

### Ablation Study (Table 3)
- Removing depth solving: Avg drops **3.7%** (largest single impact)
- Removing knowledge boundary: Avg drops only 0.5%, but unnecessary retrieval calls increase significantly
- Removing logical functions: marginal overall impact, but indispensable for knowledge graph retrieval

## Highlights & Insights
- **Structured reasoning process**: hierarchical thinking renders the reasoning process supervisable and verifiable, with logical rigor far exceeding RL-based methods
- **Dual representation**: natural language + logical functions are compatible with both general-purpose retrievers and structured knowledge bases
- **Exceptional sample efficiency**: a few hundred training samples suffice to approach SOTA; SFT training cost is far lower than RL
- **Knowledge boundary determination**: prompt + likelihood dual assessment effectively reduces noisy retrieval

## Limitations & Future Work
- SFT relies on a high-quality annotation data construction pipeline, which requires non-trivial engineering effort
- Logical functions are limited to four predefined types (Retrieval/Math/Deduce/Output); extensibility remains to be validated
- Evaluation uses only a text retriever (E5); knowledge graph retrieval experiments are not presented in the main results table
- The knowledge boundary determination threshold $\tau$ requires tuning

## Rating
- Novelty: ⭐⭐⭐⭐ — Hierarchical thinking + dual representation offers a clear methodological contribution; using SFT as an alternative to RL is practically motivated
- Experimental Thoroughness: ⭐⭐⭐⭐ — 7 datasets, multiple model scales, ablation and sensitivity analyses are all covered
- Writing Quality: ⭐⭐⭐⭐ — Framework diagrams are clear and problem definitions are precise
- Value: ⭐⭐⭐⭐⭐ — Provides direct practical guidance for the deep search / agentic RAG domain

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TIPS: Turn-Level Information-Potential Reward Shaping for Search-Augmented LLMs](../../ICLR2026/reinforcement_learning/tips_turn-level_information-potential_reward_shaping_for_search-augmented_llms.md)
- [\[AAAI 2026\] HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning](hcpo_hierarchical_conductor-based_policy_optimization_in_multi-agent_reinforceme.md)
- [\[NeurIPS 2025\] Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)
- [\[ICLR 2026\] Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents](../../ICLR2026/reinforcement_learning/information_gain-based_policy_optimization_a_simple_and_effective_approach_for_m.md)
- [\[ICLR 2026\] AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking](../../ICLR2026/reinforcement_learning/abstral_augmenting_llms_reasoning_by_reinforcing_abstract_thinking.md)

</div>

<!-- RELATED:END -->
