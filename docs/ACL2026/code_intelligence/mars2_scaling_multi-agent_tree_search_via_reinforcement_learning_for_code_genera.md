---
title: >-
  [Paper Note] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation
description: >-
  [ACL 2026][Code Intelligence][Multi-Agent] MARS2 proposes a multi-agent reinforcement tree search framework that embeds multiple independently optimized policies into a shared search tree for collaborative exploration. T…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Multi-Agent"
  - "Tree Search"
  - "Reinforcement Learning"
  - "Code Generation"
  - "GRPO"
date: 2026-05-08
content_hash: 34602b97b0a07a78
---

# MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation

**Conference**: ACL 2026
**arXiv**: [2604.14564](https://arxiv.org/abs/2604.14564)  
**Code**: [https://github.com/TsinghuaC3I/MARTI](https://github.com/TsinghuaC3I/MARTI)  
**Area**: Code Intelligence
**Keywords**: Multi-Agent, Tree Search, Reinforcement Learning, Code Generation, GRPO

## TL;DR

MARS2 proposes a multi-agent reinforcement tree search framework that embeds multiple independently optimized policies into a shared search tree for collaborative exploration. Through Thompson sampling for agent–node pair selection, tree-consistent reward shaping, and path-level group advantage estimation, the framework consistently improves single-model Pass@1 by up to 8.0% and system-level Pass@1 (MCTS) by up to 6.5% on code generation benchmarks.

## Background & Motivation

**Background**: RLVR paradigms such as GRPO have achieved significant progress on reasoning tasks including code generation. Search-augmented RL (e.g., TreeRL) introduces MCTS tree structures to provide more diverse exploration signals. Multi-agent RL (MARL) generates non-stationary data distributions through multi-policy interaction, offering a potential avenue to overcome the limitations of single-policy exploration.

**Limitations of Prior Work**: (1) Single-agent tree search is constrained — the entire search tree is driven by a single policy distribution, and during late-stage training, search behavior concentrates on a small number of high-probability branches, resulting in diminishing exploration gains. (2) Multi-agent methods are decoupled from structured search — existing multi-agent reasoning frameworks (debate, voting, etc.) perform only lightweight coordination and lack support for structured search operations such as branching, backtracking, and budget allocation.

**Key Challenge**: Single-policy search converges to local optima (Challenge 1), while multi-agent collaboration lacks search structure (Challenge 2). A unified framework is needed to address both.

**Goal**: To construct a multi-agent collaborative tree search RL framework in which heterogeneous agents cooperatively generate and refine candidate solutions within a shared search tree.

**Key Insight**: The search tree is treated as a learnable multi-agent interaction environment, where different agents contribute distinct policy priors and exploration budgets are dynamically allocated via Thompson sampling.

**Core Idea**: Multiple agents collaboratively expand nodes on a shared search tree, each optimized independently. Reward signals are shaped through tree-consistent reward shaping that incorporates parent-node and sibling-node information, while path-level group advantage estimation ensures stable credit assignment across complex search trajectories.

## Method

### Overall Architecture

Multiple heterogeneous LLMs (e.g., Qwen3 and AReaL) serve as independent agents collaboratively exploring a shared search tree. At each step, Thompson sampling first selects an agent and then a node — either a generation node (lateral expansion to produce new solutions) or a refinement node (vertical improvement of existing solutions). After expansion, test cases are executed on each node to obtain rewards, and each agent is optimized independently using tree-consistent reward shaping and path-level group advantage estimation.

### Key Designs

1. **Thompson Sampling for Agent–Node Selection**:

    - Function: Dynamically balances exploration budget allocation across heterogeneous agents.
    - Mechanism: A Beta prior is maintained for each agent–node pair. Thompson sampling first selects an agent, then samples a node from the agent's expandable nodes. Generation nodes (creating new candidates) and refinement nodes (improving existing candidates) are distinguished, enabling dynamic balance between lateral exploration and vertical deepening.
    - Design Motivation: Different agents excel at different problem types; Thompson sampling adaptively balances exploration and exploitation.

2. **Tree-Consistent Reward Shaping**:

    - Function: Enables hierarchical credit assignment on the multi-agent shared search tree.
    - Mechanism: For each non-root node $v$, a mixed baseline is defined as $b(v) = (1-\lambda) r_{p(v)} + \lambda \cdot \mu_{C(p(v)) \setminus v}$, a weighted combination of the parent node's reward and the mean reward of sibling nodes. The structural consistency gain is $\Delta(v) = r_v - b(v)$, and the shaped reward is $\hat{r}_v = r_v + \gamma \cdot \Delta(v)$.
    - Design Motivation: A child node should not only achieve a high global reward but also improve upon its parent (vertical) and outperform sibling candidates (horizontal), thereby encouraging specialization within collaboration.

3. **Path-Level Group Advantage Estimation**:

    - Function: Extends GRPO's group-relative advantage from parallel sampling to tree structures.
    - Mechanism: All nodes in the search tree originate from the same problem, forming a natural semantic group. The tree-level group-relative advantage is computed using shaped rewards: $\hat{A}_{v,j} = (\hat{r}_{v,j} - \text{mean}) / \text{std}$. Each agent is optimized only on the nodes it generated.
    - Design Motivation: The parallel trajectory assumption of standard GRPO does not hold in tree search, where parent–child and sibling hierarchical dependencies must be accounted for.

### Loss & Training

GRPO is extended to the MARS2 objective: each agent is optimized independently using DAPO's clip-higher technique with KL regularization. Training data consists of 7,992 filtered code generation problems from DeepCoder. The evaluation benchmark is LiveCodeBench v6 (2025.01–05).

## Key Experimental Results

### Main Results

| Model/System | Method | Pass@1 | Pass@1 (MCTS) | Pass@N |
|---|---|---|---|---|
| Qwen3-8B | Base | 50.3 | 54.3 | 68.6 |
| Qwen3-8B | GRPO | 52.5 (+2.2) | 57.1 (+2.8) | 73.1 |
| Qwen3-8B | RS2 | 55.4 (+5.1) | 60.6 (+6.3) | 71.4 |
| Qwen3-8B | **MARS2** | **58.3 (+8.0)** | **60.8 (+6.5)** | 72.3 |
| AReaL-14B | GRPO | 58.9 (+0.5) | 60.7 (−2.2) | 75.4 |
| AReaL-14B | **MARS2** | **64.6 (+6.2)** | **68.1 (+5.2)** | 80.2 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| GRPO (single-agent, no search) | +2.2 Pass@1 | Baseline |
| RS2 (single-agent + tree search) | +5.1 Pass@1 | Search structure is beneficial |
| MARS2 (multi-agent + tree search) | +8.0 Pass@1 | Multi-agent yields further gains |
| With weak agent (DeepCoder) | Performance still improves | Robust to agent heterogeneity |

### Key Findings

- MARS2 consistently outperforms GRPO and single-agent tree search (RS2) across all models, achieving up to 8.0% improvement in Pass@1.
- For highly optimized code models (AReaL), GRPO yields negligible gains or even degrades performance, while MARS2 still achieves a 6.2% improvement.
- System-level Pass@1 (MCTS) improves by up to 6.0%, demonstrating that multi-agent training genuinely produces complementary policies.
- Incorporating a weaker agent (DeepCoder-14B) still yields performance gains, indicating that the framework is robust to agent heterogeneity.
- AReaL-14B under MARS2 achieves 64.6% Pass@1, surpassing O4-Mini (Low) at 63.7%.

## Highlights & Insights

- Treating the search tree as a "learnable multi-agent interaction environment" rather than a static sampling process represents a paradigm-level innovation, where each node expansion constitutes a collaborative decision among agents.
- Tree-consistent reward shaping simultaneously accounts for vertical improvement (relative to the parent node) and horizontal competition (relative to sibling nodes), serving as a natural generalization of multi-agent credit assignment to tree structures.
- The experimental design is rigorous: training and inference configurations are clearly separated, and all methods share the same data budget and inference framework.

## Limitations & Future Work

- Evaluation is limited to code generation; applicability to other RLVR settings such as mathematical reasoning remains to be verified.
- The number of agents is fixed at two; the scaling behavior with more agents is unexplored.
- The prior update rule for Thompson sampling is relatively simple; more sophisticated bandit strategies may yield further improvements.
- Multi-agent training requires running multiple models concurrently, significantly increasing GPU resource requirements.

## Related Work & Insights

- **vs. TreeRL**: TreeRL drives the search tree with a single policy, resulting in diminishing exploration gains in late-stage training. MARS2 introduces multiple policies to break the constraint of single-policy priors.
- **vs. MAPoRL**: MAPoRL employs multi-agent dialogue for collaboration but lacks search structure. MARS2 embeds multiple agents within tree search, providing branching and backtracking support.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to unify multi-agent RL with tree search; tree-consistent reward shaping is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple models and scales, but is limited to code generation tasks.
- Writing Quality: ⭐⭐⭐⭐ Framework is clearly presented with rigorous formulations.
- Value: ⭐⭐⭐⭐⭐ Introduces a new paradigm for search-augmented RL with substantial performance gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](../../AAAI2026/code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[CVPR 2026\] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](../../CVPR2026/code_intelligence/mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](../../ICLR2026/code_intelligence/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[ICLR 2026\] ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](../../ICLR2026/code_intelligence/reasoningbank_scaling_agent_self-evolving_with_reasoning_memory.md)

</div>

<!-- RELATED:END -->
