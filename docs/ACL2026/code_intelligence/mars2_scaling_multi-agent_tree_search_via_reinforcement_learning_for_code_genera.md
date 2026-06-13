---
title: >-
  [Paper Note] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation
description: >-
  [ACL 2026][Code Intelligence][Multi-Agent] MARS2 proposes a multi-agent reinforcement tree search framework that embeds multiple independently optimized policies into a shared search tree for collaborative exploration. B…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Multi-Agent"
  - "Tree Search"
  - "Reinforcement Learning"
  - "Code Generation"
  - "GRPO"
date: 2026-05-08
content_hash: ce8150bde6db7e33
---

# MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation

**Conference**: ACL 2026  
**arXiv**: [2604.14564](https://arxiv.org/abs/2604.14564)  
**Code**: [https://github.com/TsinghuaC3I/MARTI](https://github.com/TsinghuaC3I/MARTI)  
**Area**: Code Intelligence  
**Keywords**: Multi-Agent, Tree Search, Reinforcement Learning, Code Generation, GRPO

## TL;DR

MARS2 proposes a multi-agent reinforcement tree search framework that embeds multiple independently optimized policies into a shared search tree for collaborative exploration. By employing Thompson sampling for agent-node selection, tree-consistent reward shaping, and path-level group advantage estimation, it consistently improves single-model Pass@1 by up to 8.0% and system-level Pass@1 (MCTS) by up to 6.5% on code generation benchmarks.

## Background & Motivation

**Background**: RLVR paradigms like GRPO have achieved significant progress in reasoning tasks such as code generation. Search-augmented RL (e.g., TreeRL) provides more diverse exploration signals by introducing MCTS structures. Multi-agent RL (MARL) generates non-stationary data distributions through multi-policy interaction, potentially overcoming the exploration limits of single policies.

**Limitations of Prior Work**: (1) Single-agent tree search is limited—the entire tree is driven by a single policy distribution, leading to diminishing returns as search behavior concentrates on a few high-probability branches during late training; (2) Multi-agent methods are decoupled from structured search—existing multi-agent reasoning frameworks (debate, voting, etc.) perform only lightweight coordination and lack structured search support like branching, backtracking, and budget allocation.

**Key Challenge**: Single-policy search converges to local optima (Challenge 1), while multi-agent collaboration lacks search structures (Challenge 2). A unified approach is needed.

**Goal**: Construct a multi-agent collaborative tree search RL framework where heterogeneous agents collaborate to generate and refine candidate solutions within a shared search tree.

**Key Insight**: Treat the search tree as a learnable multi-agent interaction environment where different agents contribute distinct policy priors, and exploration budgets are dynamically allocated via Thompson sampling.

**Core Idea**: Nodes are expanded collaboratively on a shared search tree by multiple agents, each optimized independently. Reward signals integrate information from parent and sibling nodes through tree-consistent reward shaping, while path-level group advantages ensure stable credit assignment across complex search trajectories.

## Method

### Overall Architecture

Multiple heterogeneous LLMs (e.g., Qwen3 + AReaL) act as independent agents collaborating on a shared search tree. At each step, Thompson sampling selects an agent and then a node—either a generation node (horizontal expansion of new solutions) or a refinement node (vertical improvement of existing solutions). After expansion, test cases are executed for each node to obtain rewards, and each agent is optimized independently using tree-consistent reward shaping and path-level group advantages.

### Key Designs

1.  **Thompson Sampling Agent-Node Selection**:
    *   **Function**: Dynamically balance exploration budget allocation among heterogeneous agents.
    *   **Mechanism**: Maintain Beta priors for each agent-node pair. Thompson sampling first selects an agent and then selects an expandable node for that agent. Distinguishing between generation nodes and refinement nodes balances horizontal exploration and vertical depth.
    *   **Design Motivation**: Different agents excel at different problems; Thompson sampling adaptively balances exploration and exploitation.

2.  **Tree-Consistent Reward Shaping**:
    *   **Function**: Implement hierarchical credit assignment on a multi-agent shared search tree.
    *   **Mechanism**: For each non-root node $v$, define a mixed baseline $b(v) = (1-\lambda) r_{p(v)} + \lambda \cdot \mu_{C(p(v)) \setminus v}$ (a weighted combination of parent reward and average sibling reward). The structural consistency gain is defined as $\Delta(v) = r_v - b(v)$, and the shaped reward is $\hat{r}_v = r_v + \gamma \cdot \Delta(v)$.
    *   **Design Motivation**: Child nodes must not only have high global rewards but also show improvement relative to the parent (vertical) and superiority over sibling candidates (horizontal), encouraging specialization in collaboration.

3.  **Path-level Group Advantage Estimation**:
    *   **Function**: Extend GRPO group relative advantages from parallel sampling to tree structures.
    *   **Mechanism**: All nodes in a search tree originate from the same problem, forming a natural semantic group. Tree-level group relative advantages are calculated using shaped rewards: $\hat{A}_{v,j} = (\hat{r}_{v,j} - \text{mean}) / \text{std}$. Each agent optimizes only on nodes it generated.
    *   **Design Motivation**: The parallel trajectory assumption of standard GRPO does not hold in tree search; hierarchical dependencies between parents/children and siblings must be considered.

### Loss & Training

Extend GRPO into the MARS2 objective: each agent is optimized independently using the DAPO clip-higher trick and KL regularization. Training data consists of 7,992 filtered code generation problems from DeepCoder. Evaluation benchmarks include LiveCodeBench v6 (2025.01-05).

## Key Experimental Results

### Main Results

| Model/System | Method | Pass@1 | Pass@1 (MCTS) | Pass@N |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-8B | Base | 50.3 | 54.3 | 68.6 |
| Qwen3-8B | GRPO | 52.5 (+2.2) | 57.1 (+2.8) | 73.1 |
| Qwen3-8B | RS2 | 55.4 (+5.1) | 60.6 (+6.3) | 71.4 |
| Qwen3-8B | **MARS2** | **58.3 (+8.0)** | **60.8 (+6.5)** | 72.3 |
| AReaL-14B | GRPO | 58.9 (+0.5) | 60.7 (-2.2) | 75.4 |
| AReaL-14B | **MARS2** | **64.6 (+6.2)** | **68.1 (+5.2)** | 80.2 |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| GRPO (Single agent, no search) | +2.2 Pass@1 | Baseline |
| RS2 (Single agent + Tree search) | +5.1 Pass@1 | Search structure is effective |
| MARS2 (Multi-agent + Tree search) | +8.0 Pass@1 | Multi-agent further improves performance |
| Weak Agent (DeepCoder) added | Performance still gains | Robust to agent heterogeneity |

### Key Findings

*   MARS2 consistently outperforms GRPO and single-agent tree search (RS2) across all models, with Pass@1 gains up to 8.0%.
*   For highly optimized code models (AReaL) where GRPO is nearly ineffective or even degrades, MARS2 still provides a 6.2% improvement.
*   System-level Pass@1 (MCTS) improvements reach up to 6.0%, proving that multi-agent training produces complementary policies.
*   Performance continues to improve even with the inclusion of a weak agent (DeepCoder-14B), demonstrating robustness to agent heterogeneity.
*   AReaL-14B under MARS2 achieves 64.6% Pass@1, surpassing O4-Mini (Low) at 63.7%.

## Highlights & Insights

*   Treating the search tree as a "learnable multi-agent interaction environment" rather than a static sampling process is a paradigm innovation. Every node expansion is a collaborative decision between agents.
*   Tree-consistent reward shaping simultaneously considers vertical improvement (vs. parent) and horizontal competition (vs. siblings), representing a natural extension of multi-agent credit assignment to tree structures.
*   Rigorous experimental design: training and inference configurations are explicitly separated, and all methods share the same data budget and inference framework.

## Limitations & Future Work

*   Evaluation is limited to code generation; other RLVR scenarios like mathematical reasoning remain to be verified.
*   The number of agents is fixed at 2; scaling behavior with more agents is unexplored.
*   Prior update rules for Thompson sampling are relatively simple; more complex bandit policies might be superior.
*   Multi-agent training requires running multiple models simultaneously, doubling GPU resource requirements.

## Related Work & Insights

*   **vs TreeRL**: TreeRL uses a single policy to drive the search tree, leading to diminishing exploration gains. MARS2 introduces multiple policies to break the limitations of a single-policy prior.
*   **vs MAPoRL**: MAPoRL uses multi-agent dialogue for collaboration but lacks a search structure. MARS2 embeds multiple agents into tree search, providing support for branching and backtracking.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First to unify multi-agent RL with tree search; tree-consistent reward shaping design is elegant.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models and scales, but limited to code generation tasks.
*   Writing Quality: ⭐⭐⭐⭐ Clear framework and rigorous formulas.
*   Value: ⭐⭐⭐⭐⭐ Provides a new paradigm for search-augmented RL with significant performance gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](../../AAAI2026/code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[CVPR 2026\] MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction](../../CVPR2026/code_intelligence/mm-recoder_advancing_chart-to-code_generation_with_reinforcement_learning_and_se.md)
- [\[ICLR 2026\] Breaking the SFT Plateau: Multimodal Structured Reinforcement Learning for Chart-to-Code Generation](../../ICLR2026/code_intelligence/breaking_the_sft_plateau_multimodal_structured_reinforcement_learning_for_chart-.md)
- [\[ACL 2026\] ChipSeek: Optimizing Verilog Generation via EDA-Integrated Reinforcement Learning](chipseek_optimizing_verilog_generation_via_eda-integrated_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
