---
title: >-
  [Paper Note] MARS²: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation
description: >-
  [ACL 2026][Code Intelligence][Multi-Agent Reinforcement Learning] This paper proposes MARS², which embeds multi-agent collaboration directly into tree-structured search for reinforcement learning training. It addresses c…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Multi-Agent Reinforcement Learning"
  - "Tree Search"
  - "Code Generation"
  - "GRPO"
  - "Credit Assignment"
date: 2026-05-08
content_hash: 65e2fea227c7cebd
---

# MARS²: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation

**Conference**: ACL 2026  
**arXiv**: [2604.14564](https://arxiv.org/abs/2604.14564)  
**Code**: [GitHub](https://github.com/TsinghuaC3I/MARTI)  
**Area**: Code Intelligence / Reinforcement Learning  
**Keywords**: Multi-Agent Reinforcement Learning, Tree Search, Code Generation, GRPO, Credit Assignment

## TL;DR

This paper proposes MARS², which embeds multi-agent collaboration directly into tree-structured search for reinforcement learning training. It addresses credit assignment in complex search trajectories through path-level grouped advantages and tree-consistent reward shaping, consistently outperforming single-agent methods on code generation benchmarks.

## Background & Motivation

**Background**: Reinforcement learning paradigms represented by GRPO have made significant progress in reasoning-intensive tasks such as code generation. However, exploration in single-agent RL is limited by the model's own prior distribution, leading to insufficient trajectory diversity and premature convergence to local optima. Search-enhanced RL (e.g., TreeRL) mitigates this by introducing structured search, but the search tree remains driven by a single policy.

**Limitations of Prior Work**: Two core challenges exist: (1) Diminishing exploration returns under a single-policy prior—as training progresses, search behavior concentrates on a few high-probability branches, making it difficult to expand the exploration frontier; (2) Multi-agent collaboration lacks integration with structured search—existing multi-agent LLM reasoning frameworks (debate, voting, etc.) treat agent interaction as lightweight coordination rather than a structured exploration process, lacking mechanisms for branching, backtracking, and budget allocation.

**Key Challenge**: The exploration space of single-agent tree search is limited by a shared prior, while multi-agent collaboration is detached from structured search dynamics; the respective advantages of both remain ununified.

**Goal**: Construct a unified framework where multiple independently optimized policies collaborate within a shared tree-structured search environment, while solving the credit assignment and training stability issues in multi-agent tree search.

**Key Insight**: Model the search tree as a learnable multi-agent interaction environment rather than a static sampling process.

**Core Idea**: Multiple heterogeneous agents collaborate to generate and refine candidate solutions within a shared search topology. Agent-node pairs are selected via Thompson sampling, and effective credit assignment is achieved through path-level grouped advantages and tree-consistent reward shaping.

## Method

### Overall Architecture

Shared multi-agent tree search environment → Thompson sampling to select agents and nodes for expansion → Distinguish between generation nodes (horizontal expansion of new solutions) and refinement nodes (vertical optimization of existing solutions) → Tree-consistent reward shaping (combining signals from parent and sibling nodes) → Path-level grouped advantage calculation → Independent GRPO optimization for each agent.

### Key Designs

1.  **Multi-Agent Tree Search Interaction Environment**:
    *   **Function**: Transforms the search tree into a learnable collaborative environment for multiple agents.
    *   **Mechanism**: In each expansion step, Thompson sampling is used to first select the most promising agent and then select the associated node for expansion. It distinguishes between two types of expandable nodes: generation nodes (proposing new candidates, horizontal expansion) and refinement nodes (optimizing existing solutions, vertical refinement) to dynamically balance exploitation and exploration.
    *   **Design Motivation**: Different policy priors produce diverse exploration signals, breaking the implicit boundaries of single-policy search. The stochastic nature of Thompson sampling ensures an exploration-exploitation balance in agent selection.

2.  **Tree-Consistent Reward Shaping**:
    *   **Function**: Achieves stable reward assignment under tree-structure constraints.
    *   **Mechanism**: For each non-root node $v$, a mixed baseline is defined as $b(v) = (1-\lambda)r_{p(v)} + \lambda \cdot \mu_{C(p(v))\setminus v}$, combining the parent node reward and the mean reward of sibling nodes. The structural consistency gain is $\Delta(v) = r_v - b(v)$, and the shaped reward is $\hat{r}_v = r_v + \gamma \cdot \Delta(v)$.
    *   **Design Motivation**: Pure global tree-level advantages fail to capture hierarchical dependencies. A child node should not only receive a high global reward but also show improvement relative to its parent and outperform its siblings. $\lambda$ balances vertical improvement and horizontal competition.

3.  **Path-level Grouped Advantage and Independent Optimization**:
    *   **Function**: Extends GRPO to tree search scenarios to support independent multi-agent updates.
    *   **Mechanism**: All nodes in the search tree originate from the same input problem, naturally forming a semantic group. Tree-level relative advantages are calculated on shaped rewards as $\hat{A}_{v,j} = (\hat{r}_{v,j} - \text{mean}) / \text{std}$. Each agent performs parameter updates using only its self-generated nodes, utilizing buffered asynchronous updates to avoid synchronization waits.
    *   **Design Motivation**: Concurrent updates by multiple agents in a shared search tree introduce non-stationarity; independent optimization combined with tree-consistent reward shaping alleviates this issue.

### Loss & Training

A multi-agent version of GRPO-style policy optimization is employed (Equation 7), combined with stabilization techniques such as GSPO (replacing token-level aggregation with geometric means), length penalties (DAPO-style length reward shaping), and TIS (token-level importance sampling to align vLLM inference with FSDP training distributions). all stabilization techniques are applied equally to baselines to ensure fair comparison. Training data consists of 7,992 code generation prompts from DeepCoder.

## Key Experimental Results

### Main Results

| Model/System | Method | Pass@1 | Pass@1(MCTS) | Pass@N |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-8B | Base→GRPO→RS2→MARS² | 50.3→52.5→55.4→**58.3** | 54.3→57.1→60.6→**60.8** | 68.6→73.1→71.4→**72.3** |
| AReaL-14B | Base→GRPO→RS2→MARS² | 58.4→58.9→62.3→**64.6** | 62.9→60.7→68.0→**68.1** | 74.3→75.4→81.1→**80.2** |
| Q&A-8B System | Base→GRPO→RS2→MARS² | — | 57.2→56.0→57.2→**61.7** | 69.7→72.0→72.6→**75.4** |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Introducing weak agent DeepCoder-14B | Individual Pass@1 drops, but system Pass@1(MCTS) still improves | System-level advantage does not depend on agent capability balance |
| Without reward shaping | Unstable training curves, delayed convergence | Reward shaping provides denser intermediate supervision |
| Diversity metric comparison | MARS² is optimal in AEC and DA@K | Multi-agent approach improves solution space coverage |
| MATH dataset generalization | Pass@1(MCTS) 0.756→0.804 | Tree-level credit assignment is not limited to code tasks |

### Key Findings
*   MARS² consistently improves across all model combinations and settings: Qwen3-8B Pass@1 increased from 50.3% to 58.3% (+8.0).
*   AReaL-14B reached 64.6% after MARS² training, surpassing O4-Mini (Low) at 63.7%.
*   Single-agent RS² already outperforms GRPO, but multi-agent MARS² further pushes the performance ceiling.
*   Diversity analysis shows the advantage of MARS² primarily stems from a richer pool of candidate solutions rather than repetitive use of high-reward trajectories.

## Highlights & Insights
*   The perspective of modeling the search tree as a multi-agent learning environment is novel, unifying structured search and multi-agent collaboration.
*   Tree-consistent reward shaping elegantly solves the credit assignment problem in tree structures while encouraging both vertical improvement and horizontal competition.
*   Weak agent experiments demonstrate that system-level gains do not rely on capability balance, enhancing flexibility for practical deployment.
*   Generalization from code to mathematical reasoning verifies the universality of the framework.

## Limitations & Future Work
*   Sequential interaction in multi-agent tree search reduces training efficiency (lower rollout parallelism), requiring the development of more efficient search mechanisms.
*   Experiments focused mainly on 8B and 14B models; effects on larger scale models remain to be verified.
*   While fair comparisons were made under fixed training budgets, the diminishing returns curve with increased budgets was not fully explored.
*   The collaboration mechanism between agents (Thompson sampling) is relatively simple; superior selection strategies may exist.

## Related Work & Insights
*   **vs TreeRL**: TreeRL introduces MCTS into RL training but uses a single policy; MARS² breaks single-policy prior limitations via multi-policy interaction.
*   **vs MAPoRL**: MAPoRL achieves multi-agent collaboration through multi-turn dialogue but lacks structured search; MARS² embeds collaboration directly into search trees.
*   **vs Vanilla GRPO**: i.i.d. sampling in GRPO limits the exploration space; MARS² breaks this limit through tree structures and multiple agents.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ The multi-agent tree search RL framework is a completely new unified paradigm.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive across models, scales, and metrics, with additional validation via math generalization.
*   Writing Quality: ⭐⭐⭐⭐ Technical details are complete, formulas are clear, and experiments are logically organized.
*   Value: ⭐⭐⭐⭐ Provides an effective multi-agent extension path for search-enhanced RL, actively driving improvements in reasoning capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] ReCode: Reinforcing Code Generation with Reasoning-Process Rewards](recode_reinforcing_code_generation_with_reasoning-process_rewards.md)
- [\[AAAI 2026\] ReCode: Updating Code API Knowledge with Reinforcement Learning](../../AAAI2026/code_intelligence/recode_updating_code_api_knowledge_with_reinforcement_learning.md)
- [\[ACL 2026\] ChipSeek: Optimizing Verilog Generation via EDA-Integrated Reinforcement Learning](chipseek_optimizing_verilog_generation_via_eda-integrated_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
