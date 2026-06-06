---
title: >-
  [Paper Note] SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents
description: >-
  [ACL2026][LLM Agent][Self-evolving agents] SEARL jointly optimizes agent policy parameters and external Tool Graph Memory. It addresses long-trajectory credit assignment through tool-anchored step-level advantages and pr…
tags:
  - "ACL2026"
  - "LLM Agent"
  - "Self-evolving agents"
  - "tool graph memory"
  - "RLVR"
  - "credit assignment"
  - "MCP tools"
date: 2026-05-08
content_hash: 0e6f6d062e4704fd
---

# SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents

**Conference**: ACL2026  
**arXiv**: [2604.07791](https://arxiv.org/abs/2604.07791)  
**Code**: https://github.com/circles-post/SEARL  
**Area**: LLM Agent / Reinforcement Learning  
**Keywords**: Self-evolving agents, tool graph memory, RLVR, credit assignment, MCP tools

## TL;DR
SEARL jointly optimizes agent policy parameters and external Tool Graph Memory. It addresses long-trajectory credit assignment through tool-anchored step-level advantages and process rewards, enabling small models to continuously create, reuse, and integrate tools in multi-hop QA and complex mathematical tasks.

## Background & Motivation
**Background**: RLVR has proven effective in single-turn mathematical reasoning. Agentic learning further requires models to learn from multi-step interaction trajectories, including generating tools, calling tools, and accumulating experience.

**Limitations of Prior Work**: Many tool-based agents use static tool lists, limiting their ability to adapt to new tasks. Some self-generating tool methods store tools in unstructured repositories, making reuse and composition difficult. Experience memory methods save historical trajectories but lack explicit dependency relationships. For small models, generating a "monolithic universal tool" in one go is also prone to failure.

**Key Challenge**: Training long-trajectory agents requires fine-grained credit assignment. However, the state space in open environments is massive and continuous, with identical states rarely occurring, making traditional step-level advantages based on state grouping ineffective. Meanwhile, simple process rewards are susceptible to reward hacking.

**Goal**: The authors aim to allow the agent to simultaneously optimize policy and tool-based memory during the RL training process. This ensures tools are not only created but also stored in a graph structure with dependencies, allowing them to be retrieved, merged, reused across tasks, and serve as stable anchors for step-level learning.

**Key Insight**: Tools are viewed as a more stable abstraction than raw environment states. While different tasks may have different textual states, if they call the same tool or a similar class of tools, they likely share similar sub-problem structures. Thus, step-level advantages can be aggregated by tool anchors.

**Core Idea**: Tool Graph Memory converts tools and execution dependencies into external structured states. Memory-anchored advantage is then used to assign fine-grained credit to tool creation, reuse, and execution, achieving co-evolution of policy and memory.

## Method
The core of SEARL divides agent training into two mutually reinforcing lines: the policy model learns to plan, retrieve, think, and act via RL, while the Tool Graph Memory continuously grows, merges, and connects edges within trajectories, providing reusable tools for subsequent tasks and grouping anchors for advantage estimation.

### Overall Architecture
Given a task, the agent first generates a global plan, decomposing the task into multiple subtasks. Each subtask contains four types of XML-style steps: Planning, Retrieve, Think, and Action. Retrieve selects relevant MCP tools from the Tool Graph Memory. Action can either provide a direct answer, call an existing tool, or create a new MCP tool.

During training, SEARL uses outcome rewards to judge the correctness of the final answer, while incorporating process rewards for planning, tool creation, tool execution, and formatting. Policy optimization utilizes both episode-level relative advantage and tool-memory-anchored step advantage. Tool Graph Memory acts as a directed graph preserving tool nodes and dependency edges, performing registration, retrieval, merging, and updating throughout training iterations.

### Key Designs
1. **Structured Trajectories & Composite Rewards**:
    - **Function**: Decomposes open-ended agent behavior into parsable, rewardable, and auditable steps.
    - **Mechanism**: Each task generates a high-level plan, with subtasks explicitly labeled with Retrieve, Think, and Action. Rewards consist of sparse outcome rewards and dense behavioral rewards; a correct final answer yields $r_s=11$, while local behavioral rewards include planning, tool creation, tool execution, and format rewards.
    - **Design Motivation**: If only final answer rewards are used, it is difficult to determine which action in a multi-step tool call was useful. Explicit steps allow training signals to be attributed to critical behaviors like planning and tool management.

2. **Tool-Memory-Aware Advantage Estimation**:
    - **Function**: Provides stable step-level credit assignment in long-trajectory open environments.
    - **Mechanism**: Episode-level advantage is normalized by total returns across multiple rollouts of the same task. Step-level advantage is grouped by MCP tool anchors rather than raw states. All actions related to the same or equivalent merged tools are placed in the same group, and return-to-go is used to calculate relative advantage.
    - **Design Motivation**: Identical states almost never repeat in open environments, but tools provide a more stable subtask abstraction. Grouping by tools allows comparing whether "actions related to this tool truly yield benefits" across trajectories, reducing noise caused by contextual differences.

3. **Tool Graph Memory Lifecycle**:
    - **Function**: Organizes tool knowledge and dependencies into a sustainable, growing external memory.
    - **Mechanism**: The subtask dependency graph generated by the plan is projected into the tool space to form a task subgraph. Successfully created MCP tools enter a candidate pool, and high-value tools are registered based on cumulative rewards in each training round. New tools are merged with existing ones based on cosine similarity of name/description embeddings, and edges record sequential dependencies; edges are redirected to unified nodes during merging.
    - **Design Motivation**: Unstructured tool repositories can become bloated, redundant, and difficult to reuse. The graph structure preserves not just the tools but also the knowledge of "which tools usually appear sequentially," providing combinatorial knowledge for future planning and retrieval.

### Loss & Training
The training goal is to maximize the expected reward under the policy with Tool Graph Memory, constrained by KL divergence relative to a reference policy: 
$$\max_{\pi_\theta}\mathbb{E}[r_\phi(x,y)]-\beta D_{KL}[\pi_\theta(y\mid x;\mathcal{T}_G)\|\pi_{ref}(y\mid x;\mathcal{T}_G)]$$
Specifically, the advantage includes two levels: episode-level advantage normalized by total trajectory returns, and step-level advantage normalized by return-to-go under the same tool anchor. Experiments utilized 10,000 RL training samples from Tool-star. The agent is equipped with a Python interpreter and a local Wikipedia search server. The evaluation metric is pass@1 accuracy, with Qwen3-32B as the judge.

## Key Experimental Results

### Main Results
SEARL was compared against various RL baselines in mathematical reasoning and multi-hop QA. It reached or tied for SOTA on AIME24, HotpotQA, 2Wiki, and Bamboogle, with the lowest average rank.

| Method | GSM8K | MATH500 | AIME24 | HotpotQA | 2wiki | Musique | Bamboogle | Avg Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TIR Prompt | 0.2259 | 0.0540 | 0.0000 | 0.2300 | 0.1250 | 0.0350 | 0.1200 | 5.29 |
| GRPO | 0.8870 | 0.7360 | 0.1333 | 0.2150 | 0.3450 | 0.0900 | 0.1600 | 2.43 |
| DAPO | 0.8059 | 0.5520 | 0.1333 | 0.3350 | 0.3500 | 0.0650 | 0.2480 | 3.00 |
| REINFORCE++ | 0.8658 | 0.6800 | 0.1000 | 0.1100 | 0.2600 | 0.0000 | 0.0080 | 4.57 |
| ARPO | 0.8241 | 0.6480 | 0.3333 | 0.1400 | 0.2200 | 0.0650 | 0.1760 | 3.57 |
| **SEARL** | 0.8620 | 0.6820 | 0.3333 | 0.3350 | 0.3600 | 0.0900 | 0.3040 | 1.43 |

### Ablation Study
The paper analyzes mechanisms through training curves and component ablations. While primary ablations are presented as figures without a full numerical table, the authors explicitly report that removing Step-level Grouping led to the largest degradation across most datasets, and Step Rewards also significantly impacted performance.

| Config | Key Metric / Phenomenon | Description |
| :--- | :--- | :--- |
| Full SEARL | Avg Rank 1.43 | Joint optimization of policy and tool graph memory |
| w/o Step-level Grouping | Largest degradation in AIME24, Bamboogle, etc. | Tool-anchored grouping is the core of credit assignment |
| w/o Step Rewards | Significant drop in most tasks | Final reward alone is insufficient to train tool behaviors |
| w/o Single Vanishing | Minor impact but worse stability | Prevents meaningless advantage estimates from single-element groups |
| GRPO baseline | Lower reward than SEARL, lower entropy | SEARL has denser feedback and maintains exploration |

### Key Findings
- Multi-hop QA is a strength of SEARL. It achieved 0.3350 on HotpotQA (tied with DAPO), a record 0.3600 on 2Wiki, and 0.3040 on Bamboogle, significantly higher than all baselines.
- Mathematical tasks show a trade-off. GRPO performed best on GSM8K and MATH500 with SEARL slightly lower, suggesting tool generation may introduce process noise in simple problems. However, on AIME24, SEARL tied ARPO at 0.3333, indicating complex problems benefit more from tool decomposition.
- Training dynamics show SEARL's reward consistently higher than GRPO's with higher entropy, suggesting tool-anchored advantage and process rewards maintain exploration rather than converging prematurely to fixed tool-calling patterns.
- The tool graph evolves from small, scattered subgraphs into multi-branch, cross-task functional clusters, demonstrating that memory consolidation accumulates reusable skills rather than being merely decorative.

## Highlights & Insights
- The most inspiring insight is "Tools as state abstractions." While finding identical states in open language environments is difficult, tool calls naturally aggregate similar sub-problems, providing a practical handle for long-trajectory credit assignment.
- Tool Graph Memory serves three purposes simultaneously: tool retrieval, dependency recording, and experience merging. Compared to simply storing trajectories or tool lists, the graph structure is closer to an agent’s "operational knowledge base."
- The authors do not force the model to generate one-time large tools but encourage modular tool generation for specific subtasks. This is especially important for small models, which struggle to write complex monolithic solvers.
- The slight disadvantage in GSM8K/MATH500 honestly reveals the cost of tool-based agents: simple problems do not necessarily require tools, and automated toolization might slow down or disturb reasoning.

## Limitations & Future Work
- The authors acknowledge that SEARL still lags behind methods like GRPO on GSM8K and MATH500, indicating overhead exists when generating and retrieving tools for simple problems.
- Once the toolset is formed during training, it might limit the model's adaptability to new contexts, such as direct search scenarios or highly specialized domains. The generalization boundaries of the tool graph require further testing.
- Due to model scale limitations, many generated tools remain trivial and might not be effectively reused by other LLMs. Future work may require stronger models or quality filtering mechanisms for tools.
- Reward hacking remains a risk. While process rewards mitigate sparse feedback, they might induce the model to pursue superficial signals like correct formatting or successful tool calls rather than truly improving reasoning quality.

## Related Work & Insights
- **vs GRPO / DAPO / REINFORCE++**: These methods primarily optimize the policy itself. SEARL additionally optimizes external tool graph memory, excelling in multi-hop QA where cross-step information composition is required.
- **vs ARPO**: ARPO also targets agent RL, but SEARL emphasizes tool memory structures and step-level tool anchors. Their tie on AIME24 suggests structured tools help generalization in complex math.
- **vs Alita / STELLA**: These self-generating tool methods focus on tool creation. SEARL further requires tools to be merged in a graph, retrieved, and integrated into advantage estimation.
- **Insight**: The key for long-trajectory agents is not "remembering all history" but compressing history into executable, composable, and searchable operational units. Tool graphs could be extended to API graphs, workflow graphs, or experimental procedure graphs.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The combination of Tool Graph Memory and tool-anchored advantage is innovative and addresses a core issue in agent RL credit assignment.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers math and multi-hop QA with strong baselines, though full numerical ablation tables would be preferred over charts.
- Writing Quality: ⭐⭐⭐⭐☆ The method is clearly structured, with complete formulas and workflows; some implementation details depend on the appendix.
- Value: ⭐⭐⭐⭐☆ Highly relevant for the self-evolution of small-model agents, particularly in tool-intensive and multi-hop reasoning tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](../../ICLR2026/llm_agent/exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[ACL 2026\] MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents](magma_a_multi-graph_based_agentic_memory_architecture_for_ai_agents.md)
- [\[ACL 2026\] BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search](bapo_boundary-aware_policy_optimization_for_reliable_agentic_search.md)
- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)
- [\[ACL 2026\] WebClipper: Efficient Evolution of Web Agents with Graph-based Trajectory Pruning](webclipper_efficient_evolution_of_web_agents_with_graph-based_trajectory_pruning.md)

</div>

<!-- RELATED:END -->
