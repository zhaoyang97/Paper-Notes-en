---
title: >-
  [Paper Note] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration
description: >-
  [ACL 2025][LLM (Other)][Multi-Agent Collaboration] This paper proposes AgentDropout, which optimizes the communication topology of multi-agent systems by dynamically eliminating redundant agent nodes and communication edges in each communication round. Compared to state-of-the-art methods, it reduces prompt token consumption by an average of 21.6% and completion token consumption by 18.4%, while improving performance by 1.14 points.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Multi-Agent Collaboration"
  - "Communication Topology Optimization"
  - "Node Pruning"
  - "Edge Pruning"
  - "Token Efficiency"
date: 2026-05-08
content_hash: a32ab364d6e9e49b
---

# AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration

**Conference**: ACL 2025  
**arXiv**: [2503.18891](https://arxiv.org/abs/2503.18891)  
**Code**: [https://github.com/wangzx1219/AgentDropout](https://github.com/wangzx1219/AgentDropout)  
**Area**: LLM/NLP  
**Keywords**: Multi-Agent Collaboration, Communication Topology Optimization, Node Pruning, Edge Pruning, Token Efficiency

## TL;DR
This paper proposes AgentDropout, which optimizes the communication topology of multi-agent systems by dynamically eliminating redundant agent nodes and communication edges in each communication round. Compared to state-of-the-art methods, it reduces prompt token consumption by an average of 21.6% and completion token consumption by 18.4%, while improving performance by 1.14 points.

## Background & Motivation

**Background**: LLM-based multi-agent systems (MAS) solve complex tasks by allowing multiple agents to communicate and collaborate, showing great potential in reasoning, mathematics, and code generation. The communication structure of MAS is typically modeled as a directed graph, where nodes represent agents and edges represent communication paths.

**Limitations of Prior Work**: MAS faces two core challenges: (1) high token consumption—frequent message generation and delivery among multiple agents incur huge token costs; (2) redundant communication—not all information exchanges between agents contribute positively to the final result. Methods like AgentPrune reduce communication by pruning redundant edges, but they use a fixed set of participating roles across all communication rounds, which limits the optimization effectiveness.

**Key Challenge**: Existing methods only consider "which communications are redundant" (edge pruning), while ignoring "which agent roles are redundant" (node pruning). Different discussion stages may require different agent combinations, and fixed role assignments cannot adapt to the dynamic demands of the task.

**Goal**: Simultaneously optimize agent role allocation and communication paths, dynamically adjusting the participating agents and their communication patterns in each round of discussion.

**Key Insight**: Drawing inspiration from management theories on how high-performing human teams dynamically adjust roles and collaboration styles, MAS is analogized to a human team, where the most suitable members are involved in discussions at different stages.

**Core Idea**: Through a two-step optimization strategy of Node Dropout (removing redundant agents) and Edge Dropout (removing redundant communication edges), the communication topology of MAS is dynamically adjusted to improve both efficiency and performance.

## Method

### Overall Architecture
AgentDropout models the communication structure of MAS as a multi-round directed graph. The input is a set of agents and their communication topology, and the output is the optimized communication graph for each round. The method consists of two phases: first, Node Dropout is used to determine the subset of agents that should participate in each round, and then Edge Dropout is applied to further prune redundant communication edges between agents.

### Key Designs

1. **Node Dropout (Node Pruning)**:

    - **Function**: Removes the agent nodes with the least contribution in each communication round.
    - **Mechanism**: The adjacency matrix of the communication graph is first initialized as a trainable continuous weight matrix (weights ranging from 0 to 1). The intra-round adjacency matrix is optimized using policy gradient methods, aiming to maximize task performance. After optimization, the sum of weighted in-degree and out-degree for each node in each round is calculated. The TopkNodes function is used to keep the $k = (1-\alpha) \times N$ nodes with the largest degrees and remove the remaining nodes along with all their associated edges, where $\alpha$ is the node dropout rate.
    - **Design Motivation**: Different discussion stages require different specialized roles. For example, in code generation tasks, the first round might need an architect, while the second round requires concrete coders. Fixed roles cause certain agents to redundantly consume the token budget in specific stages.

2. **Edge Dropout (Edge Pruning)**:

    - **Function**: Further removes low-contribution communication edges on the node-pruned graph.
    - **Mechanism**: After Node Dropout, the same policy gradient optimization method is used to train intra-round and inter-round adjacency matrices. Following retraining, the most important edges are retained using top-k selection based on the optimized edge weights, and lower-weight edges are removed to obtain a sparser communication graph. This yields a dynamic topology with different participating agents and communication paths for each round.
    - **Design Motivation**: Even when the participating agents for each round are determined, fully connected communication between them is unnecessary. Certain message transmission paths can be redundant or even harmful (e.g., transmitting incorrect information).

3. **DAGSample Sampling Algorithm**:

    - **Function**: Samples a discrete communication graph from the probabilistic adjacency matrix, ensuring it is a directed acyclic graph (DAG).
    - **Mechanism**: Independently samples each edge using its weight as the sampling probability, and applies post-processing to ensure the sampled result is a directed acyclic graph (DAG). This guarantees that communication between agents does not form infinite loops.
    - **Design Motivation**: Communication graphs must be directed and acyclic to ensure the correct order of message passing; random sampling might generate graphs with cycles.

### Loss & Training
The policy gradient (REINFORCE) method is used to optimize the adjacency matrix. For each training sample, $M$ communication graphs are sampled, and their performance scores $\mu(\mathcal{G}_m)$ on the task are evaluated. The expected gradient is approximated via probability-weighted averaging to perform optimization. The adjacency matrix weights are updated using gradient ascent.

## Key Experimental Results

### Main Results

| Method | Edge Pruning | Node Pruning | MMLU | GSM8K | HumanEval | Average |
|------|-----------|-----------|------|-------|-----------|------|
| Vanilla MAS (T rounds) | ✗ | ✗ | 60.13 | 71.48 | 49.17 | 65.72 |
| AgentPrune | ✓ | ✗ | 60.78 | 71.02 | 51.67 | 66.51 |
| **AgentDropout** | ✓ | ✓ | **62.75** | **73.13** | **55.84** | **68.70** |

On Qwen2.5-72B:

| Method | Edge Pruning | Node Pruning | MMLU | GSM8K | HumanEval | Average |
|------|-----------|-----------|------|-------|-----------|------|
| Vanilla MAS (T rounds) | ✗ | ✗ | 84.31 | 93.28 | 87.08 | 90.76 |
| AgentPrune | ✓ | ✗ | 83.66 | 93.67 | 86.67 | 90.81 |
| **AgentDropout** | ✓ | ✓ | **84.97** | **93.75** | **87.92** | **91.58** |

### Ablation Study

| Configuration | Average Performance | Prompt Token Reduction | Description |
|------|---------|-----------------|------|
| Full AgentDropout | 68.70 | -21.6% | Full model |
| Edge Dropout Only | 66.51 | -14.2% | Equivalent to AgentPrune |
| Node Dropout Only | 67.80 | -18.3% | Node pruning contributes more |
| No Optimization | 65.72 | 0% | Original MAS |

### Key Findings
- The contribution of Node Dropout is greater than that of Edge Dropout, indicating that "who participates in the discussion" is more important than "who communicates with whom".
- On larger and stronger models (such as Qwen2.5-72B, DeepSeek-V3), AgentDropout still brings significant improvements, indicating that communication topology optimization remains valuable even when model capability is already strong.
- Good domain transferability—the communication topology learned on one task can transfer to other tasks.
- Strong structural robustness—even if the initial topology changes, the optimized results tend to converge.

## Highlights & Insights
- Transferring Dropout from neural network regularization to multi-agent system topology optimization is conceptually simple yet effective. This analogical thinking treats agents as neurons and communication edges as connection weights, naturally repurposing the regularization effect of Dropout.
- The idea of dynamic role assignment originates from management theory, and this interdisciplinary inspiration is highly valuable. Efficient teams indeed need to dynamically adjust member roles based on the task stage.
- The method features excellent plug-and-play characteristics—it can be applied to any communication-graph-based MAS framework without modifying the underlying LLM.

## Limitations & Future Work
- Training the communication topology requires multiple LLM inferences ($M$ samplings $\times$ multiple training rounds), resulting in high computational costs.
- The node dropout rate $\alpha$ is a hyperparameter that requires tuning; different tasks may require different settings.
- The current method assumes all agents use the same LLM; the optimization for heterogeneous agent systems has not yet been explored.
- It has only been validated on non-conversational tasks and might not apply to scenarios requiring continuous participation of all roles (e.g., debates).

## Related Work & Insights
- **vs AgentPrune**: AgentPrune only performs edge pruning and uses the same strategy across rounds, while AgentDropout performs both node and edge pruning with dynamic adjustments in each round, achieving superior performance and efficiency.
- **vs GPTSwarm**: GPTSwarm optimizes MAS structures using graph neural networks but requires training an additional GNN. AgentDropout directly optimizes the adjacency matrix, which is simpler.
- **vs DyLAN**: DyLAN improves LLM inference by dynamically selecting agents but does not consider the optimization of communication edges.

## Rating
- Novelty: ⭐⭐⭐⭐ Transferring the concept of Dropout to MAS topology optimization is creative, though the overall design combines existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three base models and six datasets, with extensive ablation and transfer experiments.
- Writing Quality: ⭐⭐⭐⭐ The method is clearly described, and the motivation is convincingly presented.
- Value: ⭐⭐⭐⭐ Provides a practical solution for MAS efficiency optimization; the open-source code facilitates reproduction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)
- [\[ACL 2025\] MasRouter: Learning to Route LLMs for Multi-Agent Systems](masrouter_learning_to_route_llms_for_multi-agent_systems.md)
- [\[ACL 2025\] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)
- [\[ACL 2025\] Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System](virsci_multi_agent_idea_gen.md)
- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)

</div>

<!-- RELATED:END -->
