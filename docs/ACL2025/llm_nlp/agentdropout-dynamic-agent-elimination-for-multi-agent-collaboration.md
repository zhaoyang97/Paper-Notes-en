---
title: >-
  [Paper Note] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration
description: >-
  [ACL 2025][LLM (Other)][Multi-Agent Systems] This paper proposes AgentDropout, which dynamically eliminates redundant agents (node pruning) and redundant communication edges (edge pruning) across multiple discussion rounds, reducing prompt token consumption by 21.6% while improving task performance by 1.14 points.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Multi-Agent Systems"
  - "Communication Topology Optimization"
  - "Node Pruning"
  - "Edge Pruning"
  - "Token Efficiency"
date: 2026-05-08
content_hash: dc63908baf190fd3
---

# AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration

**Conference**: ACL 2025  
**arXiv**: [2503.18891](https://arxiv.org/abs/2503.18891)  
**Code**: [https://github.com/wangzx1219/AgentDropout](https://github.com/wangzx1219/AgentDropout)  
**Area**: LLM/NLP  
**Keywords**: Multi-Agent Systems, Communication Topology Optimization, Node Pruning, Edge Pruning, Token Efficiency

## TL;DR

This paper proposes AgentDropout, which dynamically eliminates redundant agents (node pruning) and redundant communication edges (edge pruning) across multiple discussion rounds, reducing prompt token consumption by 21.6% while improving task performance by 1.14 points.

## Background & Motivation

1. **Background**: LLM-based multi-agent systems (MAS) solve complex tasks by simulating human collaboration, but suffer from huge and inefficient communication overhead.
2. **Limitations of Prior Work**: Existing methods (such as AgentPrune) only prune communication edges but not agent nodes, and they apply the same pruning strategy across all communication rounds, overlooking the fact that different stages of discussion may require different role combinations.
3. **Key Challenge**: Redundancy in MAS arises from both unnecessary information exchanges (edge redundancy) and unnecessary participants (node redundancy). Existing methods only address the former.
4. **Goal**: Simultaneously identify and eliminate redundant agents and redundant communications, while allowing different role combinations across different discussion rounds.
5. **Key Insight**: Drawing inspiration from management theory—high-performing teams dynamically adjust member roles and responsibilities based on task demands.
6. **Core Idea**: Model the MAS as a communication graph, learn which nodes and edges are important in which rounds using trainable adjacency matrix weights, and then perform Node Dropout and Edge Dropout accordingly.

## Method

### Overall Architecture

AgentDropout optimizes the communication topology in two steps: (1) Node Dropout—training intra-round adjacency matrix weights, calculating node degrees, and removing the least contributing nodes in each discussion round; (2) Edge Dropout—further training edge weights on the node-pruned graph and pruning low-weight edges. Both steps utilize policy gradient (REINFORCE) to optimize the non-differentiable utility function.

### Key Designs

1. **Node Dropout**:

    - **Function**: Dynamically remove the least contributing agents across different discussion rounds.
    - **Mechanism**: Initialize all edge weights to 0.5, and use policy gradient to optimize the intra-round adjacency matrix $\tilde{\mathcal{A}}_{intra}$. After optimization, the sum of weighted in-degree + out-degree is calculated for each node in each round's adjacency matrix, and the nodes with the lowest degrees are removed (i.e., the TopkNodes function retains high-degree nodes and discards low-degree nodes). A dropout rate $\alpha$ is used to control the pruning ratio.
    - **Design Motivation**: Different stages of discussion require different experts—for instance, the initial stage requires broad information gathering, while the later stages require focused decision-makers. Dynamic role assignment is more efficient than fixed teams.

2. **Edge Dropout**:

    - **Function**: Finely prune redundant communication paths between nodes.
    - **Mechanism**: On the graph after node pruning, continue using policy gradient to simultaneously optimize intra-round and inter-round adjacency matrices, and then zero out edge weights below a threshold. The DAGSample algorithm is used to ensure the sampled communication graph is a directed acyclic graph (DAG).
    - **Design Motivation**: Even after reducing participants, unnecessary information exchange may still exist among the remaining agents. Edge pruning further reduces redundant communication.

3. **Policy Gradient Optimization**:

    - **Function**: Optimize the graph topology under a non-differentiable utility function.
    - **Mechanism**: Since task performance evaluation usually relies on external APIs (which are non-differentiable), REINFORCE is used to estimate the gradient: $\nabla \approx \frac{1}{M}\sum_{m=1}^{M}\mu(\mathcal{G}_m)\nabla\log(p(\mathcal{G}_m))$, where $M$ graphs are sampled to compute the probability-weighted performance.
    - **Design Motivation**: The utility function of a MAS is often a black box; policy gradient is the only viable optimization method.

### Loss & Training

- **Objective Function**: Maximize the expected task performance on sampled communication graphs.
- **Iterative update the adjacency matrix weights using a small amount of data (about dozens of samples)**.
- **Use the fixed optimized topology during inference**.

## Key Experimental Results

### Main Results

| Method | MMLU | GSM8K | HumanEval | Prompt Token↓ | Completion Token↓ |
|------|------|-------|-----------|---------------|-------------------|
| Vanilla MAS | Baseline | Baseline | Baseline | 100% | 100% |
| AgentPrune | Improved | Improved | Improved | -12% | -8% |
| **AgentDropout** | **+1.14 Avg** | **Improved** | **Improved** | **-21.6%** | **-18.4%** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Node Dropout Only | Improved | Effective individually |
| Edge Dropout Only | Improved | Effective individually |
| Node + Edge | Optimal | Complementary improvement |
| Different dropout rates | α=0.3~0.5 is optimal | Too high leads to loss of key information |

### Key Findings

- Node Dropout and Edge Dropout are complementary—the former reduces the number of participants, while the latter optimizes communication among the remaining participants.
- On larger and stronger LLMs, the performance of MAS can still be further improved through interaction optimization.
- AgentDropout demonstrates good cross-domain transferability and structural robustness.

## Highlights & Insights

- The concept of **dynamic role assignment** stems directly from management theory, successfully applying organizational behavior insights to AI system design, which is highly inspiring.
- Node Dropout allows different discussion rounds to have different team combinations, which is more flexible than fixed teams and can be transferred to any multi-agent scenario.
- Optimizing communication topology with extremely small data (dozens of samples) makes it highly practical.

## Limitations & Future Work

- Policy gradient optimization requires running MAS multiple times to estimate gradients, resulting in non-trivial initial optimization overhead.
- Currently, the dropout rate is a global hyperparameter; adaptive adjustment round-by-round might be superior.
- Tested only on reasoning, mathematics, and code tasks, with its effectiveness on creative tasks remaining unknown.

## Related Work & Insights

- **vs AgentPrune**: Only performs edge pruning with a static strategy across rounds, while AgentDropout adds node pruning and dynamic adjustment.
- **vs Standard MAS**: Fully connected communication wastes a large amount of tokens, whereas AgentDropout demonstrates that sparse communication can actually improve performance.

## Rating

- Novelty: ⭐⭐⭐⭐ Extends the concept of Dropout from neural networks to multi-agent graph topologies, offering a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task testing, with comprehensive ablation and transferability analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear diagrams and intuitive analogies with human teams.
- Value: ⭐⭐⭐⭐ High practicality with open-source code, providing practical guidance for optimizing multi-agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Red-Teaming LLM Multi-Agent Systems via Communication Attacks](red-teaming_llm_multi-agent_systems_via_communication_attacks.md)
- [\[ACL 2025\] MasRouter: Learning to Route LLMs for Multi-Agent Systems](masrouter_learning_to_route_llms_for_multi-agent_systems.md)
- [\[ACL 2025\] Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)
- [\[ACL 2025\] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)
- [\[ACL 2025\] Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System](virsci_multi_agent_idea_gen.md)

</div>

<!-- RELATED:END -->
