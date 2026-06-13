---
title: >-
  [Paper Note] Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation
description: >-
  [AAAI 2026][Multi-Agent][Multi-Agent Topology Design] This paper proposes ARG-Designer, which reformulates multi-agent system topology design as a conditional autoregressive graph generation task. Rather than pruning fro…
tags:
  - "AAAI 2026"
  - "Multi-Agent"
  - "Multi-Agent Topology Design"
  - "Autoregressive Graph Generation"
  - "Collaboration Graph"
  - "Curriculum Learning"
  - "Scalable Agents"
date: 2026-05-08
content_hash: a189082e252ac9b2
---

# Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation

**Conference**: AAAI 2026 Oral  
**arXiv**: [2507.18224](https://arxiv.org/abs/2507.18224)  
**Code**: [https://github.com/Shiy-Li/ARG-Designer](https://github.com/Shiy-Li/ARG-Designer)  
**Area**: Graph Learning
**Keywords**: Multi-Agent Topology Design, Autoregressive Graph Generation, Collaboration Graph, Curriculum Learning, Scalable Agents

## TL;DR
This paper proposes ARG-Designer, which reformulates multi-agent system topology design as a conditional autoregressive graph generation task. Rather than pruning from template graphs, the model incrementally generates agent nodes and communication edges from scratch. ARG-Designer achieves state-of-the-art performance across 6 benchmarks (average 92.78%), reduces token consumption by approximately 50% compared to G-Designer, and supports role expansion without retraining.

## Background & Motivation
**Background**: The effectiveness of LLM-based multi-agent systems critically depends on the collaborative topology — how agents are organized and exchange information. Automated topology design has become a research focus, with representative works including AgentPrune (edge pruning), AgentDropout (stochastic dropping), and G-Designer (graph autoencoder learning).

**Limitations of Prior Work**: Existing methods follow a "template graph modification" paradigm — starting from a predefined fully-connected or dense template and adapting it to the task via edge reweighting or pruning. Two key limitations arise: (1) **Redundant composition**: the template predefines all possible agent roles, so even after pruning, irrelevant agents or edges may remain; (2) **Limited scalability**: models are trained on fixed templates and cannot generalize to newly added agent roles or dynamically changing agent pools.

**Key Challenge**: The search space of the template modification paradigm is constrained by the predefined template, making truly task-tailored topologies unattainable; yet expanding the template to cover all possible roles is prohibitively expensive.

**Goal**: How to construct, from scratch, a customized multi-agent topology containing only the necessary agents and optimal communication links?

**Key Insight**: Drawing an analogy to real-world team formation — rather than hiring all possible members and then downsizing, one incrementally recruits suitable members according to task requirements. This motivates the autoregressive graph generation paradigm: iteratively adding nodes and edges until the topology is complete.

**Core Idea**: Transform multi-agent topology design from "template modification" to "conditional autoregressive graph generation," constructing from scratch a collaboration graph that is optimal in terms of agent count, roles, and connectivity.

## Method

### Overall Architecture
The MAS is modeled as a directed acyclic graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where nodes are role-assigned agents and edges are communication links. ARG-Designer is a conditional generative model $P(\mathcal{G}|\mathcal{Q}, \mathcal{R})$, conditioned on task query $\mathcal{Q}$ and role pool $\mathcal{R}$, which autoregressively factorizes into a sequence of node and edge generation steps:
$$P(\mathcal{G}|\mathcal{Q}, \mathcal{R}) = \prod_{i=1}^{|\mathcal{V}|} P(v_i|\mathcal{G}_{<i}, \mathcal{Q}, \mathcal{R}) \cdot \prod_{j=1}^{i-1} P(e_{ji}|v_i, \mathcal{G}_{<i}, \mathcal{Q})$$

### Key Designs

1. **Node Generator**:

    - **Function**: At each step, selects the role of the next agent or outputs END to terminate generation.
    - **Mechanism**: A GRU aggregates the historical embeddings of previously generated agents $\mathbf{f}_{\text{hist}}^{(i)} = \text{GRU}_{\text{prev}}([\mathbf{z}_{r_1}, \ldots, \mathbf{z}_{r_{i-1}}])$, which is fused with the task embedding $\mathbf{f}_{\mathcal{Q}}$ via dynamic gating to produce a context embedding, followed by a GRU hidden state update. Critically, role selection employs **metric learning** (dot-product) rather than a fixed classifier — the hidden state is projected into a "node intent" embedding and dot-producted against the role embedding matrix to obtain selection probabilities.
    - **Design Motivation**: The metric learning mechanism is key to scalability — new roles can be appended as embedding rows without modifying or retraining the model, naturally supporting dynamic expansion of the role pool.

2. **Edge Generator**:

    - **Function**: Determines from which existing agents $v_j$ $(j < i)$ the newly added agent $v_i$ should receive information.
    - **Mechanism**: An edge GRU is initialized with the final hidden state of the node GRU, and iterates over existing nodes to predict the existence probability of each edge via Sigmoid: $P(e_{j,i}=1) = \text{Sigmoid}(s_{\text{edge}}^{(i,j)})$.
    - **Design Motivation**: Conditioning edge generation on the node generation history and full context enables the model to learn complex structural dependencies, rather than treating each edge independently.

3. **Curriculum Learning Training Strategy**:

    - **Function**: Two-stage training — first learn to "generate correct and valid topologies," then learn to "generate compact and efficient topologies."
    - **Mechanism**: **Stage 1 (Exploration)**: Complex configurations (many agents, dense connections) are used to generate successful topologies as training data $\mathcal{D}_{\text{exp}}$, allowing the model to learn basic collaborative patterns. **Stage 2 (Efficiency)**: Three data sources are mixed — efficient topologies from simple configurations $\mathcal{D}_{\text{simple}}$, compact topologies obtained by systematically pruning the dense graphs from Stage 1 that remain successful $\mathcal{D}_{\text{pruned}}$, and replay data $\mathcal{D}_{\text{replay}}$ to prevent forgetting.
    - **Design Motivation**: Directly learning compact topologies suffers from a cold-start problem (the model has no prior sense of "correct" collaboration); learning dense-first then compact is a natural curriculum, and efficiency fine-tuning reduces token consumption by 30–34% without sacrificing performance.

### Loss & Training
The overall objective is the negative log-likelihood loss:
$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{node}} + (1-\alpha) \cdot \mathcal{L}_{\text{edge}}$$
with $\alpha=0.2$ (edge generation is weighted more heavily). Effective models can be trained with as few as 40–60 queries. Teacher Forcing is applied to accelerate training.

## Key Experimental Results

### Main Results
Accuracy comparison across 6 benchmarks (GPT-4o as the underlying LLM):

| Method | MMLU | GSM8K | AQuA | MultiArith | SVAMP | HumanEval | Avg. |
|--------|------|-------|------|------------|-------|-----------|------|
| Vanilla (Single Agent) | 80.39 | 82.30 | 71.06 | 93.09 | 86.55 | 71.39 | 80.80 |
| LLM-Debate | 84.96 | 91.40 | 77.65 | 96.36 | 90.11 | 84.70 | 87.53 |
| AgentPrune | 85.07 | 91.10 | 80.51 | 94.65 | 90.58 | 86.75 | 88.09 |
| G-Designer | 86.92 | 93.80 | 81.60 | 96.50 | 93.10 | 88.33 | 90.04 |
| **ARG-Designer** | **89.54** | **94.40** | **86.45** | **98.93** | **95.63** | **91.74** | **92.78** |

### Ablation Study

| Configuration | MMLU | GSM8K | HumanEval | Avg. |
|---------------|------|-------|-----------|------|
| ARG-Designer | **89.54** | 94.40 | **91.74** | **91.89** |
| w/o fine-tune | 88.23 | **94.70** | 90.91 | 91.28 |
| w/o task emb. | 86.93 | 93.10 | 89.26 | 89.76 |
| w/o hist. emb. | 88.23 | 93.60 | 90.08 | 90.64 |

### Key Findings
- ARG-Designer achieves state of the art on all 6 benchmarks, outperforming G-Designer by 2.74% and Vanilla single-agent by 11.98% on average.
- Token efficiency: ARG-Designer consumes only 4.1M tokens on GSM8K, approximately 50% fewer than G-Designer — because topologies generated from scratch are more compact and avoid the redundancy inherent in template modification.
- Task embedding is the most critical component: removing it causes an average drop of 2.13%, demonstrating that conditioning on the task query is central to generating customized topologies.
- Robustness: under prompt injection attacks, performance drops by only 2.15% (the lowest among compared methods), as the generated topologies naturally exhibit distributed risk and redundant communication paths.
- Scalability validation: adding a "lawyer" role without retraining, the model correctly identifies its relevance to legal questions and dynamically generates a lawyer-centric collaboration graph.

## Highlights & Insights
- The paradigm shift from "template modification" to "autoregressive generation" is the core contribution of this paper. This is not merely a methodological innovation but a reconceptualization at the problem formulation level — treating MAS topology design as conditional graph generation rather than graph editing, opening up a vastly larger design space.
- The metric learning approach to role selection is particularly elegant: while the conventional approach uses a fixed-dimension classifier (number of roles = number of classes), ARG-Designer uses similarity matching in embedding space, enabling the role pool to be dynamically extended at inference time.
- The curriculum learning strategy is worth generalizing: first learn "what works" (dense + correct), then learn "what is efficient" (compact + correct), with replay data between stages to prevent forgetting. This paradigm is applicable to other "correctness-first, efficiency-second" learning scenarios.

## Limitations & Future Work
- Node generation ordering has a known influence on autoregressive models — the paper does not thoroughly discuss the impact of different ordering strategies, which is a well-recognized issue in the graph generation literature.
- Training data is generated through trial-and-error (executing topologies and checking success), making data construction costly (requiring large numbers of LLM calls); the paper trains on only 40–60 queries, and scalability to larger settings remains unexplored.
- Experiments use only GPT-4o as the underlying LLM; performance in heterogeneous LLM settings (different nodes using LLMs of varying capability) is not investigated.
- The number of communication rounds $K=3$ is a fixed hyperparameter with no adaptive mechanism.

## Related Work & Insights
- **vs. G-Designer**: G-Designer employs a graph autoencoder to learn topologies on predefined templates; ARG-Designer generates topologies autoregressively from scratch. The former is constrained by the template and cannot alter agent count or role set, while the latter can dynamically determine both.
- **vs. AgentPrune/AgentDropout**: These methods perform "subtraction" (removing edges/agents from a dense graph); ARG-Designer performs "addition" (constructing from an empty graph). Addition more naturally avoids redundancy.
- **vs. BAMAS**: BAMAS focuses on budget-constrained LLM selection and topology selection (from 4 predefined options); ARG-Designer supports unconstrained generation of arbitrary topology structures. The two are complementary — BAMAS can handle budget-constrained LLM selection upstream, while ARG-Designer generates the optimal topology downstream.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The paradigm shift (template modification → autoregressive generation) is a significant conceptual contribution; the metric learning design for scalable role selection is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across 6 benchmarks with ablations, robustness analysis, scalability tests, and case studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem formulation is clear; comparisons with existing paradigms are intuitively illustrated; the logical flow is coherent.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new direction for automated multi-agent topology design with an elegant framework and strong empirical support.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A Graph-Theoretical Perspective on Law Design for Multiagent Systems](a_graph-theoretical_perspective_on_law_design_for_multiagent_systems.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](../../ACL2026/multi_agent/cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)
- [\[ICML 2026\] RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation](../../ICML2026/multi_agent/radar_redundancy-aware_diffusion_for_multi-agent_communication_structure_generat.md)
- [\[AAAI 2026\] SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication](safesieve_from_heuristics_to_experience_in_progressive_pruning_for_llm-based_mul.md)

</div>

<!-- RELATED:END -->
