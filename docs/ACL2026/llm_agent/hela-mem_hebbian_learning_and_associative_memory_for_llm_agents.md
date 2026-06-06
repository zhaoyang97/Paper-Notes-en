---
title: >-
  [Paper Note] HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents
description: >-
  [ACL 2026][LLM Agent][Hebbian Learning] HeLa-Mem proposes a neuroscience-inspired memory architecture for LLM agents that models conversation history as a dynamic graph with Hebbian learning dynamics. It reinforces inter…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Hebbian Learning"
  - "Associative Memory"
  - "Long-term Dialogue"
  - "Episodic-Semantic Dual-path"
  - "Spreading Activation"
date: 2026-05-08
content_hash: 8b082c78cd58d159
---

# HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents

**Conference**: ACL 2026  
**arXiv**: [2604.16839](https://arxiv.org/abs/2604.16839)  
**Code**: [GitHub](https://github.com/hela-mem)  
**Area**: LLM Agent / Memory Systems  
**Keywords**: Hebbian Learning, Associative Memory, Long-term Dialogue, Episodic-Semantic Dual-path, Spreading Activation

## TL;DR
HeLa-Mem proposes a neuroscience-inspired memory architecture for LLM agents that models conversation history as a dynamic graph with Hebbian learning dynamics. It reinforces inter-memory connections through co-activation, condenses hub memories into semantic knowledge via reflective distillation, and utilizes dual-path retrieval—combining semantic similarity with Hebbian spreading activation—to achieve state-of-the-art performance on the LoCoMo benchmark with significantly fewer tokens.

## Background & Motivation

**Background**: Long-term memory in LLM agents is a critical challenge, as fixed context windows cannot maintain coherence across extended interactions. Existing memory systems represent conversation history as unstructured embeddings and retrieve information based on semantic similarity.

**Limitations of Prior Work**: (1) Embedding-based retrieval fails to capture the associative structure inherent in human memory, where related experiences strengthen their connections through repeated co-activation. (2) Current methods optimize single dimensions (structure, retrieval, or updates) in isolation, ignoring their interplay. (3) More fundamentally, the dynamic evolution of memory is overlooked, as current systems treat storage and retrieval as independent static processes, failing to capture the evolving connectivity between memories.

**Key Challenge**: Semantic similarity only captures surface-level associations. However, associations in human memory run deeper—a topic discussed today might trigger a memory from a month ago not because of surface keywords, but because they belong to the same evolving narrative.

**Goal**: To build an LLM agent memory architecture that simulates three biological memory mechanisms: association, consolidation, and spreading activation.

**Key Insight**: Drawing on the principles of Hebbian learning ("neurons that fire together, wire together") and the dual-system theory of episodic and semantic memory.

**Core Idea**: Represent episodic memory using a dynamic graph driven by Hebbian learning dynamics, generate semantic memory through reflective distillation, and implement retrieval via dual-path spreading activation.

## Method

### Overall Architecture
A three-module cognitive cycle: (1) Online Encoding and Association—conversation turns are encoded as graph nodes, and edge weights are reinforced via Hebbian dynamics upon co-activation. (2) Reflective Consolidation—a reflection agent detects hub nodes and performs Hebbian distillation, condensing densely connected memory clusters into semantic knowledge. (3) Dual-path Retrieval—information is retrieved during queries through both semantic similarity (Base Path) and Hebbian spreading activation (Flip Path).

### Key Designs

1. **Hebbian Online Association**:

    - **Function**: Capture latent memory associations that cannot be discovered by semantic embeddings.
    - **Mechanism**: Edge weights are updated using the formula $w_{ij}^{(t+1)} = (1-\lambda) \cdot w_{ij}^{(t)} + \eta \cdot \mathbb{I}(v_i, v_j \in \mathcal{K}_t)$, where $\lambda$ controls synaptic decay, $\eta$ is the learning rate, and $\mathbb{I}$ indicates whether two nodes are co-activated in the current retrieval set. Repeatedly co-occurring memories gradually strengthen their connections, while unused links decay over time.
    - **Design Motivation**: Transcend semantic similarity—two memories may be superficially dissimilar but establish strong associations through repeated simultaneous retrieval (similar to human associative memory).

2. **Hebbian Distillation (Reflective Consolidation)**:

    - **Function**: Prevent memory graph explosion while preserving key information.
    - **Mechanism**: When the cumulative edge weight $D(v_i) = \sum_{j \in \mathcal{N}(i)} w_{ij} > \delta_{hub}$ exceeds a threshold, a reflection agent retrieves the hub node and its strongly connected neighbors. The LLM then synthesizes this cluster into structured semantic knowledge (user profiles, factual memories, agent knowledge) stored in semantic memory. Simultaneously, adaptive forgetting is applied to low-weight nodes that have not been visited for a long time.
    - **Design Motivation**: Simulate the sleep consolidation process in the brain—frequently activated memory clusters are abstracted into stable long-term knowledge.

3. **Dual-path Spreading Activation Retrieval**:

    - **Function**: Retrieve both semantically and associatively related memories.
    - **Mechanism**: Base activation is defined as $S_{base}(v_i) = (\text{sim}(\mathbf{q}, \mathbf{e}_i) + \alpha \cdot \text{keyword}) \cdot \gamma(v_i)$ (semantic + temporal decay). Spreading activation is $S(v_j) = S_{base}(v_j) + \beta \sum_{i \in \mathcal{N}(j)} S_{base}(v_i) \cdot w_{ij}$. The final retrieval set = Top-k (Base Path) ∪ Top-m (Flip Path: nodes with high scores after spreading that were not selected by the Base Path).
    - **Design Motivation**: The Flip Path captures memories that are "semantically distant but associatively near," which is particularly beneficial for multi-hop reasoning.

### Loss & Training
Training-free. All parameters ($\eta, \lambda, \beta, \tau$, etc.) are hyperparameters. Hebbian learning occurs online during the retrieval process.

## Key Experimental Results

### Main Results (LoCoMo Benchmark)

| Method | Multi-hop F1 | Temporal F1 | Open-domain F1 | Single-hop F1 | Token Usage ↓ |
|------|--------|--------|---------|--------|---------|
| MemGPT | - | - | - | - | High |
| A-Mem | - | - | - | - | Med |
| **HeLa-Mem** | **SOTA** | **SOTA** | **SOTA** | **SOTA** | **Lowest** |

### Ablation Study

| Configuration | Description |
|------|------|
| w/o Hebbian Learning | Degenerates to pure semantic retrieval; multi-hop performance drops significantly. |
| w/o Reflective Distillation | Graph expands indefinitely; retrieval noise increases. |
| w/o Spreading Activation | Base path only; fails to discover associative memories. |

### Key Findings
- HeLa-Mem achieves SOTA across all four question categories while using significantly fewer tokens (located in the "ideal top-left region" of the performance-efficiency chart).
- Spreading activation contributes most to multi-hop reasoning, highlighting the value of latent associations captured by Hebbian learning.
- The average rank is 1.25 (nearly first in all categories), showing consistent superior performance across four different LLM backbones.

## Highlights & Insights
- **Unified modeling of three biological memory mechanisms** (association, consolidation, spreading activation) provides an elegant cognitive science perspective.
- The "hub detection → cluster synthesis → semantic knowledge" pipeline of Hebbian distillation effectively simulates memory consolidation in the human brain.
- Token efficiency advantages demonstrate that more precise retrieval (rather than more retrieval) is the key.

## Limitations & Future Work
- Hyperparameters ($\eta, \lambda, \delta_{hub}$, etc.) require manual tuning and may be sensitive to different scenarios.
- Evaluation was conducted only on the LoCoMo benchmark; verification in more long-term dialogue scenarios is needed.
- Computational overhead for graph operations increases with conversation length.

## Related Work & Insights
- **vs A-Mem**: A-Mem uses a Zettelkasten-style note network, while HeLa-Mem employs a dynamic graph driven by Hebbian dynamics, where connections are "learned" from interactions.
- **vs Mem0/MemGPT**: These methods optimize single dimensions; HeLa-Mem unifies association, consolidation, and retrieval.
- **vs APEX-MEM**: APEX-MEM uses attribute graphs with append-only storage, whereas HeLa-Mem evolves the graph structure dynamically via Hebbian learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Hebbian learning-driven memory architecture is a first for LLM agents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four backbones + four categories + ablation, though limited to a single benchmark.
- Writing Quality: ⭐⭐⭐⭐⭐ Cognitive science motivations are clear, and the architecture description is systematic.
- Value: ⭐⭐⭐⭐⭐ Provides a new bio-inspired paradigm for long-term memory in LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mem^p: Exploring Agent Procedural Memory](memp_exploring_agent_procedural_memory.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)
- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](../../NeurIPS2025/llm_agent/a-mem_agentic_memory_for_llm_agents.md)
- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)
- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)

</div>

<!-- RELATED:END -->
