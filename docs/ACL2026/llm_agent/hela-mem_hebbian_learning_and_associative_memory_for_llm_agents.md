---
title: >-
  [Paper Note] HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents
description: >-
  [ACL 2026][LLM Agent][To be supplemented] To be supplemented after a thorough reading of the paper.
tags:
  - ACL 2026
  - LLM Agent
  - To be supplemented
date: 2026-05-08
content_hash: a5c3760eb0df761a
---

# HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents

**Conference**: ACL 2026
**arXiv**: [2604.16839](https://arxiv.org/abs/2604.16839)
**Code**: [GitHub](https://github.com/hela-mem)
**Area**: LLM Agent / Memory Systems
**Keywords**: Hebbian Learning, Associative Memory, Long-term Dialogue, Episodic-Semantic Dual Pathway, Spreading Activation

## TL;DR
HeLa-Mem proposes a neuroscience-inspired memory architecture for LLM agents that models conversation history as a dynamic graph driven by Hebbian learning dynamics — strengthening inter-memory connections through co-activation, distilling hub memories into semantic knowledge via reflective consolidation, and retrieving via a dual-pathway combining semantic similarity with Hebbian spreading activation. It achieves state-of-the-art performance on LoCoMo with significantly fewer tokens.

## Background & Motivation

**Background**: Long-term memory for LLM agents is a critical challenge — fixed context windows cannot maintain coherence across extended interactions. Existing memory systems represent conversation history as unstructured embedding vectors and retrieve information via semantic similarity.

**Limitations of Prior Work**: (1) Embedding-based retrieval fails to capture the associative structure found in human memory, where related experiences gradually strengthen connections through repeated co-activation; (2) Existing methods independently optimize a single dimension (structure/retrieval/update), neglecting their interactions; (3) More fundamentally, the dynamic evolution of memory is ignored — current systems treat storage and retrieval as separate static processes, unable to capture continuously reorganizing inter-memory connectivity.

**Key Challenge**: Semantic similarity captures only surface-level associations, whereas associations in human memory run deeper — a topic discussed today may trigger a memory from a month ago not because of superficial keyword overlap, but because both belong to the same evolving narrative.

**Goal**: To construct an LLM agent memory architecture that emulates three mechanisms found in biological memory: association, consolidation, and spreading activation.

**Key Insight**: Drawing on the Hebbian learning principle ("neurons that fire together, wire together") and the dual-system theory of episodic-semantic memory.

**Core Idea**: Dynamic graph representation of episodic memory driven by Hebbian learning dynamics + semantic memory generated through reflective distillation + dual-pathway spreading activation retrieval.

## Method

### Overall Architecture
A three-module cognitive cycle: (1) Online encoding and association — dialogue turns are encoded as graph nodes, and Hebbian learning reinforces edge weights upon co-activation; (2) Reflective consolidation — a reflection agent detects hub nodes and performs Hebbian distillation, condensing densely connected memory clusters into semantic knowledge; (3) Dual-pathway retrieval — at query time, retrieval proceeds simultaneously via semantic similarity (base pathway) and Hebbian spreading activation (flip pathway).

### Key Designs

1. **Hebbian Online Association**:

    - Function: Captures latent inter-memory associations that semantic embeddings cannot detect.
    - Mechanism: Edge weight update formula $w_{ij}^{(t+1)} = (1-\lambda) \cdot w_{ij}^{(t)} + \eta \cdot \mathbb{I}(v_i, v_j \in \mathcal{K}_t)$, where $\lambda$ controls synaptic decay, $\eta$ controls the learning rate, and $\mathbb{I}$ indicates whether both nodes were co-activated in the current retrieval set. Repeatedly co-occurring memories progressively strengthen their connection, while unused connections decay over time.
    - Design Motivation: To go beyond semantic similarity — two memories may appear superficially dissimilar yet establish strong associations by being repeatedly retrieved together (analogous to associative links in the human brain).

2. **Hebbian Distillation (Reflective Consolidation)**:

    - Function: Prevents unbounded growth of the memory graph while preserving critical information.
    - Mechanism: When the cumulative edge weight of node $v_i$, $D(v_i) = \sum_{j \in \mathcal{N}(i)} w_{ij} > \delta_{hub}$, exceeds a threshold, the reflection agent retrieves the hub node and its strongly connected neighbors, and synthesizes the cluster into structured semantic knowledge (user profiles, factual memories, agent knowledge) using an LLM, which is stored in the semantic memory store. Nodes with low weights and long access intervals are adaptively forgotten.
    - Design Motivation: To simulate the brain's sleep consolidation process — frequently activated memory clusters are abstracted into stable long-term knowledge.

3. **Dual-Pathway Spreading Activation Retrieval**:

    - Function: Retrieves both semantically relevant and associatively relevant memories.
    - Mechanism: Base activation $S_{base}(v_i) = (\text{sim}(\mathbf{q}, \mathbf{e}_i) + \alpha \cdot \text{keyword}) \cdot \gamma(v_i)$ (semantic + temporal decay). Spreading activation $S(v_j) = S_{base}(v_j) + \beta \sum_{i \in \mathcal{N}(j)} S_{base}(v_i) \cdot w_{ij}$. Final retrieval set = Top-k (base pathway) ∪ Top-m (flip pathway: nodes with high post-diffusion scores not selected by the base pathway).
    - Design Motivation: The flip pathway captures memories that are "semantically distant but associatively close" — particularly beneficial for multi-hop reasoning.

### Loss & Training
No training is required. All parameters ($\eta, \lambda, \beta, \tau$, etc.) are hyperparameters. Hebbian learning occurs online during the retrieval process.

## Key Experimental Results

### Main Results (LoCoMo Benchmark)

| Method | Multi-hop F1 | Temporal F1 | Open-domain F1 | Single-hop F1 | Tokens↓ |
|--------|-------------|-------------|----------------|---------------|---------|
| MemGPT | - | - | - | - | High |
| A-Mem | - | - | - | - | Medium |
| **HeLa-Mem** | **Best** | **Best** | **Best** | **Best** | **Fewest** |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| w/o Hebbian Learning | Degrades to pure semantic retrieval; significant drop on multi-hop |
| w/o Reflective Distillation | Graph grows unboundedly; retrieval noise increases |
| w/o Spreading Activation | Base pathway only; fails to discover associative memories |

### Key Findings
- HeLa-Mem achieves state-of-the-art performance across all four question categories while consuming significantly fewer tokens (occupying the ideal upper-left region of the performance–efficiency plot).
- Spreading activation contributes most to multi-hop reasoning — demonstrating the value of latent associations captured by Hebbian learning.
- Average rank of 1.25 (near-perfect across all categories), with consistently strong performance across four LLM backbones.

## Highlights & Insights
- **Unified modeling of three biological memory mechanisms** (association + consolidation + spreading activation) offers an elegant cognitive science perspective.
- The Hebbian distillation pipeline — hub detection → cluster synthesis → semantic knowledge — simulates memory consolidation in the human brain.
- The token efficiency advantage demonstrates that more precise retrieval, rather than more retrieval, is the key factor.

## Limitations & Future Work
- Hyperparameters ($\eta, \lambda, \delta_{hub}$, etc.) require manual tuning and may be sensitive to different scenarios.
- Evaluation is limited to the single LoCoMo benchmark; generalization to broader long-term dialogue settings remains to be verified.
- Computational overhead of graph operations grows with conversation length.

## Related Work & Insights
- **vs. A-Mem**: A-Mem uses a Zettelkasten-style note network; HeLa-Mem uses a dynamic graph driven by Hebbian dynamics, where connections are *learned* from interaction rather than manually structured.
- **vs. Mem0/MemGPT**: These methods each optimize a single dimension independently; HeLa-Mem unifies association, consolidation, and retrieval.
- **vs. APEX-MEM**: APEX-MEM employs an attribute graph with append-only storage; HeLa-Mem dynamically evolves the graph structure through Hebbian learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Hebbian learning-driven memory architecture is pioneering in the LLM Agent domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four backbones × four question categories × ablation study, but limited to a single benchmark.
- Writing Quality: ⭐⭐⭐⭐⭐ Cognitive science motivation is clearly articulated; architectural description is systematic.
- Value: ⭐⭐⭐⭐⭐ Offers a biologically inspired new paradigm for long-term memory in LLMs.

**Code**: To be confirmed
**Area**: llm_agent
**Keywords**: To be supplemented

## TL;DR
To be supplemented after a thorough reading of the paper.

## Background & Motivation
To be supplemented after a thorough reading of the paper.

## Method
To be supplemented after a thorough reading of the paper.

## Key Experimental Results
To be supplemented after a thorough reading of the paper.

## Highlights & Insights
To be supplemented after a thorough reading of the paper.

## Limitations & Future Work
To be supplemented after a thorough reading of the paper.

## Related Work & Insights
To be supplemented after a thorough reading of the paper.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4](discover_and_prove_an_open-source_agentic_framework_for_hard_mode_automated_theo.md)
- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](../../NeurIPS2025/llm_agent/a-mem_agentic_memory_for_llm_agents.md)
- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)
- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)

</div>

<!-- RELATED:END -->
