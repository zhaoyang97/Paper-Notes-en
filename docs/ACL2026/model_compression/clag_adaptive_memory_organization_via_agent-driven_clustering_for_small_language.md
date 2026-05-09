---
title: >-
  [Paper Note] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents
description: >-
  [ACL 2026][Model Compression][Small Language Models] This paper proposes CLAG, a clustering-based agent memory framework that organizes memories into semantically coherent clusters via SLM-driven routing, performs local evolution updates within clusters, and filters noise through two-stage retrieval, achieving significant improvements over global memory pool baselines across multiple QA datasets.
tags:
  - ACL 2026
  - Model Compression
  - Small Language Models
  - Clustering Memory
  - Agent Memory Management
  - Local Evolution
  - Two-Stage Retrieval
date: 2026-05-08
content_hash: cfebda6203e7e959
---

# CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents

**Conference**: ACL 2026
**arXiv**: [2603.15421](https://arxiv.org/abs/2603.15421)
**Code**: [https://github.com/dmis-lab/CLAG](https://github.com/dmis-lab/CLAG)
**Area**: Model Compression
**Keywords**: Small Language Models, Clustering Memory, Agent Memory Management, Local Evolution, Two-Stage Retrieval

## TL;DR
This paper proposes CLAG, a clustering-based agent memory framework that organizes memories into semantically coherent clusters via SLM-driven routing, performs local evolution updates within clusters, and filters noise through two-stage retrieval, achieving significant improvements over global memory pool baselines across multiple QA datasets.

## Background & Motivation

**Background**: LLM agents increasingly rely on external memory systems to support knowledge reuse and complex reasoning. Existing agent memory systems have evolved from static RAG (append-only, global retrieval) to frameworks supporting active evolution (e.g., A-mem, MemoryOS, GAM), enabling reflection, compression, and rewriting.

**Limitations of Prior Work**: These evolving memory systems still operate within a single global memory pool. As memories accumulate, global retrieval faces two coupled problems: (1) the search space expands, increasing the probability of retrieving semantically similar but task-irrelevant memories; (2) memory evolution mechanisms are exposed to topically mixed neighborhoods, potentially misleading updates and gradually degrading memory quality.

**Key Challenge**: These issues are particularly severe for small language models (SLMs), which are highly sensitive to irrelevant context. Cross-topic interference in a global memory pool can significantly impair SLM response quality.

**Goal**: Design a lightweight structured memory system for SLM agents that preserves self-evolution capabilities while reducing cross-topic interference.

**Key Insight**: Drawing from the cognitive science principle that "new information should refine relevant schemas without disturbing unrelated memory structures," clustering is treated as an agent-actively-controlled operation rather than a static preprocessing step.

**Core Idea**: Use SLM agent-driven online clustering to organize memories into semantically coherent neighborhoods, confining both evolution and retrieval to local clusters, thereby fundamentally reducing cross-topic interference.

## Method

### Overall Architecture
CLAG comprises three core components: (1) Agent Routing — assigning new memories to semantically coherent clusters; (2) Local Evolution — updating and consolidating memories within clusters; (3) Two-Stage Cluster-Aware Retrieval — first filtering clusters, then performing fine-grained retrieval within selected clusters. All agent decisions (routing/evolution/selection) are produced by the same SLM backbone, invoked through role-specific prompts.

### Key Designs

1. **Agent-Driven Memory Routing**:

    - Function: Assigns each new memory to the most semantically relevant cluster.
    - Mechanism: Operates in two phases. During cold-start (number of processed memories < $n$), memories are buffered; once sufficient data accumulates, initial clusters are established via InitializeClusters. Subsequently, for each new memory, Top-K candidate clusters are coarsely filtered by vector distance, and the SLM agent then reviews candidate cluster semantic profiles to make the final assignment. If cosine similarity falls below threshold $\tau$, a new cluster is created. When a cluster exceeds size threshold $\tau_{split}$, it is automatically bisected (K-Means) to prevent semantic drift.
    - Design Motivation: Pure vector distance may fail to distinguish subtle semantic differences; the SLM agent can make more accurate judgments by reading cluster profiles. The adaptive splitting mechanism ensures clusters do not grow excessively large.

2. **Intra-Cluster Local Evolution**:

    - Function: Updates and consolidates memories within a semantically coherent neighborhood, avoiding cross-topic contamination.
    - Mechanism: After a new memory $m_{new}$ is routed to a cluster, the Top-K most similar neighbors $\mathcal{M}_{local}$ are identified within that cluster. The SLM agent first analyzes fine-grained relationships (causal, temporal, etc.) to generate links $L_{new}$, then determines for each neighbor whether an update is needed to reflect new information: $m_j^* \leftarrow \text{SLM}(m_{new} \| \mathcal{M}_{local} \setminus \{m_j\} \| m_j)$. Evolved memories replace their originals, and cluster profiles are updated accordingly.
    - Design Motivation: Analogous to human cognition — new experiences primarily reshape understanding of related concepts (i.e., within the same cluster) without affecting unrelated knowledge domains. Local evolution enables each cluster to function as a self-optimizing module.

3. **Two-Stage Cluster-Aware Retrieval**:

    - Function: Reduces the search space and suppresses noise from semantically similar but task-irrelevant memories.
    - Mechanism: Stage 1 (Agent Cluster Selection) — the query is embedded and distances to cluster centroids are computed; Top-K candidates are selected, and the SLM agent evaluates the alignment between cluster profiles and the query, returning a variable-size subset. Stage 2 (Intra-Cluster Retrieval) — fine-grained vector retrieval is performed exclusively among members of the selected clusters.
    - Design Motivation: Semantically similar but topically distinct memories constitute the primary noise source in global retrieval. The two-stage design applies high-level semantic filtering via cluster profiles first, which is especially beneficial for noise-sensitive SLMs.

### Loss & Training
No training is required; the framework operates entirely at inference time. Routing, evolution, and retrieval decisions are all completed through SLM prompt invocations.

## Key Experimental Results

### Main Results (Three SLM Backbones × Three Datasets)

| Model | Method | LoCoMo F1 | HotpotQA F1 | BioASQ F1 |
|-------|--------|-----------|-------------|-----------|
| Qwen3-0.6B | RAG | 12.90 | 11.75 | 2.40 |
| Qwen3-0.6B | A-mem | 14.29 | 12.04 | 3.61 |
| Qwen3-0.6B | GAM | 16.05 | 7.81 | 3.40 |
| Qwen3-0.6B | **CLAG** | **20.99** | **15.50** | **22.01** |
| Llama3.2-1B | GAM | 22.63 | 13.85 | 6.52 |
| Llama3.2-1B | **CLAG** | **21.05** | **14.20** | **10.16** |

### Latency Comparison (Qwen3-0.6B)

| Method | Retrieval Latency (ms) | End-to-End Latency (ms) |
|--------|----------------------|------------------------|
| RAG | 17.80 | 289.60 |
| GAM | 8303.41 | 17934.32 |
| CLAG | 142.43 | 514.14 |

### Key Findings
- CLAG achieves the most substantial gains on BioASQ (Qwen3-0.6B: 2.40→22.01), as the large topical diversity of biomedical QA makes cluster-based filtering most effective.
- In terms of latency, CLAG substantially outperforms GAM (514ms vs. 17934ms) by avoiding global evolution operations.
- On adversarial question subsets in LoCoMo, CLAG's advantage is most pronounced (50.34 vs. GAM 41.25), demonstrating that clustering effectively suppresses interfering memories.
- MemoryOS performs anomalously poorly on small models (far below the RAG baseline), suggesting its design assumes stronger model capacity.

## Highlights & Insights
- **Clustering as an Agent Operation**: Clustering is elevated from static preprocessing to an online operation actively controlled by the agent, with cluster quality continuously improving as interactions accumulate. This "structure as intelligence" paradigm is generalizable to other long-term memory scenarios.
- **Local Evolution Prevents Global Contamination**: Confining evolution operations within clusters makes each cluster a self-contained optimization unit — analogous to the design philosophy of modular neural networks, where local updates do not affect global structure.
- **SLM-Friendly Design**: The two-stage retrieval substantially narrows the search space before the SLM processes it, effectively mitigating small models' vulnerability to irrelevant context.

## Limitations & Future Work
- Cluster initialization requires accumulating a sufficient volume of memories (cold-start problem), precluding cluster-based benefits in early stages.
- The quality ceiling of the SLM-as-router is bounded by model capability; routing may be inaccurate for topics with ambiguous semantic boundaries.
- Experiments on large language models are absent, leaving it unclear whether clustering provides equivalent benefits at larger scales.
- Automatic cluster count management (splitting thresholds) requires domain-specific tuning.

## Related Work & Insights
- **vs. A-mem**: A-mem evolves memories in a global pool, causing severe cross-topic interference. CLAG confines evolution within clusters, improving F1 from 14.29 to 20.99 on Qwen3-0.6B.
- **vs. GAM**: GAM's global evolution incurs extremely high latency (17934ms). CLAG's intra-cluster local evolution requires only 514ms — a 35× speedup.
- **vs. MemoryOS**: MemoryOS collapses in performance on SLMs (below the RAG baseline), indicating its design implicitly assumes stronger model capacity. CLAG is specifically optimized for SLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ Agent-driven online clustering combined with local evolution constitutes a meaningful and novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets and three model backbones, with thorough latency analysis.
- Writing Quality: ⭐⭐⭐⭐ Conceptual diagrams are clear; the analogy to cognitive science is appropriate.
- Value: ⭐⭐⭐⭐ Provides a practical solution to the memory management challenges of SLM agents.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Mem^p: Exploring Agent Procedural Memory](memp_exploring_agent_procedural_memory.md)
- [\[ACL 2026\] Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)
- [\[AAAI 2026\] AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation](../../AAAI2026/model_compression/agentodrl_a_large_language_model-based_multi-agent_system_fo.md)
- [\[ACL 2026\] ChemAmp: Amplified Chemistry Tools via Composable Agents](chemamp_amplified_chemistry_tools_via_composable_agents.md)
- [\[ACL 2026\] Enabling Agents to Communicate Entirely in Latent Space](enabling_agents_to_communicate_entirely_in_latent_space.md)

<!-- RELATED:END -->
