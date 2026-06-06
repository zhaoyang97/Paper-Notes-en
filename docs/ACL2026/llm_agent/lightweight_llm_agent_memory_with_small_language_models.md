---
title: >-
  [Paper Note] Lightweight LLM Agent Memory with Small Language Models
description: >-
  [ACL 2026][LLM Agent][Agent Memory] This paper proposes LightMem, a lightweight LLM agent memory system driven by multiple specialized small language models (SLMs). By modularizing memory operations into a Controller (SL…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Agent Memory"
  - "Small Language Models"
  - "Lightweight Retrieval"
  - "Online-Offline Decoupling"
  - "Long-term Dialogue"
date: 2026-05-08
content_hash: bcb8de6bc3831c43
---

# Lightweight LLM Agent Memory with Small Language Models

**Conference**: ACL 2026
**arXiv**: [2604.07798](https://arxiv.org/abs/2604.07798)  
**Code**: None  
**Area**: LLM Agents / Memory Systems
**Keywords**: Agent Memory, Small Language Models, Lightweight Retrieval, Online-Offline Decoupling, Long-term Dialogue

## TL;DR

This paper proposes LightMem, a lightweight LLM agent memory system driven by multiple specialized small language models (SLMs). By modularizing memory operations into a Controller (SLM-1), a Selector (SLM-2), and a Writer (SLM-3), and decoupling online processing from offline consolidation, LightMem achieves an average F1 improvement of approximately 2.5 over A-MEM on the LoCoMo benchmark, while attaining a retrieval latency of 83ms and an end-to-end latency of 581ms.

## Background & Motivation

**Background**: LLM-driven agents excel at long-term dialogue, multi-step reasoning, and task interaction, but are constrained by context window limits, necessitating external memory to maintain cross-turn consistency. Existing memory systems fall into two categories: retrieval-based external memory (e.g., MemoryBank, ReadAgent), which is efficient but susceptible to retrieval noise and unstable accuracy; and LLM-driven memory operation systems (e.g., A-MEM, HiAgent), which achieve higher accuracy but incur significant cumulative latency from repeated large model invocations.

**Limitations of Prior Work**: (1) Retrieval-based methods are limited by simplistic query construction and candidate filtering, introducing retrieval noise that destabilizes answer accuracy. (2) LLM-driven methods accumulate non-trivial runtime overhead through repeated model calls during long-term interaction. (3) Existing systems lack explicit online/offline decoupling, making it difficult to optimize the trade-off between efficiency and effectiveness.

**Key Challenge**: High-frequency online memory operations demand low latency and controllability, yet improving memory accuracy typically requires stronger model reasoning capacity. Incorporating heavy abstraction and consolidation operations into the online path severely degrades response speed.

**Goal**: To design a lightweight memory system that delegates high-frequency online memory operations to specialized SLMs while deferring heavy abstraction and consolidation to offline processing, achieving efficient and accurate memory retrieval under a constrained computational budget.

**Key Insight**: Recent advances in SLMs have enabled them to reliably handle structured decision tasks—such as intent routing, query construction, and semantic filtering—where predictable behavior and low overhead are prioritized over maximal generative capacity.

**Core Idea**: Multiple specialized SLMs collaboratively handle online memory operations (query parsing, retrieval, writing), while heavy consolidation is delegated to an offline large model, achieving an optimal balance between efficiency and effectiveness.

## Method

### Overall Architecture

LightMem modularizes memory operations into online and offline paths. The online path is driven by three specialized SLMs: SLM-1 (Controller) performs intent modeling and retrieval control, transforming user input into hypothetical queries (HQ) and allocating retrieval budgets; SLM-2 (Selector) executes two-stage retrieval—coarse vector retrieval followed by semantic consistency re-ranking; SLM-3 (Writer) compresses interactions into compact MTM entries and maintains them incrementally. The offline path employs a large-context model to distill high-value MTM fragments into de-identified long-term semantic knowledge (LTM), stored as a graph-structured knowledge base.

### Key Designs

1. **Three-Tier Memory Storage (STM/MTM/LTM)**:

    - Function: Organizes memory by temporal and access characteristics, supporting full coverage from immediate context to long-term knowledge.
    - Mechanism: STM serves as working memory within the SLM context window, updated turn-by-turn without persistence. MTM is the sole carrier of personalized episodic memory, storing semantic summaries, temporal information, access statistics, and user identifiers, with a capacity limit of $|M_u^{\text{MTM}}| \leq B$ ($B=10^4$). LTM stores de-identified semantic knowledge offline-distilled from high-value MTM fragments, organized as a lightweight graph structure to support multi-hop reasoning.
    - Design Motivation: Information at different time scales requires distinct storage and retrieval strategies. User identifiers enable user-level logical isolation, balancing privacy, consistency, and scalability.

2. **Two-Stage Retrieval**:

    - Function: Retrieves the most relevant memory set $R_t$ from the memory store under a fixed Top-$K$ budget.
    - Mechanism: Stage 1 performs metadata-constrained coarse vector retrieval, returning candidate sets for each hypothetical query with a total budget of $2K$ (allocating $2K/n$ per HQ). Stage 2 uses SLM-2 to perform semantic consistency checking and relevance judgment over $|C|=2K$ candidates, compressing results to $|R_t| \leq K$. The 2:1 compression achieves: (i) stable computation with fixed candidate size, (ii) semantic refinement beyond vector similarity, and (iii) noise suppression by explicitly discarding approximately half of the candidates.
    - Design Motivation: Pure vector retrieval fails to capture fine-grained semantic consistency, yet having an SLM retrieve directly from the full store is computationally prohibitive. The two-stage design uses efficient retrieval to ensure coverage and SLM verification to ensure precision.

3. **Offline Consolidation**:

    - Function: Incrementally distills high-value MTM fragments into long-term semantic knowledge, sustaining continuous evolution of LTM.
    - Mechanism: A large-context LLM processes incremental batches (newly written or reactivated MTM entries) in the offline path, abstracting fragments into privacy-preserving knowledge candidates. Similarity search locates the nearest semantic anchors in LTM, and new candidates are incrementally inserted and linked within the local neighborhood. A confidence decay mechanism is applied to weakly supported candidates to enable natural forgetting.
    - Design Motivation: Strictly decoupling heavy abstraction operations from the online path avoids increasing online retrieval and write latency. Incremental processing rather than full reconstruction maintains computational efficiency.

### Loss & Training

SLM-2 is fine-tuned with LoRA on 2,000 constructed (Query, Subgraph, Path) samples. Other SLMs use quantized Llama-3.2-1B-Instruct (default) or Qwen2.5-1.5B-Instruct. The MTM capacity limit is $B=10^4$; when exceeded, maintenance is performed by evicting stale or low-value entries and compressing redundant content. Offline consolidation is handled by a large-context LLM and is fully decoupled from the online path.

## Key Experimental Results

### Main Results

**LoCoMo Benchmark Key Results (GPT-4o-mini as response generator)**

| Method | Single-hop F1 | Multi-hop F1 | Temporal F1 | Open-domain F1 | Adversarial F1 | Token Length |
|--------|--------------|-------------|------------|---------------|---------------|-------------|
| LoCoMo | 40.36 | 25.02 | 18.41 | 12.04 | 69.23 | 16,910 |
| MemGPT | 41.04 | 26.65 | 25.52 | 9.15 | 43.29 | 16,977 |
| A-MEM | 44.65 | 27.02 | 45.85 | 12.14 | 50.03 | 2,520 |
| LightMem | **45.81** | **28.85** | **46.28** | **13.52** | **54.57** | 1,150 |

**DialSim Benchmark Results (GPT-4o-mini)**

| Method | F1 | BLEU-1 | ROUGE-L | METEOR | SBERT |
|--------|----|--------|---------|--------|-------|
| LoCoMo | 2.55 | 3.13 | 2.75 | 1.64 | 15.76 |
| A-MEM | 3.45 | 3.37 | 3.54 | 2.05 | 19.51 |
| LightMem | **4.12** | **3.95** | **4.20** | **2.48** | **23.40** |

### Ablation Study

**DialSim Ablation (Llama-3.2-1B)**

| Configuration | F1 | SBERT |
|---------------|----|-------|
| LightMem (full) | 4.12 | 23.40 |
| w/o semantic re-ranking | 3.83 | 22.82 |
| w/o HQ and retrieval routing | 3.87 | — |
| w/o MTM | 3.75 | — |
| w/o offline consolidation | 3.96 | — |
| w/o graph structure | — | 22.82 |

**Latency Analysis (GPT-4o-mini)**

| Method | Retrieval P50 (ms) | Retrieval P95 (ms) | End-to-End P50 (ms) | End-to-End P95 (ms) |
|--------|-------------------|-------------------|--------------------|--------------------|
| A-MEM | 856 | 1583 | 914 | 3682 |
| MemGPT | 143 | 451 | 2087 | 3451 |
| LightMem | **83** | **167** | **581** | **1325** |

### Key Findings

- LightMem consistently outperforms baselines across all model scales (from GPT-4o to Llama-3.2-1B), demonstrating that its gains are not dependent on a specific backbone model.
- Compared to A-MEM, LightMem reduces retrieval latency by 10× (856ms → 83ms P50) and end-to-end latency by approximately 36%.
- LightMem surpasses full-context methods using 16K+ tokens while consuming only ~1K tokens of effective context, substantially reducing inference cost.
- As MTM grows to 10,000 entries, LightMem maintains stable performance due to Stage 2 semantic filtering, whereas pure vector retrieval F1 degrades from 3.95 to 3.83.
- Error-injection stress tests reveal that SLM-2 semantic re-ranking is the most critical component, as its removal causes the largest performance drop.

## Highlights & Insights

- The principle of "letting the right-sized model do the right job" is well embodied in this memory system—SLMs handle high-frequency structured tasks while large models handle low-frequency heavy tasks.
- The 2:1 compression strategy of two-stage retrieval is elegant and effective, using a fixed candidate size to ensure computational stability while suppressing retrieval noise through semantic verification.
- The graph-structured LTM supports multi-hop reasoning and cross-user knowledge sharing while protecting privacy through de-identification.

## Limitations & Future Work

- SLM-2 requires fine-tuning on constructed data; its generalization to new domains warrants further investigation.
- Offline consolidation relies on a large-context LLM, which may be infeasible in fully edge-deployed scenarios.
- The specific effectiveness of LTM graph structure maintenance and the natural forgetting mechanism lack detailed analysis.
- Evaluation is conducted on only two dialogue benchmarks; applicability to more complex agent tasks (e.g., tool use, multi-step planning) remains to be validated.

## Related Work & Insights

- **vs. A-MEM**: A-MEM constructs a self-organizing memory network via LLM-driven notes and automatic linking but does not emphasize online/offline decoupling. LightMem replaces online LLM calls with SLMs, reducing latency by 10×.
- **vs. MemGPT**: MemGPT treats the context window as virtual memory with paging but relies on long-context replay (~16K tokens). LightMem achieves superior performance with only ~1K tokens.
- **vs. MemoryBank/ReadAgent**: These pure retrieval methods are substantially weaker than LightMem across all categories, especially on multi-hop and temporal reasoning tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The SLM-driven modular memory system and online/offline decoupling represent a meaningful architectural innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6 backbone models, 5 baselines, detailed ablations, latency analysis, and stress tests—highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and technical details are sufficient, though some notation definitions are scattered.
- Value: ⭐⭐⭐⭐ Provides a practical and efficient memory solution for long-term dialogue agents; the SLM-driven paradigm has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ACL 2026\] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)
- [\[ACL 2026\] Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)

</div>

<!-- RELATED:END -->
