---
title: >-
  [Paper Note] Lightweight LLM Agent Memory with Small Language Models
description: >-
  [ACL 2026][LLM Agent][Agent Memory] This paper proposes LightMem, a lightweight LLM agent memory system driven by multiple specialized Small Language Models (SLMs). By modularizing memory operations into a Controller (SL…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Agent Memory"
  - "Small Language Models"
  - "Lightweight Retrieval"
  - "Online-Offline Decoupling"
  - "Long-term Dialogue"
date: 2026-05-08
content_hash: 34c2be3cb21a0cec
---

# Lightweight LLM Agent Memory with Small Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.07798](https://arxiv.org/abs/2604.07798)  
**Code**: None  
**Area**: LLM Agent / Memory Systems  
**Keywords**: Agent Memory, Small Language Models, Lightweight Retrieval, Online-Offline Decoupling, Long-term Dialogue

## TL;DR

This paper proposes LightMem, a lightweight LLM agent memory system driven by multiple specialized Small Language Models (SLMs). By modularizing memory operations into a Controller (SLM-1), a Selector (SLM-2), and a Writer (SLM-3), and decoupling online processing from offline consolidation, it achieves an average F1 improvement of approximately 2.5 on the LoCoMo benchmark (compared to A-MEM), while attaining an 83ms retrieval latency and 581ms end-to-end latency.

## Background & Motivation

**Background**: LLM-driven agents excel in long-term dialogues, multi-step reasoning, and task interactions but are limited by context windows, necessitating external memory to maintain cross-turn consistency. Existing memory systems can be categorized into two types: retrieval-based external memory (e.g., MemoryBank, ReadAgent), which is efficient but suffers from retrieval noise and unstable accuracy; and LLM-driven memory operations (e.g., A-MEM, HiAgent), which offer higher accuracy but accumulate significant latency due to repeated large model calls.

**Limitations of Prior Work**: (1) Retrieval-based methods are limited by the simplicity of query construction and candidate filtering, introducing retrieval noise that leads to unstable response accuracy; (2) LLM-driven methods implement memory operations through repeated model calls in long-term interactions, accumulating non-trivial runtime overhead; (3) Existing systems lack explicit decoupling of online/offline processes, making it difficult to optimize the trade-off between efficiency and effectiveness.

**Key Challenge**: High-frequency online memory operations require low latency and controllability, yet improving memory accuracy typically demands stronger model reasoning capabilities; mixing heavy abstraction and consolidation operations into the online path severely slows down response speeds.

**Goal**: Design a lightweight memory system that delegates high-frequency online memory operations to specialized SLMs and defers heavy abstraction and consolidation to offline processing, achieving efficient and accurate memory invocation within a limited computational budget.

**Key Insight**: Recent advances in SLMs enable them to reliably handle structured decision tasks (such as intent routing, query construction, and semantic filtering). These tasks emphasize predictable behavior and low overhead rather than maximizing generative capacity.

**Core Idea**: Orchestrate multiple specialized SLMs to cooperatively handle online memory operations (query parsing, retrieval, writing), while delegating heavy consolidation to an offline large model to achieve an optimal balance between efficiency and effectiveness.

## Method

### Overall Architecture

LightMem modularizes memory operations into online and offline paths. The online path is driven by three specialized SLMs: SLM-1 (Controller) handles intent modeling and retrieval control, converting user input into Hypothetical Queries (HQ) and allocating retrieval budgets; SLM-2 (Selector) performs two-stage retrieval—vector-based coarse retrieval followed by semantic consistency reranking; SLM-3 (Writer) compresses interactions into compact MTM entries and maintains them incrementally. The offline path utilizes a large-context model to distill high-value MTM fragments into de-identified Long-Term semantic knowledge (LTM), stored as a graph-structured knowledge base.

### Key Designs

1.  **Three-Tier Memory Storage (STM/MTM/LTM)**:
    - **Function**: Organizes memory by time and access characteristics, supporting full coverage from immediate context to long-term knowledge.
    - **Mechanism**: STM is working memory in the SLM context window, updated per turn without persistence; MTM is the sole carrier of personalized episodic memory, storing semantic summaries, temporal information, access statistics, and user identifiers, with a capacity limit $|M_u^{\text{MTM}}| \leq B$ ($B=10^4$); LTM stores de-identified semantic knowledge distilled offline from high-value MTM fragments, organized in a lightweight graph structure to support multi-hop reasoning.
    - **Design Motivation**: Information at different timescales requires different storage and retrieval strategies; user identifiers achieve user-level logical isolation, balancing privacy, consistency, and scalability.

2.  **Two-Stage Retrieval**:
    - **Function**: Retrieves the most relevant memory set $R_t$ from the memory bank under a fixed Top-$K$ budget.
    - **Mechanism**: Stage 1 uses metadata-constrained vector coarse retrieval to return candidate sets for each hypothetical query, with a total budget of $2K$ (allocating $2K/n$ per HQ); Stage 2 employs SLM-2 to perform semantic consistency checks and relevance judgments on $|C|=2K$ candidates, compressing them into $|R_t| \leq K$ final results. This 2-to-1 compression achieves: (i) stable computation with fixed candidate sizes, (ii) semantic refinement beyond vector similarity, and (iii) noise suppression by explicitly discarding approximately half of the candidates.
    - **Design Motivation**: Pure vector retrieval struggles to capture fine-grained semantic consistency, but direct retrieval by SLMs from the entire database is computationally prohibitive; the two-stage design ensures coverage via efficient retrieval and precision via SLM verification.

3.  **Offline Consolidation**:
    - **Function**: Incrementally distills high-value MTM fragments into long-term semantic knowledge, maintaining the continuous evolution of the LTM.
    - **Mechanism**: A large-context LLM processes incremental batches (newly written or reactivated MTM entries) in the offline path, abstracting fragments into privacy-preserving knowledge candidates. It locates the nearest semantic anchors in the LTM via similarity search and incrementally inserts and links them within local neighborhoods. A confidence decay mechanism is applied to weakly supported candidates to enable natural forgetting.
    - **Design Motivation**: Strictly decouples heavy abstraction operations from the online path to avoid increasing online retrieval and writing latency; uses incremental processing instead of rebuilding from scratch to maintain computational efficiency.

### Loss & Training

SLM-2 is fine-tuned using LoRA on 2,000 constructed (Query, Subgraph, Path) samples. Other SLMs use quantized deployments of Llama-3.2-1B-Instruct (default) or Qwen2.5-1.5B-Instruct. The MTM capacity limit is $B=10^4$; when exceeded, maintenance is performed by evicting stale/low-value entries and compressing redundant content. Offline consolidation is handled by a large-context LLM, fully decoupled from the online path.

## Key Experimental Results

### Main Results

**Key Results on LoCoMo Benchmark (GPT-4o-mini as response generator)**

| Method | Single-hop F1 | Multi-hop F1 | Temporal F1 | Open-domain F1 | Adversarial F1 | Token Length |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LoCoMo | 40.36 | 25.02 | 18.41 | 12.04 | 69.23 | 16,910 |
| MemGPT | 41.04 | 26.65 | 25.52 | 9.15 | 43.29 | 16,977 |
| A-MEM | 44.65 | 27.02 | 45.85 | 12.14 | 50.03 | 2,520 |
| LightMem | **45.81** | **28.85** | **46.28** | **13.52** | **54.57** | 1,150 |

**Results on DialSim Benchmark (GPT-4o-mini)**

| Method | F1 | BLEU-1 | ROUGE-L | METEOR | SBERT |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LoCoMo | 2.55 | 3.13 | 2.75 | 1.64 | 15.76 |
| A-MEM | 3.45 | 3.37 | 3.54 | 2.05 | 19.51 |
| LightMem | **4.12** | **3.95** | **4.20** | **2.48** | **23.40** |

### Ablation Study

**DialSim Ablation (Llama-3.2-1B)**

| Configuration | F1 | SBERT |
| :--- | :--- | :--- |
| LightMem (Full) | 4.12 | 23.40 |
| w/o Semantic Reranking | 3.83 | 22.82 |
| w/o HQ and Retrieval Routing | 3.87 | - |
| w/o MTM | 3.75 | - |
| w/o Offline Consolidation | 3.96 | - |
| w/o Graph Structure | - | 22.82 |

**Latency Analysis (GPT-4o-mini)**

| Method | Retrieval Latency P50 (ms) | Retrieval Latency P95 (ms) | End-to-End P50 (ms) | End-to-End P95 (ms) |
| :--- | :--- | :--- | :--- | :--- |
| A-MEM | 856 | 1583 | 914 | 3682 |
| MemGPT | 143 | 451 | 2087 | 3451 |
| LightMem | **83** | **167** | **581** | **1325** |

### Key Findings

- LightMem consistently outperforms baselines across all model scales (from GPT-4o to Llama-3.2-1B), proving that its gains do not depend on a specific backbone model.
- Compared to A-MEM, LightMem reduces retrieval latency by 10x (856ms $\rightarrow$ 83ms P50) and end-to-end latency by approximately 36%.
- LightMem surpasses full-context methods using 16K+ tokens by using an effective context of only about 1K tokens, significantly reducing inference costs.
- When MTM grows to 10,000 entries, LightMem maintains stable performance due to Stage 2 semantic filtering, whereas pure vector retrieval F1 drops from 3.95 to 3.83.
- Error injection stress tests show that SLM-2 semantic reranking is the most critical component; its removal leads to the largest performance decline.

## Highlights & Insights

- The philosophy of "using the right-sized model for the right task" is well-reflected in the memory system—SLMs handle high-frequency structured tasks, while large models handle low-frequency heavy tasks.
- The 2:1 compression strategy in two-stage retrieval is simple and effective, ensuring computational stability with fixed candidate sizes while suppressing retrieval noise through semantic verification.
- The graph structure design of the LTM supports multi-hop reasoning and cross-user knowledge sharing while protecting privacy through de-identification.

## Limitations & Future Work

- SLM-2 requires fine-tuning on constructed data; its generalization capability to new domains needs further validation.
- Offline consolidation depends on large-context LLMs, which may not be feasible in complete edge deployment scenarios.
- The specific effects of the LTM graph structure maintenance and natural forgetting mechanism lack detailed analysis.
- Evaluation was limited to two dialogue benchmarks; applicability to more complex agent tasks (e.g., tool use, multi-step planning) remains to be verified.

## Related Work & Insights

- **vs A-MEM**: A-MEM builds self-organizing memory networks through LLM-driven notes and auto-linking but does not emphasize online/offline decoupling; LightMem replaces online LLM calls with SLMs, reducing latency by 10x.
- **vs MemGPT**: MemGPT treats the context window as virtual memory for paging but relies on long-context replay (~16K tokens); LightMem achieves better performance with only ~1K tokens.
- **vs MemoryBank/ReadAgent**: These pure retrieval methods are significantly weaker than LightMem across all categories, especially in multi-hop and temporal reasoning tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ SLM-driven modular memory systems and online/offline decoupling represent meaningful architectural innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering 6 backbone models, 5 baselines, detailed ablations, latency analysis, and stress testing.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and sufficient technical details, although some notation definitions are somewhat dispersed.
- Value: ⭐⭐⭐⭐ Provides a practical and efficient memory solution for long-term dialogue agents; the SLM-driven approach has broad application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ACL 2026\] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)
- [\[ACL 2026\] Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)
- [\[ACL 2026\] Polaris: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair](polaris_a_gödel_agent_framework_for_small_language_models_through_experience-abs.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)

</div>

<!-- RELATED:END -->
