---
title: >-
  [Paper Note] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models
description: >-
  [ACL 2026][LLM Agent][LLM Memory] Proposes the AnchorMem memory framework, inspired by the Proust phenomenon, which decouples retrieval units (atomic facts) from generation contexts (original interactions). It connects f…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "LLM Memory"
  - "Atomic Facts"
  - "Associative Event Graph"
  - "Retrieval Augmentation"
  - "Long-term Conversation"
date: 2026-05-08
content_hash: 0eb281dd706500f5
---

# AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.17377](https://arxiv.org/abs/2604.17377)  
**Code**: [https://github.com/RayNeo-AI-2025/AnchorMem](https://github.com/RayNeo-AI-2025/AnchorMem)  
**Area**: LLM Evaluation  
**Keywords**: LLM Memory, Atomic Facts, Associative Event Graph, Retrieval Augmentation, Long-term Conversation  

## TL;DR

Proposes the AnchorMem memory framework, inspired by the Proust phenomenon, which decouples retrieval units (atomic facts) from generation contexts (original interactions). It connects fragmented memories through an associative event graph, significantly outperforming existing memory systems like A-Mem and Mem0 on the LoCoMo benchmark.

## Background & Motivation

**Background**: LLMs require memory systems to utilize historical experiences in long-term multi-turn interactions. Existing methods are categorized into generative memory systems (A-Mem, Mem0), which integrate new information through frequent rewriting, and graph indexing methods (GraphRAG, HippoRAG 2), which structure knowledge via entity-relationship graphs.

**Limitations of Prior Work**: Generative memory systems suffer from "context blurring," where frequent rewriting dilutes or loses subtle nuances of original interactions. Graph indexing methods face the "false connection" problem, where high-frequency but semantically generalized entities (e.g., "oil") establish misleading bridges, causing retrieval paths to erroneously confuse unrelated events.

**Key Challenge**: There is a fundamental tension between fine-grained retrieval (requiring precise retrieval units) and context integrity (requiring preservation of original interaction nuances). Existing methods either sacrifice context precision for retrieval efficiency or introduce noisy connections due to over-structuring.

**Goal**: To design a memory framework that achieves both precise atomic-level retrieval and preserves the integrity of the original context.

**Key Insight**: Inspired by the "Proust phenomenon" in cognitive science (where a specific sensory cue triggers complete episodic memory), the framework explicitly decouples retrieval anchors from generation contexts.

**Core Idea**: Extract atomic facts as retrieval anchors, retain original interactions as immutable contexts, and establish logical cross-memory connections via an associative event graph.

## Method

### Overall Architecture

AnchorMem is constructed as a heterogeneous graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$, where nodes include three types: interaction contexts $\mathcal{V}_C$, atomic facts $\mathcal{V}_F$, and associative events $\mathcal{V}_E$. The edge set includes mapping edges (surjection from facts to contexts) and associative edges (semantic connections between facts). The process consists of three steps: Fact-Context Construction → Associative Event Graph Construction → Memory Retrieval and Generation.

### Key Designs

1. **Fact-Context Units**:
    - **Function**: Decouples each interaction into retrievable atomic facts and immutable original contexts.
    - **Mechanism**: For each interaction content $C_i$, an LLM extracts a set of atomic facts $\mathcal{F}_i = \{f_{i,1}, ..., f_{i,n}\}$, where each fact is a concise independent statement. A surjective mapping $\mathcal{M}(f) = C_i$ is established so that any retrieved fact can immediately locate the original context. Interaction contents remain immutable without rewriting or compression.
    - **Design Motivation**: To avoid context blurring caused by frequent rewriting while attaining precise retrieval via atomic facts.

2. **Associative Event Graph**:
    - **Function**: Establishes logical cross-memory connections without relying on generalized entity bridges.
    - **Mechanism**: Aggregates semantically related atomic facts into "associative event" nodes. High-order event links bind related fact sets into shared event representations rather than using generic entities as bridges. This integrates fragmented information while avoiding false connections caused by high-frequency entities.
    - **Design Motivation**: To address the imprecision of entity-level connections in traditional graph indexing by implementing more reliable cross-memory integration via event-level aggregation.

3. **Memory Retrieval and Context Reconstruction**:
    - **Function**: Anchors the query to specific facts and events to reconstruct the full generation context.
    - **Mechanism**: During retrieval, the query is anchored to specific atomic facts and associative events to locate relevant memories. However, in the generation stage, the associated original interaction fragments and events are used to reconstruct the context. Thus, retrieval precision stems from atomic fact matching, while generation quality comes from the integrity of the original context.
    - **Design Motivation**: To pursue precision during retrieval and completeness during generation, seamlessly linked through mapping relationships.

### Loss & Training

AnchorMem is a training-free framework that does not require additional training. Fact extraction and event construction are completed via LLM prompt engineering.

## Key Experimental Results

### Main Results

Evaluated on the LoCoMo benchmark using GPT-4o-mini:

| Method | Avg F1 | Avg BLEU | Avg ACC |
|------|--------|----------|--------|
| NaiveRAG | 33.06 | 25.90 | 47.92 |
| HippoRAG 2 | 43.11 | 30.52 | 61.82 |
| Mem0 | 29.07 | 23.34 | 39.31 |
| A-Mem | 32.39 | 26.57 | 51.35 |
| LightMem | 41.71 | 32.14 | 61.95 |
| **AnchorMem** | **49.87** | **38.85** | **70.52** |
| **AnchorMem†** | **51.91** | **40.25** | **73.83** |

### Ablation Study

| Configuration | Key Advantage | Description |
|------|---------|------|
| vs Mem0 | +22.80 F1 | Avoids context loss from frequent rewriting |
| vs HippoRAG 2 | +6.76 F1 | Avoids entity-level false connections |
| vs LightMem | +8.16 F1 | Atomic facts + event graph outperforms thematic summaries |
| AnchorMem† (Enhanced) | +2.04 F1 | Further optimization of retrieval strategies |

### Key Findings

- AnchorMem outperforms baselines across all four question categories (single-hop, multi-hop, open-domain, temporal), particularly in single-hop questions where it achieves an F1 of 55.84, leading LightMem by over 11 percentage points.
- It demonstrates consistent advantages on open-source models like Qwen2.5-32B, proving model-agnosticism.
- AnchorMem achieves the fastest memory construction speed among all methods by avoiding frequent rewriting.

## Highlights & Insights

- The analogy to the Proust phenomenon is ingenious: just as the taste of a Madeleine evokes complete childhood memories, atomic facts serve as "anchors" to trigger the recall of complete interaction scenes. This cognitive science inspiration translates directly into effective technical design.
- The decoupling of retrieval units and generation contexts is the core innovation: it lets retrieval perform its function (precise matching) and generation perform its function (complete context), connected via mappings, which is more flexible than end-to-end compression.
- The associative event graph uses "events" rather than "entities" as cross-memory bridges, effectively avoiding false associations caused by common entities (e.g., names, common nouns). This design is transferable to other RAG scenarios.

## Limitations & Future Work

- The quality of atomic fact extraction depends on LLM prompt engineering; different LLMs may produce facts of varying quality.
- As interaction history grows, the number of atomic facts and event nodes increases linearly, necessitating long-term maintenance strategies.
- Currently evaluated only on the LoCoMo benchmark; verification in real-world application scenarios (e.g., personal assistants, customer service) is lacking.
- Strategies for associative event construction (how to determine which facts belong to the same event) warrant further exploration.

## Related Work & Insights

- **vs A-Mem**: A-Mem drives note linking and evolution through retrieval but loses context via frequent rewriting; AnchorMem keeps the original context immutable.
- **vs Mem0**: Mem0 performs fine-grained extraction and state updates, but information loss during updates is significant; AnchorMem's F1 is 22.8 points higher.
- **vs HippoRAG 2**: Uses entity graphs as association signals for reranking, but entity-level connections are unreliable; AnchorMem is more precise using event-level connections.

## Rating
- Novelty: ⭐⭐⭐⭐ The design idea of decoupling retrieval-generation combined with cognitive science inspiration is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across multiple models and question types.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear cognitive science motivation and standardized formal definitions.
- Value: ⭐⭐⭐⭐ Provides a practical and effective new paradigm for LLM long-term memory systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](implicitmembench_measuring_unconscious_behavioral_adaptation_in_large_language_m.md)
- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)
- [\[ACL 2026\] HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents](hela-mem_hebbian_learning_and_associative_memory_for_llm_agents.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)

</div>

<!-- RELATED:END -->
